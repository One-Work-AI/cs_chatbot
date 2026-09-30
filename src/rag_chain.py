"""
[파일명]: src/rag_chain.py
[역할]: 
  1. 1차 Bi-Encoder(BAAI/bge-m3)로 Top-5 후보 약관 검색
  2. 2차 Cross-Encoder(BAAI/bge-reranker-v2-m3)로 재정렬하여 상위 Top-3 약관 선별
  3. EXAONE-3.5-7.8B-Instruct 베이스 모델(리비전 고정) + 튜닝팀 LoRA 어댑터 결합 로드
  4. 튜닝팀 학습 규격과 동일한 프롬프트(문의 유형 intent 반영)로 답변 생성 및 소요 시간 측정
"""
import re
import time
import inspect
import subprocess
from pathlib import Path
import chromadb
import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# =====================================================================
model.eval()
model.config.use_cache = True

# =====================================================================
# 4. 튜닝팀 + RAG팀 공통 시스템 프롬프트
# =====================================================================
SYSTEM_PROMPT = """
당신은 쇼핑몰 고객센터 상담 AI입니다.

고객 문의와 문의 유형(Intent), 그리고 검색된 정책 문서를 참고하여
고객에게 전달할 최종 답변을 작성하세요.

규칙:
- 검색된 정책 문서의 내용을 우선 근거로 사용하세요.
- 정책 문서에 없는 정책, 금액, 기간, 연락처, 사실은 임의로 만들지 마세요.
- 검색된 문서만으로 확실히 답할 수 없는 내용은 추측하지 마세요.
- 한국어 존댓말을 사용하세요.
- 답변은 간결하고 명확하게 작성하세요.
- 불필요한 분석 과정은 출력하지 마세요.
- 고객에게 전달할 최종 답변만 작성하세요.
""".strip()


def retrieve_and_rerank(query: str, top_k: int = RETRIEVAL_TOP_K) -> list:
    """
    1차 Bi-Encoder(bge-m3)로 Top-5 문서를 검색한 뒤,
    2차 Cross-Encoder(bge-reranker-v2-m3) 점수가 높은 상위 top_k개 문서를 반환합니다.
    """
    cand_docs = policy_retriever.invoke(query)
    if not cand_docs:
        return []
    pairs = [[query, doc.page_content] for doc in cand_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(cand_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


def format_policy_docs(docs: list) -> str:
    """검색된 약관 문서 리스트에 출처·조항·페이지 헤더를 붙여 프롬프트용 문자열로 변환합니다."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        raw_src = meta.get("source_file") or meta.get("source") or "정책문서"
        src = Path(str(raw_src)).name

        header_parts = [src]
        for key in ["section", "subsection", "article"]:
            if meta.get(key):
                header_parts.append(str(meta[key]))

        if meta.get("page_start") and meta.get("page_end"):
            header_parts.append(f"p.{meta['page_start']}~{meta['page_end']}")
        elif meta.get("page") is not None:
            header_parts.append(f"p.{int(meta['page']) + 1}")

        header = " | ".join(header_parts)
        blocks.append(f"[정책 문서 {i}: {header}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


# 50건마다 GitHub 원격 저장소에 중간 결과를 백업하기 위한 카운터
_eval_counter = 0


def get_cs_answer(
    query: str,
    intent: str | None = None,
    case_context: str | None = None,
    return_time: bool = False,
):
    """
    고객 문의(query)와 문의 유형(intent)을 입력받아 최종 상담 답변을 생성합니다.
    - return_time=True 설정 시: (답변 텍스트, 검색된 정책 문서 텍스트, 생성 소요 시간(초)) 반환
    - return_time=False 설정 시: (답변 텍스트, 검색된 정책 문서 텍스트) 반환
    """
    global _eval_counter
    if _eval_counter > 0 and _eval_counter % 50 == 0:
        try:
            subprocess.run(
                ["git", "add", "rag_full_val_results.csv", "src/rag_chain.py"],
                cwd=str(BASE_DIR),
                check=False,
            )
            subprocess.run(
                ["git", "commit", "-m", f"chore: {_eval_counter}건 생성 자동 중간 백업"],
                cwd=str(BASE_DIR),
                check=False,
            )
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=str(BASE_DIR),
                check=False,
            )
            print(f"\n-> [자동 백업 완료] {_eval_counter}건까지의 결과가 깃허브에 푸시되었습니다.")
        except Exception as e:
            print(f"\n-> [자동 백업 경고] 중간 푸시 실패: {e}")
    _eval_counter += 1

    # 1. 리랭커 기반 Top-3 약관 검색
    retrieved_policies = retrieve_and_rerank(query, top_k=RETRIEVAL_TOP_K)
    policy_context = (
        format_policy_docs(retrieved_policies)
        if retrieved_policies
        else "검색된 관련 정책 문서가 없습니다."
    )

    # 2. 튜닝 모델 학습 포맷과 동일한 유저 프롬프트 구성 (문의 유형 포함)
    intent_str = str(intent).strip() if intent and str(intent).strip() != "nan" else "일반 문의"
    user_prompt = f"""
문의 유형: {intent_str}

고객 문의:
{query}

검색된 정책 문서:
{policy_context}
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 3. 모델 추론 및 순수 답변 생성 시간(generation_seconds) 측정
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_seconds = time.perf_counter() - t0

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 4. 내부 문서 번호 태그 노출 제거 후처리 (예: [정책 문서 1] 등 제거)
    response = re.sub(
        r"\s*[\[\(](?:참고|문서|정책\s*문서|정책)\s*\d+[^\]\)]*[\]\)]",
        "",
        response,
    ).strip()

    if return_time:
        return response, policy_context, gen_seconds
    return response, policy_context