"""
[파일명]: src/rag_chain.py
[역할]: 쇼핑몰 CS 챗봇의 핵심 RAG 파이프라인 모듈
        1) 1차 검색 (Bi-Encoder): BAAI/bge-m3 임베딩으로 Chroma DB에서 후보 약관 Top-5 검색
        2) 2차 재정렬 (Cross-Encoder): BAAI/bge-reranker-v2-m3 리랭커로 정밀 채점 후 상위 Top-3 약관 확정
        3) 프롬프트 조립: 팀 공통 시스템 프롬프트(10대 규칙) + 검색된 약관(Top-3) + 고객 문의 결합
        4) 답변 생성 및 후처리: EXAONE-3.5-7.8B-Instruct(4-bit 양자화) 생성 후 내부 참조 기호 정규식 제거
        5) 안전장치: 코랩 런타임 끊김 대비 50건 생성마다 GitHub 자동 커밋/푸시 수행
"""
import re
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
# [호환성 패치] 최신 transformers 버전과 EXAONE 모델의 어텐션 마스크 인자명 충돌 방지
# =====================================================================
_orig_create_causal_mask = masking_utils.create_causal_mask
_ccm_params = inspect.signature(_orig_create_causal_mask).parameters

def _patched_create_causal_mask(*args, **kwargs):
    if "input_embeds" in kwargs and "inputs_embeds" in _ccm_params:
        kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _ccm_params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in _ccm_params}
    return _orig_create_causal_mask(*args, **kwargs)

masking_utils.create_causal_mask = _patched_create_causal_mask

# =====================================================================
# 1. 경로 및 사용 모델 ID 설정
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"  # 생성용 LLM
EMBEDDING_MODEL_ID = "BAAI/bge-m3"                 # 1차 검색용 임베딩 모델
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"      # 2차 재정렬용 리랭커 모델

# =====================================================================
# 2. 임베딩 모델, Chroma DB(Top-5 검색기), 리랭커(Top-3 재정렬기) 로드
# =====================================================================
print("-> [RAG] 1차 임베딩 모델(BAAI/bge-m3) 및 Chroma DB 연결 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# Chroma DB 내 실제 청크(41개)가 저장된 컬렉션 자동 연결
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
active_collection = "cs_policy_kb"
for col in client.list_collections():
    col_name = col.name if hasattr(col, "name") else str(col)
    if client.get_collection(col_name).count() > 0:
        active_collection = col_name
        break

policy_db = Chroma(
    collection_name=active_collection,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)
print(f"-> [RAG] 연결된 컬렉션: '{active_collection}' (저장된 청크 수: {policy_db._collection.count()}개)")

# [핵심 1] 리랭커가 재정렬할 수 있도록 1차 후보군을 5개(k=5)로 넉넉하게 검색
policy_retriever = policy_db.as_retriever(search_kwargs={"k": 5})

# [핵심 2] 질문과 후보 문서의 문맥을 직접 비교하는 Cross-Encoder 리랭커 로드
print(f"-> [Reranker] 2차 재정렬 모델({RERANKER_MODEL_ID}) 로드 중...")
reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, device=device)
print("-> [Reranker] 준비 완료! (Top-5 후보 검색 -> Top-3 정밀 재정렬)")

# =====================================================================
# 3. EXAONE-3.5-7.8B 모델 4-bit 양자화 로드 (튜닝 어댑터 결합 가능 지점)
# =====================================================================
print("-> [LLM] EXAONE-3.5-7.8B 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# [참고: 튜닝팀 LoRA 어댑터 결합 시 아래 3줄의 주석을 해제하여 연결]
# from peft import PeftModel
# ADAPTER_PATH = str(BASE_DIR / "models" / "lora_adapter")
# model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()
print("-> [LLM] 모델 준비 완료!")

# =====================================================================
# 4. 팀 공통 시스템 프롬프트 (환각 방지 및 정책 우선 10대 원칙)
# =====================================================================
SYSTEM_PROMPT = """
당신은 쇼핑몰 고객센터 AI 상담원입니다.

아래 규칙을 반드시 따르세요.

1. 제공된 정책 문서를 가장 우선적인 근거로 사용하세요.
2. 상담 사례는 답변 표현과 문의 유형을 참고하는 보조 자료로만 사용하세요.
3. 정책 문서와 상담 사례가 충돌하면 반드시 정책 문서를 따르세요.

4. 답변에 포함하는 정책, 조건, 기간, 금액, 절차, 필요 정보는
반드시 제공된 정책 문서에서 확인할 수 있는 내용이어야 합니다.

5. 일반적인 쇼핑몰 지식이나 상식을 이용해
정책 문서에 없는 절차, 서류, 조건을 추가하지 마세요.

6. 정책 문서에 명시되지 않은 영수증, 증빙서류, 신청 양식,
개인정보 등의 제출을 고객에게 임의로 요구하지 마세요.

7. 여러 정책 문서가 고객 문의와 관련되어 있다면
하나의 문서만 사용하지 말고 관련된 정책들을 함께 검토하여 답변하세요.

8. 고객이 구체적인 상황을 제공하지 않아 정확한 판단이 어렵다면,
정책에서 확인되는 일반적인 조건까지만 안내하고
추가 정보가 필요하다는 점을 설명하세요.

9. 고객 문의와 직접 관련된 내용만 답변하세요.

10. 고객에게 자연스럽고 간결한 한국어 존댓말로 답변하세요.
다른 언어의 단어나 표현을 섞지 마세요.
""".strip()


def retrieve_and_rerank(query: str, top_k: int = 3) -> list:
    """
    고객 질문(query)에 대해 1차로 Top-5 문서를 검색한 뒤,
    리랭커(CrossEncoder)로 [질문, 문서] 쌍의 관련도 점수를 매겨 상위 Top-3 문서만 반환합니다.
    """
    cand_docs = policy_retriever.invoke(query)
    if not cand_docs:
        return []

    # [질문, 후보 문서 본문] 쌍 리스트 생성 및 리랭커 점수 예측
    pairs = [[query, doc.page_content] for doc in cand_docs]
    scores = reranker.predict(pairs)

    # 점수가 높은 순으로 내림차순 정렬 후 상위 top_k(3개)만 추출
    scored_docs = sorted(zip(cand_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


def format_policy_docs(docs: list) -> str:
    """
    검색된 Document 객체 리스트의 메타데이터(출처 파일명, 목차, 조항, 페이지)와 본문을
    LLM이 읽기 쉬운 '[문서 N: 출처 | 조항 | 페이지]' 형태의 문자열로 포맷팅합니다.
    """
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        raw_src = meta.get("source_file") or meta.get("source") or "정책문서"
        src = Path(str(raw_src)).name

        header_parts = [src]
        if meta.get("section"):
            header_parts.append(str(meta["section"]))
        if meta.get("subsection"):
            header_parts.append(str(meta["subsection"]))
        if meta.get("article"):
            header_parts.append(str(meta["article"]))

        if meta.get("page_start") and meta.get("page_end"):
            header_parts.append(f"p.{meta['page_start']}~{meta['page_end']}")
        elif meta.get("page") is not None:
            header_parts.append(f"p.{int(meta['page']) + 1}")

        header = " | ".join(header_parts)
        blocks.append(f"[문서 {i}: {header}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


# 코랩 장시간 평가 시 50건 단위 GitHub 자동 백업을 위한 카운터
_eval_counter = 0


def get_cs_answer(query: str, case_context: str | None = None):
    """
    단일 고객 문의(query)를 입력받아:
    1) 리랭커 기반 Top-3 약관 검색 -> 2) 프롬프트 구성 -> 3) EXAONE 추론 -> 4) 정규식 클리닝을 거쳐
    최종 (생성된 답변, 참조한 약관 원문) 튜플을 반환합니다.
    """
    global _eval_counter
    # 직전 50건이 CSV에 저장된 시점마다 자동으로 GitHub에 중간 백업 푸시
    if _eval_counter > 0 and _eval_counter % 50 == 0:
        try:
            subprocess.run(["git", "add", "rag_full_val_results.csv", "src/rag_chain.py"], cwd=str(BASE_DIR), check=False)
            subprocess.run(["git", "commit", "-m", f"chore: {_eval_counter}건 생성 자동 중간 백업"], cwd=str(BASE_DIR), check=False)
            subprocess.run(["git", "push", "origin", "main"], cwd=str(BASE_DIR), check=False)
            print(f"\n-> [자동 백업 완료] {_eval_counter}건까지의 결과가 깃허브에 푸시되었습니다.")
        except Exception as e:
            print(f"\n-> [자동 백업 경고] 중간 푸시 실패(생성은 계속 진행됨): {e}")
    _eval_counter += 1

    # 1. 리랭커를 통해 가장 관련성 높은 약관 Top-3 추출 및 텍스트 포맷팅
    retrieved_policies = retrieve_and_rerank(query, top_k=3)
    policy_context = format_policy_docs(retrieved_policies)

    if not case_context:
        case_context = "제공된 상담 사례 없음"

    # 2. 팀 공통 유저 프롬프트 구성
    user_prompt = f"""
[정책 문서]

{policy_context}

[상담 사례]

{case_context}

[고객 문의]

{query}

위 정책 문서를 우선적인 근거로 사용하여 고객 문의에 답변하세요.

관련된 정책이 여러 개라면 함께 검토하여 답변하세요.
정책 문서에서 확인할 수 없는 조건, 절차, 서류 또는 정보를
일반적인 지식으로 추가하지 마세요.

상담 사례가 정책 문서와 다를 경우 반드시 정책 문서를 따르세요.
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. 결정론적 생성(do_sample=False, max_new_tokens=300)으로 재현성 확보
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 4. [후처리] '[문서 1]', '(참고 2)' 등 내부 프롬프트 태그가 답변에 노출되지 않도록 제거 (Clean Answer 100% 보장)
    response = re.sub(r"\s*[\[\(](?:참고|문서|정책)\s*\d+[^\]\)]*[\]\)]", "", response).strip()
    return response, policy_context