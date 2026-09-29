"""
[파일명]: src/rag_chain.py
[역할]: BGE-m3(Top-5) + BGE-Reranker(Top-3) + EXAONE(Base 및 튜닝 LoRA 어댑터 자동 결합)
        + 튜닝팀 호환 프롬프트(문의 유형 intent 반영) + 문항별 생성 시간 측정
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
# [호환성 패치] 최신 transformers 버전과 EXAONE 어텐션 마스크 인자 충돌 방지
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
# 1. 경로 및 모델 설정
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
ADAPTER_DIR = BASE_DIR / "models" / "lora_adapter"  # 튜닝팀 어댑터 넣을 폴더

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# =====================================================================
# 2. 임베딩(Top-5) + Chroma DB + 리랭커(Top-3) 로드
# =====================================================================
print("-> [RAG] 1차 임베딩 모델(BAAI/bge-m3) 및 Chroma DB 연결 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

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
print(f"-> [RAG] 연결된 컬렉션: '{active_collection}' (청크 수: {policy_db._collection.count()}개)")

policy_retriever = policy_db.as_retriever(search_kwargs={"k": 5})

print(f"-> [Reranker] 2차 재정렬 모델({RERANKER_MODEL_ID}) 로드 중...")
reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, device=device)
print("-> [Reranker] 준비 완료! (Top-5 검색 -> Top-3 재정렬)")

# =====================================================================
# 3. EXAONE Base 모델 및 튜닝팀 LoRA 어댑터 로드
# =====================================================================
print("-> [LLM] EXAONE-3.5-7.8B 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    trust_remote_code=True,
)

# models/lora_adapter 폴더에 튜닝팀 어댑터 파일이 있으면 자동으로 결합
if ADAPTER_DIR.exists() and any(ADAPTER_DIR.iterdir()):
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    print(f"-> [LLM] 튜닝팀 LoRA 어댑터 결합 완료! ({ADAPTER_DIR})")
else:
    model = base_model
    print("-> [LLM] 순수 Base 모델 로드 완료 (models/lora_adapter 폴더에 어댑터 넣으면 자동 결합됨)")

model.eval()

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


def retrieve_and_rerank(query: str, top_k: int = 3) -> list:
    """Top-5 후보 문서를 검색한 뒤 리랭커 점수로 재정렬하여 상위 Top-3를 반환합니다."""
    cand_docs = policy_retriever.invoke(query)
    if not cand_docs:
        return []
    pairs = [[query, doc.page_content] for doc in cand_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(cand_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]


def format_policy_docs(docs: list) -> str:
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


_eval_counter = 0


def get_cs_answer(
    query: str,
    intent: str | None = None,
    case_context: str | None = None,
    return_time: bool = False,
):
    """
    고객 문의(query)와 문의 유형(intent)을 받아 답변과 검색된 정책 문서를 반환합니다.
    - return_time=True 설정 시 (response, policy_context, generation_seconds) 3개 값을 반환합니다.
    """
    global _eval_counter
    if _eval_counter > 0 and _eval_counter % 50 == 0:
        try:
            subprocess.run(["git", "add", "rag_full_val_results.csv", "src/rag_chain.py"], cwd=str(BASE_DIR), check=False)
            subprocess.run(["git", "commit", "-m", f"chore: {_eval_counter}건 생성 자동 중간 백업"], cwd=str(BASE_DIR), check=False)
            subprocess.run(["git", "push", "origin", "main"], cwd=str(BASE_DIR), check=False)
            print(f"\n-> [자동 백업 완료] {_eval_counter}건까지의 결과가 깃허브에 푸시되었습니다.")
        except Exception as e:
            print(f"\n-> [자동 백업 경고] 중간 푸시 실패: {e}")
    _eval_counter += 1

    # 1. 리랭커 기반 Top-3 약관 검색
    retrieved_policies = retrieve_and_rerank(query, top_k=3)
    policy_context = format_policy_docs(retrieved_policies) if retrieved_policies else "검색된 관련 정책 문서가 없습니다."

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
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. 답변 생성 및 순수 생성 시간(generation_seconds) 측정
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_seconds = time.perf_counter() - t0

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # 4. 내부 참조 기호 제거 후처리
    response = re.sub(r"\s*[\[\(](?:참고|문서|정책\s*문서|정책)\s*\d+[^\]\)]*[\]\)]", "", response).strip()

    if return_time:
        return response, policy_context, gen_seconds
    return response, policy_context