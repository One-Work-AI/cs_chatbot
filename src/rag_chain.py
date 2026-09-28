"""
[파일명]: src/rag_chain.py
[역할]: Base LLM(EXAONE-3.5-7.8B) 로드 + 구조 기반 정책/상담사례 검색 + 팀 공통 프롬프트 적용
"""
import re
import inspect
from pathlib import Path
import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 0. EXAONE-3.5 & 최신 transformers 호환성 패치
_orig_create_causal_mask = masking_utils.create_causal_mask
_ccm_params = inspect.signature(_orig_create_causal_mask).parameters

def _patched_create_causal_mask(*args, **kwargs):
    if "input_embeds" in kwargs and "inputs_embeds" in _ccm_params:
        kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _ccm_params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in _ccm_params}
    return _orig_create_causal_mask(*args, **kwargs)

masking_utils.create_causal_mask = _patched_create_causal_mask

# 1. 경로 및 모델 기본 설정
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
POLICY_COLLECTION_NAME = "cs_policy_kb"
CASE_COLLECTION_NAME = "cs_case_kb"  # 상담 사례 별도 적재 시 자동 연동

# 2. 임베딩 모델 및 Chroma Retriever 로드
print("-> [RAG] 임베딩 모델 및 Chroma DB 연결 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

policy_db = Chroma(
    collection_name=POLICY_COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)
policy_retriever = policy_db.as_retriever(search_kwargs={"k": 4})

# 상담 사례 컬렉션이 존재할 경우에만 활성화 (없으면 빈 값 처리)
try:
    case_db = Chroma(
        collection_name=CASE_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    case_retriever = case_db.as_retriever(search_kwargs={"k": 2}) if case_db._collection.count() > 0 else None
except Exception:
    case_retriever = None

# 3. Base LLM 로드 (4-bit NF4 양자화)
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
model.eval()
print("-> [LLM] 모델 준비 완료!")

# 4. 팀 공통 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 쇼핑몰 고객센터 AI 상담원입니다.

아래 규칙을 반드시 따르세요.

제공된 정책 문서를 가장 우선적인 근거로 사용하세요.
상담 사례는 답변 표현과 문의 유형을 참고하는 보조 자료로만 사용하세요.
정책 문서와 상담 사례가 충돌하면 반드시 정책 문서를 따르세요.

답변에 포함하는 정책, 조건, 기간, 금액, 절차, 필요 정보는
반드시 제공된 정책 문서에서 확인할 수 있는 내용이어야 합니다.

일반적인 쇼핑몰 지식이나 상식을 이용해
정책 문서에 없는 절차, 서류, 조건을 추가하지 마세요.

정책 문서에 명시되지 않은 영수증, 증빙서류, 신청 양식,
개인정보 등의 제출을 고객에게 임의로 요구하지 마세요.

여러 정책 문서가 고객 문의와 관련되어 있다면
하나의 문서만 사용하지 말고 관련된 정책들을 함께 검토하여 답변하세요.

고객이 구체적인 상황을 제공하지 않아 정확한 판단이 어렵다면,
정책에서 확인되는 일반적인 조건까지만 안내하고
추가 정보가 필요하다는 점을 설명하세요.

고객 문의와 직접 관련된 내용만 답변하세요.

고객에게 자연스럽고 간결한 한국어 존댓말로 답변하세요.
다른 언어의 단어나 표현을 섞지 마세요.
""".strip()


def format_policy_docs(docs: list) -> str:
    """구조 기반 청킹 메타데이터(section, subsection, article, page)를 포함해 포맷팅합니다."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        src = meta.get("source_file", "정책문서")
        header_parts = [src]
        if meta.get("section"):
            header_parts.append(str(meta["section"]))
        if meta.get("subsection"):
            header_parts.append(str(meta["subsection"]))
        if meta.get("article"):
            header_parts.append(str(meta["article"]))
        if meta.get("page_start") and meta.get("page_end"):
            header_parts.append(f"p.{meta['page_start']}~{meta['page_end']}")

        header = " | ".join(header_parts)
        blocks.append(f"[문서 {i}: {header}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def get_cs_answer(query: str, case_context: str | None = None):
    # 1) 정책 문서 검색
    retrieved_policies = policy_retriever.invoke(query)
    policy_context = format_policy_docs(retrieved_policies)

    # 2) 상담 사례 검색 (직접 전달받지 않은 경우 DB 확인 후 없으면 기본 문구 적용)
    if case_context is None:
        if case_retriever is not None:
            retrieved_cases = case_retriever.invoke(query)
            case_context = "\n\n".join([d.page_content.strip() for d in retrieved_cases])
        else:
            case_context = "제공된 상담 사례 없음 (정책 문서만 참고하여 답변할 것)"

    # 3) 팀 공통 유저 프롬프트 구성
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

    # Clean Answer 100% 유지를 위한 내부 문서 번호 꼬리표 제거 후처리
    response = re.sub(r"\s*[\[\(](?:참고|문서|정책)\s*\d*[^\]\)]*[\]\)]", "", response).strip()
    return response, policy_context


if __name__ == "__main__":
    sample_q = "결제 완료 상태인데 상품 옵션을 변경하고 싶어요."
    ans, ctx = get_cs_answer(sample_q)
    print(f"[질문]: {sample_q}\n[답변]: {ans}")