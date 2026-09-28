"""
[파일명]: src/rag_chain.py
[역할]: Base LLM(EXAONE-3.5-7.8B) 로드 + 정책 검색(k=3) + 팀 공통 프롬프트 적용
"""
import re
import inspect
from pathlib import Path
import chromadb
import torch
import transformers.masking_utils as masking_utils
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

_orig_create_causal_mask = masking_utils.create_causal_mask
_ccm_params = inspect.signature(_orig_create_causal_mask).parameters

def _patched_create_causal_mask(*args, **kwargs):
    if "input_embeds" in kwargs and "inputs_embeds" in _ccm_params:
        kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in _ccm_params.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in _ccm_params}
    return _orig_create_causal_mask(*args, **kwargs)

masking_utils.create_causal_mask = _patched_create_causal_mask

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"

print("-> [RAG] 임베딩 모델 및 Chroma DB 연결 중...")
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
print(f"-> [RAG] 연결된 컬렉션: '{active_collection}' (저장된 청크 수: {policy_db._collection.count()}개)")
policy_retriever = policy_db.as_retriever(search_kwargs={"k": 3})

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

def format_policy_docs(docs: list) -> str:
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

def get_cs_answer(query: str, case_context: str | None = None):
    retrieved_policies = policy_retriever.invoke(query)
    policy_context = format_policy_docs(retrieved_policies)

    if not case_context:
        case_context = "제공된 상담 사례 없음"

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

    response = re.sub(r"\s*[\[\(](?:참고|문서|정책)\s*\d+[^\]\)]*[\]\)]", "", response).strip()
    return response, policy_context

if __name__ == "__main__":
    sample_q = "결제 완료 상태인데 상품 옵션을 변경하고 싶어요."
    ans, ctx = get_cs_answer(sample_q)
    print("=" * 60)
    print(f"[질문]: {sample_q}")
    print(f"[답변]: {ans}")
    print("=" * 60)
