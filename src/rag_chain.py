"""
[파일명]: src/rag_chain.py
[역할]: Base LLM(EXAONE-3.5-7.8B) 로드 + RAG 검색 결합 + CS 답변 생성 함수 제공
"""
import os
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

# 1. 경로 및 모델 설정
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
COLLECTION_NAME = "cs_policy_kb"

# 2. Chroma Retriever 로드 (k=4)
print("-> [RAG] 임베딩 모델 및 Chroma DB 연결 중...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)

doc_count = vector_db._collection.count()
print(f"-> [RAG] 현재 Chroma DB에 로드된 약관 청크 수: {doc_count}개")
if doc_count == 0:
    raise RuntimeError("Chroma DB에 저장된 문서가 0개입니다! ingest.py를 먼저 실행하세요.")

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 3. Base LLM (EXAONE 7.8B) 로드 (float16 + 4-bit 양자화)
print("-> [LLM] EXAONE-3.5-7.8B 4-bit 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("-> [LLM] 모델 준비 완료!")

# 4. 시스템 프롬프트 설정
SYSTEM_PROMPT = """당신은 고객센터의 친절하고 전문적인 AI 상담원입니다.
반드시 아래 [참고 문서]에 주어진 사실에만 근거하여 고객에게 정중한 존댓말로 답변하세요.

[필수 준수 사항]
1. 마크다운 제목(###)이나 글머리 기호(-, 1.)를 쓰지 말고, 핵심 내용만 3~4문장(300자 이내)의 자연스러운 한 문단 줄글로 간결하게 답변하세요.
2. 답변 안에 '[참고 문서]', '(출처: ...)' 같은 내부 문서 표기를 절대 포함하지 마세요.
3. 배송지 변경은 '입금 대기/결제 완료' 단계에서만 가능하며, 이미 결제된 주문의 상품 품목·옵션(색상/사이즈)·수량 변경이나 일부 상품 제외는 불가능하므로 기존 주문을 취소한 후 다시 주문하도록 안내하세요.
4. 주문 취소/변경/배송 조회 문의 시 [마이페이지 > 주문내역] 경로와 주문 상태(입금대기/결제완료/상품준비중/배송중)별 기준 및 수수료·기간 수치를 정확히 포함하세요.
5. 문서에 없는 내용은 지어내지 말고 "해당 사항은 정확한 확인이 필요하여 고객센터(1:1 문의)로 접수해 주시면 확인 후 상세히 안내해 드리겠습니다."라고 안내하세요."""

def get_cs_answer(query: str):
    retrieved_docs = retriever.invoke(query)
    
    context_blocks = []
    for i, doc in enumerate(retrieved_docs):
        src = doc.metadata.get("source_file", "정책문서")
        context_blocks.append(f"[참고 {i+1} - 출처: {src}]\n{doc.page_content.strip()}")
    context_text = "\n\n".join(context_blocks)

    user_message = f"""[참고 문서]
{context_text}

[고객 문의]
{query}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()

    response = re.sub(r"\s*[\[\(]참고[^\]\)]*[\]\)]", "", response).strip()
    return response, context_text
