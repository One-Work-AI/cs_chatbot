"""
[파일명]: src/cs_chatbot/rag_chain.py
[역할]: Base LLM(EXAONE-3.5-7.8B) 로드 + RAG 검색 결합 + CS 답변 생성 함수 제공
"""
import os

# 1. 용량이 넉넉한 D 드라이브에 캐시 폴더 생성 및 지정
os.environ["HF_HOME"] = "D:/hf_cache"

# 2. Windows I/O 에러를 일으키는 Xet 다운로더 비활성화
os.environ["HF_HUB_DISABLE_XET"] = "1"

from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==========================================
# 1. 경로 및 모델 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
COLLECTION_NAME = "cs_policy_kb"

# ==========================================
# 2. Chroma Retriever 로드 (k=3: 가장 유사한 3개 조항 발췌)
# ==========================================
print("-> [RAG] 임베딩 모델 및 Chroma DB 연결 중...")
# 수정 후 (VRAM 2.3GB 절약)
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_ID,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vector_db = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)
# Top-3 관련 문서를 추출하도록 설정
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 3. Base LLM (EXAONE 7.8B) 로드 (4-bit 양자화)
# ==========================================
print("-> [LLM] EXAONE-3.5-7.8B 4-bit 모델 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
print("-> [LLM] 모델 준비 완료!")

# ==========================================
# 4. 프롬프트 엔지니어링 (Base LLM의 환각 방지용 가드레일)
# ==========================================
SYSTEM_PROMPT = """당신은 고객센터의 친절하고 전문적인 AI 상담원입니다.
반드시 아래 [참고 문서]에 주어진 사실에만 근거하여 고객에게 정중한 존댓말로 답변하세요.

[필수 준수 사항]
1. [참고 문서]에 명시된 내용만 사용하여 사실에 입각해 답변하세요.
2. [참고 문서]에 고객 문의에 대한 구체적 해결책이나 정책이 없는 경우, 절대 임의로 지어내지 마세요.
3. 문서에 없는 내용인 경우 "해당 사항은 정확한 확인이 필요하여 고객센터(1:1 문의)로 접수해 주시면 확인 후 상세히 안내해 드리겠습니다."와 같이 정중히 안내하세요."""

def get_cs_answer(query: str):
    """
    고객 문의(query)를 받아 RAG 검색 후 EXAONE 답변을 반환하는 함수
    :return: (생성된 답변, 검색된 참고 문서 텍스트)
    """
    # 1. 지식 검색 (Retrieval)
    retrieved_docs = retriever.invoke(query)
    
    # 검색된 문서들을 하나의 문자열로 결합 (출처 파일명 포함)
    context_blocks = []
    for i, doc in enumerate(retrieved_docs):
        src = doc.metadata.get("source_file", "정책문서")
        context_blocks.append(f"[참고 {i+1} - 출처: {src}]\n{doc.page_content.strip()}")
    context_text = "\n\n".join(context_blocks)

    # 2. 사용자 메시지 조합
    user_message = f"""[참고 문서]
{context_text}

[고객 문의]
{query}"""

    # 3. EXAONE 공식 대화 템플릿(Chat Template) 적용
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. 답변 생성 (Generation)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,    # 고객 응대용 표준 길이
            temperature=0.2,       # 임의 창작을 억제하고 사실 위주 생성을 위해 낮게 설정
            top_p=0.9,
            repetition_penalty=1.1,# 동일 문장 반복 방지
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    # 프롬프트 입력 부분을 제외하고 순수 생성된 텍스트만 디코딩
    response = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()

    return response, context_text