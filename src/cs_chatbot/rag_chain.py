# src/cs_chatbot/rag_chain.py
"""
[CS 챗봇 핵심 RAG 파이프라인]
작성 목적: 사용자 질문을 받아 약관 DB를 검색하고, EXAONE 모델을 통해 최종 답변을 생성합니다.
주요 아키텍처:
  1. 1차 검색 (Retriever): BAAI/bge-m3 (빠른 의미 기반 후보군 탐색)
  2. 2차 검증 (Reranker): BAAI/bge-reranker-v2-m3 (문맥 교차 검증을 통한 정확도 향상)
  3. 텍스트 생성 (LLM): LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct + CS 전용 LoRA 가중치
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# -------------------------------------------------------------------------
# [전역 변수 캐싱] FastAPI 서버가 구동될 때 최초 1회만 메모리에 적재하기 위함
# -------------------------------------------------------------------------
_model = None
_tokenizer = None
_reranker_tokenizer = None
_reranker_model = None
_vectorstore = None

def initialize_pipeline():
    """
    [초기화 함수] 웹 서버(FastAPI) 기동 시 호출되어 무거운 AI 모델들을 GPU에 미리 올려둡니다(Warm-up).
    사용자 요청이 들어올 때마다 모델을 로드하면 수 분이 소요되므로, 이를 방지하기 위한 필수 조치입니다.
    """
    global _model, _tokenizer, _reranker_tokenizer, _reranker_model, _vectorstore
    
    # 이미 메모리에 적재되어 있다면 중복 로드를 방지합니다.
    if _model is not None:
        return

    print("=> [System] AI 파이프라인 초기화 및 모델 GPU 적재를 시작합니다...")
    
    # [1] 임베딩 모델 및 Chroma 벡터 DB 세팅
    # 현재 파일 위치(src/cs_chatbot)를 기준으로 최상위 폴더의 chroma_db 경로를 동적으로 추적합니다.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db_path = os.path.join(base_dir, "chroma_db")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    _vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    # [2] 리랭커 모델 세팅 (Top-5 후보를 Top-3 진성 데이터로 압축하는 역할)
    _reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    _reranker_model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16
    ).to('cuda')
    _reranker_model.eval() # 추론 모드로 고정하여 불필요한 메모리 사용 방지

    # [3] LLM 텍스트 생성 모델 세팅 (EXAONE 베이스 모델 + 파인튜닝된 LoRA 어댑터 결합)
    lora_path = os.path.join(base_dir, "models", "lora_adapter")
    model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        revision="main" # 최신 transformers 라이브러리와의 충돌을 막기 위해 리비전 강제 해제
    )
    # 베이스 모델 위에 튜닝팀이 만든 CS 특화 말투 가중치(LoRA)를 덮어씌웁니다.
    _model = PeftModel.from_pretrained(base_model, lora_path)
    _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, revision="main")
    
    print("=> [System] AI 파이프라인 적재 완료. 실시간 서비스 준비가 끝났습니다.")

def get_cs_answer(query: str):
    """
    [핵심 추론 함수] 프론트엔드 API에서 직접 호출하며, 입력된 질문에 대한 최종 답변과 참조 근거를 반환합니다.
    
    Args:
        query (str): 사용자가 웹 화면에서 입력한 질문 텍스트
        
    Returns:
        tuple: (생성된 답변 텍스트, 답변의 근거가 된 약관 텍스트)
    """
    # 서버 기동 시 로드되지 않은 경우를 대비한 방어 코드
    initialize_pipeline()
    
    # -------------------------------------------------------------------------
    # [Step 1] 1차 검색 (Dense Retrieval)
    # BGE-m3 임베딩을 사용하여 벡터 DB에서 질문과 의미상 가장 가까운 약관 5개를 빠르게 뽑아옵니다.
    # -------------------------------------------------------------------------
    docs = _vectorstore.similarity_search(query, k=5)
    
    # -------------------------------------------------------------------------
    # [Step 2] 2차 검증 (Cross-Encoder Reranking)
    # 1차로 뽑힌 5개 문서와 질문의 문맥을 교차 검증하여, 환각을 유발할 수 있는 오답 문서를 걸러내고 Top-3로 압축합니다.
    # -------------------------------------------------------------------------
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        inputs = _reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to('cuda')
        scores = _reranker_model(**inputs, return_dict=True).logits.view(-1,).float()
        
    # 점수 기준으로 내림차순 정렬 후 최상위 3개 문서만 추출
    scored_docs = list(zip(docs, scores.cpu().tolist()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_3_docs = [doc for doc, score in scored_docs[:3]]
    
    # 검색된 3개의 약관을 하나의 문자열(컨텍스트)로 병합합니다. (이때 {{order_no}} 등의 플레이스홀더가 포함되어 있습니다)
    context = "\n\n".join([doc.page_content for doc in top_3_docs])
    
    # -------------------------------------------------------------------------
    # [Step 3] 프롬프트 엔지니어링 (Prompting)
    # 검색된 약관을 LLM에게 주입하며, 반드시 주어진 약관 내에서만 대답하도록 강력히 통제합니다.
    # -------------------------------------------------------------------------
    system_prompt = (
        "당신은 쇼핑몰의 전문 CS 상담원입니다. "
        "반드시 아래 제공된 [참고 약관]만을 근거로 고객의 질문에 답변하세요. "
        "만약 [참고 약관]에 고객의 질문에 대한 정확한 정보가 없다면, 절대 유추하거나 기존에 학습한 지식을 섞어 지어내지 말고 "
        "'제공된 약관에서 해당 내용을 찾을 수 없어 정확한 안내가 어렵습니다.'라고 정중하게 답변하세요. "
        "답변은 고객이 이해하기 쉽도록 명확하고 자연스러운 한국어 문장으로 작성하세요."
    )
    prompt = f"[참고 약관]\n{context}\n\n[고객 질문]\n{query}\n\n[답변]:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    input_ids = _tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    # -------------------------------------------------------------------------
    # [Step 4] 최종 답변 텍스트 생성 (LLM Generation)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        output = _model.generate(
            input_ids,
            max_new_tokens=300,       # 원래 세팅으로 롤백
            do_sample=False,          # temperature 삭제 (무조건 가장 확률 높은 단어만 선택)
            pad_token_id=_tokenizer.pad_token_id,  # 거슬리는 터미널 경고(Warning) 로그 방지
            eos_token_id=_tokenizer.eos_token_id   # 정답 생성 완료 시 칼같은 마무리를 위해 필수
        )
        
    generated_text = _tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    
    # 프론트엔드 화면 구성을 위해 '최종 텍스트'와 '참조한 약관 원본'을 동시에 반환합니다.
    return generated_text.strip(), context