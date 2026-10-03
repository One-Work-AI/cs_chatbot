"""
[FastAPI 백엔드 서버 엔드포인트]
프론트엔드(웹 뷰)와 AI 코어 로직(rag_chain.py)을 연결해 주는 통신 서버입니다.
팀원 연동 가이드:
  - Base URL: http://[서버IP]:8000
  - 데이터 통신 방식: JSON (REST API)
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==============================================================================
# 1. 경로 설정 및 모듈 임포트
# ==============================================================================
# api/main.py 위치 기준 2단계 상위가 프로젝트 루트(cs_chatbot)이므로 동적 경로 추가
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.cs_chatbot.rag_chain import get_cs_answer, initialize_pipeline

# ==============================================================================
# 2. FastAPI 애플리케이션 초기화 및 CORS 보안 설정
# ==============================================================================
app = FastAPI(
    title="쇼핑몰 CS 자동화 챗봇 백엔드 API",
    description="EXAONE 3.5 기반 RAG + Reranker 파이프라인. 외부 API 의존 없이 독립 구동됩니다.",
    version="1.1.0"
)

# [핵심] CORS(Cross-Origin Resource Sharing) 허용 설정
# 팀원의 프론트엔드 서버(포트가 다름)에서 들어오는 API 요청을 브라우저가 차단하지 않도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실무 배포 시에는 ["http://localhost:3000"] 등 프론트 실제 주소로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST 등 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# ==============================================================================
# 3. 서버 기동 이벤트 (Warm-up)
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    """
    서버가 켜질 때 단 한 번 실행됩니다.
    무거운 AI 파이프라인(LLM, 임베딩, 크로스인코더 등)을 GPU에 미리 적재하여 사용자 첫 질문 시의 로딩 타임을 최소화합니다.
    """
    print("=> 백엔드 서버 기동. AI 모델 Warm-up을 시작합니다...")
    initialize_pipeline()

# ==============================================================================
# 4. 데이터 통신 규약 (Schemas)
# ==============================================================================
class ChatRequest(BaseModel):
    """[요청(Request)] 프론트엔드 -> 백엔드로 쏠 때 사용하는 JSON 규격"""
    query: str                # 사용자가 입력한 실제 질문 (예: "배송지 변경 어떻게 해?")
    user_id: str = "guest"    # 세션 관리 및 DB 로깅을 위한 사용자 식별자 (선택)

class ChatResponse(BaseModel):
    """[응답(Response)] 백엔드 -> 프론트엔드로 돌려주는 JSON 규격"""
    answer: str               # AI가 생성한 최종 챗봇 답변
    referenced_context: str   # 답변의 근거가 된 약관 원본 (프론트 UI 툴팁/출처 표기용)
    processing_time: float    # 답변 생성 소요 시간 (프론트 로딩 UI 성능 측정용)

# ==============================================================================
# 5. API 엔드포인트 라우터
# ==============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    [핵심 통신 API] 사용자의 질문을 받아 AI 답변을 생성하여 반환합니다.
    프론트엔드는 사용자 질문 입력 시 본 엔드포인트를 호출(POST)하고 응답을 대기(await)해야 합니다.
    """
    try:
        start_time = time.time()
        
        # AI 코어 로직(get_cs_answer) 호출
        answer, context = get_cs_answer(request.query)
        
        process_time = round(time.time() - start_time, 2)
        
        # 사전에 정의한 규약(ChatResponse)에 맞춰 JSON 딕셔너리 형태로 반환
        return ChatResponse(
            answer=answer,
            referenced_context=context,
            processing_time=process_time
        )
    except Exception as e:
        # GPU 메모리 부족 등 서버 내부 예외 발생 시 프론트 화면이 뻗지 않도록 500 에러 명시 반환
        raise HTTPException(status_code=500, detail=f"AI 추론 중 서버 오류 발생: {str(e)}")


@app.get("/health")
async def health_check():
    """
    [서버 상태 점검 API]
    프론트엔드 렌더링 전, 현재 백엔드 서버가 살아있는지(Ping) 찔러보는 용도로 사용합니다.
    """
    return {"status": "ok", "message": "CS 챗봇 서버가 정상 작동 중입니다. (RAG 파이프라인 대기중)"}