# api/main.py
"""
[FastAPI 백엔드 서버 엔드포인트]
프론트엔드(웹 뷰)와 AI 코어 로직(rag_chain.py)을 연결해 주는 통신 서버입니다.
팀원 연동 가이드:
  - Base URL: http://[서버IP]:8000
  - 데이터 통신 방식: JSON (REST API)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import sys
import os

# src 폴더의 모듈을 인식하도록 시스템 환경 변수에 루트 경로를 등록합니다.
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.cs_chatbot.rag_chain import get_cs_answer, initialize_pipeline

# FastAPI 애플리케이션 정의
app = FastAPI(
    title="쇼핑몰 CS 자동화 챗봇 백엔드 API",
    description="EXAONE 3.5 기반 RAG + Reranker 파이프라인. 외부 API 의존 없이 100% 독립 구동됩니다.",
    version="1.1.0"
)

@app.on_event("startup")
async def startup_event():
    """
    [서버 기동 이벤트] 서버가 켜질 때 단 한 번 실행됩니다.
    무거운 AI 파이프라인을 GPU에 미리 적재(Warm-up)하여 사용자 첫 질문 시의 병목 현상을 방지합니다.
    """
    print("=> 백엔드 서버 기동. AI 모델 Warm-up을 시작합니다...")
    initialize_pipeline()

# -------------------------------------------------------------------------
# [데이터 통신 규약 (Schemas)]
# 프론트엔드 팀원은 아래의 데이터 모양에 맞춰서 JSON을 쏘고 받으면 됩니다.
# -------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """
    [요청(Request) 데이터 포맷] 프론트엔드 -> 백엔드로 보낼 때 사용
    """
    query: str                # 사용자가 입력한 실제 질문 (예: "배송지 변경 어떻게 해?")
    user_id: str = "guest"    # 추후 세션 관리 및 DB 로깅을 위한 사용자 식별자

class ChatResponse(BaseModel):
    """
    [응답(Response) 데이터 포맷] 백엔드 -> 프론트엔드로 돌려줄 때 사용
    """
    answer: str               # AI가 생성한 최종 답변
    referenced_context: str   # 답변의 근거가 된 약관 원본 (웹 화면의 툴팁이나 출처 표기용으로 사용)
    processing_time: float    # 답변 생성에 걸린 시간 (웹 화면의 로딩 UI 통계용)

# -------------------------------------------------------------------------
# [API 엔드포인트 라우터]
# -------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    [핵심 통신 API] 사용자의 질문을 받아 AI 답변을 생성하여 반환합니다.
    - 프론트엔드에서는 사용자 입력 후 본 엔드포인트를 호출하고 대기(await)해야 합니다.
    - 평균 응답 시간: 5 ~ 10초 내외
    """
    try:
        start_time = time.time()
        
        # AI 코어 로직(get_cs_answer)을 호출하여 처리 결과를 받아옵니다.
        answer, context = get_cs_answer(request.query)
        
        process_time = round(time.time() - start_time, 2)
        
        # 규약(ChatResponse)에 맞춰 JSON 형태로 프론트엔드에 응답을 쏴줍니다.
        return ChatResponse(
            answer=answer,
            referenced_context=context,
            processing_time=process_time
        )
    except Exception as e:
        # AI 추론 중 VRAM 부족 등 예외 상황 발생 시 프론트엔드에 500 에러를 반환합니다.
        raise HTTPException(status_code=500, detail=f"AI 추론 중 서버 오류 발생: {str(e)}")

@app.get("/health")
async def health_check():
    """
    [서버 상태 점검 API]
    프론트엔드 화면을 띄우기 전, 현재 AI 서버가 살아있는지(Ping) 확인하는 용도입니다.
    """
    return {"status": "ok", "message": "CS 챗봇 서버가 정상 작동 중입니다. (RAG 파이프라인 대기중)"}