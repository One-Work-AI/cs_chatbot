"""
[파일명]: src/cs_chatbot/ingest.py
[역할]: 사내 정책 PDF 문서(약관, 운영정책)를 읽어와 벡터 DB(Chroma)에 색인(Indexing)
[실행]: uv run python src/cs_chatbot/ingest.py
"""

import os
from pathlib import Path
from tqdm import tqdm
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==========================================
# 1. 경로 및 주요 설정
# ==========================================
# 프로젝트 루트 디렉터리 기준 상대 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "policy"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
COLLECTION_NAME = "cs_policy_kb"

def main():
    print("=" * 60)
    print("1. 정책 문서(PDF) 로드 및 텍스트 분할 시작")
    print("=" * 60)

    # 색인할 PDF 파일 목록 확인
    pdf_files = list(POLICY_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"'{POLICY_DIR}' 경로에 PDF 파일이 없습니다.")

    all_raw_docs = []
    for pdf_path in pdf_files:
        print(f"-> 로드 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        raw_docs = loader.load()
        # 메타데이터에 파일명 기록 (추후 출처 표기용)
        for doc in raw_docs:
            doc.metadata["source_file"] = pdf_path.name
        all_raw_docs.extend(raw_docs)

    print(f"총 {len(all_raw_docs)}개 페이지 로드 완료.")

    # 2. 텍스트 청킹 (약관/정책 문맥 유지를 위해 오버랩 15% 적용)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # 한 청크당 500자 (한국어 기준 약 2~3개 단락)
        chunk_overlap=100,      # 문맥 단절을 방지하기 위한 75자 중복
        separators=["\n\n", "\n", "제", "조", ".", " "]
    )
    split_docs = text_splitter.split_documents(all_raw_docs)
    print(f"총 {len(split_docs)}개 청크로 분할 완료.")

    # ==========================================
    # 3. 임베딩 모델 준비 (BGE-m3)
    # ==========================================
    print("\n2. BGE-m3 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}  # 코사인 유사도 검색 최적화
    )

    # ==========================================
    # 4. Chroma DB 생성 및 배치 단위 add_documents 저장
    # ==========================================
    print(f"\n3. Chroma DB 저장 시작 (경로: {CHROMA_DIR})...")
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # 한 번에 대량 문서를 add할 때 생기는 메모리 병목 방지를 위한 배치 처리
    BATCH_SIZE = 100
    for i in tqdm(range(0, len(split_docs), BATCH_SIZE), desc="Vector DB 색인 중"):
        batch = split_docs[i : i + BATCH_SIZE]
        vector_db.add_documents(batch)

    print("=" * 60)
    print("지식 베이스(Vector DB) 인덱싱이 성공적으로 완료되었습니다!")
    print("=" * 60)

if __name__ == "__main__":
    main()