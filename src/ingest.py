"""
[파일명]: src/ingest.py
[역할]: 사내 정책 PDF 문서(약관, 운영정책)를 조항 단위가 끊기지 않게 잘라서 벡터 DB(Chroma)에 저장
"""
import re
import shutil
from pathlib import Path
from tqdm import tqdm
import torch

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
POLICY_DIR = BASE_DIR / "data" / "policy"
CHROMA_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_ID = "BAAI/bge-m3"
COLLECTION_NAME = "cs_policy_kb"

def main():
    print("=" * 60)
    print("1. 정책 문서(PDF) 로드 및 텍스트 정제 시작")
    print("=" * 60)

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("-> 기존 chroma_db 폴더 초기화 완료.")

    pdf_files = sorted(list(POLICY_DIR.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"'{POLICY_DIR}' 경로에 PDF 파일이 없습니다.")

    merged_docs = []
    for pdf_path in pdf_files:
        print(f"-> 로드 및 정제 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        
        full_text = "\n".join([p.page_content for p in pages if p.page_content])
        full_text = re.sub(r"[\t \xa0]+", " ", full_text)
        full_text = re.sub(r"\n ", "\n", full_text)
        
        # 조항(제1조 등) 및 운영정책 목차(1. , 1.1 등) 앞에 빈 줄 삽입
        full_text = re.sub(r"\n(제\s*\d+\s*조)", r"\n\n\1", full_text)
        full_text = re.sub(r"\n(\d{1,2}\.(?:\d\s+|\s+)[가-힣])", r"\n\n\1", full_text)
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)
        
        merged_docs.append(Document(
            page_content=full_text.strip(),
            metadata={"source_file": pdf_path.name}
        ))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    split_docs = text_splitter.split_documents(merged_docs)
    print(f"-> 총 {len(split_docs)}개의 약관 청크로 분할 완료.")

    print("\n2. BGE-m3 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"\n3. Chroma DB 저장 시작 (경로: {CHROMA_DIR})...")
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    BATCH_SIZE = 100
    for i in tqdm(range(0, len(split_docs), BATCH_SIZE), desc="Vector DB 색인 중"):
        batch = split_docs[i : i + BATCH_SIZE]
        vector_db.add_documents(batch)

    print(f"-> Chroma DB 내 저장 완료된 총 청크 수: {vector_db._collection.count()}개")
    print("=" * 60)
    print("지식 베이스(Vector DB) 재구축 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()
