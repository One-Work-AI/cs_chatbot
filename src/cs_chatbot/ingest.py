# src/cs_chatbot/ingest.py
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def build_vector_db():
    """
    [벡터 DB 구축 파이프라인]
    data/csv 폴더에 있는 약관 원본 데이터를 읽어 BGE-m3 임베딩 후 ChromaDB에 저장합니다.
    """
    # 1. 파일 위치가 변경되었으므로, 현재 위치(src/cs_chatbot)에서 3단계 위로 올라가 루트 경로를 잡습니다.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 2. 바뀐 폴더 구조에 맞춘 절대 경로 맵핑
    csv_path = os.path.join(base_dir, "data", "csv", "cs_data.csv")
    db_path = os.path.join(base_dir, "chroma_db")
    
    print(f"=> [데이터 로드] {csv_path} 파일을 읽어옵니다.")
    
    if not os.path.exists(csv_path):
        print(f"[오류] 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요: {csv_path}")
        return

    # 3. CSV 데이터 로드 (실제 약관 내용이 들어있는 컬럼명에 맞춰 수정 필요할 수 있음)
    df = pd.read_csv(csv_path)
    
    documents = []
    for _, row in df.iterrows():
        # 데이터프레임의 첫 번째 컬럼을 텍스트로 사용한다고 가정. (필요시 컬럼명 지정: row['content'])
        content = str(row.iloc[0]) 
        doc = Document(page_content=content, metadata={"source": "cs_policy_data"})
        documents.append(doc)
        
    print(f"=> [임베딩] 총 {len(documents)}개의 청크를 BGE-m3 모델로 임베딩합니다.")
    
    # 4. 임베딩 모델 세팅 (rag_chain.py와 동일한 모델 사용 확인 완료)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 5. Chroma DB 생성
    print("=> [DB 생성] Chroma DB에 벡터 데이터를 저장 중입니다...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    print(f"=> [완료] 벡터 DB 구축이 완료되었습니다. 저장 경로: {db_path}")

if __name__ == "__main__":
    build_vector_db()