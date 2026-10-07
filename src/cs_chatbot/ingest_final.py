# src/cs_chatbot/ingest.py
import os
import re
import shutil

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# cs_data.csv 열: 플래그, 문의 내용, 카테고리, 의도, 응답, source_file
QUESTION_COL = "문의 내용"
ANSWER_COL = "응답"
PLACEHOLDER = re.compile(r"\{\{.*?\}\}")   # {{order_no}}, {{shipping.address}} 같은 치환자

# 답변에 치환자({{…}})가 있는 상담 예시도 검색 DB에 넣을지 여부.
#   True  : 넣음 — LLM이 "주문번호 {{orders.order_no}}는 …" 같은 답을 만들고, 백엔드가 팀 DB 값으로 채워서 고객에게 보냄
#   False : 뺌   — 백엔드 치환 없이 쓸 때 (치환자가 고객 화면에 나가는 것을 줄임)
KEEP_PLACEHOLDER_ANSWERS = True


def build_documents(df: pd.DataFrame) -> list[Document]:
    """
    [문서 만들기] 상담 데이터 한 행(질문 + 답변)을 검색용 문서 1개로 만듭니다.
      - 문서 내용 = "질문: … / 답변: …"  (예전에는 첫 번째 열 '플래그'(BIL, BIZ 등)가 들어가서 검색 결과가 쓸모없었음)
      - 답변의 치환자({{…}})는 그대로 둠: 백엔드가 팀 DB 값(주문번호·배송지 등)으로 채워서 고객에게 보냄
        (KEEP_PLACEHOLDER_ANSWERS = False 로 바꾸면 치환자가 있는 답변은 제외)
      - 답변이 같은 행은 하나만 남김: 검색 Top-K가 같은 내용으로 채워지는 것을 방지
    """
    missing = [c for c in (QUESTION_COL, ANSWER_COL) if c not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing} (현재 열: {list(df.columns)})")

    documents = []
    seen_answers = set()
    for _, row in df.iterrows():
        question = str(row[QUESTION_COL]).strip()
        answer = str(row[ANSWER_COL]).strip()
        if not answer or answer == "nan" or answer in seen_answers:
            continue
        if "{{" in answer and not KEEP_PLACEHOLDER_ANSWERS:
            continue
        seen_answers.add(answer)
        # 질문의 치환자는 검색에 방해가 되므로 지우고 공백을 정리 ("주문번호 {{order_no}} 건을" → "주문번호 건을")
        question = " ".join(PLACEHOLDER.sub("", question).split())
        content = f"질문: {question}\n답변: {answer}"
        metadata = {
            "source": "cs_policy_data",
            "category": str(row.get("카테고리", "")),
            "intent": str(row.get("의도", "")),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def build_vector_db():
    """
    [벡터 DB 구축 파이프라인]
    data/csv 폴더에 있는 상담 데이터를 읽어 BGE-m3 임베딩 후 ChromaDB에 저장합니다.
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

    # 3. CSV 데이터 로드 → 검색용 문서 만들기
    df = pd.read_csv(csv_path)
    documents = build_documents(df)
    print(f"=> [문서 생성] 전체 {len(df)}행 중 {len(documents)}개를 문서로 만들었습니다. (중복 답변 제외)")
    if not documents:
        print("[오류] 만들 문서가 없습니다. CSV 내용을 확인하세요.")
        return
    print(f"=> [문서 예시]\n{documents[0].page_content}")

    # 4. 예전 벡터 DB가 남아 있으면 지웁니다. (지우지 않으면 예전 문서와 새 문서가 섞여서 검색됨)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        print(f"=> [정리] 기존 벡터 DB를 지웠습니다: {db_path}")

    print(f"=> [임베딩] 총 {len(documents)}개의 청크를 BGE-m3 모델로 임베딩합니다.")

    # 5. 임베딩 모델 세팅 (rag_chain.py와 동일한 모델 사용 확인 완료)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 6. Chroma DB 생성
    print("=> [DB 생성] Chroma DB에 벡터 데이터를 저장 중입니다...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_path
    )

    print(f"=> [완료] 벡터 DB 구축이 완료되었습니다. 저장 경로: {db_path}")


if __name__ == "__main__":
    build_vector_db()
