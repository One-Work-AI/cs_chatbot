"""
[파일명]: src/evaluate_hit.py
[역할]: 루트 폴더의 retrieval_gold_candid... 파일을 읽어 동일한 기준으로 Hit@K(기본 Hit@3) 산출
        (3명의 청킹/메타데이터 방식이 달라도 본문+메타데이터 통합 매칭으로 공정하게 평가)
"""
import re
from pathlib import Path
import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
COLLECTION_NAME = "cs_policy_kb"


def find_gold_csv() -> Path:
    """루트 폴더 또는 data 폴더에서 retrieval_gold 파일을 자동으로 찾습니다."""
    candidates = list(BASE_DIR.glob("retrieval_gold*.csv")) + list((BASE_DIR / "data").glob("retrieval_gold*.csv"))
    if not candidates:
        raise FileNotFoundError("retrieval_gold로 시작하는 CSV 파일을 찾을 수 없습니다.")
    return candidates[0]


def normalize_text(text: str) -> str:
    """띄어쓰기 및 특수문자를 제거하여 '제 12 조'와 '제12조'를 동일하게 비교합니다."""
    return re.sub(r"\s+", "", str(text)).lower()


def extract_policy_keys(raw_target: str) -> list[str]:
    """
    정답 정책 컬럼에서 핵심 조항 식별자(예: '1.1', '1.2', '제12조' 등)와 원문 키워드를 추출합니다.
    여러 개가 적혀 있는 경우(',', ';', '/', '|' 구분) 모두 분리합니다.
    """
    if pd.isna(raw_target):
        return []
    raw_str = str(raw_target).strip()
    if not raw_str or raw_str.lower() == "nan":
        return []

    # 조항 번호 패턴(예: 1.1, 2.3, 제12조, 제12조의2) 우선 추출
    code_matches = re.findall(r"(?:제\s*\d+\s*조(?:\s*의\s*\d+)?|\d+\.\d+)", raw_str)
    if code_matches:
        return [normalize_text(m) for m in code_matches]

    # 조항 번호가 없는 텍스트 형태면 구분자로 분리 후 정규화
    parts = [p.strip() for p in re.split(r"[,;|/]", raw_str) if p.strip()]
    return [normalize_text(p) for p in parts]


def is_document_hit(doc, target_keys: list[str]) -> bool:
    """검색된 단일 Chunk(본문 + 메타데이터) 안에 정답 조항 키가 포함되어 있는지 판정합니다."""
    meta_values = " ".join(str(v) for v in (doc.metadata or {}).values())
    combined_doc_text = normalize_text(meta_values + " " + doc.page_content)

    # 정답 키 중 하나라도 검색 문서에 포함되면 적중(Hit)으로 인정
    for key in target_keys:
        if key and key in combined_doc_text:
            return True
    return False


def main(k: int = 3):
    gold_path = find_gold_csv()
    df = pd.read_csv(gold_path)
    print("=" * 70)
    print(f"1. Gold Set 파일 로드 완료: {gold_path.name} (총 {len(df)}건)")
    print(f"   컬럼 목록: {list(df.columns)}")
    print("=" * 70)

    # 질문 컬럼 및 정답 정책 컬럼 자동 매핑
    q_col = next((c for c in ["문의 내용", "question", "query", "질문", "문의내용"] if c in df.columns), df.columns[0])
    target_candidates = [
        "gold_policy", "target_policy", "policy", "section", "article",
        "정답_정책", "기대_정책", "정책", "조항", "gold_chunk", "answer_policy"
    ]
    t_col = next((c for c in target_candidates if c in df.columns), None)
    if t_col is None:
        # 이름에 policy, gold, section, article, 정책, 조항이 들어간 컬럼 탐색
        t_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["policy", "gold", "sec", "art", "정책", "조항", "chunk"]) and c != q_col),
            df.columns[1] if len(df.columns) > 1 else df.columns[0]
        )

    print(f"-> 매핑된 질문 컬럼: [{q_col}] / 정답 기준 컬럼: [{t_col}]")

    # Chroma DB 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    hits_at_1 = 0
    hits_at_k = 0
    valid_count = 0
    detail_rows = []

    for idx, row in df.iterrows():
        query = str(row[q_col]).strip()
        target_keys = extract_policy_keys(row[t_col])

        # 정답 라벨이 비어 있는 행은 평가 모수에서 제외
        if not target_keys:
            continue

        valid_count += 1
        retrieved_docs = retriever.invoke(query)

        hit_rank = None
        for rank, doc in enumerate(retrieved_docs, 1):
            if is_document_hit(doc, target_keys):
                hit_rank = rank
                break

        if hit_rank == 1:
            hits_at_1 += 1
        if hit_rank is not None and hit_rank <= k:
            hits_at_k += 1

        detail_rows.append({
            "번호": idx + 1,
            "질문": query,
            "정답_기준": row[t_col],
            "추출된_매칭키": ", ".join(target_keys),
            f"Hit@{k}_성공여부": 1 if hit_rank is not None else 0,
            "적중_순위": hit_rank if hit_rank is not None else "미적중",
        })

    hit1_rate = (hits_at_1 / valid_count * 100) if valid_count > 0 else 0.0
    hitk_rate = (hits_at_k / valid_count * 100) if valid_count > 0 else 0.0

    out_path = BASE_DIR / "rag_gold_hit_results.csv"
    pd.DataFrame(detail_rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"       [Gold Set 검색 적중률 평가 결과 (총 {valid_count}개 유효 문항)]")
    print("=" * 70)
    print(f" - Hit@1 (1순위 적중률) : {hit1_rate:.2f}% ({hits_at_1}/{valid_count}건)")
    print(f" - Hit@{k} (상위 {k}개 적중률): {hitk_rate:.2f}% ({hits_at_k}/{valid_count}건)")
    print("=" * 70)
    print(f"-> 상세 채점 내역 저장 완료: {out_path.name}")


if __name__ == "__main__":
    main(k=3)