"""
[파일명]: evaluation/scripts/evaluate_hit.py
[역할]: Gold Set(145건)을 기준으로 아래 두 가지 검색 방식의 적중률(Hit@1, Hit@3)을 동시 비교합니다.
        1) 단독 검색 (Bi-Encoder): BAAI/bge-m3 임베딩만으로 상위 3개(Top-3) 추출
        2) 리랭커 적용 (Cross-Encoder): BAAI/bge-m3로 상위 5개(Top-5) 후보 추출 후,
           BAAI/bge-reranker-v2-m3 모델이 질문-문서 쌍을 정밀 재채점하여 최종 상위 3개(Top-3) 확정
"""
import re
from pathlib import Path
import pandas as pd
import torch
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# [핵심 수정] 파일 위치가 evaluation/scripts/ 로 깊어졌으므로 parent를 3번 호출하여 루트를 잡음
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"              # 1차 벡터 검색용 Bi-Encoder 임베딩 모델
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"   # 2차 정밀 재정렬용 Cross-Encoder 리랭커 모델


def find_gold_csv() -> Path:
    """
    [핵심 수정] data/csv 폴더 안에서 검색 평가용 Gold Set CSV 파일을 정확히 탐색합니다.
    """
    csv_dir = BASE_DIR / "data" / "csv"
    candidates = list(csv_dir.glob("retrieval_gold*.csv"))
    
    if not candidates:
        raise FileNotFoundError(f"'retrieval_gold'로 시작하는 파일이 없습니다. 경로를 확인하세요: {csv_dir}")
    return candidates[0]


def normalize_text(text: str) -> str:
    """
    문자열 비교 시 띄어쓰기나 대소문자 차이로 인한 오판정을 막기 위해 공백을 제거하고 소문자로 통일합니다.
    (예: '제 12 조' -> '제12조')
    """
    return re.sub(r"\s+", "", str(text)).lower()


def extract_policy_keys(raw_target: str) -> list[str]:
    """
    Gold Set의 정답 정책 컬럼에서 핵심 조항 번호(예: '1.1', '제12조', '제12조의2')를 추출합니다.
    팀원마다 청킹 기준이나 메타데이터 표기법이 달라도 공정하게 채점하기 위한 전처리 함수입니다.
    """
    if pd.isna(raw_target):
        return []
    raw_str = str(raw_target).strip()
    if not raw_str or raw_str.lower() == "nan":
        return []

    # 1순위: 조항 번호 패턴('제N조', '제N조의M', 'N.M') 정규식 추출
    code_matches = re.findall(r"(?:제\s*\d+\s*조(?:\s*의\s*\d+)?|\d+\.\d+)", raw_str)
    if code_matches:
        return [normalize_text(m) for m in code_matches]

    # 2순위: 조항 번호가 없는 일반 텍스트 라벨인 경우 구분자(',', ';', '/', '|') 기준으로 분리
    parts = [p.strip() for p in re.split(r"[,;|/]", raw_str) if p.strip()]
    return [normalize_text(p) for p in parts]


def is_document_hit(doc, target_keys: list[str]) -> bool:
    """
    검색된 단일 문서 청크(메타데이터 + 본문 텍스트) 안에 정답 조항 키가 포함되어 있는지 판정합니다.
    복수 정답 키 중 하나라도 포함되면 적중(Hit)으로 인정합니다.
    """
    meta_values = " ".join(str(v) for v in (doc.metadata or {}).values())
    combined_doc_text = normalize_text(meta_values + " " + doc.page_content)

    for key in target_keys:
        if key and key in combined_doc_text:
            return True
    return False


def get_hit_rank(docs: list, target_keys: list[str]) -> int | None:
    """
    검색된 문서 리스트(Top-K)를 1순위부터 순회하며 정답 문서가 몇 순위에서 처음 등장하는지 반환합니다.
    Top-K 안에 정답이 없으면 None을 반환합니다.
    """
    for rank, doc in enumerate(docs, 1):
        if is_document_hit(doc, target_keys):
            return rank
    return None


def main(initial_k: int = 5, final_k: int = 3):
    """
    [메인 평가 로직]
    - initial_k (기본값 5): 1차 임베딩 검색으로 가져올 후보 문서 수 (Top-5)
    - final_k   (기본값 3): 리랭커 재정렬 후 최종적으로 남길 문서 수 (Top-3)
    """
    # 1. Gold Set 데이터 로드 및 질문/정답 컬럼 자동 매핑
    gold_path = find_gold_csv()
    df = pd.read_csv(gold_path)
    print("=" * 75)
    print(f"1. Gold Set 파일 로드 완료: {gold_path.name} (총 {len(df)}건)")
    print("=" * 75)

    q_col = next((c for c in ["문의 내용", "question", "query", "질문", "문의내용"] if c in df.columns), df.columns[0])
    target_candidates = [
        "gold_policy", "target_policy", "policy", "section", "article",
        "정답_정책", "기대_정책", "정책", "조항", "gold_chunk", "answer_policy"
    ]
    t_col = next((c for c in target_candidates if c in df.columns), None)
    if t_col is None:
        t_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["policy", "gold", "sec", "art", "정책", "조항", "chunk"]) and c != q_col),
            df.columns[1] if len(df.columns) > 1 else df.columns[0]
        )

    # 2. 임베딩 모델 및 Chroma Vector DB 연결
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> 실행 디바이스: {device}")
    print(f"-> 1차 임베딩 모델({EMBEDDING_MODEL_ID}) 및 Chroma DB 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # 코사인 유사도 계산을 위한 L2 정규화
    )

    # DB 내 데이터가 존재하는 컬렉션 자동 탐색 (기본값: cs_policy_kb)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    active_collection = "cs_policy_kb"
    for col in client.list_collections():
        col_name = col.name if hasattr(col, "name") else str(col)
        if client.get_collection(col_name).count() > 0:
            active_collection = col_name
            break

    vectorstore = Chroma(
        collection_name=active_collection,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    # 1차 검색기는 리랭커에 넘겨줄 후보군 5개(Top-5)를 뽑도록 설정
    retriever_top5 = vectorstore.as_retriever(search_kwargs={"k": initial_k})

    # 3. 2차 재정렬을 수행할 Cross-Encoder 리랭커 모델 로드
    print(f"-> 2차 리랭커 모델({RERANKER_MODEL_ID}) 로드 중...")
    reranker = CrossEncoder(RERANKER_MODEL_ID, max_length=512, device=device)

    # 집계 변수 초기화
    base_hit1, base_hitk = 0, 0      # 리랭커 미사용 시 적중 건수
    rerank_hit1, rerank_hitk = 0, 0  # 리랭커 적용 시 적중 건수
    valid_count = 0
    detail_rows = []

    # 4. 문항별 검색 및 리랭킹 비교 평가 수행
    for idx, row in df.iterrows():
        query = str(row[q_col]).strip()
        target_keys = extract_policy_keys(row[t_col])
        if not target_keys:
            continue  # 정답 조항 라벨이 없는 행은 평가 모수에서 제외

        valid_count += 1

        # [Step 1] BGE-m3 임베딩으로 상위 5개(Top-5) 후보 문서 검색
        cand_docs = retriever_top5.invoke(query)

        # [비교 A: 단독 검색] 리랭커 없이 상위 3개(Top-3)만 잘라서 적중 여부 확인
        base_top3 = cand_docs[:final_k]
        base_rank = get_hit_rank(base_top3, target_keys)
        if base_rank == 1:
            base_hit1 += 1
        if base_rank is not None and base_rank <= final_k:
            base_hitk += 1

        # [비교 B: 리랭커 적용] Top-5 후보 각각에 대해 [질문, 문서본문] 쌍을 구성하여 리랭커 점수 산출
        if cand_docs:
            pairs = [[query, doc.page_content] for doc in cand_docs]
            scores = reranker.predict(pairs)
            # 리랭커 점수 기준 내림차순 정렬 후 상위 3개(Top-3) 선택
            scored_docs = sorted(zip(cand_docs, scores), key=lambda x: x[1], reverse=True)
            reranked_top3 = [doc for doc, _ in scored_docs[:final_k]]
        else:
            reranked_top3 = []

        rerank_rank = get_hit_rank(reranked_top3, target_keys)
        if rerank_rank == 1:
            rerank_hit1 += 1
        if rerank_rank is not None and rerank_rank <= final_k:
            rerank_hitk += 1

        # 문항별 상세 비교 결과 저장
        detail_rows.append({
            "번호": idx + 1,
            "질문": query,
            "정답_기준": row[t_col],
            "추출된_매칭키": ", ".join(target_keys),
            f"기존_Hit@{final_k}": 1 if base_rank is not None else 0,
            "기존_적중순위": base_rank if base_rank is not None else "미적중",
            f"리랭커_Hit@{final_k}": 1 if rerank_rank is not None else 0,
            "리랭커_적중순위": rerank_rank if rerank_rank is not None else "미적중",
        })

    # 5. 상세 결과 CSV 저장 및 최종 성적표 출력
    # [핵심 수정] 결과를 evaluation/results 폴더에 저장하도록 경로 변경
    out_path = BASE_DIR / "evaluation" / "results" / "rag_gold_hit_results.csv"
    pd.DataFrame(detail_rows).to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 75)
    print(f"   [Gold Set 검색 적중률 비교 결과 (총 {valid_count}개 유효 문항)]")
    print("=" * 75)
    print(f" 1) 단독 검색 (Top-{final_k})              : Hit@1 {base_hit1/valid_count*100:6.2f}% ({base_hit1}/{valid_count}) | Hit@{final_k} {base_hitk/valid_count*100:6.2f}% ({base_hitk}/{valid_count})")
    print(f" 2) 리랭커 적용 (Top-{initial_k} -> Top-{final_k}) : Hit@1 {rerank_hit1/valid_count*100:6.2f}% ({rerank_hit1}/{valid_count}) | Hit@{final_k} {rerank_hitk/valid_count*100:6.2f}% ({rerank_hitk}/{valid_count})")
    print("=" * 75)
    print(f"-> 상세 비교 내역 저장 완료: {out_path.name}")


if __name__ == "__main__":
    main(initial_k=5, final_k=3)