"""
[파일명]: evaluation/scripts/score_rag.py
[역할]: RAG 평가 결과 CSV 파일을 읽어와 정량 평가지표 3종 계산 및 성적표 출력
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# [핵심 수정] 3단계 위로 올라가 루트(cs_chatbot)를 잡음
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# 입출력 경로를 evaluation/results 폴더 안으로 강제
FULL_CSV_PATH = RESULTS_DIR / "rag_full_val_results.csv"
SAMPLE_CSV_PATH = RESULTS_DIR / "rag_base_val_results.csv"
SCORED_CSV_PATH = RESULTS_DIR / "rag_scored_results.csv"

def parse_ref_answers(ref_raw: str) -> list:
    """참조_정답 컬럼에서 정답 문장 리스트를 추출"""
    ref_str = str(ref_raw).strip()
    try:
        parsed = json.loads(ref_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [ref_str]

def calc_char_ngram_metrics(pred: str, target: str, n: int = 2):
    """한국어 조사 차이를 보정하기 위해 음절 바이그램(2글자 묶음) 기준 F1 및 재현율(Recall) 계산"""
    pred_clean = re.sub(r"[^가-힣0-9]", "", pred)
    targ_clean = re.sub(r"[^가-힣0-9]", "", target)
    
    pred_ngrams = set([pred_clean[i:i+n] for i in range(len(pred_clean)-n+1)])
    targ_ngrams = set([targ_clean[i:i+n] for i in range(len(targ_clean)-n+1)])
    
    if not pred_ngrams or not targ_ngrams:
        return 0.0, 0.0
        
    overlap = len(pred_ngrams & targ_ngrams)
    precision = overlap / len(pred_ngrams)
    recall = overlap / len(targ_ngrams)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, recall

def main():
    target_csv = FULL_CSV_PATH if FULL_CSV_PATH.exists() else SAMPLE_CSV_PATH
    if not target_csv.exists():
        raise FileNotFoundError(f"채점할 결과 파일이 없습니다. 경로 확인 필요: {target_csv}")

    df = pd.read_csv(target_csv)
    print("=" * 75)
    print(f"1. 채점 대상 파일 로드: {target_csv.name} (총 {len(df)}건)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-> 채점용 임베딩 모델(BAAI/bge-m3) 로드 중 (장치: {device})...")
    eval_model = SentenceTransformer("BAAI/bge-m3", device=device)

    semantic_scores = []   # 지표 1: 참조 정답과의 의미 유사도 (0~100점)
    keyword_recalls = []   # 지표 2: 참조 정답 핵심어 포함률 (Recall, 0~100점)
    ground_scores = []     # 지표 3: 검색된 약관 문서(Context) 근거 충실도 (0~100점)

    for _, row in df.iterrows():
        pred = str(row["생성된_RAG_답변"]).strip()
        ctx = str(row.get("검색된_참고_문서", "")).strip()
        ref_list = parse_ref_answers(row["참조_정답"])

        # 1) 참조 정답과의 의미 유사도 계산
        pred_emb = eval_model.encode([pred], normalize_embeddings=True)
        ref_embs = eval_model.encode(ref_list, normalize_embeddings=True)
        sim_matrix = cosine_similarity(pred_emb, ref_embs)[0]
        best_sem_sim = float(np.max(sim_matrix)) * 100.0
        semantic_scores.append(round(best_sem_sim, 2))

        # 2) 참조 정답 핵심어 포함률 계산
        best_recall = max([calc_char_ngram_metrics(pred, r, 2)[1] for r in ref_list]) * 100.0
        keyword_recalls.append(round(best_recall, 2))

        # 3) 약관 근거 충실도 계산
        if ctx and ctx != "nan":
            ctx_emb = eval_model.encode([ctx], normalize_embeddings=True)
            ctx_sim = float(cosine_similarity(pred_emb, ctx_emb)[0][0]) * 100.0
        else:
            ctx_sim = 0.0
        ground_scores.append(round(ctx_sim, 2))

    df["의미_유사도_점수"] = semantic_scores
    df["정답_핵심어_포함률"] = keyword_recalls
    df["약관_근거_충실도"] = ground_scores

    df.to_csv(SCORED_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n" + "#" * 75)
    print(f"          [RAG 파이프라인 정량 평가 성적표 (총 {len(df)}건 기준)]")
    print("#" * 75)
    print(f"1. 평균 의미 유사도 (BGE-m3 Semantic Similarity) : {np.mean(semantic_scores):.2f}점 / 100점")
    print(f"2. 평균 정답 핵심어 포함률 (2-gram Recall)       : {np.mean(keyword_recalls):.2f}점 / 100점")
    print(f"3. 평균 약관 근거 충실도 (Context Groundedness)  : {np.mean(ground_scores):.2f}점 / 100점")
    print("-" * 75)

    if "카테고리" in df.columns:
        print("\n[카테고리별 평균 점수 요약]")
        cat_summary = df.groupby("카테고리")[["의미_유사도_점수", "정답_핵심어_포함률", "약관_근거_충실도"]].mean().round(2)
        print(cat_summary.to_string())
    print("#" * 75)
    print(f"\n-> 상세 채점 결과가 '{SCORED_CSV_PATH}' 로 저장되었습니다.")

if __name__ == "__main__":
    main()