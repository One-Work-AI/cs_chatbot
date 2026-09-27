"""
[파일명]: src/evaluate_rag.py
[역할]: validation_canonical_981.csv에서 30개 질문을 균등 추출해 RAG 성능 평가 및 저장
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from rag_chain import get_cs_answer

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_CSV_PATH = BASE_DIR / "data" / "csv" / "validation_canonical_981.csv"
OUTPUT_CSV_PATH = BASE_DIR / "rag_base_val_results.csv"

def run_evaluation(sample_n: int = None):
    print("=" * 60)
    print(f"1. 검증 데이터셋 로드: {VAL_CSV_PATH.name}")
    print("=" * 60)
    
    val_df = pd.read_csv(VAL_CSV_PATH)
    total_count = len(val_df)
    
    if sample_n:
        step = max(1, total_count // sample_n)
        val_df = val_df.iloc[::step].head(sample_n).copy()
        print(f"균등 샘플링 모드: 전체 {total_count}개 중 {step}칸 간격으로 {len(val_df)}개를 골고루 평가합니다.")
    else:
        print(f"전체 평가 모드: 총 {total_count}개 질문을 평가합니다.")

    results = []
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="RAG 답변 생성 중"):
        query = str(row["문의 내용"]).strip()
        ref_answers = row.get("reference_responses_json", row.get("응답", ""))

        try:
            pred_answer, context = get_cs_answer(query)
        except Exception as e:
            pred_answer = f"[오류 발생]: {str(e)}"
            context = ""

        results.append({
            "원본_행번호": idx,
            "카테고리": row.get("카테고리", ""),
            "의도": row.get("의도", ""),
            "문의 내용": query,
            "생성된_RAG_답변": pred_answer,
            "참조_정답": ref_answers,
            "검색된_참고_문서": context
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n평가 완료! 저장 경로: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_evaluation(sample_n=30)
