"""
[파일명]: src/evaluate_rag.py
[역할]: validation_canonical_981.csv 981개 전체 평가 (10건마다 중간 저장 및 이어하기 지원)
"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from rag_chain import get_cs_answer

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_CSV_PATH = BASE_DIR / "data" / "csv" / "validation_canonical_981.csv"
OUTPUT_CSV_PATH = BASE_DIR / "rag_full_val_results.csv"

def run_evaluation(sample_n: int = None):
    print("=" * 65)
    print(f"1. 검증 데이터셋 로드: {VAL_CSV_PATH.name}")
    print("=" * 65)
    
    val_df = pd.read_csv(VAL_CSV_PATH)
    total_count = len(val_df)
    
    if sample_n:
        step = max(1, total_count // sample_n)
        val_df = val_df.iloc[::step].head(sample_n).copy()
        print(f"균등 샘플링 모드: 전체 {total_count}개 중 {len(val_df)}개를 평가합니다.")
    else:
        print(f"전체 평가 모드: 총 {total_count}개 질문 전체를 평가합니다.")

    # 기존에 중간 저장된 파일이 있으면 불러와서 이어하기 수행
    results = []
    completed_indices = set()
    if OUTPUT_CSV_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        valid_df = existing_df[~existing_df["생성된_RAG_답변"].astype(str).str.startswith("[오류 발생]")]
        results = valid_df.to_dict("records")
        completed_indices = set(valid_df["원본_행번호"].tolist())
        print(f"-> [이어하기 감지] 기존 완료된 {len(completed_indices)}건을 건너뛰고 이어서 진행합니다.")

    new_count = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="RAG 981개 답변 생성 중"):
        if idx in completed_indices:
            continue

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
        new_count += 1

        # 코랩 끊김 방지를 위해 10개마다 즉시 중간 저장
        if new_count % 10 == 0:
            pd.DataFrame(results).sort_values("원본_행번호").to_csv(
                OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig"
            )

    result_df = pd.DataFrame(results).sort_values("원본_행번호")
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 65)
    print(f"평가 완료! 총 {len(result_df)}개의 결과 파일이 저장되었습니다.")
    print(f"저장 경로: {OUTPUT_CSV_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    # 981개 전체 평가 실행
    run_evaluation(sample_n=None)
