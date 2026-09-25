"""
[파일명]: src/cs_chatbot/evaluate_rag.py
[역할]: validation_canonical_981.csv 데이터셋으로 RAG 파이프라인 성능 평가 및 결과 저장
[실행]: uv run python src/cs_chatbot/evaluate_rag.py
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 앞서 구현한 RAG 답변 생성 함수 import
from cs_chatbot.rag_chain import get_cs_answer

# ==========================================
# 1. 파일 경로 설정
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
VAL_CSV_PATH = BASE_DIR / "data" / "csv" / "validation_canonical_981.csv"
OUTPUT_CSV_PATH = BASE_DIR / "rag_base_val_results.csv"

def run_evaluation(sample_n: int = None):
    """
    Validation 데이터셋을 평가하고 CSV로 저장하는 함수
    :param sample_n: 테스트용 샘플 개수 (None이면 981개 전체 실행)
    """
    print("=" * 60)
    print(f"1. 검증 데이터셋 로드: {VAL_CSV_PATH.name}")
    print("=" * 60)
    
    val_df = pd.read_csv(VAL_CSV_PATH)
    total_count = len(val_df)
    
    # 초기에 5~10개만 돌려볼 수 있도록 슬라이싱 지원
    if sample_n:
        val_df = val_df.head(sample_n)
        print(f"테스트 모드: 전체 {total_count}개 중 상위 {sample_n}개만 평가합니다.")
    else:
        print(f"전체 평가 모드: 총 {total_count}개 질문을 평가합니다.")

    results = []
    
    # 2. 질문별 RAG 추론 반복 루프
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="RAG 답변 생성 중"):
        query = str(row["문의 내용"]).strip()
        # 여러 정답이 보존된 JSON 컬럼 우선 사용, 없으면 '응답' 컬럼 사용
        ref_answers = row.get("reference_responses_json", row.get("응답", ""))

        try:
            pred_answer, context = get_cs_answer(query)
        except Exception as e:
            pred_answer = f"[오류 발생]: {str(e)}"
            context = ""

        results.append({
            "문의 내용": query,
            "생성된_RAG_답변": pred_answer,
            "참조_정답": ref_answers,
            "검색된_참고_문서": context
        })

    # 3. 평가 결과 CSV 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    
    print("\n" + "=" * 60)
    print(f"평가 완료! 결과 파일이 저장되었습니다.")
    print(f"저장 경로: {OUTPUT_CSV_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    # 처음 실행 시에는 10개만 돌려서 포맷과 동작을 먼저 확인하세요.
    run_evaluation(sample_n=10)
    
    # 10개가 정상 작동하는 것을 확인한 후, 아래 주석을 풀고 전체를 실행하시면 됩니다.
    # run_evaluation(sample_n=None)