"""
[파일명]: src/evaluate_rag.py
[역할]: final_test_canonical_981.csv 981개 최종 본 평가
        (50건마다 저장 및 GitHub 자동 커밋/푸시 + 이어하기 지원)
"""
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from rag_chain import get_cs_answer

BASE_DIR = Path(__file__).resolve().parent.parent
VAL_CSV_PATH = BASE_DIR / "data" / "csv" / "final_test_canonical_981.csv"

OUTPUT_CSV_PATH = BASE_DIR / "rag_full_val_results.csv"
FINAL_CSV_PATH = BASE_DIR / "rag_full_final_results.csv"
OLD_VAL_BACKUP = BASE_DIR / "rag_full_val_results_validation_backup.csv"


def git_auto_push(commit_msg: str):
    """중간 결과 CSV 파일을 GitHub 원격 저장소에 자동 커밋 및 푸시합니다."""
    try:
        subprocess.run(
            ["git", "add", "-f", "rag_full_val_results.csv", "rag_full_final_results.csv"],
            cwd=str(BASE_DIR),
            check=False,
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=str(BASE_DIR),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        res = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(BASE_DIR),
            check=False,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            print(f"\n-> [GitHub 자동 푸시 성공] {commit_msg}")
        else:
            print(f"\n-> [GitHub 푸시 알림] 로컬 저장은 정상 완료되었습니다.")
    except Exception as e:
        print(f"\n-> [GitHub 자동 푸시 경고] {e}")


def run_evaluation(sample_n: int | None = None):
    print("=" * 65)
    print(f"1. 최종 테스트 데이터셋 로드: {VAL_CSV_PATH.name}")
    print("=" * 65)

    val_df = pd.read_csv(VAL_CSV_PATH)
    total_count = len(val_df)
    first_final_query = str(val_df.iloc[0]["문의 내용"]).strip()

    results = []
    completed_indices = set()

    if FINAL_CSV_PATH.exists() and not OUTPUT_CSV_PATH.exists():
        shutil.copy2(FINAL_CSV_PATH, OUTPUT_CSV_PATH)

    if OUTPUT_CSV_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        existing_first_query = str(existing_df.iloc[0]["문의 내용"]).strip() if len(existing_df) > 0 else ""

        if "generation_seconds" not in existing_df.columns or existing_first_query != first_final_query:
            shutil.copy2(OUTPUT_CSV_PATH, OLD_VAL_BACKUP)
            OUTPUT_CSV_PATH.unlink()
            print(f"-> [초기화 완료] 기존 validation 결과를 '{OLD_VAL_BACKUP.name}'으로 분리 백업했습니다.")
            print("-> final_test_canonical_981.csv 1번 문항부터 새로 시작합니다.")
        else:
            valid_df = existing_df[
                ~existing_df["생성된_RAG_답변"].astype(str).str.startswith("[오류 발생]")
            ]
            results = valid_df.to_dict("records")
            completed_indices = set(valid_df["원본_행번호"].tolist())
            print(f"-> [이어하기 감지] 기존 완료된 {len(completed_indices)}건을 건너뛰고 이어서 진행합니다.")

    new_count = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Final 981개 답변 생성 중"):
        if idx in completed_indices:
            continue

        query = str(row["문의 내용"]).strip()
        intent = str(row.get("의도", "")).strip()
        ref_answers = row.get("reference_responses_json", row.get("응답", ""))

        try:
            pred_answer, context, gen_sec = get_cs_answer(
                query=query,
                intent=intent,
                return_time=True,
            )
        except Exception as e:
            pred_answer = f"[오류 발생]: {str(e)}"
            context = ""
            gen_sec = 0.0

        results.append({
            "원본_행번호": idx,
            "카테고리": row.get("카테고리", ""),
            "의도": intent,
            "문의 내용": query,
            "생성된_RAG_답변": pred_answer,
            "참조_정답": ref_answers,
            "검색된_참고_문서": context,
            "generation_seconds": round(float(gen_sec), 4),
        })
        new_count += 1
        total_done = len(results)

        # 50건마다 저장 및 GitHub 원격 저장소 자동 푸시
        if new_count % 50 == 0:
            df_temp = pd.DataFrame(results).sort_values("원본_행번호")
            df_temp.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
            df_temp.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")
            git_auto_push(f"chore: Final 평가 {total_done}/{total_count}건 자동 백업")

    result_df = pd.DataFrame(results).sort_values("원본_행번호")
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    result_df.to_csv(FINAL_CSV_PATH, index=False, encoding="utf-8-sig")

    git_auto_push(f"feat: Final 981건 전체 생성 완료 ({len(result_df)}건)")

    valid_times = result_df["generation_seconds"].dropna()
    avg_sec = valid_times.mean() if len(valid_times) > 0 else 0.0
    p95_sec = valid_times.quantile(0.95) if len(valid_times) > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"최종 평가 완료! 총 {len(result_df)}개의 결과 파일이 저장 및 푸시되었습니다.")
    print(f"- 저장 경로          : {FINAL_CSV_PATH}")
    print(f"- EXAONE 생성 시간   : 평균 {avg_sec:.3f}초 / P95 {p95_sec:.3f}초")
    print("=" * 65)


if __name__ == "__main__":
    run_evaluation(sample_n=None)
