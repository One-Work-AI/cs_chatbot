import os
import pandas as pd
from pypdf import PdfReader

# ==============================================================================
# 0. 경로 설정
# ==============================================================================
# __file__: 현재 실행 중인 check_data.py 파일의 절대 경로 (.../evaluation/scripts/check_data.py)
# 3단계 위로 올라가야 최상위 프로젝트 루트(cs_chatbot)를 정확히 바라봄
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 데이터가 보관된 디렉터리 경로 설정
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")
POLICY_DIR = os.path.join(BASE_DIR, "data", "policy")


# ==============================================================================
# 1. train.csv 데이터 점검
# ==============================================================================
print("=" * 65)
print("1. train.csv 데이터셋 점검")
print("=" * 65)

train_path = os.path.join(CSV_DIR, "train.csv")

if os.path.exists(train_path):
    # 한글 윈도우 환경 등에서 인코딩 문제(UnicodeDecodeError)가 자주 발생하므로 예외 처리
    try:
        df = pd.read_csv(train_path, encoding="utf-8")
        print("[정보] UTF-8 인코딩으로 정상 로드되었습니다.")
    except UnicodeDecodeError:
        df = pd.read_csv(train_path, encoding="cp949")
        print("[정보] CP949 인코딩으로 정상 로드되었습니다.")

    # 1) 기본 메타 정보 출력 (행/열 크기, 컬럼 목록)
    # [수정] 열 개수 출력 시 df.shape[1]로 수정하여 정상 출력되도록 조치
    print(f"\n* 전체 행/열 크기: {df.shape} (행: {df.shape[0]}개, 열: {df.shape[1]}개)")
    print(f"* 컬럼 목록: {list(df.columns)}")

    # 2) 결측치(Null) 확인 - RAG 또는 분류 시 결측값 처리가 필요한지 점검
    print("\n* 컬럼별 결측치 현황:")
    print(df.isnull().sum())

    # 3) 범주형 데이터 분포 확인 (카테고리, 의도, 질문 그룹 등)
    # 데이터셋에 어떤 종류의 고객 문의가 주로 들어있는지 파악
    target_columns = ["카테고리", "의도", "question_group"]
    for col in target_columns:
        if col in df.columns:
            print(f"\n* [{col}] 고유값 개수: {df[col].nunique()}개")
            print(f"  - 상위 5개 빈도:\n{df[col].value_counts().head(5)}")

    # 4) 실제 데이터 샘플(상위 2개 행) 확인
    print("\n* 샘플 데이터 (첫 2개 행):")
    for idx, row in df.head(2).iterrows():
        print(f"\n--- [샘플 {idx + 1}] ---")
        for col in df.columns:
            val_str = str(row[col])
            # 내용이 너무 길면 앞부분 120자만 잘라서 미리보기
            preview = val_str[:120] + "..." if len(val_str) > 120 else val_str
            print(f"  - {col}: {preview}")
else:
    print(f"[오류] 파일을 찾을 수 없습니다: {train_path}")
    print("      'data/csv/' 폴더 안에 train.csv가 있는지 확인해 주세요.")


# ==============================================================================
# 2. 정책 PDF 문서 점검
# ==============================================================================
print("\n" + "=" * 65)
print("2. 정책 PDF 문서 점검 (RAG 지식 베이스)")
print("=" * 65)

# 검사할 정책 PDF 파일 목록
pdf_files = [
    "쇼핑몰 이용약관 및 운영정책.pdf",
    "전자상거래_표준약관.pdf"
]

for pdf_name in pdf_files:
    pdf_path = os.path.join(POLICY_DIR, pdf_name)
    
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"\n* 문서명: {pdf_name}")
        print(f"  - 총 페이지 수: {total_pages}장")

        # 1페이지 텍스트 추출 테스트 (인코딩/깨짐 여부 확인)
        first_page = reader.pages[0]
        extracted_text = first_page.extract_text() or ""
        
        # 줄바꿈 정리 후 앞 200자 미리보기
        clean_preview = extracted_text.strip().replace("\n", " ")[:200]
        print(f"  - 1페이지 텍스트 추출 확인:\n    \"{clean_preview}...\"")
    else:
        print(f"\n[오류] 파일을 찾을 수 없습니다: {pdf_path}")
        print(f"      'data/policy/' 폴더 안에 {pdf_name} 파일이 있는지 확인해 주세요.")