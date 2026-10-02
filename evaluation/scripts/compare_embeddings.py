# evaluation/scripts/compare_embeddings.py
# ==============================================================================
# 0. 필요한 도구(라이브러리) 가져오기
# ==============================================================================
import os      # 컴퓨터의 폴더 경로와 파일을 다루기 위한 도구
import time    # 시간(초, 밀리초)을 측정하기 위한 스톱워치 도구
import pandas as pd  # 엑셀처럼 표(CSV) 데이터를 쉽게 읽고 다루는 도구
from pypdf import PdfReader  # PDF 파일에서 글자를 읽어오는 도구

# LangChain 관련 도구들
from langchain_core.documents import Document  # 본문 글과 출처(파일명, 페이지)를 하나로 묶어주는 상자
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 긴 글을 의미 단위로 잘라주는 가위
from langchain_community.embeddings import HuggingFaceEmbeddings  # 글자를 컴퓨터용 숫자로 바꿔주는 번역기(임베딩)
from langchain_community.vectorstores import Chroma  # 숫자로 바뀐 글들을 보관하고 검색해 주는 도서관(벡터 DB)


# ==============================================================================
# 1. 폴더 경로 설정 (어디에 파일이 있는지 컴퓨터에 알려주기)
# ==============================================================================
# [핵심 수정] BASE_DIR: 현재 프로젝트의 최상위 폴더 (cs_chatbot)
# 현재 파일이 evaluation/scripts/에 위치하므로 3단계 위로 올라감
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CSV 파일들이 모여있는 폴더 (cs_chatbot/data/csv)
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

# 약관 PDF 파일들이 모여있는 폴더 (cs_chatbot/data/policy)
POLICY_DIR = os.path.join(BASE_DIR, "data", "policy")


# ==============================================================================
# 2. 검증용 CSV 파일에서 실제 고객 질문 30개 뽑아오기
# ==============================================================================
def load_validation_test_queries(sample_size=30):
    """
    validation_canonical_981.csv 파일에서
    실제 고객들이 남긴 질문을 골고루 30개 뽑아오는 함수입니다.
    """
    val_path = os.path.join(CSV_DIR, "validation_canonical_981.csv")
    
    # 파일이 진짜 있는지 확인
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"[오류] 파일이 없습니다: {val_path}")
        
    # 한글 깨짐 방지를 위해 UTF-8로 먼저 읽고, 안 되면 CP949로 읽기
    try:
        df = pd.read_csv(val_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(val_path, encoding="cp949")
        
    print("=" * 80)
    print(f"1. 검증 데이터 파일 열기 성공: validation_canonical_981.csv (총 {len(df)}개 질문 보유)")
    print("=" * 80)
    
    # 질문이 적힌 열(컬럼) 이름 찾기 (보통 '문의 내용'으로 되어 있음)
    candidate_cols = ["문의 내용", "질문", "question", "query", "text"]
    q_col = None
    for col in candidate_cols:
        if col in df.columns:
            q_col = col
            break
    if not q_col:
        q_col = df.columns[0]  # 못 찾으면 맨 첫 번째 열을 질문으로 사용
        
    print(f"* 찾은 질문 열 이름: [{q_col}]")
    
    # 981개 중에서 골고루 30개를 건너뛰며 뽑기 (편향 방지)
    step = max(1, len(df) // sample_size)
    sampled_queries = df[q_col].iloc[::step].head(sample_size).dropna().astype(str).tolist()
    
    print(f"* 테스트에 사용할 실제 고객 질문 {len(sampled_queries)}개 추출 완료!\n")
    return sampled_queries


# ==============================================================================
# 3. 실제 서비스용 CSV 데이터를 읽어서 청크 준비 (41개)
# ==============================================================================
def prepare_chunks():
    """
    기계적인 PDF 자르기 대신, 실제 서비스 DB 구축(ingest.py)과 완벽히 동일하게 
    사람이 직접 정제한 cs_data.csv 파일의 원본 데이터를 읽어옵니다.
    """
    csv_path = os.path.join(CSV_DIR, "cs_data.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[오류] 데이터 파일이 없습니다: {csv_path}")
        
    df = pd.read_csv(csv_path)
    documents = []
    
    for idx, row in df.iterrows():
        # 첫 번째 열을 텍스트 본문으로 사용 (ingest.py 로직과 동일)
        content = str(row.iloc[0]) 
        # 메타데이터에 출처 기록
        doc = Document(page_content=content, metadata={"source": "cs_data.csv", "row_idx": idx})
        documents.append(doc)
        
    print(f"[약관 준비 완료] 실제 서비스와 동일한 총 {len(documents)}개의 진성 약관 조각 로드 완료.\n")
    return documents


# ==============================================================================
# 4. 모델 하나씩 불러와서 성능 테스트하기
# ==============================================================================
def benchmark_model(model_idx, model_info, chunks, queries):
    """
    모델 1개를 골라서:
    1) 모델 켜지는 시간
    2) 33개 약관을 외우는(인덱싱) 시간
    3) 30개 고객 질문을 검색하는 평균 속도와 정확도 점수
    를 측정합니다.
    """
    model_name = model_info["name"]     # 모델 다운로드용 이름
    model_alias = model_info["alias"]   # 화면에 보여줄 별명
    dimension = model_info["dim"]       # 모델의 숫자 크기(차원)
    
    print("=" * 80)
    print(f"▶ [{model_idx}/3] 모델 평가 중: {model_alias}")
    print(f"  (모델명: {model_name} | 벡터 크기: {dimension}차원)")
    print("=" * 80)
    
    # 1) 모델 로드 시간 측정
    t_load_start = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},               # 내 컴퓨터 CPU로 실행
        encode_kwargs={"normalize_embeddings": True}   # 정확도를 높이기 위한 정규화
    )
    load_time = time.time() - t_load_start
    print(f"1) 모델 켜진 시간: {load_time:.2f}초")
    
    # 2) 약관 조각들을 벡터 DB에 집어넣는 시간 측정
    # (컬렉션 이름을 모델마다 다르게 주어서 충돌을 막음)
    unique_collection = f"val_benchmark_{model_idx}"
    t_idx_start = time.time()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=unique_collection
    )
    index_time = time.time() - t_idx_start
    print(f"2) 33개 약관 외우는(인덱싱) 시간: {index_time:.2f}초")
    
    # 3) 몸풀기(워밍업) 한 번 실행
    vectorstore.similarity_search_with_score("몸풀기 질문", k=1)
    
    # 4) 실제 고객 질문 30개 검색해보기
    latencies = []  # 검색 속도를 모을 바구니
    distances = []  # 점수를 모을 바구니
    
    print(f"\n3) 실제 고객 질문 {len(queries)}개 검색 속도 측정 중...")
    for q in queries:
        t_q_start = time.time()
        
        # 질문과 가장 비슷한 약관 1개 찾기
        results = vectorstore.similarity_search_with_score(q, k=1)
        
        # 검색 걸린 시간 계산 (밀리초 ms 단위)
        latency_ms = (time.time() - t_q_start) * 1000
        latencies.append(latency_ms)
        
        # results[0]은 (문서, 점수) 세트이므로, 그 중에서 두 번째인 '점수(score)'만 바구니에 담기!
        best_doc, best_score = results[0]
        distances.append(best_score)
        
    # 30개 평균 계산
    avg_latency = sum(latencies) / len(latencies)
    avg_distance = sum(distances) / len(distances)
    
    # 첫 번째 질문에 대해 뭘 찾아왔는지 맛보기로 하나 보여주기
    sample_res_doc, sample_res_dist = vectorstore.similarity_search_with_score(queries[0], k=1)[0]
    preview = sample_res_doc.page_content.replace("\n", " ").strip()[:80]
    
    print(f"   • [첫 번째 질문 샘플] \"{queries[0][:35]}...\"")
    print(f"   • [찾아온 약관] {sample_res_doc.metadata.get('source')} (p.{sample_res_doc.metadata.get('page')})")
    print(f"   • [찾아온 내용] {preview}...")
    print(f"\n-> [결과] 30개 평균 검색 속도: {avg_latency:.1f} ms | 평균 거리 점수: {avg_distance:.4f}")
    print("-" * 80 + "\n")
    
    # 표로 출력할 결과 딕셔너리로 반환
    return {
        "alias": model_alias,
        "dim": dimension,
        "index_time": f"{index_time:.2f}s",
        "avg_latency": f"{avg_latency:.1f}ms",
        "avg_dist": f"{avg_distance:.4f}"
    }


# ==============================================================================
# 5. 전체 실행 시작
# ==============================================================================
if __name__ == "__main__":
    # 1. 약관 청크 만들기
    chunks = prepare_chunks()
    
    # 2. validation_canonical_981.csv에서 질문 30개 뽑기
    queries = load_validation_test_queries(sample_size=30)
    
    # 3. 비교해 볼 3개 모델 목록
    candidate_models = [
        {
            "alias": "BAAI/bge-m3 (SOTA 대형 다국어)",
            "name": "BAAI/bge-m3",
            "dim": 1024
        },
        {
            "alias": "intfloat/multilingual-e5-base (MS 다국어 E5)",
            "name": "intfloat/multilingual-e5-base",
            "dim": 768
        },
        {
            "alias": "jhgan/ko-sroberta-multitask (한국어 경량 특화)",
            "name": "jhgan/ko-sroberta-multitask",
            "dim": 768
        }
    ]
    
    report_summary = []
    
    # 모델 3개 차례대로 벤치마크 돌리기
    for idx, model_info in enumerate(candidate_models, start=1):
        try:
            summary = benchmark_model(idx, model_info, chunks, queries)
            report_summary.append(summary)
        except Exception as error:
            print(f"[오류 발생] {model_info['alias']}: {error}\n")
            
    # ==========================================================================
    # 포트폴리오에 바로 넣을 수 있는 최종 요약표 출력!
    # ==========================================================================
    print("\n" + "#" * 85)
    print("       [validation_canonical_981 기반 임베딩 모델 벤치마크 최종 요약표]")
    print("#" * 85)
    print(f"{'모델명':<42} | {'차원':<6} | {'인덱싱':<8} | {'평균 검색속도':<12} | {'평균 거리점수':<10}")
    print("-" * 85)
    for res in report_summary:
        print(f"{res['alias']:<42} | {res['dim']:<6} | {res['index_time']:<8} | {res['avg_latency']:<12} | {res['avg_dist']:<10}")
    print("#" * 85 + "\n")