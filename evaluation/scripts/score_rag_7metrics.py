"""
[파일명]: evaluation/scripts/score_rag_7metrics.py
[역할]: 튜닝 조원의 7개 평가지표와 동일한 기준으로 981개 RAG 결과를 채점하고 1:1 비교표 출력
        (rag_scored_results.csv의 임베딩 점수를 재활용하여 GPU 없이 3초 만에 채점 완료)
"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

# [핵심 수정] 3단계 위로 올라가 루트(cs_chatbot)를 잡음
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# 입출력 경로를 evaluation/results 폴더 안으로 강제
SCORED_3_CSV_PATH = RESULTS_DIR / "rag_scored_results.csv"
FULL_CSV_PATH = RESULTS_DIR / "rag_full_val_results.csv"
SAMPLE_CSV_PATH = RESULTS_DIR / "rag_base_val_results.csv"
SCORED_7_CSV_PATH = RESULTS_DIR / "rag_scored_7metrics.csv"

def parse_ref_answers(ref_raw: str) -> list:
    ref_str = str(ref_raw).strip()
    try:
        parsed = json.loads(ref_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [ref_str]

# 1. Key Fact F1 (핵심정보 보존도: 주문번호/placeholder, 금액/기간/숫자, [메뉴명] 보존 여부)
def extract_key_facts(text: str) -> set:
    facts = set()
    for m in re.findall(r"\{\{[^}]+\}\}", text):
        facts.add(m.strip())
    for m in re.findall(r"\[([^\]]+)\]", text):
        for sub in re.split(r"[>/]", m):
            if sub.strip():
                facts.add(sub.strip())
    for m in re.findall(r"\d+(?:[-,~:]\d+)*(?:원|일|시간|시|분|개월|%|번)?", text):
        facts.add(m.strip())
    return facts

def calc_key_fact_f1(pred: str, ref: str) -> float:
    pf = extract_key_facts(pred)
    rf = extract_key_facts(ref)
    if not rf:
        return 1.0
    if not pf:
        return 0.0
    inter = len(pf & rf)
    p = inter / len(pf)
    r = inter / len(rf)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

# 2. ROUGE-L (답변 표현 일치도: 어절 단위 최장 공통 부분수열 F1)
def calc_rouge_l(pred: str, ref: str) -> float:
    p_tokens = pred.split()
    r_tokens = ref.split()
    if not p_tokens or not r_tokens:
        return 0.0
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

# 3. chrF (문장 구성 유사도: 글자 단위 n-gram F-score)
def calc_chrf(pred: str, ref: str, max_n: int = 3, beta: float = 2.0) -> float:
    pred_c = re.sub(r"\s+", "", pred)
    ref_c = re.sub(r"\s+", "", ref)
    if not pred_c or not ref_c:
        return 0.0
    prec_sum, rec_sum, valid_n = 0.0, 0.0, 0
    for n in range(1, max_n + 1):
        if len(pred_c) < n or len(ref_c) < n:
            continue
        p_ngrams = [pred_c[i:i+n] for i in range(len(pred_c) - n + 1)]
        r_ngrams = [ref_c[i:i+n] for i in range(len(ref_c) - n + 1)]
        p_counts, r_counts = {}, {}
        for g in p_ngrams: p_counts[g] = p_counts.get(g, 0) + 1
        for g in r_ngrams: r_counts[g] = r_counts.get(g, 0) + 1
        overlap = sum(min(p_counts[g], r_counts.get(g, 0)) for g in p_counts)
        prec_sum += overlap / len(p_ngrams)
        rec_sum += overlap / len(r_ngrams)
        valid_n += 1
    if valid_n == 0 or (prec_sum + rec_sum) == 0:
        return 0.0
    avg_p = prec_sum / valid_n
    avg_r = rec_sum / valid_n
    beta2 = beta ** 2
    return (1 + beta2) * (avg_p * avg_r) / ((beta2 * avg_p) + avg_r)

# 4~6. 안정성 지표 3종 (Repetition Free, Korean Response, Clean Answer)
def check_repetition_free(pred: str) -> float:
    sents = [s.strip() for s in re.split(r"[.!?]\s+|\n+", pred) if len(s.strip()) >= 10]
    if len(sents) <= 1:
        return 1.0
    return 1.0 if (len(set(sents)) / len(sents)) >= 0.8 else 0.0

def check_korean_response(pred: str) -> float:
    hangul = len(re.findall(r"[가-힣]", pred))
    letters = len(re.findall(r"[가-힣a-zA-Z\u4e00-\u9fff]", pred))
    return 1.0 if (letters > 0 and (hangul / letters) >= 0.5) else 0.0

def check_clean_answer(pred: str) -> float:
    dirty_patterns = [r"<think>", r"</think>", r"<\|im_start\|>", r"<\|endoftext\|>", r"\[시스템\]", r"\[System\]", r"\[참고\s*문서"]
    for pat in dirty_patterns:
        if re.search(pat, pred, re.IGNORECASE):
            return 0.0
    return 1.0

def main():
    # 이미 1차 채점이 완료된 rag_scored_results.csv가 있으면 우선 로드
    if SCORED_3_CSV_PATH.exists():
        target_csv = SCORED_3_CSV_PATH
    elif FULL_CSV_PATH.exists():
        target_csv = FULL_CSV_PATH
    else:
        target_csv = SAMPLE_CSV_PATH

    df = pd.read_csv(target_csv)
    print(f"-> 채점 대상 로드 완료: {target_csv.name} (총 {len(df)}건)")

    has_precomputed = ("의미_유사도_점수" in df.columns) and ("약관_근거_충실도" in df.columns)
    if has_precomputed:
        print("-> [초고속 모드] 기존에 계산된 BGE-m3 의미 유사도 및 약관 충실도 점수를 그대로 활용합니다.")
        eval_model = None
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = SentenceTransformer("BAAI/bge-m3", device=device)

    sem_list, kf_list, rl_list, chrf_list = [], [], [], []
    rep_list, kor_list, cln_list, grd_list = [], [], [], []

    for _, row in df.iterrows():
        pred = str(row["생성된_RAG_답변"]).strip()
        ctx = str(row.get("검색된_참고_문서", "")).strip()
        refs = parse_ref_answers(row["참조_정답"])

        if has_precomputed:
            sem_sim = float(row["의미_유사도_점수"])
            grd = float(row["약관_근거_충실도"])
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            pred_emb = eval_model.encode([pred], normalize_embeddings=True)
            ref_embs = eval_model.encode(refs, normalize_embeddings=True)
            sem_sim = float(np.max(cosine_similarity(pred_emb, ref_embs)[0])) * 100.0
            if ctx and ctx != "nan":
                ctx_emb = eval_model.encode([ctx], normalize_embeddings=True)
                grd = float(cosine_similarity(pred_emb, ctx_emb)[0][0]) * 100.0
            else:
                grd = 0.0

        kf = max(calc_key_fact_f1(pred, r) for r in refs) * 100.0
        rl = max(calc_rouge_l(pred, r) for r in refs) * 100.0
        chrf = max(calc_chrf(pred, r) for r in refs) * 100.0
        rep = check_repetition_free(pred) * 100.0
        kor = check_korean_response(pred) * 100.0
        cln = check_clean_answer(pred) * 100.0

        sem_list.append(round(sem_sim, 2))
        kf_list.append(round(kf, 2))
        rl_list.append(round(rl, 2))
        chrf_list.append(round(chrf, 2))
        rep_list.append(round(rep, 2))
        kor_list.append(round(kor, 2))
        cln_list.append(round(cln, 2))
        grd_list.append(round(grd, 2))

    df["Semantic_Similarity"] = sem_list
    df["Key_Fact_F1"] = kf_list
    df["ROUGE_L"] = rl_list
    df["chrF"] = chrf_list
    df["Repetition_Free"] = rep_list
    df["Korean_Response"] = kor_list
    df["Clean_Answer"] = cln_list
    df["Context_Groundedness"] = grd_list
    df.to_csv(SCORED_7_CSV_PATH, index=False, encoding="utf-8-sig")

    summary_rows = [
        ("Semantic Similarity", "답변 의미 적합도", f"{np.mean(sem_list):.2f}%", "87.68%"),
        ("Key Fact F1", "핵심정보 보존도", f"{np.mean(kf_list):.2f}%", "62.65%"),
        ("ROUGE-L", "답변 표현 일치도", f"{np.mean(rl_list):.2f}%", "26.03%"),
        ("chrF", "문장 구성 유사도", f"{np.mean(chrf_list):.2f}%", "63.14%"),
        ("Repetition Free", "반복 없는 응답률", f"{np.mean(rep_list):.2f}%", "100%"),
        ("Korean Response", "한국어 정상 응답률", f"{np.mean(kor_list):.2f}%", "100%"),
        ("Clean Answer", "고객 전달 가능 응답률", f"{np.mean(cln_list):.2f}%", "100%"),
        ("Context Groundedness", "약관 근거 충실도(RAG 강점)", f"{np.mean(grd_list):.2f}%", "-"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["기존 지표", "CS 관점 표현", "RAG 981개 결과", "튜닝 팀 현재 결과"])
    print("\n" + "=" * 80)
    print("                [RAG vs 튜닝 7대 지표 1:1 비교 성적표]")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

if __name__ == "__main__":
    main()