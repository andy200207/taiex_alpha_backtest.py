# %%
# =====================================================================
# ONE-CELL: News → Sentiment (Gemini) → TAIEX Signal (0~10 scale)
# - 今天 / 指定月份 / 任意起訖 / 近 N 天
# - 修正版：放大幅度 + 自適應門檻 + TODAY 強制方向備援
# - 可選：抓 meta description（預設）／抓正文（READ_FULL_BODY=True）
# =====================================================================

# ===== 0) 參數設定 =====================================================
MODE         = "MONTH"          # "TODAY" | "MONTH" | "RANGE" | "LAST_N"
MONTH        = "2025-10"        # MODE="MONTH" 時使用
START_DATE   = "2025-10-01"     # MODE="RANGE" 時使用
END_DATE     = "2025-10-31"     # MODE="RANGE" 時使用
LAST_N_DAYS  = 30               # MODE="LAST_N" 時使用（含今天）

# 決策門檻（對應 S_t 0~10 量尺）
TH_LONG      = 5.3              # 放寬
TH_SHORT     = 4.7              # 收窄中性帶
MIN_ITEMS    = 30               # 降低樣本數門檻

# 放大日分數幅度（補丁①）
AMPLIFY_W    = 1.6              # 對 W_t 乘上後再映射到 0~10

# 自適應門檻設定（補丁②）
SIGNAL_MODE  = "adaptive"       # "fixed" | "adaptive"
ADAPT_LONG_Q = 0.66             # 上 1/3 做多
ADAPT_SHORT_Q= 0.34             # 下 1/3 放空（可改 0.7/0.3）

# TODAY 強制方向備援（補丁③）
TODAY_BAND   = 0.15             # S_t > 5+0.15 → LONG； S_t < 5-0.15 → SHORT

# Gemini 模型與成本
MODEL_ID     = "gemini-2.5-flash-lite"   # 可改 flash / pro
TEMPERATURE  = 0.4
BATCH_SIZE   = 8
TIMEOUT      = 12
MAX_ITEMS_PER_DAY = 200
ENRICH_META_TOPK  = 80           # 抓前 K 則 meta 描述
READ_FULL_BODY    = False        # 要不要抓正文（關掉較快）
ENRICH_BODY_TOPK  = 40           # 抓正文的前 K 則
BODY_MAX_CHARS    = 1000         # 正文最大拼接字數

# 權重與關鍵詞
CRED_WEIGHT = {"official":1.00,"pro_media":0.90,"mainstream":0.80,"forum":0.50}
IMPACT_KW = {
    "台股":1.5,"台指":1.5,"台指期":1.5,"加權":1.3,"台積電":1.4,"半導體":1.2,
    "CPI":1.0,"FOMC":1.1,"美元":1.0,"利率":1.0,"油價":0.9,"費半":1.0,"納斯達克":0.9,
    "地震":1.2,"颱風":1.1,"關稅":1.1,"法說":1.0,"AI":0.8
}

# 白名單（可關掉以擴大樣本）
USE_DOMAIN_WHITELIST = True
DOMAIN_WHITELIST = [
    "ltn.com.tw","udn.com","cna.com.tw","ctee.com.tw","technews.tw","digitimes.com",
    "money.udn.com","yahoo.com","reuters.com","bloomberg.com","ft.com","wsj.com",
    "economist.com","cnyes.com","businessweekly.com.tw","storm.mg"
]

# 其他
TOPN_REASON  = 10
LAG_NOTE     = "建議隔日 08:45 入場以避免未來資訊 (台北時區)"
SHOW_PREVIEW = True

# ===== 1) 安裝套件 =====================================================
try:
    get_ipython()
    %pip -q install -U feedparser google-generativeai beautifulsoup4 lxml numpy pandas python-dateutil
except Exception:
    import sys, subprocess
    for pkg in ["feedparser","google-generativeai","beautifulsoup4","lxml","numpy","pandas","python-dateutil"]:
        subprocess.check_call([sys.executable,"-m","pip","install","-U",pkg])

# ===== 2) 匯入與工具 ===================================================
import os, re, json, time, math, hashlib, requests
import numpy as np, pandas as pd, feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
from urllib.parse import urlparse

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

TZ     = timezone(timedelta(hours=8))
RUN_TS = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")
UA     = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}

def md5(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest()

def parse_time_any(s):
    try:
        return dateparser.parse(s).astimezone(TZ)
    except Exception:
        return datetime.now(TZ)

def clean(x: str) -> str:
    x = (x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.","")
    except:
        return ""

def clip(v, lo, hi):
    return max(lo, min(hi, v))

def date_range(start_date, end_date):
    days = pd.date_range(start=start_date, end=end_date, freq="D")
    return [d.to_pydatetime().astimezone(TZ) for d in days]

# ===== 3) Gemini 初始化 ================================================
if not os.environ.get("GEMINI_API_KEY"):
    try:
        from getpass import getpass
        os.environ["GEMINI_API_KEY"] = getpass("請輸入 GEMINI_API_KEY（輸入時不顯示）：")
    except Exception:
        pass
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("尚未提供 GEMINI_API_KEY。請設定環境變數或依提示輸入。")

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
gen_model = genai.GenerativeModel(
    model_name=MODEL_ID,
    generation_config=GenerationConfig(temperature=TEMPERATURE, response_mime_type="application/json"),
)

# ===== 4) 來源（今天） ==================================================
def fetch_reuters_business(n_max=60):
    url="http://feeds.reuters.com/reuters/businessNews"
    try:
        feed=feedparser.parse(url); out=[]
        for e in feed.entries[:n_max]:
            out.append({"source":domain_of(getattr(e,"link","")) or "reuters.com","source_tag":"pro_media",
                        "title":clean(getattr(e,"title","")),
                        "summary":clean(getattr(e,"summary","")),
                        "url":getattr(e,"link",""),
                        "published_at":parse_time_any(getattr(e,"published","") or getattr(e,"updated",""))})
        return out
    except Exception as ex:
        print("[warn] Reuters 抓取失敗:", ex); return []

def fetch_google_news_rss(query="台股 OR 台灣 股市 OR 台指期 OR 半導體 OR 台積電", n_max=120, lang="zh-TW", region="TW"):
    import urllib.parse as up
    q=up.quote(query)
    url=f"https://news.google.com/rss/search?q={q}&hl={lang}&gl={region}&ceid={region}:{'zh-Hant' if lang.startswith('zh') else lang}"
    try:
        feed=feedparser.parse(url); out=[]
        for e in feed.entries[:n_max]:
            link=getattr(e,"link",""); src=domain_of(link)
            out.append({"source":src or "news.google.com","source_tag":"mainstream",
                        "title":clean(getattr(e,"title","")),
                        "summary":clean(getattr(e,"summary","")),
                        "url":link,
                        "published_at":parse_time_any(getattr(e,"published","") or getattr(e,"updated",""))})
        return out
    except Exception as ex:
        print("[warn] GoogleNews 抓取失敗:", ex); return []

def fetch_ptt_board(board="Stock", pages=1):
    base=f"https://www.ptt.cc/bbs/{board}/index.html"; cookies={"over18":"1"}; out=[]
    try:
        url=base
        for _ in range(max(1,pages)):
            r=requests.get(url, headers=UA, cookies=cookies, timeout=TIMEOUT); r.raise_for_status()
            soup=BeautifulSoup(r.text,"lxml")
            for ent in soup.select("div.r-ent"):
                a=ent.select_one("div.title a")
                if not a: continue
                title=clean(a.text); link="https://www.ptt.cc"+a.get("href","")
                out.append({"source":"ptt.cc","source_tag":"forum",
                            "title":title,"summary":title,"url":link,
                            "published_at":datetime.now(TZ)})
            prev=soup.select_one("a.btn.wide:contains('上頁')")
            if not prev or pages==1: break
            url="https://www.ptt.cc"+prev.get("href")
        return out
    except Exception as ex:
        print(f"[warn] PTT {board} 抓取失敗:", ex); return []

# ===== 5) 歷史來源（GDELT，publisher = 真實網域） =====================
def fetch_gdelt_by_date(d, query="(Taiwan OR 台灣 OR 台股 OR 台指期 OR TSMC OR 台積電 OR semiconductor OR 半導體 OR finance OR 股市)",
                        max_records=300):
    start_local=datetime(d.year,d.month,d.day,0,0,0,tzinfo=TZ)
    end_local  =datetime(d.year,d.month,d.day,23,59,59,tzinfo=TZ)
    start_utc=start_local.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    end_utc  = end_local.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")
    url="https://api.gdeltproject.org/api/v2/doc/doc"
    params={"query":query,"mode":"ArtList","maxrecords":str(max_records),
            "format":"json","startdatetime":start_utc,"enddatetime":end_utc}
    try:
        r=requests.get(url, params=params, headers=UA, timeout=TIMEOUT); r.raise_for_status()
        js=r.json(); arts=js.get("articles",[]); out=[]
        for a in arts[:max_records]:
            link=a.get("url",""); src=domain_of(link)
            t=a.get("seendate","")
            published_at=parse_time_any(t) if t else start_local
            out.append({"source":src or "unknown","source_tag":"mainstream",
                        "title":clean(a.get("title","")), "summary":"", "url":link,
                        "published_at":published_at})
        if USE_DOMAIN_WHITELIST:
            out=[x for x in out if domain_of(x["url"]) in DOMAIN_WHITELIST]
        return out[:MAX_ITEMS_PER_DAY]
    except Exception as ex:
        print(f"[warn] GDELT 抓取失敗: {ex}"); return []

# ===== 6) 內容補強：meta / 正文 =======================================
def enrich_meta_descriptions(rows, topk=ENRICH_META_TOPK, timeout=TIMEOUT):
    k=min(topk, len(rows))
    for i in range(k):
        url=rows[i].get("url","")
        if not url: continue
        try:
            r=requests.get(url, headers=UA, timeout=timeout)
            soup=BeautifulSoup(r.text,"lxml")
            meta = (soup.select_one('meta[name="description"]') or
                    soup.select_one('meta[property="og:description"]') or
                    soup.select_one('meta[name="twitter:description"]'))
            if meta and meta.get("content"):
                desc = clean(meta.get("content"))
                rows[i]["summary"] = (rows[i].get("summary") or "") + " " + desc
        except Exception:
            pass
    return rows

def enrich_body(rows, topk=ENRICH_BODY_TOPK, timeout=TIMEOUT, max_chars=BODY_MAX_CHARS):
    if not READ_FULL_BODY: return rows
    k=min(topk, len(rows))
    for i in range(k):
        url=rows[i].get("url","")
        if not url: continue
        try:
            r=requests.get(url, headers=UA, timeout=timeout)
            soup=BeautifulSoup(r.text,"lxml")
            # 粗略抓正文：<article> or 主要 <p>
            article = soup.find("article")
            if article:
                ps = [clean(p.get_text(" ")) for p in article.find_all("p")]
            else:
                ps = [clean(p.get_text(" ")) for p in soup.find_all("p")]
            text = " ".join([p for p in ps if p and len(p) > 40])[:max_chars]
            if text:
                rows[i]["summary"] = (rows[i].get("summary") or "") + " " + text
        except Exception:
            pass
    return rows

# ===== 7) 打分工具（向量化 + JSON 重試 + fallback） ====================
KW_ARRAY=np.array(list(IMPACT_KW.keys()))
KW_W    =np.array(list(IMPACT_KW.values()))

def impact_weight(texts):
    out=[]
    for txt in texts:
        t=txt or ""
        hits=np.array([1.0 if kw in t else 0.0 for kw in KW_ARRAY])
        score = 1.0 + float(hits @ KW_W)/5.0   # 基準 1.0，最多 ~1.5
        out.append(float(clip(score, 0.5, 1.5)))
    return np.array(out)

def credibility_weight(tags):
    return np.array([CRED_WEIGHT.get(tag,0.7) for tag in tags])

SYSTEM_PROMPT = (
    "You are a financial news rater for Taiwan index futures sentiment.\n"
    "For each item (title+summary) return ONLY a JSON array of floats in [-0.6, 0.6], "
    "one per item, same order. Use small continuous values (avoid many zeros). "
    "If unsure, return +/-0.05 instead of 0. Output JSON array only."
)

def extract_json_array(s: str):
    if s is None: return None
    txt = s.strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if not m: return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        frag = re.sub(r",\s*\]", "]", frag)
        try:
            return json.loads(frag)
        except Exception:
            return None

def score_chunk(texts):
    content=[{"role":"user","parts":[
        "Items:\n" + "\n".join([f"{i+1}. {t[:8000]}" for i,t in enumerate(texts)]),
        "\n\nRespond with JSON array only."
    ]}]
    try:
        resp = gen_model.generate_content(system_instruction=SYSTEM_PROMPT, contents=content)
        raw  = getattr(resp, "text", None) or str(resp)
        return extract_json_array(raw)
    except Exception:
        return None

def fallback_score(text):
    POS = ["強勁","上修","優於預期","成長","擴產","擴張","上漲","飆升","走高","回升","反彈","創高",
           "降息","寬鬆","利多","增持","買進","回暖","旺季","突破","beat","beats","beat estimates",
           "rally","rebound","soar","surge","bull","bullish","upgrade","growth","strong"]
    NEG = ["疲弱","下修","遜於預期","衰退","裁員","停工","下跌","暴跌","走低","利空","升息","緊縮",
           "降價","減產","停產","停擺","地震","災害","戰爭","制裁","關稅","downgrade","miss","missed",
           "slump","plunge","tumble","bear","bearish","weak","shortage"]
    t=(text or "").lower()
    p=sum(1 for w in POS if w.lower() in t)
    n=sum(1 for w in NEG if w.lower() in t)
    if p==0 and n==0: return 0.05 if "上" in text else (-0.05 if "下" in text else 0.0)
    s=(p-n)/max(1,(p+n))
    return float(clip(s*0.6, -0.6, 0.6))

def gemini_scores_with_retry(texts, batch=BATCH_SIZE):
    def recur(ts):
        if not ts: return []
        # 先嘗試一批
        arr = score_chunk(ts)
        if isinstance(arr, list) and len(arr)==len(ts):
            return [float(clip(float(x), -0.6, 0.6)) for x in arr]
        # 若失敗且批量 >1，切半重試
        if len(ts) > 1:
            mid=len(ts)//2
            return recur(ts[:mid]) + recur(ts[mid:])
        # 單一條還是失敗 → fallback
        return [fallback_score(ts[0])]
    # 分批送
    out=[]
    for i in range(0, len(texts), batch):
        out.extend(recur(texts[i:i+batch]))
        time.sleep(0.2)
    return np.array(out, dtype=float)

# ===== 8) 單日打分與聚合（S_t 0~10） =================================
def run_scoring(df_day):
    if df_day is None or len(df_day)==0:
        return None, 5.0, 0.0, "HOLD"

    df=df_day.copy()
    df["title"]=df["title"].astype(str)
    df["summary"]=df["summary"].astype(str)
    df["published_at"]=pd.to_datetime(df["published_at"])
    df["key"]=df["url"].fillna("")+"|"+df["title"].fillna("")
    df["key_hash"]=df["key"].apply(md5)
    df=df.drop_duplicates("key_hash").reset_index(drop=True)

    if USE_DOMAIN_WHITELIST:
        df = df[df["source"].apply(lambda s: (s in DOMAIN_WHITELIST))]

    # 供模型用的文字
    df["text_for_llm"] = (df["title"].fillna("") + ". " + df["summary"].fillna("")).str.slice(0, 4000)

    # 向量權重
    df["impact_w"] = impact_weight(df["text_for_llm"].tolist())    # 0.5~1.5
    df["cred_w"]   = credibility_weight(df["source_tag"].tolist())  # 0.5~1.0

    # 極性（Gemini + 重試 + fallback），範圍 [-0.6,0.6]
    df["polarity"] = gemini_scores_with_retry(df["text_for_llm"].tolist(), batch=BATCH_SIZE)

    # 新穎度：重複標題降權
    title_hash = df["title"].str.lower().apply(md5)
    dup_counts = title_hash.map(title_hash.value_counts())
    df["novelty_w"] = (1.0/dup_counts.clip(lower=1).astype(float))

    # 單篇分數（-1~1，實務上會落在較小區間）
    raw = df["polarity"] * df["impact_w"] * df["cred_w"] * df["novelty_w"]
    df["item_score"] = np.clip(raw, -1.0, 1.0)

    # 加權平均 W_t（2 端 5% 截尾）
    weights = (df["impact_w"]*df["cred_w"]).to_numpy()
    weights = np.where(weights<=0, 1.0, weights)
    s = df["item_score"].to_numpy()
    if len(s) >= 20:
        lo, hi = np.quantile(s, 0.05), np.quantile(s, 0.95)
        s = np.clip(s, lo, hi)
    W_t = float(np.average(s, weights=weights))

    # 放大後映射到 0~10（補丁①）
    W_t_clip = clip(W_t * AMPLIFY_W, -0.8, 0.8)
    S_t = round(5 + 5 * (W_t_clip/0.8), 2)
    S_t = float(clip(S_t, 0.0, 10.0))

    # 初步方向（固定門檻）
    n_items = int(len(df))
    if n_items < MIN_ITEMS:
        side = "HOLD"
    else:
        side = "LONG" if S_t >= TH_LONG else ("SHORT" if S_t <= TH_SHORT else "HOLD")

    # 顯示友善欄位
    df["polarity10"]   = np.round(5 + (df["polarity"]/0.6)*5, 2)   # [-0.6,0.6] → [0,10]
    df["item_score10"] = np.round(5 + 5*df["item_score"], 2)       # [-1,1] → [0,10]

    return df, S_t, W_t, side

# ===== 9) 主流程（抓資料 → 內容補強 → 打分） ===========================
print("="*96)
print(f"[News→Signal] 執行時間（台北）：{RUN_TS}；模式：{MODE}")
print("="*96)

rows_all_days = []
signals_rows  = []
reasons_rows  = []

def run_for_day(d, rows):
    # 台北當日範圍
    day_start=datetime(d.year,d.month,d.day,0,0,0,tzinfo=TZ)
    day_end  =datetime(d.year,d.month,d.day,23,59,59,tzinfo=TZ)
    df=pd.DataFrame(rows)
    if df.empty: return None, 5.0, 0.0, "HOLD", 0
    df=df[(df["published_at"]>=day_start)&(df["published_at"]<=day_end)].reset_index(drop=True)
    if df.empty: return None, 5.0, 0.0, "HOLD", 0
    df_scored, S_t, W_t, side = run_scoring(df)
    return df_scored, S_t, W_t, side, len(df_scored) if df_scored is not None else 0

if MODE.upper() == "TODAY":
    start_date = end_date = datetime.now(TZ)
    rows = []
    rows += fetch_reuters_business(n_max=80)
    rows += fetch_google_news_rss(n_max=150)
    rows += fetch_ptt_board(board="Stock", pages=1)
    rows = rows[:MAX_ITEMS_PER_DAY]
    rows = enrich_meta_descriptions(rows, topk=ENRICH_META_TOPK)
    rows = enrich_body(rows, topk=ENRICH_BODY_TOPK)
    df_scored, S_t, W_t, side, nn = run_for_day(datetime.now(TZ), rows)

    # TODAY 強制方向（補丁③）
    if side == "HOLD" and (nn >= MIN_ITEMS):
        side = "LONG" if S_t > (5 + TODAY_BAND) else ("SHORT" if S_t < (5 - TODAY_BAND) else "HOLD")

    dstr = datetime.now(TZ).strftime("%Y-%m-%d")
    if df_scored is not None:
        df_scored.insert(0,"date", dstr)
        df_scored["day_score"]=S_t
        df_scored["day_side"]=side
        rows_all_days.append(df_scored[["date","published_at","source","title","polarity","impact_w","cred_w","novelty_w","item_score","polarity10","item_score10","url","day_score","day_side"]])
        df_scored["abs_contrib"]=df_scored["item_score"].abs()*(df_scored["impact_w"]*df_scored["cred_w"])
        reasons_rows.append(df_scored.sort_values("abs_contrib",ascending=False).head(TOPN_REASON)[["date","source","title","item_score","url"]])

    signals_rows.append({"date":dstr,"S_t":S_t,"W_t":W_t,"side":side,"n_items":nn})

else:
    # 日期範圍
    if MODE.upper()=="MONTH":
        y,m = map(int, MONTH.split("-"))
        start_date = datetime(y,m,1,tzinfo=TZ)
        end_date   = (datetime(y+1,1,1,tzinfo=TZ)-timedelta(days=1)) if m==12 else (datetime(y,m+1,1,tzinfo=TZ)-timedelta(days=1))
    elif MODE.upper()=="RANGE":
        start_date = parse_time_any(START_DATE)
        end_date   = parse_time_any(END_DATE)
    elif MODE.upper()=="LAST_N":
        end_date   = datetime.now(TZ)
        start_date = end_date - timedelta(days=int(LAST_N_DAYS)-1)
    else:
        raise ValueError("MODE 需為 TODAY / MONTH / RANGE / LAST_N")

    for d in date_range(start_date.date(), end_date.date()):
        print(f"\n---- {d.strftime('%Y-%m-%d')} ----")
        rows = fetch_gdelt_by_date(d, max_records=MAX_ITEMS_PER_DAY)
        rows = enrich_meta_descriptions(rows, topk=ENRICH_META_TOPK)
        rows = enrich_body(rows, topk=ENRICH_BODY_TOPK)
        df_scored, S_t, W_t, side, nn = run_for_day(d, rows)
        if df_scored is not None:
            df_scored.insert(0,"date", d.strftime("%Y-%m-%d"))
            df_scored["day_score"]=S_t
            df_scored["day_side"]=side
            rows_all_days.append(df_scored[["date","published_at","source","title","polarity","impact_w","cred_w","novelty_w","item_score","polarity10","item_score10","url","day_score","day_side"]])
            df_scored["abs_contrib"]=df_scored["item_score"].abs()*(df_scored["impact_w"]*df_scored["cred_w"])
            reasons_rows.append(df_scored.sort_values("abs_contrib",ascending=False).head(TOPN_REASON)[["date","source","title","item_score","url"]])

        signals_rows.append({"date":d.strftime("%Y-%m-%d"),"S_t":S_t,"W_t":W_t,"side":side,"n_items":nn})
        print(f"  n={nn} | S_t={S_t:.2f} | 初步建議：{side}")

# ===== 10) 自適應門檻重劃（補丁②） ===================================
df_signals = pd.DataFrame(signals_rows)
if SIGNAL_MODE == "adaptive" and len(df_signals) >= 7 and MODE.upper() != "TODAY":
    q_hi = float(df_signals["S_t"].quantile(ADAPT_LONG_Q))
    q_lo = float(df_signals["S_t"].quantile(ADAPT_SHORT_Q))
    df_signals["side"] = np.where(
        (df_signals["S_t"] >= q_hi) & (df_signals["n_items"] >= MIN_ITEMS), "LONG",
        np.where((df_signals["S_t"] <= q_lo) & (df_signals["n_items"] >= MIN_ITEMS), "SHORT", "HOLD")
    )

# ===== 11) 輸出 CSV 與圖 ==============================================
start_tag = (start_date.strftime("%Y%m%d") if MODE.upper()!="TODAY" else datetime.now(TZ).strftime("%Y%m%d"))
end_tag   = (end_date.strftime("%Y%m%d")   if MODE.upper()!="TODAY" else datetime.now(TZ).strftime("%Y%m%d"))

sig_name = f"signals_{start_tag}_{end_tag}.csv"
df_signals.to_csv(sig_name, index=False, encoding="utf-8-sig")

if rows_all_days:
    df_items = pd.concat(rows_all_days, ignore_index=True)
    items_name = f"news_items_{start_tag}_{end_tag}.csv"
    df_items.to_csv(items_name, index=False, encoding="utf-8-sig")

if reasons_rows:
    df_reasons = pd.concat(reasons_rows, ignore_index=True)
    reasons_name = f"reasons_top{TOPN_REASON}_{start_tag}_{end_tag}.csv"
    df_reasons.to_csv(reasons_name, index=False, encoding="utf-8-sig")

print("\n" + "-"*96)
print(f"【每日信號】已輸出：{sig_name}")
if rows_all_days: print(f"【新聞明細】已輸出：{items_name}")
if reasons_rows:  print(f"【Top{TOPN_REASON} 理由】已輸出：{reasons_name}")
print(f"滯後/入場備註：{LAG_NOTE}")
print("-"*96)

# 繪圖（0~10 量尺 + LONG/SHORT 標記）
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    x = pd.to_datetime(df_signals["date"])
    y = df_signals["S_t"].astype(float)
    plt.plot(x, y, marker="o")
    for xi, yi, side in zip(x, y, df_signals["side"]):
        if side=="LONG":
            plt.scatter([xi],[yi], s=60, marker="^")
        elif side=="SHORT":
            plt.scatter([xi],[yi], s=60, marker="v")
    plt.axhline(TH_LONG, linestyle="--")
    plt.axhline(TH_SHORT, linestyle="--")
    plt.ylim(0,10)
    plt.title(f"Daily Sentiment Score (0~10)  {start_tag}~{end_tag}")
    plt.xlabel("Date"); plt.ylabel("S_t (0~10)")
    plt.tight_layout()
    png_name = f"sentiment_plot_{start_tag}_{end_tag}.png"
    plt.savefig(png_name, dpi=150)
    print(f"【情緒走勢圖】已輸出：{png_name}")
except Exception as ex:
    print("[warn] 繪圖失敗：", ex)

# 預覽
if SHOW_PREVIEW and rows_all_days:
    try:
        from IPython.display import display
        display(df_items.head(50))
    except Exception:
        pass



