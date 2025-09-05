import streamlit as st
import pandas as pd
import json, io, hashlib, re
from datetime import date, datetime
from urllib.parse import quote, unquote
import qrcode

# === New: Google Sheets ===
import gspread
from google.oauth2.service_account import Credentials
from gspread.exceptions import WorksheetNotFound

# --- 頁面設定 ---
st.set_page_config(
    page_title="護持活動集點(for幹部)",
    page_icon="🔢",
    layout="wide",
)

# ================= Google Sheet Helpers =================
SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

def _get_gspread_client():
    # 1) Streamlit Cloud: 使用 st.secrets["gcp_service_account"]
    if "gcp_service_account" in st.secrets:
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=SCOPES
        )
        return gspread.authorize(creds)
    # 2) 本地端: 使用 credentials.json
    try:
        creds = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
        return gspread.authorize(creds)
    except Exception as e:
        st.error("找不到服務帳號憑證。請在 st.secrets 加入 gcp_service_account，或將 credentials.json 放在專案根目錄。")
        raise e

def _parse_sheet_id(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if "docs.google.com" in s:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", s)
        if m:
            return m.group(1)
    return s

@st.cache_resource(show_spinner=False)
def open_spreadsheet(sheet_id: str):
    client = _get_gspread_client()
    return client.open_by_key(sheet_id)

def get_or_create_ws(sh, title: str, headers: list[str]):
    try:
        ws = sh.worksheet(title)
        # 確保至少有表頭
        values = ws.get_all_values()
        if not values:
            ws.update([headers])
        else:
            ex_header = [h.strip() for h in values[0]]
            # 若缺少欄位，保留既有資料，後續在讀取/寫入補齊
            for col in headers:
                if col not in ex_header:
                    ex_header.append(col)
            if ex_header != values[0]:
                # 只更新表頭列；資料列保持
                ws.update([ex_header] + values[1:])
        return ws
    except WorksheetNotFound:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(headers)))
        ws.update([headers])
        return ws

def ws_to_df(ws, expected_cols: list[str]) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        ws.update([expected_cols])
        return pd.DataFrame(columns=expected_cols)
    header = values[0]
    data = values[1:]
    df = pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)
    # 補齊缺欄、只保留需要欄
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    return df[expected_cols]

def df_to_ws(ws, df: pd.DataFrame, expected_cols: list[str]):
    if df is None:
        ws.clear()
        ws.update([expected_cols])
        return
    # 確保欄位順序
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[expected_cols].copy()
    # 轉字串避免 gspread 型別問題
    data = [expected_cols] + df.astype(str).values.tolist()
    ws.clear()
    ws.update(data)

# ================= Domain Helpers（你原本的） =================
def normalize_names(s: str):
    if not s:
        return []
    raw = (s.replace("、", ",")
             .replace(" ", " ")  # 全形空白
             .replace("，", ",")
             .replace("（", "(")
             .replace("）", ")")
             .replace(" ", ","))
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "(" in token and ")" in token:
            token = token.split("(")[0].strip()
        out.append(token)
    return [n for n in out if n]

def aggregate(df, points_map, rewards):
    if df.empty:
        return pd.DataFrame(columns=["participant", "總點數"])
    df = df.copy()
    df["points"] = df["category"].map(points_map).fillna(0).astype(int)
    summary = (
        df.pivot_table(index="participant", columns="category",
                       values="points", aggfunc="count", fill_value=0)
          .sort_index()
    )
    summary["總點數"] = 0
    for cat, pt in points_map.items():
        if cat in summary.columns:
            summary["總點數"] += summary[cat] * pt
    thresholds = sorted([int(r["threshold"]) for r in rewards if str(r.get("threshold","")).strip()!=""])
    def reward_badge(x):
        gain = [t for t in thresholds if x >= t]
        return (max(gain) if gain else 0)
    summary["已達門檻"] = summary["總點數"].apply(reward_badge)
    return summary.reset_index().sort_values(["總點數","participant"], ascending=[False,True])

def make_code(title: str, category: str, iso_date: str, length: int = 8) -> str:
    """根據(標題, 類別, 日期)產生穩定短代碼；固定長度，英數字"""
    base = f"{iso_date}|{category}|{title}".encode("utf-8")
    h = hashlib.md5(base).hexdigest()
    return h[:length].upper()

def upsert_link(links_df: pd.DataFrame, code: str, title: str, category: str, iso_date: str) -> pd.DataFrame:
    links_df = links_df.copy()
    if "code" not in links_df.columns:
        links_df = pd.DataFrame(columns=["code","title","category","date"])
    mask = links_df["code"] == code
    row = {"code": code, "title": title, "category": category, "date": iso_date}
    if mask.any():
        links_df.loc[mask, ["title","category","date"]] = [title, category, iso_date]
    else:
        links_df = pd.concat([links_df, pd.DataFrame([row])], ignore_index=True)
    return links_df

# ================== Sheet-backed Storage API ==================
def load_config_from_sheet(sh):
    # scoring_items(category, points) + rewards(threshold, reward)
    ws_items = get_or_create_ws(sh, "scoring_items", ["category","points"])
    ws_rewards = get_or_create_ws(sh, "rewards", ["threshold","reward"])
    items_df = ws_to_df(ws_items, ["category","points"])
    rewards_df = ws_to_df(ws_rewards, ["thr]()
