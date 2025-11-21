# --- 頁面設定 ---
import streamlit as st
st.set_page_config(page_title="護持活動集點(for幹部)", page_icon="🔢", layout="wide")

import requests
import pandas as pd
import json, io, hashlib, re
from datetime import date, datetime
from urllib.parse import quote, unquote, urlsplit, urlunsplit, quote_plus
import qrcode
import time, random
import os, hmac

# ================== Excel 匯出工具 ==================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Sheet1") -> bytes:
    engine = None
    try:
        import openpyxl  # noqa
        engine = "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            raise RuntimeError("需要 openpyxl 或 xlsxwriter 其中之一，請在 requirements.txt 安裝其中一個")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()

# ================== 安全網址 + QR ==================
def _sanitize_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    parts = urlsplit(u if "://" in u else "https://" + u)
    scheme = parts.scheme or "https"
    netloc = parts.netloc or parts.path
    path = quote(parts.path or "", safe="/-._~")
    if parts.query:
        pairs = []
        for p in parts.query.split("&"):
            if "=" in p:
                k, v = p.split("=", 1)
                pairs.append(f"{quote_plus(k)}={quote_plus(v)}")
            else:
                pairs.append(quote_plus(p))
        query = "&".join(pairs)
    else:
        query = ""
    fragment = quote(parts.fragment or "", safe="-._~")
    return urlunsplit((scheme, netloc, path, query, fragment))

def build_checkin_url(public_base: str, code: str) -> str:
    base = (public_base or "").strip()
    if base.endswith("/"):
        base = base[:-1]
    return _sanitize_url(f"{base}/?mode=checkin&c={quote_plus(str(code))}")

def show_safe_link_box(url: str, title: str = "分享報到短連結（LINE 友善）"):
    safe = _sanitize_url(url)
    st.subheader(title)
    st.markdown(
        f'<a href="{safe}" target="_blank" rel="noopener noreferrer" style="font-size:18px;">點我前往報到</a>',
        unsafe_allow_html=True,
    )
    st.caption("LINE 請直接複製這段給大家（單獨一行，不要加文字或符號）")
    st.code(safe, language="text")
    img = qrcode.make(safe)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="掃碼報到", use_column_width=False)

# ================= Google Sheet Helpers =================
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import WorksheetNotFound, APIError

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _get_gspread_client():
    info = st.secrets["gcp_service_account"]
    if isinstance(info, str):
        info = json.loads(info)

    # 私鑰自動修正
    pk = info.get("private_key", "")
    if not isinstance(pk, str) or "BEGIN PRIVATE KEY" not in pk:
        raise RuntimeError("secrets 裡的 gcp_service_account.private_key 看起來不對，請確認有 BEGIN/END 標頭")
    if "\\n" in pk and "\n" not in pk:
        pk = pk.replace("\\n", "\n")
    pk = pk.strip()
    if not pk.startswith("-----BEGIN PRIVATE KEY-----"):
        pk = "-----BEGIN PRIVATE KEY-----\n" + pk.split("-----BEGIN PRIVATE KEY-----")[-1].lstrip()
    if not pk.endswith("-----END PRIVATE KEY-----"):
        pk = pk.split("-----BEGIN PRIVATE KEY-----")[0].rstrip() + "\n-----END PRIVATE KEY-----"
    info = dict(info)
    info["private_key"] = pk

    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

FIXED_SHEET_ID = st.secrets["google_sheets"]["sheet_id"]

@st.cache_resource(show_spinner=False)
def open_spreadsheet_by_fixed_id():
    client = _get_gspread_client()
    return client.open_by_key(FIXED_SHEET_ID)

def _explain_api_error(e: APIError) -> str:
    try:
        status = e.response.status_code
        try:
            body = e.response.json()
        except Exception:
            body = e.response.text
        return f"status={status}, detail={body}"
    except Exception:
        return str(e)

def _with_retry(func, *args, **kwargs):
    for i in range(5):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429, 500, 502, 503, 504):
                time.sleep((1.6 ** i) + random.random() * 0.4)
                continue
            raise

def get_or_create_ws(sh, title: str, headers: list[str]):
    try:
        ws = _with_retry(sh.worksheet, title)
    except WorksheetNotFound:
        try:
            ws = _with_retry(sh.add_worksheet, title=title, rows=2000, cols=max(10, len(headers)))
            _with_retry(ws.update, [headers])
            return ws
        except APIError as e:
            st.error(f"無法建立工作表「{title}」。{_explain_api_error(e)}")
            st.stop()
    except APIError as e:
        st.error(f"讀取工作表「{title}」失敗。{_explain_api_error(e)}")
        st.stop()

    if ws is None:
        st.error(f"取得工作表「{title}」失敗（ws=None）。請檢查 sheet_id/權限。")

    try:
        values = _with_retry(ws.get_all_values)
        if not values:
            _with_retry(ws.update, [headers])
            return ws
        ex_header = [h.strip() for h in values[0]]
        changed = False
        for col in headers:
            if col not in ex_header:
                ex_header.append(col)
                changed = True
        if changed:
            _with_retry(ws.update, [ex_header] + values[1:])
        return ws
    except APIError as e:
        st.error(f"更新工作表「{title}」表頭失敗。{_explain_api_error(e)}")
        st.stop()

def ws_to_df(ws, expected_cols: list[str]) -> pd.DataFrame:
    values = _with_retry(ws.get_all_values)
    if not values:
        _with_retry(ws.update, [expected_cols])
        return pd.DataFrame(columns=expected_cols)
    header = values[0]
    data = values[1:]
    df = pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    return df[expected_cols]

def safe_write_ws(ws, df: pd.DataFrame, expected_cols: list[str], *, allow_clear: bool = False):
    if df is None:
        return
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[expected_cols].copy()

    if df.empty and not allow_clear:
        vals = _with_retry(ws.get_all_values)
        if not vals:
            _with_retry(ws.update, [expected_cols])
        return

    data = [expected_cols] + df.astype(str).values.tolist()
    _with_retry(ws.clear)
    _with_retry(ws.update, data)

def df_to_ws(ws, df: pd.DataFrame, expected_cols: list[str]):
    safe_write_ws(ws, df, expected_cols, allow_clear=True)

# ============ 管理密碼 ============
def _get_admin_pass() -> str:
    return (
        st.secrets.get("app", {}).get("admin_password")
        or os.getenv("ADMIN_PASSWORD", "")
    )

ADMIN_PASS = _get_admin_pass()

def _check_pw(pw_input: str) -> bool:
    return bool(ADMIN_PASS) and hmac.compare_digest(str(pw_input), str(ADMIN_PASS))

try:
    HAVE_DIALOG = hasattr(st, "dialog")
except Exception:
    HAVE_DIALOG = False

def _need_pw(action_key: str, payload: dict | None = None):
    st.session_state["pending_action"] = action_key
    st.session_state["pending_payload"] = payload or {}
    st.rerun()

# 四個主鍵欄
KEY_COLS = ["date", "title", "category", "participant"]

# 🔧 正規化：把 'None' / 'nan' / 'NaN' 當成空白
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in df.columns:
        s = out[c].astype(str)
        s = s.replace({"None": "", "nan": "", "NaN": ""})
        out[c] = s.str.strip()
    return out

def _is_blank_row(row) -> bool:
    return all((str(row.get(c, "")).strip() == "") for c in KEY_COLS)

def _count_deleted_rows(before_df: pd.DataFrame, after_df: pd.DataFrame) -> int:
    def keyset(df: pd.DataFrame) -> set[str]:
        if "idempotency_key" in df.columns and df["idempotency_key"].astype(str).str.len().gt(0).any():
            return set(df["idempotency_key"].astype(str))
        combo = (
            df["date"].astype(str)
            + "|"
            + df["title"].astype(str)
            + "|"
            + df["category"].astype(str)
            + "|"
            + df["participant"].astype(str)
        )
        return set(combo)

    return len(keyset(before_df) - keyset(after_df))

# ---------- 穩定 append ----------
def safe_append(ws, rows: list[list], *, value_input_option: str = "RAW") -> bool:
    if not rows:
        return True
    for i in range(5):
        try:
            ws.append_rows(rows, value_input_option=value_input_option, table_range="A1")
            return True
        except APIError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429, 500, 502, 503, 504):
                time.sleep((1.6 ** i) + random.random() * 0.4)
                continue
            raise
    return False

# ✅ 只呼叫一次
sh = open_spreadsheet_by_fixed_id()

# ================= Domain Helpers =================
def normalize_names(s: str):
    if not s:
        return []
    raw = (
        s.replace("、", ",")
        .replace(" ", " ")
        .replace("，", ",")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", ",")
    )
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
        df.pivot_table(
            index="participant",
            columns="category",
            values="points",
            aggfunc="count",
            fill_value=0,
        ).sort_index()
    )
    summary["總點數"] = 0
    for cat, pt in points_map.items():
        if cat in summary.columns:
            summary["總點數"] += summary[cat] * pt
    thresholds = sorted(
        [int(r["threshold"]) for r in rewards if str(r.get("threshold", "")).strip() != ""]
    )

    def reward_badge(x):
        gain = [t for t in thresholds if x >= t]
        return max(gain) if gain else 0

    summary["已達門檻"] = summary["總點數"].apply(reward_badge)
    return summary.reset_index().sort_values(["總點數", "participant"], ascending=[False, True])

def make_code(title: str, category: str, iso_date: str, length: int = 8) -> str:
    base = f"{iso_date}|{category}|{title}".encode("utf-8")
    h = hashlib.md5(base).hexdigest()
    return h[:length].upper()

def make_idempotency_key(name: str, title: str, category: str, iso_date: str) -> str:
    raw = f"{iso_date}|{title}|{category}|{name}".strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()

def upsert_link(links_df: pd.DataFrame, code: str, title: str, category: str, iso_date: str) -> pd.DataFrame:
    links_df = links_df.copy()
    if "code" not in links_df.columns:
        links_df = pd.DataFrame(columns=["code", "title", "category", "date"])
    mask = links_df["code"] == code
    row = {"code": code, "title": title, "category": category, "date": iso_date}
    if mask.any():
        links_df.loc[mask, ["title", "category", "date"]] = [title, category, iso_date]
    else:
        links_df = pd.concat([links_df, pd.DataFrame([row])], ignore_index=True)
    return links_df

# ================== Sheet-backed Storage API ==================
EVENT_COLS = ["date", "title", "category", "participant", "idempotency_key"]
# 🔄 已刪除紀錄：多一個 deleted_at 欄位
EVENT_DELETED_COLS = EVENT_COLS + ["deleted_at"]

def load_config_from_sheet(sh):
    ws_items = get_or_create_ws(sh, "scoring_items", ["category", "points"])
    ws_rewards = get_or_create_ws(sh, "rewards", ["threshold", "reward"])
    items_df = ws_to_df(ws_items, ["category", "points"])
    rewards_df = ws_to_df(ws_rewards, ["threshold", "reward"])
    return {
        "scoring_items": items_df.to_dict(orient="records"),
        "rewards": rewards_df.to_dict(orient="records"),
    }

def save_config_to_sheet(sh, cfg):
    ws_items = get_or_create_ws(sh, "scoring_items", ["category", "points"])
    ws_rewards = get_or_create_ws(sh, "rewards", ["threshold", "reward"])
    items = (
        pd.DataFrame(cfg.get("scoring_items", []))
        if cfg.get("scoring_items")
        else pd.DataFrame(columns=["category", "points"])
    )
    rewards = (
        pd.DataFrame(cfg.get("rewards", []))
        if cfg.get("rewards")
        else pd.DataFrame(columns=["threshold", "reward"])
    )
    if "points" in items.columns:
        items["points"] = (
            pd.to_numeric(items["points"], errors="coerce").fillna(0).astype(int)
        )
    if "threshold" in rewards.columns:
        rewards["threshold"] = (
            pd.to_numeric(rewards["threshold"], errors="coerce").fillna(0).astype(int)
        )
    df_to_ws(ws_items, items, ["category", "points"])
    df_to_ws(ws_rewards, rewards, ["threshold", "reward"])

def load_events_from_sheet(sh) -> pd.DataFrame:
    ws = get_or_create_ws(sh, "events", EVENT_COLS)
    df = ws_to_df(ws, EVENT_COLS)
    return _normalize_df(df)

def save_events_to_sheet(sh, df: pd.DataFrame, *, allow_clear: bool = False):
    ws = get_or_create_ws(sh, "events", EVENT_COLS)
    safe_write_ws(ws, df, EVENT_COLS, allow_clear=allow_clear)

def load_links_from_sheet(sh) -> pd.DataFrame:
    ws = get_or_create_ws(sh, "links", ["code", "title", "category", "date"])
    return ws_to_df(ws, ["code", "title", "category", "date"])

def save_links_to_sheet(sh, df: pd.DataFrame):
    ws = get_or_create_ws(sh, "links", ["code", "title", "category", "date"])
    df_to_ws(ws, df, ["code", "title", "category", "date"])

# 🗑 已刪除紀錄（讀取）
def load_deleted_from_sheet(sh) -> pd.DataFrame:
    ws = get_or_create_ws(sh, "events_deleted", EVENT_DELETED_COLS)
    df = ws_to_df(ws, EVENT_DELETED_COLS)
    return _normalize_df(df)

# 🗑 已刪除紀錄（追加寫入）
def append_deleted_rows(sh, deleted_df: pd.DataFrame):
    if deleted_df is None or deleted_df.empty:
        return
    ws_del = get_or_create_ws(sh, "events_deleted", EVENT_DELETED_COLS)
    df = deleted_df.copy()
    # 補上 deleted_at 欄位
    now_iso = datetime.now().isoformat(timespec="seconds")
    if "deleted_at" not in df.columns:
        df["deleted_at"] = now_iso
    else:
        df["deleted_at"] = df["deleted_at"].replace("", now_iso)
    df = df[EVENT_DELETED_COLS]
    rows = df.astype(str).values.tolist()
    safe_append(ws_del, rows, value_input_option="USER_ENTERED")

# ---------- event_keys（去重用） ----------
def load_event_keys_ws(sh):
    return get_or_create_ws(
        sh, "event_keys", ["idempotency_key", "date", "title", "category", "participant"]
    )

# 🆕：不快取，每次即時讀取，避免 DUP 誤判
def load_event_keyset(sh) -> set:
    ws_keys = load_event_keys_ws(sh)
    vals = _with_retry(ws_keys.get_all_values)
    if not vals or len(vals) <= 1:
        return set()
    return set([r[0] for r in vals[1:] if r and r[0]])

# ============ 後端 API ============
AS_URL = st.secrets.get("apps_script", {}).get("web_app_url", "").strip()

def send_checkin_via_api(date_str: str, title: str, category: str, name: str, *, max_retries: int = 5) -> str:
    if not AS_URL:
        return "ERR: NO_URL"

    payload = {
        "date": date_str,
        "title": title,
        "category": category,
        "participant": name,
        "idempotency_key": make_idempotency_key(name, title, category, date_str),
    }

    last_err = ""
    for i in range(max_retries):
        try:
            r = requests.post(AS_URL, json=payload, timeout=12)
            try:
                data = r.json()
                status = (data.get("status") or "").upper()
                msg = data.get("message", "")
                if status in ("OK", "DUP"):
                    return status
                if status == "ERR":
                    return f"ERR: {msg}"
                last_err = f"HTTP {r.status_code} JSON={data}"
            except Exception:
                last_err = f"HTTP {r.status_code} TEXT={r.text[:200]}"
        except Exception as e:
            last_err = f"EXC {e}"

        time.sleep(min(2 ** i, 8) + random.random() * 0.3)

    return f"ERR: {last_err or 'unknown'}"

def append_events_rows(sh, rows: list[dict]):
    """統一入口：優先用 API；沒有 API 時退回直接寫表（含冪等鍵與索引維護）"""
    if not rows:
        return {"added": [], "skipped": []}

    # 透過 API
    if WRITE_MODE.startswith("透過後端") and AS_URL:
        added, skipped = [], []
        for r in rows:
            d, t, c, p = r["date"], r["title"], r["category"], r["participant"]
            res = send_checkin_via_api(d, t, c, p)
            if res == "OK":
                added.append(p)
            elif res == "DUP":
                skipped.append(p)
            else:
                st.warning(f"{p} 寫入失敗：{res}")
        return {"added": added, "skipped": skipped}

    # 直接寫 Sheet：先讀 event_keys，確保即時
    ws_events = get_or_create_ws(sh, "events", EVENT_COLS)
    ws_keys = load_event_keys_ws(sh)
    keyset = load_event_keyset(sh)

    evt_payload, key_payload = [], []
    added, skipped = [], []
    for r in rows:
        d, t, c, p = r["date"], r["title"], r["category"], r["participant"]
        k = make_idempotency_key(p, t, c, d)
        if k in keyset:
            skipped.append(p)
            continue
        evt_payload.append([d, t, c, p, k])
        key_payload.append([k, d, t, c, p])
        keyset.add(k)
        added.append(p)

    ok1 = safe_append(ws_events, evt_payload, value_input_option="USER_ENTERED") if evt_payload else True
    ok2 = safe_append(ws_keys, key_payload, value_input_option="USER_ENTERED") if key_payload else True
    if not (ok1 and ok2):
        st.warning("部分寫入失敗，請稍後在『完整記錄』確認。")
    return {"added": added, "skipped": skipped}

# ==== 寫入模式（Sidebar 會改寫 WRITE_MODE）====
use_api_default = bool(AS_URL)
WRITE_MODE = "透過後端 API（推薦）" if use_api_default else "直接寫入 Google Sheet"

# ================= Query Params =================
qp = st.query_params
mode = qp.get("mode", "")
code_param = qp.get("c", "")
event_param = qp.get("event", "")

# ============ Public check-in via URL ============
if mode == "checkin":
    st.markdown("### ✅ 線上報到")

    if not sh:
        st.error("找不到 Google Sheet。")
        st.stop()

    events_df = load_events_from_sheet(sh)
    links_df = load_links_from_sheet(sh)

    title, category, target_date = "未命名活動", "活動護持（含宿訪）", date.today().isoformat()
    resolved = False

    if code_param:
        rec = links_df.loc[links_df["code"].astype(str) == str(code_param)]
        if not rec.empty:
            title = rec.iloc[0]["title"]
            category = rec.iloc[0]["category"]
            target_date = rec.iloc[0]["date"]
            resolved = True

    if (not resolved) and event_param:
        try:
            decoded = unquote(event_param)
            if decoded.strip().startswith("{"):
                o = json.loads(decoded)
                title = o.get("title", title)
                category = o.get("category", category)
                target_date = o.get("date", target_date)
                resolved = True
            else:
                title = decoded or title
        except Exception:
            pass

    st.info(f"活動：**{title}**｜類別：**{category}**｜日期：{target_date}")
    st.markdown(
        """
        <div style="color:#d32f2f; font-weight:700;">請務必輸入全名</div>
        <div style="color:#000;">（例：陳曉瑩。可一次多人報到，用「、」「，」或空白分隔）</div>
        """,
        unsafe_allow_html=True,
    )

    names_input = st.text_area(
        label="姓名清單",
        key="pub_names_area",
        placeholder="例如：陳曉瑩、劉宜儒，許崇萱、黃佳宜 徐睿妤",
        label_visibility="collapsed",
    )

    if st.button("送出報到", key="pub_submit_btn"):
        names = normalize_names(names_input)
        if not names:
            st.error("請至少輸入一位姓名。")
        else:
            to_add = [
                {"date": target_date, "title": title, "category": category, "participant": n}
                for n in names
            ]
            result = append_events_rows(sh, to_add) or {"added": [], "skipped": []}
            if result["added"]:
                st.success(f"已報到 {len(result['added'])} 人：{'、'.join(result['added'])}")
            if result["skipped"]:
                st.warning(f"以下人員先前已報到，已跳過：{'、'.join(result['skipped'])}")
    st.stop()

# ================= Admin UI =================
st.title("🔢護持活動集點(for幹部)")

st.sidebar.title("⚙️ 設定（Google Sheet）")
st.sidebar.success(f"已綁定試算表：{st.secrets['google_sheets']['sheet_id']}")

WRITE_MODE = st.sidebar.radio(
    "寫入模式",
    options=["透過後端 API（推薦）", "直接寫入 Google Sheet"],
    index=0 if use_api_default else 1,
    help="大量同秒報到時，建議用後端 API（Apps Script）避免撞限額。",
    key="write_mode_radio",
)
if not ADMIN_PASS:
    st.sidebar.warning("尚未設定管理密碼（app.admin_password 或環境變數 ADMIN_PASSWORD）。")

def api_healthcheck() -> str:
    if not AS_URL:
        return "未設定 API URL"
    try:
        r = requests.get(AS_URL, timeout=6)
        return f"可連線（HTTP {r.status_code}）"
    except Exception as e:
        return f"不可連線：{e}"

with st.sidebar.expander("🔌 後端 API 狀態", expanded=False):
    st.write(f"API URL：{AS_URL or '（未設定）'}")
    if st.button("測試連線", key="btn_api_ping"):
        st.info(api_healthcheck())

# 載入設定 / 資料
if "config" not in st.session_state:
    st.session_state.config = load_config_from_sheet(sh)
if "events" not in st.session_state:
    st.session_state.events = load_events_from_sheet(sh)
if "links" not in st.session_state:
    st.session_state.links = load_links_from_sheet(sh)

config = st.session_state.config
scoring_items = config.get("scoring_items", [])
rewards = config.get("rewards", [])

points_map = {}
for i in scoring_items:
    if "category" in i:
        try:
            points_map[i["category"]] = int(i.get("points", 0))
        except Exception:
            points_map[i["category"]] = 0

# Sidebar editors
with st.sidebar.expander("➕ 編輯集點項目與點數", expanded=False):
    st.caption("新增或調整表格後點『儲存設定』。")
    items_df = (
        pd.DataFrame(scoring_items)
        if scoring_items
        else pd.DataFrame(columns=["category", "points"])
    )
    edited_items = st.data_editor(
        items_df, num_rows="dynamic", use_container_width=True, key="sb_items_editor"
    )
    if st.button("💾 儲存設定（集點項目）", key="sb_save_items_btn"):
        cfg = st.session_state.config
        if not edited_items.empty:
            edited_items["category"] = edited_items["category"].astype(str)
            edited_items["points"] = (
                pd.to_numeric(edited_items["points"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            edited_items = edited_items.dropna(subset=["category"])
        cfg["scoring_items"] = edited_items.to_dict(orient="records")
        st.session_state.config = cfg
        save_config_to_sheet(sh, cfg)
        st.success("已儲存集點項目。")

with st.sidebar.expander("🎁 編輯獎勵門檻", expanded=False):
    rew_df = (
        pd.DataFrame(rewards)
        if rewards
        else pd.DataFrame(columns=["threshold", "reward"])
    )
    rew_edit = st.data_editor(
        rew_df, num_rows="dynamic", use_container_width=True, key="sb_rewards_editor"
    )
    if st.button("💾 儲存設定（獎勵）", key="sb_save_rewards_btn"):
        cfg = st.session_state.config
        if not rew_edit.empty:
            rew_edit["reward"] = rew_edit["reward"].astype(str)
            rew_edit["threshold"] = (
                pd.to_numeric(rew_edit["threshold"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            rew_edit = rew_edit.dropna(subset=["threshold", "reward"])
        cfg["rewards"] = rew_edit.to_dict(orient="records")
        st.session_state.config = cfg
        save_config_to_sheet(sh, cfg)
        st.success("已儲存獎勵門檻。")

# ============== Tabs ==============
tabs = st.tabs(
    [
        "🟪 產生 QRcode",
        "📝 現場報到",
        "📆 依日期查看參與者",
        "👤 個人明細",
        "📒 完整記錄",
        "🏆 排行榜",
        "🗑 已刪除紀錄",
    ]
)

# -------- 0) 產生 QRcode -------
with tabs[0]:
    st.subheader("生成報到 QR Code")

    public_base = st.text_input(
        "公開網址（本頁網址）", value="", key="qr_public_url_input"
    ).rstrip("/")

    qr_title = st.text_input("活動標題", value="迎新晚會", key="qr_title_input")
    qr_category = st.selectbox(
        "類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="qr_category_select"
    )
    qr_date = st.date_input("活動日期", value=date.today(), key="qr_date_picker")

    iso = qr_date.isoformat()
    code = make_code(qr_title or qr_category, qr_category, iso, length=8)

    links_df = st.session_state.links
    links_df = upsert_link(
        links_df, code=code, title=(qr_title or qr_category), category=qr_category, iso_date=iso
    )
    st.session_state.links = links_df
    save_links_to_sheet(sh, links_df)

    short_url = build_checkin_url(public_base, code)

    if public_base:
        show_safe_link_box(short_url)
        img = qrcode.make(short_url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.download_button(
            "⬇️ 下載 QR 圖片",
            data=buf.getvalue(),
            file_name=f"checkin_{code}.png",
            mime="image/png",
            key="qr_download_btn",
        )
    else:
        st.info("請貼上你的 .streamlit.app 根網址（本頁網址）。")

    with st.expander("🔎 目前所有短代碼一覽", expanded=False):
        st.dataframe(
            links_df.sort_values("date", ascending=False),
            use_container_width=True,
            height=220,
        )
        st.download_button(
            "⬇️ 下載連結代碼 Excel（匯出）",
            data=df_to_excel_bytes(links_df, "links"),
            file_name="links.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="links_download_excel_btn",
        )
        if st.button("🧹 清空所有短代碼（links）", key="links_clear_btn"):
            st.session_state.links = st.session_state.links.iloc[0:0]
            save_links_to_sheet(sh, st.session_state.links)
            st.success("已清空所有短代碼。")

# -------- 1) 現場報到 --------
with tabs[1]:
    st.subheader("現場快速報到")
    on_title = st.text_input("活動標題", value="未命名活動", key="on_title_input")
    on_category = st.selectbox(
        "類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="on_category_select"
    )
    on_date = st.date_input("日期", value=date.today(), key="on_date_picker")
    st.markdown(
        """
        <div style="color:#d32f2f; font-weight:700;">請務必輸入全名（例：陳曉瑩）</div>
        <div style="color:#000;">（可一次多人報到，用「、」「，」或空白分隔）</div>
        """,
        unsafe_allow_html=True,
    )
    names_input = st.text_area(
        "姓名清單",
        placeholder="例如：陳曉瑩、蕭雅云，張詠禎 徐睿妤",
        key="on_names_area",
        label_visibility="collapsed",
    )
    if st.button("➕ 加入報到名單", key="on_add_btn"):
        target_date = on_date.isoformat()
        names = normalize_names(names_input)
        if not names:
            st.warning("請至少輸入一位姓名。")
        else:
            to_add = [
                {
                    "date": target_date,
                    "title": on_title,
                    "category": on_category,
                    "participant": n,
                }
                for n in names
            ]
            result = append_events_rows(sh, to_add) or {"added": [], "skipped": []}
            if result["added"]:
                st.session_state.events = load_events_from_sheet(sh)
                st.success(f"已加入 {len(result['added'])} 人：{'、'.join(result['added'])}")
            if result["skipped"]:
                st.warning(f"已跳過（先前已報到）：{'、'.join(result['skipped'])}")

# -------- 2) 依日期查看參與者 --------
with tabs[2]:
    st.subheader("📆 依日期查看參與者")

    import calendar
    from datetime import date as _date

    color_map = {
        "帶人開法會": "#FF0000",
        "法會護持一天": "#FF7F00",
        "參與獻供": "#E6B800",
        "活動護持 (含宿訪)": "#00C300",
        "參與晨讀": "#00FFFF",
        "參與讀書會/ 與學長姐有約": "#0066FF",
        "上研究班 (新民、至善)": "#8A2BE2",
        "參與辦道": "#FF1493",
        "參與成全/道務會議": "#A0522D",
        "帶人求道": "#FFFF66",
        "帶人進研究班": "#BEBEBE",
        "練講": "#F4A3A3",
        "背誦經典": "#808080",
    }

    def canon_cat(s: str) -> str:
        s = str(s or "").strip()
        s = s.replace("（", "(").replace("）", ")").replace("／", "/").replace("　", " ")
        s = re.sub(r"\s+", " ", s)
        s = s.replace(" /", "/").replace("/ ", "/")
        s = s.replace("參與讀書會/ 與學長姐有約", "參與讀書會/與學長姐有約")
        s = s.replace("活動護持(含宿訪)", "活動護持 (含宿訪)")
        return s

    color_map_canon = {canon_cat(k): v for k, v in color_map.items()}
    label_map_canon = {canon_cat(k): k for k in color_map.keys()}

    if st.session_state.events.empty:
        st.info("目前尚無活動紀錄。")
    else:
        events_df = st.session_state.events.copy()
        events_df["date_str"] = pd.to_datetime(
            events_df["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        events_df["date"] = pd.to_datetime(
            events_df["date"], errors="coerce"
        ).dt.date
        events_df["cat_norm"] = events_df["category"].map(canon_cat)

        today = _date.today()
        coly, colm = st.columns(2)
        with coly:
            year = st.number_input(
                "年份", min_value=2020, max_value=2035, value=today.year, step=1
            )
        with colm:
            month = st.number_input(
                "月份", min_value=1, max_value=12, value=today.month, step=1
            )

        cal = calendar.TextCalendar(firstweekday=calendar.SUNDAY)
        month_weeks = cal.monthdayscalendar(year, month)
        weekday_labels = [
            calendar.day_abbr[(i + cal.getfirstweekday()) % 7] for i in range(7)
        ]

        c1, c2 = st.columns([3, 1])
        with c1:
            html = "<table style='border-collapse: collapse; width:100%; text-align:center;'>"
            html += f"<tr><th colspan='7' style='font-size:20px;padding:8px;'>{calendar.month_name[month]} {year}</th></tr>"
            html += "<tr>" + "".join([f"<th>{d}</th>" for d in weekday_labels]) + "</tr>"

            dots_size = 6
            for week in month_weeks:
                html += "<tr>"
                for day in week:
                    if day == 0:
                        html += "<td style='padding:8px;border:1px solid #333;'></td>"
                    else:
                        day_date = _date(year, month, day)
                        day_str = day_date.isoformat()
                        ddf = events_df[events_df["date_str"] == day_str]
                        dots = ""
                        for cat_norm in sorted(ddf["cat_norm"].dropna().unique()):
                            col = color_map_canon.get(cat_norm, "#9E9E9E")
                            label = label_map_canon.get(cat_norm, cat_norm)
                            dots += (
                                f"<div title='{label}' "
                                f"style='width:{dots_size}px;height:{dots_size}px;"
                                f"border-radius:50%;background:{col};display:inline-block;margin:1px;'></div>"
                            )
                        html += (
                            f"<td style='padding:4px;border:1px solid #333;'>{day}<br>{dots}</td>"
                        )
                html += "</tr>"
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

        with c2:
            st.markdown(
                "<div style='font-size:16px; font-weight:600;'>📌 類別</div>",
                unsafe_allow_html=True,
            )
            legend_html = ""
            for cat, col in color_map.items():
                legend_html += (
                    f"<div style='margin:2px 0;'>"
                    f"<span style='display:inline-block;width:7px;height:7px;"
                    f"border-radius:50%;background:{col};margin-right:6px;'></span>{cat}</div>"
                )
            st.markdown(legend_html, unsafe_allow_html=True)

        sel_date = st.date_input("選擇日期", value=today, key="bydate_date_picker")
        sel_date_str = sel_date.isoformat()
        day_df = events_df[events_df["date_str"] == sel_date_str][
            ["date_str", "title", "category", "participant"]
        ]

        if day_df.empty:
            st.info(f"{sel_date_str} 沒有任何紀錄。")
        else:
            cat_options = sorted(day_df["category"].astype(str).unique())
            sel_cats = st.multiselect(
                "篩選類別（可多選）",
                options=cat_options,
                default=cat_options,
                key="bydate_cats_multiselect",
            )
            show_df = day_df[day_df["category"].isin(sel_cats)].copy()

            names = sorted(show_df["participant"].astype(str).unique())
            st.write(f"**共 {len(names)} 人**：", "、".join(names) if names else "（無）")
            st.dataframe(
                show_df[["participant", "title", "category"]].sort_values(
                    ["category", "participant"]
                ),
                use_container_width=True,
                height=300,
            )
            st.download_button(
                "⬇️ 下載當日明細 Excel（匯出）",
                data=df_to_excel_bytes(show_df, "events"),
                file_name=f"events_{sel_date_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="bydate_download_btn",
            )

# -------- 3) 個人明細 --------
with tabs[3]:
    st.subheader("個人參加明細")
    if st.session_state.events.empty:
        st.info("目前尚無活動紀錄。")
    else:
        c1, c2 = st.columns(2)
        with c1:
            participants = (
                st.session_state.events["participant"]
                .astype(str)
                .fillna("")
                .unique()
                .tolist()
            )
            participants = sorted([p for p in participants if p])
            person = st.selectbox("選擇參加者", participants, key="detail_person_select")
        with c2:
            only_cat = st.multiselect(
                "篩選類別（可多選）",
                options=sorted(st.session_state.events["category"].unique()),
                default=None,
                key="detail_cats_multiselect",
            )
        dfp = st.session_state.events.query(
            "participant == @person"
        )[["date", "title", "category", "participant"]].copy()
        if only_cat:
            dfp = dfp[dfp["category"].isin(only_cat)]
        st.dataframe(
            dfp[["date", "title", "category"]].sort_values("date"),
            use_container_width=True,
            height=350,
        )
        st.download_button(
            "⬇️ 下載此人明細 Excel（匯出）",
            data=df_to_excel_bytes(dfp, "records"),
            file_name=f"{person}_records.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="detail_download_btn",
        )

# -------- 4) 完整記錄（可編輯） --------
with tabs[4]:
    st.subheader("完整記錄（可編輯）")
    st.caption("欄位：date, title, category, participant, idempotency_key（請勿修改 id 欄）")

    # 先把資料 normalize，一次處理 'None' / 'nan'
    base_df = _normalize_df(st.session_state.events)
    original_df = base_df.copy()

    # 把四個主鍵都空的幽靈列（以前刪成 None 的）直接去掉
    base_df = base_df[~base_df.apply(_is_blank_row, axis=1)].reset_index(drop=True)

    # 🔽 排序方式（只影響畫面）
    sort_mode = st.selectbox(
        "排序方式（只影響畫面）",
        [
            "依日期：新 → 舊",
            "依日期：舊 → 新",
            "姓名：A → Z",
            "姓名：Z → A",
        ],
        key="full_sort_mode",
    )

    df = base_df.copy()
    if sort_mode == "依日期：新 → 舊":
        df = df.sort_values("date", ascending=False, na_position="last")
    elif sort_mode == "依日期：舊 → 新":
        df = df.sort_values("date", ascending=True, na_position="last")
    elif sort_mode == "姓名：A → Z":
        df = df.sort_values("participant", ascending=True, na_position="last")
    elif sort_mode == "姓名：Z → A":
        df = df.sort_values("participant", ascending=False, na_position="last")

    df = df.reset_index(drop=True)

    # -------- 顯示可編輯表格 --------
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="full_editor_table",
        column_config={
            "idempotency_key": st.column_config.TextColumn("idempotency_key", disabled=True),
        },
    )

    edited_norm = _normalize_df(edited)
    edited_nonblank = edited_norm[~edited_norm.apply(_is_blank_row, axis=1)].reset_index(drop=True)

    def _keyset(df: pd.DataFrame) -> set[str]:
        if "idempotency_key" in df.columns and df["idempotency_key"].astype(str).str.len().gt(0).any():
            return set(df["idempotency_key"].astype(str))
        combo = (
            df["date"].astype(str) + "|" +
            df["title"].astype(str) + "|" +
            df["category"].astype(str) + "|" +
            df["participant"].astype(str)
        )
        return set(combo)

    # 找出這次「會被刪掉」的列（只用來顯示／之後保存時也會備份到 events_deleted）
    deleted_keys = _keyset(original_df) - _keyset(edited_nonblank)
    if "idempotency_key" in original_df.columns and original_df["idempotency_key"].astype(str).str.len().gt(0).any():
        deleted_preview = original_df[original_df["idempotency_key"].astype(str).isin(deleted_keys)]
    else:
        combo_orig = (
            original_df["date"].astype(str) + "|" +
            original_df["title"].astype(str) + "|" +
            original_df["category"].astype(str) + "|" +
            original_df["participant"].astype(str)
        )
        deleted_preview = original_df[combo_orig.isin(deleted_keys)]

    deleted_count = len(deleted_preview)
    st.info(f"本次變更偵測到：刪除 {deleted_count} 筆（按『保存變更』才會寫回，也會備份到『已刪除紀錄』）。")

    if deleted_count > 0:
        with st.expander("🔍 即將刪除的紀錄預覽", expanded=False):
            st.dataframe(
                deleted_preview[["date", "title", "category", "participant"]],
                use_container_width=True,
                height=260,
            )

    # 把這次的刪除預覽存起來，待會按下保存、輸入密碼後實際寫入 deleted sheet
    st.session_state["deleted_preview_df"] = deleted_preview

    if st.button("💾 保存變更", key="full_save_btn"):
        _need_pw("delete_rows", {"edited_df": edited_nonblank})

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ 下載 Excel（匯出）",
            data=df_to_excel_bytes(st.session_state.events, "events"),
            file_name="events_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="full_download_btn",
        )
    with c2:
        st.markdown("**🗄️ 歸檔並清空（建立新工作表備份）**")
        if st.button("執行歸檔並清空", key="full_archive_btn"):
            backup_title = f"events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            _need_pw("archive_clear", {"backup_title": backup_title})
    with c3:
        st.markdown("**♻️ 只清空（不備份）**")
        if st.button("執行只清空", key="full_clear_btn"):
            _need_pw("clear_only", {})

# -------- 5) 排行榜 --------
with tabs[5]:
    st.subheader("排行榜（依總點數）")
    ev4 = st.session_state.events[["date", "title", "category", "participant"]].copy()
    summary = aggregate(ev4, points_map, rewards)
    st.dataframe(summary, use_container_width=True, height=520)

    if summary.empty:
        st.info("目前沒有可匯出的排行榜資料。")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ 下載排行榜 Excel",
                data=df_to_excel_bytes(summary, "leaderboard"),
                file_name="leaderboard.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="leaderboard_download_excel_btn",
            )
        with c2:
            if st.button("📤 匯出排行榜到 Google Sheet（leaderboard）", key="leaderboard_export_btn"):
                ws_lb = get_or_create_ws(sh, "leaderboard", list(summary.columns))
                df_to_ws(ws_lb, summary, list(summary.columns))
                st.success("已匯出到工作表：leaderboard（已覆蓋）。")
        with c3:
            if st.button("📸 建立排行榜快照（新分頁）", key="leaderboard_snapshot_btn"):
                snap_title = f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ws_snap = get_or_create_ws(sh, snap_title, list(summary.columns))
                df_to_ws(ws_snap, summary, list(summary.columns))
                st.success(f"已建立快照：{snap_title}")

# -------- 6) 已刪除紀錄 --------
with tabs[6]:
    st.subheader("🗑 已刪除紀錄（備份區，不再計入點數）")

    try:
        deleted_df = load_deleted_from_sheet(sh)
    except Exception as e:
        st.error(f"讀取已刪除紀錄失敗：{e}")
        deleted_df = pd.DataFrame(columns=EVENT_DELETED_COLS)

    if deleted_df.empty:
        st.info("目前沒有已刪除紀錄。")
    else:
        # 盡量依 deleted_at 新→舊 排序
        if "deleted_at" in deleted_df.columns:
            deleted_df["deleted_at_sort"] = pd.to_datetime(
                deleted_df["deleted_at"], errors="coerce"
            )
            deleted_df = deleted_df.sort_values(
                ["deleted_at_sort", "date"], ascending=[False, False]
            ).drop(columns=["deleted_at_sort"])
        st.dataframe(
            deleted_df[["deleted_at", "date", "title", "category", "participant"]],
            use_container_width=True,
            height=520,
        )
        st.download_button(
            "⬇️ 下載已刪除紀錄 Excel（備份）",
            data=df_to_excel_bytes(deleted_df, "events_deleted"),
            file_name="events_deleted.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="deleted_download_btn",
        )

# === 密碼對話框 + 實際動作 ===
def _exec_pending_action():
    """依 pending_action 執行實際工作。"""
    action = st.session_state.get("pending_action")
    payload = st.session_state.get("pending_payload") or {}

    if action == "delete_rows":
        edited = payload["edited_df"]

        # 1) 讀取當前 events（舊資料，用來算真正刪掉哪些）
        before_df = load_events_from_sheet(sh)

        # 2) 算出真正被刪掉的列，並備份到 events_deleted
        def _keyset_inner(df: pd.DataFrame) -> set[str]:
            if "idempotency_key" in df.columns and df["idempotency_key"].astype(str).str.len().gt(0).any():
                return set(df["idempotency_key"].astype(str))
            combo = (
                df["date"].astype(str) + "|" +
                df["title"].astype(str) + "|" +
                df["category"].astype(str) + "|" +
                df["participant"].astype(str)
            )
            return set(combo)

        before_norm = _normalize_df(before_df)
        edited_norm = _normalize_df(edited)

        deleted_keys = _keyset_inner(before_norm) - _keyset_inner(edited_norm)
        if "idempotency_key" in before_norm.columns and before_norm["idempotency_key"].astype(str).str.len().gt(0).any():
            deleted_rows = before_norm[before_norm["idempotency_key"].astype(str).isin(deleted_keys)]
        else:
            combo_before = (
                before_norm["date"].astype(str) + "|" +
                before_norm["title"].astype(str) + "|" +
                before_norm["category"].astype(str) + "|" +
                before_norm["participant"].astype(str)
            )
            deleted_rows = before_norm[combo_before.isin(deleted_keys)]

        if not deleted_rows.empty:
            append_deleted_rows(sh, deleted_rows)

        # 3) 寫回新的 events
        save_events_to_sheet(sh, edited_norm, allow_clear=True)
        st.session_state.events = load_events_from_sheet(sh)

    elif action == "archive_clear":
        backup_title = payload["backup_title"]
        ws_backup = get_or_create_ws(sh, backup_title, EVENT_COLS)
        df_to_ws(ws_backup, st.session_state.events, EVENT_COLS)

        empty_df = st.session_state.events.iloc[0:0]
        save_events_to_sheet(sh, empty_df, allow_clear=True)
        st.session_state.events = load_events_from_sheet(sh)

    elif action == "clear_only":
        empty_df = st.session_state.events.iloc[0:0]
        save_events_to_sheet(sh, empty_df, allow_clear=True)
        st.session_state.events = load_events_from_sheet(sh)

def _show_pw_dialog():
    action = st.session_state.get("pending_action")
    if not action:
        return

    title = {
        "delete_rows": "刪除資料需要管理密碼",
        "archive_clear": "歸檔並清空需要管理密碼",
        "clear_only": "清空資料需要管理密碼",
    }.get(action, "需要管理密碼")

    def render_inner():
        pw = st.text_input("請輸入管理密碼", type="password", key="__admin_pw")
        c1, c2 = st.columns(2)
        if c1.button("確認"):
            if _check_pw(pw):
                _exec_pending_action()
                st.session_state["pending_action"] = ""
                st.session_state["pending_payload"] = {}
                st.success("已完成。")
                st.rerun()
            else:
                st.error("密碼錯誤")
        if c2.button("取消"):
            st.session_state["pending_action"] = ""
            st.session_state["pending_payload"] = {}
            st.rerun()

    if HAVE_DIALOG:
        @st.dialog(title)
        def _dlg():
            render_inner()
        _dlg()
    else:
        st.warning(title)
        render_inner()

# 若有待執行動作，顯示密碼對話框
_show_pw_dialog()