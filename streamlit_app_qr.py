import streamlit as st
import pandas as pd
import json, io, hashlib, re
from datetime import date, datetime
from urllib.parse import quote, unquote
import qrcode

# --- 頁面設定（建議放最上面） ---
st.set_page_config(
    page_title="護持活動集點(for幹部)",
    page_icon="🔢",
    layout="wide",
)

# ================= Google Sheet Helpers =================
from google.oauth2.service_account import Credentials
import gspread
from gspread.exceptions import WorksheetNotFound

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

def _get_gspread_client():
    # ✅ 從 secrets 讀 gcp_service_account
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES
    )
    return gspread.authorize(creds)

# === 固定使用這一份 Google Sheet（從 secrets 讀取） ===
FIXED_SHEET_ID = st.secrets["google_sheets"]["sheet_id"]

@st.cache_resource(show_spinner=False)
def open_spreadsheet_by_fixed_id():
    client = _get_gspread_client()
    return client.open_by_key(FIXED_SHEET_ID)

# ✅ 一開始就打開固定的 spreadsheet（只保留一次）
sh = open_spreadsheet_by_fixed_id()

# ================= Google Sheet Helpers =================
from google.oauth2.service_account import Credentials
import gspread

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]

def _get_gspread_client():
    # ✅ 從 secrets 讀 gcp_service_account
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES
    )
    return gspread.authorize(creds)

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
    rewards_df = ws_to_df(ws_rewards, ["threshold","reward"])
    cfg = {
        "scoring_items": items_df.to_dict(orient="records"),
        "rewards": rewards_df.to_dict(orient="records"),
    }
    return cfg

def save_config_to_sheet(sh, cfg):
    ws_items = get_or_create_ws(sh, "scoring_items", ["category","points"])
    ws_rewards = get_or_create_ws(sh, "rewards", ["threshold","reward"])
    items = pd.DataFrame(cfg.get("scoring_items", [])) if cfg.get("scoring_items") else pd.DataFrame(columns=["category","points"])
    rewards = pd.DataFrame(cfg.get("rewards", [])) if cfg.get("rewards") else pd.DataFrame(columns=["threshold","reward"])
    # 整數化
    if "points" in items.columns:
        items["points"] = pd.to_numeric(items["points"], errors="coerce").fillna(0).astype(int)
    if "threshold" in rewards.columns:
        rewards["threshold"] = pd.to_numeric(rewards["threshold"], errors="coerce").fillna(0).astype(int)
    df_to_ws(ws_items, items, ["category","points"])
    df_to_ws(ws_rewards, rewards, ["threshold","reward"])

def load_events_from_sheet(sh) -> pd.DataFrame:
    ws = get_or_create_ws(sh, "events", ["date","title","category","participant"])
    return ws_to_df(ws, ["date","title","category","participant"])

def save_events_to_sheet(sh, df: pd.DataFrame):
    ws = get_or_create_ws(sh, "events", ["date","title","category","participant"])
    df_to_ws(ws, df, ["date","title","category","participant"])

def load_links_from_sheet(sh) -> pd.DataFrame:
    ws = get_or_create_ws(sh, "links", ["code","title","category","date"])
    return ws_to_df(ws, ["code","title","category","date"])

def save_links_to_sheet(sh, df: pd.DataFrame):
    ws = get_or_create_ws(sh, "links", ["code","title","category","date"])
    df_to_ws(ws, df, ["code","title","category","date"])

# ================= Query Params / Sheet ID bootstrap =================
qp = st.query_params
mode = qp.get("mode", "")
code_param  = qp.get("c", "")
event_param = qp.get("event", "")

# ============ Public check-in via URL ============
if mode == "checkin":
    st.markdown("### ✅ 線上報到")

    # sh 來自 open_spreadsheet_by_fixed_id()，固定你的那份 sheet
    if not sh:
        st.error("找不到 Google Sheet。")
        st.stop()

    events_df = load_events_from_sheet(sh)
    links_df  = load_links_from_sheet(sh)
    ...


    # 取得活動資訊：優先用 c 代碼查 links；若沒有 c 才嘗試舊的 event JSON
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
        <div style="color:#d32f2f; font-weight:700;">
          請務必輸入全名
        </div>
        <div style="color:#000;">
        （例：陳曉瑩）
        （可一次多人報到，用「、」「，」或空白分隔）
        </div>
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
            existing = set(
                events_df.loc[
                    (events_df["date"] == target_date) &
                    (events_df["title"] == title) &
                    (events_df["category"] == category),
                    "participant"
                ].astype(str).tolist()
            )
            to_add, skipped = [], []
            for n in names:
                if n in existing:
                    skipped.append(n)
                else:
                    to_add.append({"date": target_date, "title": title,
                                   "category": category, "participant": n})
                    existing.add(n)
            if to_add:
                events_df = pd.concat([events_df, pd.DataFrame(to_add)], ignore_index=True)
                save_events_to_sheet(sh, events_df)
                st.success(f"已報到 {len(to_add)} 人：{'、'.join([r['participant'] for r in to_add])}")
            if skipped:
                st.warning(f"以下人員已經報到過，已跳過：{'、'.join(skipped)}")
    st.stop()

# ================= Admin UI =================
st.title("🔢護持活動集點(for幹部)")

# Sidebar settings（用 Google Sheet 而不是檔案路徑）
st.sidebar.title("⚙️ 設定（Google Sheet）")
st.sidebar.success(
    f"已綁定試算表：{st.secrets['google_sheets']['sheet_id']}"
)


# 載入設定 / 資料
if "config" not in st.session_state:
    st.session_state.config = load_config_from_sheet(sh)
if "events" not in st.session_state:
    st.session_state.events = load_events_from_sheet(sh)
if "links" not in st.session_state:
    st.session_state.links = load_links_from_sheet(sh)

config = st.session_state.config
# 統一 key：scoring_items
scoring_items = config.get("scoring_items", [])
rewards = config.get("rewards", [])
# 轉換 points_map
points_map = {}
for i in scoring_items:
    if "category" in i:
        try:
            points_map[i["category"]] = int(i.get("points", 0))
        except:
            points_map[i["category"]] = 0

# Sidebar editors
with st.sidebar.expander("➕ 編輯集點項目與點數", expanded=False):
    st.caption("新增或調整表格後點『儲存設定』。")
    items_df = pd.DataFrame(scoring_items) if scoring_items else pd.DataFrame(columns=["category","points"])
    edited = st.data_editor(items_df, num_rows="dynamic", use_container_width=True, key="sb_items_editor")
    if st.button("💾 儲存設定（集點項目）", key="sb_save_items_btn"):
        cfg = st.session_state.config
        # 清理空列與型別
        if not edited.empty:
            edited["category"] = edited["category"].astype(str)
            edited["points"] = pd.to_numeric(edited["points"], errors="coerce").fillna(0).astype(int)
            edited = edited.dropna(subset=["category"])
        cfg["scoring_items"] = edited.to_dict(orient="records")
        st.session_state.config = cfg
        save_config_to_sheet(sh, cfg)
        st.success("已儲存集點項目。")

with st.sidebar.expander("🎁 編輯獎勵門檻", expanded=False):
    rew_df = pd.DataFrame(rewards) if rewards else pd.DataFrame(columns=["threshold","reward"])
    rew_edit = st.data_editor(rew_df, num_rows="dynamic", use_container_width=True, key="sb_rewards_editor")
    if st.button("💾 儲存設定（獎勵）", key="sb_save_rewards_btn"):
        cfg = st.session_state.config
        if not rew_edit.empty:
            rew_edit["reward"] = rew_edit["reward"].astype(str)
            rew_edit["threshold"] = pd.to_numeric(rew_edit["threshold"], errors="coerce").fillna(0).astype(int)
            rew_edit = rew_edit.dropna(subset=["threshold","reward"])
        cfg["rewards"] = rew_edit.to_dict(orient="records")
        st.session_state.config = cfg
        save_config_to_sheet(sh, cfg)
        st.success("已儲存獎勵門檻。")

# ============== Tabs (custom order) ==============
tabs = st.tabs([
    "🟪 產生 QRcode",        # 0
    "📝 現場報到",           # 1
    "📆 依日期查看參與者",   # 2
    "👤 個人明細",           # 3
    "📒 完整記錄",           # 4
    "🏆 排行榜",             # 5
])

# -------- 0) 產生 QRcode（含短代碼） --------
with tabs[0]:
    st.subheader("生成報到 QR Code")
    public_base = st.text_input("公開網址（本頁網址）", value="", key="qr_public_url_input")
    if public_base.endswith("/"):
        public_base = public_base[:-1]
    qr_title    = st.text_input("活動標題", value="迎新晚會", key="qr_title_input")
    qr_category = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="qr_category_select")
    qr_date     = st.date_input("活動日期", value=date.today(), key="qr_date_picker")

    iso = qr_date.isoformat()
    code = make_code(qr_title or qr_category, qr_category, iso, length=8)

    # 更新/寫入 links（Google Sheet）
    links_df = st.session_state.links
    links_df = upsert_link(links_df, code=code, title=(qr_title or qr_category),
                           category=qr_category, iso_date=iso)
    st.session_state.links = links_df
    save_links_to_sheet(sh, links_df)

    # ✅ 短連結：使用固定的 sheet，不需 sid
    short_url = f"{public_base}/?mode=checkin&c={code}"
    st.write("**短連結（建議分享這個）**")
    st.code(short_url, language="text")

    # 產生 QR（用短連結）
    if public_base:
        img = qrcode.make(short_url)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption=f"掃描報到 ｜ 代碼：{code}", width=260)
        st.download_button("⬇️ 下載 QR 圖片", data=buf.getvalue(),
                           file_name=f"checkin_{code}.png",
                           mime="image/png", key="qr_download_btn")
    else:
        st.info("請貼上你的 .streamlit.app 根網址（本頁網址）。")

    with st.expander("🔎 目前所有短代碼一覽", expanded=False):
        st.dataframe(links_df.sort_values("date", ascending=False), use_container_width=True, height=220)
        st.download_button("⬇️ 下載連結代碼 CSV（匯出）",
                           data=links_df.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="links.csv", mime="text/csv",
                           key="links_download_btn")
    # 清空 links
    if st.button("🧹 清空所有短代碼（links）", key="links_clear_btn"):
        st.session_state.links = st.session_state.links.iloc[0:0]
        save_links_to_sheet(sh, st.session_state.links)
        st.success("已清空所有短代碼。")

# -------- 1) 現場報到 --------
with tabs[1]:
    st.subheader("現場快速報到")
    on_title    = st.text_input("活動標題", value="未命名活動", key="on_title_input")
    on_category = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="on_category_select")
    on_date     = st.date_input("日期", value=date.today(), key="on_date_picker")
    st.markdown(
        """
        <div style="color:#d32f2f; font-weight:700;">
          請務必輸入全名（例：陳曉瑩）
        </div>
        <div style="color:#000;">
          （可一次多人報到，用「、」「，」或空白分隔）
        </div>
        """,
        unsafe_allow_html=True,
    )
    names_input = st.text_area("姓名清單", placeholder="例如：陳曉瑩、蕭雅云，張詠禎 徐睿妤",
                               key="on_names_area", label_visibility="collapsed")
    if st.button("➕ 加入報到名單", key="on_add_btn"):
        ev = st.session_state.events.copy()
        target_date = on_date.isoformat()
        names = normalize_names(names_input)
        if not names:
            st.warning("請至少輸入一位姓名。")
        else:
            existing = set(
                ev.loc[
                    (ev["date"] == target_date) &
                    (ev["title"] == on_title) &
                    (ev["category"] == on_category),
                    "participant"
                ].astype(str).tolist()
            )
            to_add, skipped = [], []
            for n in names:
                if n in existing:
                    skipped.append(n)
                else:
                    to_add.append({"date": target_date, "title": on_title,
                                   "category": on_category, "participant": n})
                    existing.add(n)
            if to_add:
                ev = pd.concat([ev, pd.DataFrame(to_add)], ignore_index=True)
                st.session_state.events = ev
                save_events_to_sheet(sh, ev)
                st.success(f"已加入 {len(to_add)} 人：{'、'.join([r['participant'] for r in to_add])}")
            if skipped:
                st.warning(f"已跳過（重複）：{'、'.join(skipped)}")

# -------- 2) 依日期查看參與者 --------
with tabs[2]:
    st.subheader("依日期查看參與者")
    if st.session_state.events.empty:
        st.info("目前尚無活動紀錄。")
    else:
        sel_date = st.date_input("選擇日期", value=date.today(), key="bydate_date_picker")
        sel_date_str = sel_date.isoformat()
        day_df = st.session_state.events[st.session_state.events["date"].astype(str) == sel_date_str].copy()
        if day_df.empty:
            st.info(f"{sel_date_str} 沒有任何紀錄。")
        else:
            cat_options = sorted(day_df["category"].astype(str).unique())
            sel_cats = st.multiselect("篩選類別（可多選）",
                                      options=cat_options, default=cat_options,
                                      key="bydate_cats_multiselect")
            show_df = day_df[day_df["category"].isin(sel_cats)].copy()
            names = sorted(show_df["participant"].astype(str).unique())
            st.write(f"**共 {len(names)} 人**：", "、".join(names) if names else "（無）")
            st.dataframe(show_df[["participant","title","category"]]
                         .sort_values(["category","participant"]),
                         use_container_width=True, height=300)
            st.download_button("⬇️ 下載當日明細 CSV（匯出）",
                               data=show_df.to_csv(index=False, encoding="utf-8-sig"),
                               file_name=f"events_{sel_date_str}.csv", mime="text/csv",
                               key="bydate_download_btn")

# -------- 3) 個人明細 --------
with tabs[3]:
    st.subheader("個人參加明細")
    if st.session_state.events.empty:
        st.info("目前尚無活動紀錄。")
    else:
        c1, c2 = st.columns(2)
        with c1:
            person = st.selectbox("選擇參加者",
                                  sorted(st.session_state.events["participant"].unique()),
                                  key="detail_person_select")
        with c2:
            only_cat = st.multiselect("篩選類別（可多選）",
                                      options=sorted(st.session_state.events["category"].unique()),
                                      default=None, key="detail_cats_multiselect")
        dfp = st.session_state.events.query("participant == @person").copy()
        if only_cat:
            dfp = dfp[dfp["category"].isin(only_cat)]
        st.dataframe(dfp[["date","title","category"]].sort_values("date"),
                     use_container_width=True, height=350)
        st.download_button("⬇️ 下載此人明細 CSV（匯出）",
                           data=dfp.to_csv(index=False, encoding="utf-8-sig"),
                           file_name=f"{person}_records.csv", mime="text/csv",
                           key="detail_download_btn")

# -------- 4) 完整記錄 --------
with tabs[4]:
    st.subheader("完整記錄（可編輯）")
    st.caption("欄位：date, title, category, participant")
    edited = st.data_editor(st.session_state.events, num_rows="dynamic",
                            use_container_width=True, key="full_editor_table")
    st.session_state.events = edited
    save_events_to_sheet(sh, edited)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ 下載 CSV（匯出）",
                           data=edited.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="events_export.csv", mime="text/csv",
                           key="full_download_btn")
    with c2:
        if st.button("🗄️ 歸檔並清空（建立新工作表備份）", key="full_archive_btn"):
            backup_title = f"events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ws_backup = get_or_create_ws(sh, backup_title, ["date","title","category","participant"])
            df_to_ws(ws_backup, edited, ["date","title","category","participant"])
            st.session_state.events = edited.iloc[0:0]
            save_events_to_sheet(sh, st.session_state.events)
            st.success(f"已備份到工作表：{backup_title} 並清空。")
    with c3:
        if st.button("♻️ 只清空（不備份）", key="full_clear_btn"):
            st.session_state.events = edited.iloc[0:0]
            save_events_to_sheet(sh, st.session_state.events)
            st.success("已清空所有資料（未備份）。")

# -------- 5) 排行榜 --------
with tabs[5]:
    st.subheader("排行榜（依總點數）")
    summary = aggregate(st.session_state.events, points_map, rewards)
    st.dataframe(summary, use_container_width=True, height=520)
