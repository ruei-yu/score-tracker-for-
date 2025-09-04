import streamlit as st
import pandas as pd
import json, io, hashlib
from datetime import date, datetime
from urllib.parse import quote, unquote
import qrcode

# --- 頁面設定 ---
st.set_page_config(
    page_title="護持活動集點(for幹部)",
    page_icon="🔢",
    layout="wide",
)

# ================= Helpers =================
def load_config(file):
    try:
        return json.load(open(file, "r", encoding="utf-8"))
    except Exception:
        return {" scoring_items": [], "rewards": []}

def save_config(cfg, file):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

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
    thresholds = sorted([r["threshold"] for r in rewards])
    def reward_badge(x):
        gain = [t for t in thresholds if x >= t]
        return (max(gain) if gain else 0)
    summary["已達門檻"] = summary["總點數"].apply(reward_badge)
    return summary.reset_index().sort_values(["總點數","participant"], ascending=[False,True])

def save_events(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_events(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["date","title","category","participant"])

# -------- Short link registry (for clean URLs) --------
def load_links(path):
    try:
        df = pd.read_csv(path, dtype=str)
        need_cols = {"code","title","category","date"}
        if not need_cols.issubset(set(df.columns)):
            return pd.DataFrame(columns=list(need_cols))
        return df
    except Exception:
        return pd.DataFrame(columns=["code","title","category","date"])

def save_links(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def make_code(title: str, category: str, iso_date: str, length: int = 8) -> str:
    """根據(標題, 類別, 日期)產生穩定短代碼；固定長度，英數字"""
    base = f"{iso_date}|{category}|{title}".encode("utf-8")
    h = hashlib.md5(base).hexdigest()  # 穩定且夠短
    return h[:length].upper()

def upsert_link(links_df: pd.DataFrame, code: str, title: str, category: str, iso_date: str) -> pd.DataFrame:
    """新增或更新 links.csv 中某個代碼的活動資訊（同 code 覆蓋）"""
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

# ============ Public check-in via URL ============
qp = st.query_params
mode = qp.get("mode", "")
# 新增短代碼參數 c；保留舊參數 event 做相容
code_param  = qp.get("c", "")
event_param = qp.get("event", "")

if mode == "checkin":
    st.markdown("### ✅ 線上報到（公開頁）")
    data_file  = st.text_input("資料儲存CSV路徑", value="events.csv", key="pub_datafile_input")
    links_file = st.text_input("連結代碼CSV路徑", value="links.csv", key="pub_linksfile_input")

    events_df = load_events(data_file)
    links_df  = load_links(links_file)

    # 取得活動資訊：優先用 c 代碼查 links.csv；若沒有 c 才嘗試舊的 event JSON
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

    # 多名同時報到（標示：紅字粗體 + 黑字說明）
st.markdown(
    """
    <div style="color:#d32f2f; font-weight:700; font-size:1rem;">
      請務必輸入全名
    </div>
    <div style="color:#000;">
     （例：陳曉瑩）（可一次多人報到，用「、」「，」或空白分隔）
    </div>
    """,
    unsafe_allow_html=True,
)

names_input = st.text_area(
    label="姓名清單",
    key="pub_names_area",
    placeholder="例如：陳曉瑩、劉宜儒、許崇萱 黃佳宜 徐睿妤",
    label_visibility="collapsed",  # 把文字輸入框上方預設標籤藏起來（我們用上面的自訂說明）
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
                to_add.append({
                    "date": target_date, "title": title,
                    "category": category, "participant": n
                })
                existing.add(n)
        if to_add:
            events_df = pd.concat([events_df, pd.DataFrame(to_add)], ignore_index=True)
            save_events(events_df, data_file)
            st.success(f"已報到 {len(to_add)} 人：{'、'.join([r['participant'] for r in to_add])}")
        if skipped:
            st.warning(f"以下人員已經報到過，已跳過：{'、'.join(skipped)}")
st.stop()

# ================= Admin UI =================
st.title("🔢護持活動集點(for幹部)")

# Sidebar settings
st.sidebar.title("⚙️ 設定")
cfg_file   = st.sidebar.text_input("設定檔路徑", value="points_config.json", key="sb_cfg_path")
data_file  = st.sidebar.text_input("資料儲存CSV路徑", value="events.csv",        key="sb_data_path")
links_file = st.sidebar.text_input("連結代碼CSV路徑", value="links.csv",         key="sb_links_path")

if "config" not in st.session_state:
    st.session_state.config = load_config(cfg_file)
if "events" not in st.session_state:
    st.session_state.events = load_events(data_file)
if "links" not in st.session_state:
    st.session_state.links = load_links(links_file)

config = st.session_state.config
scoring_items = config.get(" scoring_items", [])
rewards = config.get("rewards", [])
points_map = {i["category"]: int(i["points"]) for i in scoring_items}

# Sidebar editors
with st.sidebar.expander("➕ 編輯集點項目與點數", expanded=False):
    st.caption("新增或調整表格後點『儲存設定』。")
    items_df = pd.DataFrame(scoring_items) if scoring_items else pd.DataFrame(columns=["category","points"])
    edited = st.data_editor(items_df, num_rows="dynamic", use_container_width=True, key="sb_items_editor")
    if st.button("💾 儲存設定（集點項目）", key="sb_save_items_btn"):
        config[" scoring_items"] = edited.dropna(subset=["category"]).to_dict(orient="records")
        st.session_state.config = config
        save_config(config, cfg_file)
        st.success("已儲存集點項目。")

with st.sidebar.expander("🎁 編輯獎勵門檻", expanded=False):
    rew_df = pd.DataFrame(rewards) if rewards else pd.DataFrame(columns=["threshold","reward"])
    rew_edit = st.data_editor(rew_df, num_rows="dynamic", use_container_width=True, key="sb_rewards_editor")
    if st.button("💾 儲存設定（獎勵）", key="sb_save_rewards_btn"):
        config["rewards"] = [
            {"threshold": int(r["threshold"]), "reward": r["reward"]}
            for r in rew_edit.dropna(subset=["threshold","reward"]).to_dict(orient="records")
        ]
        st.session_state.config = config
        save_config(config, cfg_file)
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
    st.subheader("生成報到 QR Code（短連結）")
    public_base = st.text_input("公開網址（本頁網址）", value="", key="qr_public_url_input")
    if public_base.endswith("/"):
        public_base = public_base[:-1]
    qr_title    = st.text_input("活動標題", value="迎新晚會", key="qr_title_input")
    qr_category = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="qr_category_select")
    qr_date     = st.date_input("活動日期", value=date.today(), key="qr_date_picker")

    iso = qr_date.isoformat()
    code = make_code(qr_title or qr_category, qr_category, iso, length=8)

    # 更新/寫入 links.csv
    links_df = st.session_state.links
    links_df = upsert_link(links_df, code=code, title=(qr_title or qr_category),
                           category=qr_category, iso_date=iso)
    st.session_state.links = links_df
    save_links(links_df, links_file)

    # 短連結：使用 ?mode=checkin&c=CODE
    short_url = f"{public_base}/?mode=checkin&c={code}"

    # 同時保留舊長連結（相容）
    payload = json.dumps({"title": qr_title or qr_category,
                          "category": qr_category,
                          "date": iso}, ensure_ascii=False)
    encoded = quote(payload, safe="")
    long_url = f"{public_base}/?mode=checkin&event={encoded}"

    st.write("**短連結（建議分享這個）**")
    st.code(short_url, language="text")

    st.write("（備用）長連結")
    st.code(long_url, language="text")

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
        st.download_button("⬇️ 下載連結代碼 CSV",
                           data=links_df.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="links.csv", mime="text/csv",
                           key="links_download_btn")

# -------- 1) 現場報到 --------
with tabs[1]:
    st.subheader("現場快速報到（多名一起）")
    on_title    = st.text_input("活動標題", value="未命名活動", key="on_title_input")
    on_category = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="on_category_select")
    on_date     = st.date_input("日期", value=date.today(), key="on_date_picker")
    st.caption("提示：可一次輸入多位，以「、」「，」「空白」分隔，可含括號註記。")

    names_input = st.text_area("姓名清單", placeholder="曉瑩、筱晴（六） 佳宜 睿妤", key="on_names_area")
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
                save_events(ev, data_file)
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
            st.download_button("⬇️ 下載當日明細 CSV",
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
        st.download_button("⬇️ 下載此人明細 CSV",
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
    save_events(edited, data_file)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ 下載 CSV",
                           data=edited.to_csv(index=False, encoding="utf-8-sig"),
                           file_name="events_export.csv", mime="text/csv",
                           key="full_download_btn")
    with c2:
        if st.button("🗄️ 歸檔並清空", key="full_archive_btn"):
            backup_name = f"events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            edited.to_csv(backup_name, index=False, encoding="utf-8-sig")
            st.session_state.events = edited.iloc[0:0]
            save_events(st.session_state.events, data_file)
            st.success(f"已備份到 {backup_name} 並清空。")
    with c3:
        if st.button("♻️ 只清空（不備份）", key="full_clear_btn"):
            st.session_state.events = edited.iloc[0:0]
            save_events(st.session_state.events, data_file)
            st.success("已清空所有資料（未備份）。")

# -------- 5) 排行榜 --------
with tabs[5]:
    st.subheader("排行榜（依總點數）")
    summary = aggregate(st.session_state.events, points_map, rewards)
    st.dataframe(summary, use_container_width=True, height=520)
