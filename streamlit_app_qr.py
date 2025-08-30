import streamlit as st
import pandas as pd
import json, io
from datetime import date, datetime
from urllib.parse import quote, unquote
import qrcode

st.set_page_config(page_title="集點計分器 + 報到QR", page_icon="🔢", layout="wide")

# ---------- Helpers ----------
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
    raw = (
        s.replace("、", ",")
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
    summary = summary.reset_index().sort_values(
        ["總點數", "participant"], ascending=[False, True]
    )
    return summary

def save_events(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_events(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["date","title","category","participant"])

# ---------- Query Param: Public check-in mode ----------
qp = st.query_params                           # ← 新API
mode = qp.get("mode", "")
event_param = qp.get("event", "")

if mode == "checkin":
    st.markdown("### ✅ 線上報到")
    data_file = st.text_input("資料儲存CSV路徑", value="events.csv", key="ci_datafile")
    events_df = load_events(data_file)

    # event info from URL
    title, category, target_date = "未命名活動", "活動護持（含宿訪）", date.today().isoformat()
    try:
        decoded = unquote(event_param)
        if decoded.strip().startswith("{"):
            o = json.loads(decoded)
            title = o.get("title", title)
            category = o.get("category", category)
            target_date = o.get("date", target_date)
        else:
            title = decoded or title
    except Exception:
        pass

    st.info(f"活動：**{title}**｜類別：**{category}**｜日期：{target_date}")

    # ✅ 一次輸入多位：用「、」「，」或空白分隔；可含括號註記（會自動去除）
    names_input = st.text_area(
        "請輸入姓名（可用「、」「，」或空白分隔；可含括號註記）",
        key="ci_names",
        placeholder="例如：曉瑩、筱晴、崇萱（六） 佳宜 睿妤"
    )

    if st.button("送出報到", key="ci_submit"):
        names = normalize_names(names_input)
        if not names:
            st.error("請至少輸入一位姓名。")
        else:
            # 加速查重：同日、同標題、同類別既有名單
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
                    skipped.append(n)   # 已報到，跳過
                else:
                    to_add.append({
                        "date": target_date, "title": title,
                        "category": category, "participant": n
                    })
                    existing.add(n)    # 防止同批重覆

            if to_add:
                events_df = pd.concat([events_df, pd.DataFrame(to_add)], ignore_index=True)
                save_events(events_df, data_file)
                st.success(f"已報到 {len(to_add)} 人：{'、'.join([r['participant'] for r in to_add])}")

            if skipped:
                st.warning(f"以下人員已經報到過，已跳過：{'、'.join(skipped)}")

    st.stop()

# ---------- Admin / Normal UI ----------
st.title("🔢 集點計分器 + 報到QR")

st.sidebar.title("⚙️ 設定")
cfg_file = st.sidebar.text_input("設定檔路徑", value="points_config.json", key="cfg_path")
data_file = st.sidebar.text_input("資料儲存CSV路徑", value="events.csv", key="data_path")

if "config" not in st.session_state:
    st.session_state.config = load_config(cfg_file)
if "events" not in st.session_state:
    st.session_state.events = load_events(data_file)

config = st.session_state.config
scoring_items = config.get(" scoring_items", [])
rewards = config.get("rewards", [])
points_map = {i["category"]: int(i["points"]) for i in scoring_items}

# 👉 在 App 內編輯 scoring_items & rewards
with st.sidebar.expander("➕ 編輯集點項目與點數", expanded=False):
    st.caption("新增或調整右側表格後點『儲存設定』。")
    items_df = pd.DataFrame(scoring_items) if scoring_items else pd.DataFrame(columns=["category", "points"])
    edited = st.data_editor(items_df, num_rows="dynamic", use_container_width=True, key="items_editor")
    if st.button("💾 儲存設定（集點項目）", key="save_items"):
        config[" scoring_items"] = edited.dropna(subset=["category"]).to_dict(orient="records")
        st.session_state.config = config
        save_config(config, cfg_file)
        st.success("已儲存集點項目。")

with st.sidebar.expander("🎁 編輯獎勵門檻", expanded=False):
    rew_df = pd.DataFrame(rewards) if rewards else pd.DataFrame(columns=["threshold", "reward"])
    rew_edit = st.data_editor(rew_df, num_rows="dynamic", use_container_width=True, key="rewards_editor")
    if st.button("💾 儲存設定（獎勵）", key="save_rewards"):
        config["rewards"] = [
            {"threshold": int(r["threshold"]), "reward": r["reward"]}
            for r in rew_edit.dropna(subset=["threshold", "reward"]).to_dict(orient="records")
        ]
        st.session_state.config = config
        save_config(config, cfg_file)
        st.success("已儲存獎勵門檻。")

# --- Tabs ---
tabs = st.tabs(["📥 管理與統計", "📱 產生報到 QR"])

# --- Tab 1: 管理與統計 ---
with tabs[0]:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("📥 匯入或建立出席資料")
        uploaded = st.file_uploader("上傳 CSV", type=["csv"], key="upload_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.events = df
            save_events(df, data_file)
            st.success(f"已載入 {len(df)} 筆")

        # quick add
        d = st.date_input("日期", value=date.today(), key="add_date")
        cat = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="add_cat")
        title = st.text_input("標題", value="", key="add_title")
        names_text = st.text_area("參與名單（以、或，或空白分隔，可含註記）", key="add_names")
        if st.button("➕ 新增到列表", key="add_btn"):
            names = normalize_names(names_text)
            new_rows = pd.DataFrame([{
                "date": d.isoformat(), "title": title or cat,
                "category": cat, "participant": n
            } for n in names])
            st.session_state.events = pd.concat([st.session_state.events, new_rows], ignore_index=True)
            save_events(st.session_state.events, data_file)
            st.success(f"已新增 {len(new_rows)} 筆")

        st.markdown("#### 🧰 歸檔與重置")
        if st.button("🗄️ 歸檔並清空", key="archive_clear"):
            backup_name = f"events_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.session_state.events.to_csv(backup_name, index=False, encoding="utf-8-sig")
            st.session_state.events = st.session_state.events.iloc[0:0]
            save_events(st.session_state.events, data_file)
            st.success(f"已備份到 {backup_name} 並清空。")
        if st.button("♻️ 只清空（不備份）", key="just_clear"):
            st.session_state.events = st.session_state.events.iloc[0:0]
            save_events(st.session_state.events, data_file)
            st.success("已清空所有資料（未備份）。")

        st.download_button(
            "⬇️ 下載事件CSV",
            data=st.session_state.events.to_csv(index=False, encoding="utf-8-sig"),
            file_name="events_export.csv", mime="text/csv", key="dl_events"
        )

    with right:
        st.subheader("📊 統計與獎勵")
        summary = aggregate(st.session_state.events, points_map, rewards)
        st.dataframe(summary, use_container_width=True, height=520)

        # 額外：活動明細表
        st.markdown("#### 📅 個人參加明細")
        if not st.session_state.events.empty:
            selected_person = st.selectbox(
                "選擇要查看的參加者",
                sorted(st.session_state.events["participant"].unique()),
                key="detail_person"
            )
            person_events = st.session_state.events.query("participant == @selected_person")
            st.dataframe(
                person_events[["date", "title", "category"]].sort_values("date"),
                use_container_width=True
            )
        else:
            st.info("目前尚無活動紀錄。")
        # === 依日期查看參與者 ===
st.markdown("#### 📆 依日期查看參與者")
if not st.session_state.events.empty:
    # 用 date_input 讓管理者自由挑日期
    sel_date = st.date_input("選擇日期", value=date.today(), key="bydate_date")
    sel_date_str = sel_date.isoformat()

    day_df = st.session_state.events[
        st.session_state.events["date"].astype(str) == sel_date_str
    ].copy()

    if day_df.empty:
        st.info(f"{sel_date_str} 沒有任何紀錄。")
    else:
        # 可選擇要看的類別（預設全選）
        cat_options = sorted(day_df["category"].astype(str).unique())
        sel_cats = st.multiselect(
            "篩選類別（可多選）", options=cat_options, default=cat_options, key="bydate_cats"
        )
        show_df = day_df[day_df["category"].isin(sel_cats)].copy()

        # 顯示參與者名單（去重、排序）
        names = sorted(show_df["participant"].astype(str).unique())
        st.write(f"**共 {len(names)} 人**：", "、".join(names) if names else "（無）")

        # 明細表：誰參加了什麼活動
        st.dataframe(
            show_df[["participant", "title", "category"]]
                .sort_values(["category", "participant"]),
            use_container_width=True,
            height=300,
        )

        # 下載當日名單/明細
        st.download_button(
            "⬇️ 下載當日明細 CSV",
            data=show_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"events_{sel_date_str}.csv",
            mime="text/csv",
            key="bydate_download",
        )
else:
    st.info("目前尚無活動紀錄。")

# --- Tab 2: 產生報到 QR ---
with tabs[1]:
    st.subheader("生成報到 QR Code")

    # 加上唯一 key，避免 DuplicateElementId
    public_base = st.text_input("公開網址", value="", key="qr_public_url")
    if public_base.endswith("/"):
        public_base = public_base[:-1]

    title = st.text_input("活動標題", value="迎新晚會", key="qr_title")
    category = st.selectbox("類別", list(points_map.keys()) or ["活動護持（含宿訪）"], key="qr_category")
    qr_date = st.date_input("活動日期", value=date.today(), key="qr_date")

    event_payload = json.dumps({
        "title": title or category,
        "category": category,
        "date": qr_date.isoformat()
    }, ensure_ascii=False)
    encoded = quote(event_payload, safe="")

    if public_base:
        checkin_url = f"{public_base}/?mode=checkin&event={encoded}"
        st.write("**報到連結：**")
        st.code(checkin_url, language="text")

        # 產生 QR
        img = qrcode.make(checkin_url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="請讓大家掃描此 QR 報到", width=260)
        st.download_button(
            "⬇️ 下載 QR 圖片", data=buf.getvalue(),
            file_name=f"checkin_qr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png", key="qr_download"
        )
    else:
        st.info("請先貼上當前公開網址（例如本頁的根網址 https://xxx.streamlit.app）。")
