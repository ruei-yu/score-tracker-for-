# utils_safe_url.py  — LINE 友善版
import io
import qrcode
import streamlit as st
from urllib.parse import urlsplit, urlunsplit, quote, quote_plus

# ---------- URL 清洗與生成 ----------

def _sanitize_url(url: str) -> str:
    """清洗並轉義網址，避免特殊字元造成解析問題（LINE 內建瀏覽器友善）。"""
    u = (url or "").strip()
    if not u:
        return ""

    # 若未帶 scheme，預設 https
    parts = urlsplit(u if "://" in u else "https://" + u)

    scheme = parts.scheme or "https"
    netloc = parts.netloc or parts.path

    # 路徑做百分比轉義（保留常見安全字元）
    path = quote(parts.path or "", safe="/-._~")

    # 查詢參數逐一轉義（保留 & 與 = 的結構）
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
    """
    以根網址 + 報到代碼，產生 checkin 連結。
    會自動清掉 public_base 結尾的 '/'，並轉義代碼。
    """
    base = (public_base or "").strip()
    if base.endswith("/"):
        base = base[:-1]
    return _sanitize_url(f"{base}/?mode=checkin&c={quote_plus(str(code))}")


# ---------- Streamlit 展示（只顯示一個最佳網址 + QR） ----------

def show_safe_link_box(url: str, title: str = "分享報到短連結（LINE 友善）"):
    """
    畫面上只顯示：
    1) 一個可點的超連結（a 標籤）
    2) 一段純文字網址（建議直接貼到 LINE）
    3) 一張 QR Code（給不方便點的人掃）
    """
    safe = _sanitize_url(url)
    st.subheader(title)

    # 1) 可點連結（用 <a>，避免 markdown autolink 的額外處理）
    st.markdown(
        f'<a href="{safe}" target="_blank" rel="noopener noreferrer" style="font-size:18px;">點我前往報到</a>',
        unsafe_allow_html=True,
    )

    # 2) 純文字（貼 LINE 推薦直接貼這一段，單獨一行、不要加任何符號或字）
    st.caption("LINE 請直接複製這段給大家（單獨一行，不要加文字或符號）")
    st.code(safe, language="text")

    # 3) QR Code
    img = qrcode.make(safe)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="掃碼報到", use_column_width=False)