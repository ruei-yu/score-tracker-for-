# utils_safe_url.py
import io, qrcode
from urllib.parse import urlsplit, urlunsplit, quote, quote_plus

def sanitize_url(url: str) -> str:
    """清洗並轉義網址，避免特殊字元造成解析問題。"""
    u = url.strip()
    parts = urlsplit(u if "://" in u else "https://" + u)

    scheme = parts.scheme or "https"
    netloc = parts.netloc or parts.path
    path = quote(parts.path or "", safe="/-._~")

    # 參數處理
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


def share_variants(url: str):
    """輸出各平台較安全的分享格式"""
    safe = sanitize_url(url)
    return {
        "plain": safe,
        "slack_discord": f"<{safe}>",
        "markdown_code": f"`{safe}`",
        "html_anchor": f'<a href="{safe}" target="_blank" rel="noopener noreferrer">點我前往</a>',
    }


# ===== Streamlit 元件 =====
import streamlit as st

def show_safe_link_box(url: str, title: str = "報到連結"):
    v = share_variants(url)
    st.subheader(title)

    # 可點的連結
    st.markdown(v["html_anchor"], unsafe_allow_html=True)

    # 純網址
    st.caption("建議直接複製這段給大家：")
    st.code(v["plain"])

    # Slack/Discord
    st.caption("Slack/Discord 建議格式：")
    st.code(v["slack_discord"])

    # Markdown 格式
    st.caption("Markdown 安全格式：")
    st.code(v["markdown_code"])

    # QR Code
    img = qrcode.make(v["plain"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    st.image(buf.getvalue(), caption="掃碼報到", use_column_width=False)
