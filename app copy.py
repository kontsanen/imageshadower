"""
Harvia Labs — Cast-a-shadow beta
Automated perspective-shadow compositing for e-commerce product images.
"""

import base64
import io
import os
from collections import deque
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cast-a-shadow beta — Harvia Labs",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (only what config.toml cannot do) ──────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');
  @font-face { font-family:'Chevin Pro'; src:local('Chevin Pro'),local('ChevinPro'); }
  @font-face { font-family:'Gotham';     src:local('Gotham'),local('GothamBook'); }

  html, body, [class*="css"] { font-family:'Chevin Pro','Gotham','Barlow',sans-serif !important; }

  :root {
    --bg:      #EAE8E0;
    --bg-card: #F5F4F0;
    --red:     #ED1C24;
    --red-h:   #C41520;
    --text:    #505045;
    --muted:   #727266;
    --border:  #D9D6C8;
    --chip-bg: #EAE8E0;
  }

  /* Layout */
  .block-container { padding-top:1rem !important; padding-left:1.2rem !important; padding-right:1.2rem !important; max-width:none !important; }
  [data-testid="stHorizontalBlock"] { align-items:flex-start !important; }

  /* Sidebar header gap */
  [data-testid="stSidebarHeader"] { height:0 !important; min-height:0 !important; padding:0 !important; margin:0 !important; overflow:hidden !important; }
  section[data-testid="stSidebar"] > div,
  [data-testid="stSidebarContent"] { padding-top:0 !important; }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { margin-top:0 !important; padding-top:0 !important; gap:0.65rem !important; }
  section[data-testid="stSidebar"] hr { margin:0.3rem 0 !important; }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] small,
  section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }

  /* File uploader */
  [data-testid="stFileUploaderDropzone"] { background:var(--bg-card) !important; border:1.5px dashed var(--border) !important; border-radius:10px !important; min-height:110px !important; padding:1rem !important; }
  [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--red) !important; }
  [data-testid="stFileUploaderDropzone"] button { background:transparent !important; border:1px solid var(--border) !important; border-radius:6px !important; box-shadow:none !important; color:var(--muted) !important; font-size:0.80rem !important; }
  [data-testid="stFileUploaderDropzone"] button:hover { color:var(--red) !important; border-color:var(--red) !important; }
  [data-testid="stFileUploaderDropzone"] small { color:var(--muted) !important; font-size:0.68rem !important; }
  [data-testid="stFileUploaderDropzoneDragOver"],
  [data-testid="stFileUploaderDropzone"]:focus-within { background-color:rgba(217,214,200,0.25) !important; border-color:var(--border) !important; }

  /* Slider: suppress tick-bar red box on hover (thumb handled by config.toml) */
  .stSlider [data-baseweb="slider"] > div:last-child { background-color:transparent !important; }
  .stSlider [role="slider"]:focus,
  .stSlider [role="slider"]:focus-visible { outline:none !important; box-shadow:none !important; }

  /* Buttons */
  .stButton > button, .stDownloadButton > button { border-radius:6px !important; font-weight:600 !important; letter-spacing:0.03em; }

  /* Custom spinner */
  @keyframes cas-spin { to { transform:rotate(360deg); } }
  .cas-status { display:flex; align-items:center; gap:0.55rem; color:#505045; font-size:0.90rem; margin:0.6rem 0; }
  .cas-ring { flex-shrink:0; width:16px; height:16px; border:2px solid #D9D6C8; border-top-color:#727266; border-radius:50%; animation:cas-spin 0.75s linear infinite; }

  /* Custom components */
  .section-label { font-size:0.70rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:0.35rem; }
  .result-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:0.9rem 1.5rem; margin-top:2rem; margin-bottom:1.25rem; }
  .header-bar { display:flex; align-items:center; gap:0.8rem; margin-bottom:1.4rem; }
  .chip { display:inline-block; background:var(--chip-bg); border:1px solid var(--border); border-radius:20px; padding:0.15rem 0.6rem; font-size:0.74rem; font-weight:500; color:var(--text); }

  /* Expander */
  [data-testid="stExpander"] > details { background-color:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary p { background-color:var(--bg-card) !important; color:var(--text) !important; }

  /* st.code */
  [data-testid="stCode"] { background-color:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:6px !important; }
  [data-testid="stCode"] code { color:var(--text) !important; background:transparent !important; }

  /* Alert boxes */
  [data-testid="stAlert"] { background-color:var(--bg-card) !important; border:1px solid var(--border) !important; border-left:3px solid var(--muted) !important; border-radius:8px !important; }
  [data-testid="stAlert"] p { color:var(--muted) !important; }
  [data-testid="stAlert"] svg { display:none !important; }

  #MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_SHADOW_FILE = "default_shadow.png"
LOGO_FILE           = "harvia_labs_red.png"
ALPHA_THRESHOLD     = 25
WHITE_THRESHOLD     = 235
BLACK_THRESHOLD     = 20


# ── Pixel masks ────────────────────────────────────────────────────────────────

def valid_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """Product pixels: reject transparent, near-white, near-black."""
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    return ~(
        (a < ALPHA_THRESHOLD) |
        ((r > WHITE_THRESHOLD) & (g > WHITE_THRESHOLD) & (b > WHITE_THRESHOLD)) |
        ((r < BLACK_THRESHOLD) & (g < BLACK_THRESHOLD) & (b < BLACK_THRESHOLD))
    )


def shadow_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """Shadow pixels: reject only fully transparent (shadow body is near-black)."""
    return arr[:,:,3] >= ALPHA_THRESHOLD


# ── Background removal ─────────────────────────────────────────────────────────

def has_transparency(img: Image.Image) -> bool:
    if img.mode not in ("RGBA", "LA", "PA"):
        return False
    arr = np.array(img.convert("RGBA"))
    return bool((arr[:,:,3] < 200).mean() > 0.005)


def remove_solid_background(img: Image.Image, tolerance: int = 30) -> Image.Image:
    """Trimap / alpha-matting background removal."""
    img_rgba = img.convert("RGBA")
    arr = np.array(img_rgba, dtype=np.float32)
    H, W = arr.shape[:2]

    # Background colour from corners
    c = min(max(10, int(min(H, W) * 0.03)), H // 4, W // 4)
    corners = np.vstack([
        arr[:c,  :c,  :3].reshape(-1, 3),
        arr[:c,  -c:, :3].reshape(-1, 3),
        arr[-c:, :c,  :3].reshape(-1, 3),
        arr[-c:, -c:, :3].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    # Colour-distance map
    dist = np.sqrt(((arr[:,:,:3] - bg_color) ** 2).sum(axis=2))
    is_bg_candidate = dist <= tolerance

    # BFS flood-fill on downscaled mask
    WORK_SIZE = 400
    scale = min(1.0, WORK_SIZE / max(H, W))
    sh, sw = max(1, int(H * scale)), max(1, int(W * scale))

    mask_small = np.array(
        Image.fromarray((is_bg_candidate * 255).astype(np.uint8), "L")
        .resize((sw, sh), Image.Resampling.NEAREST)
    ) > 128

    bg_small = np.zeros((sh, sw), dtype=bool)
    queue: deque = deque()

    def _seed(y: int, x: int) -> None:
        if mask_small[y, x] and not bg_small[y, x]:
            bg_small[y, x] = True
            queue.append((y, x))

    for y in range(sh): _seed(y, 0); _seed(y, sw - 1)
    for x in range(sw): _seed(0, x); _seed(sh - 1, x)

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < sh and 0 <= nx < sw and not bg_small[ny, nx] and mask_small[ny, nx]:
                bg_small[ny, nx] = True
                queue.append((ny, nx))

    # Dilate BG in lower half to absorb floor-shadow halos
    split = sh // 2
    bg_lower = bg_small.copy()
    bg_lower[:split, :] = False
    bg_pil_low = Image.fromarray((bg_lower * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(11))
    bg_small = bg_small | (np.array(bg_pil_low) > 128)

    # Upsample to full resolution
    fg_mask = np.array(
        Image.fromarray(((~bg_small) * 255).astype(np.uint8), "L")
        .resize((W, H), Image.Resampling.NEAREST)
    ) > 128

    # Transition zone
    trans_px = min(30, max(12, int(max(H, W) * 0.015)))
    ksize = 2 * trans_px + 1
    fg_pil  = Image.fromarray((fg_mask * 255).astype(np.uint8), "L")
    fg_core = np.array(fg_pil.filter(ImageFilter.MinFilter(ksize))) > 128
    fg_halo = np.array(fg_pil.filter(ImageFilter.MaxFilter(ksize))) > 128
    transition = fg_halo & ~fg_core

    # Alpha: core=1, transition=colour-ramp, bg=0
    alpha = np.where(fg_core, 1.0, np.where(transition, np.clip(dist / tolerance, 0, 1), 0.0))
    alpha = np.array(
        Image.fromarray((alpha * 255).astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(0.8)),
        dtype=np.float32,
    ) / 255.0

    # Defringe semi-transparent pixels
    rgb  = arr[:,:,:3].copy()
    semi = (alpha > 0.01) & (alpha < 0.99)
    if semi.any():
        a3  = alpha[:,:,np.newaxis]
        bg3 = bg_color[np.newaxis, np.newaxis, :]
        rgb = np.where(semi[:,:,np.newaxis], np.clip((rgb - (1 - a3) * bg3) / np.clip(a3, 0.01, 1), 0, 255), rgb)

    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[:,:,:3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:,:, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


# ── Shadow stripping ───────────────────────────────────────────────────────────

def strip_baked_shadow(img: Image.Image) -> Image.Image:
    """Crop out the baked-in floor shadow below the solid product base."""
    arr  = np.array(img.convert("RGBA"))
    mask = valid_pixel_mask(arr)
    row_counts  = mask.sum(axis=1).astype(float)
    body_counts = row_counts[:max(1, int(img.height * 0.70))]
    peak = body_counts.max()
    if peak == 0:
        return img
    threshold = peak * 0.08
    base_row = img.height - 1
    for row in range(img.height - 1, -1, -1):
        if row_counts[row] >= threshold:
            base_row = row
            break
    margin = max(20, int(img.height * 0.03))
    return img.crop((0, 0, img.width, min(base_row + margin, img.height)))


# ── Anchor detection ───────────────────────────────────────────────────────────

def _bottom_profile(mask: np.ndarray) -> np.ndarray:
    """Per-column index of the bottommost valid pixel, or -1 if none."""
    H = mask.shape[0]
    has_any = mask.any(axis=0)
    bottom  = H - 1 - np.argmax(mask[::-1, :], axis=0)
    return np.where(has_any, bottom, -1).astype(int)


def find_bottom_left_anchor(img: Image.Image) -> Optional[Tuple[int, int]]:
    arr  = np.array(img.convert("RGBA"))
    mask = valid_pixel_mask(arr)
    prof = _bottom_profile(mask)

    valid_cols = np.where(prof >= 0)[0]
    if len(valid_cols) == 0:
        return None

    # Column-density guard: reject sparse columns (shadow remnants, fringe)
    col_counts = mask.sum(axis=0)
    min_count  = max(3, int(mask.shape[0] * 0.02))
    valid_cols = valid_cols[col_counts[valid_cols] >= min_count]
    if len(valid_cols) == 0:
        return None

    global_bottom = int(prof[valid_cols].max())
    top_rows  = np.where(mask.any(axis=1))[0]
    obj_h     = global_bottom - int(top_rows.min()) if len(top_rows) else 100
    tolerance = max(8, int(obj_h * 0.03))

    candidates = valid_cols[prof[valid_cols] >= global_bottom - tolerance]
    if len(candidates) == 0:
        candidates = valid_cols

    ax = int(candidates.min())
    return ax, int(prof[ax])


def find_bottom_right_anchor(img: Image.Image) -> Optional[Tuple[int, int]]:
    arr  = np.array(img.convert("RGBA"))
    mask = shadow_pixel_mask(arr)
    prof = _bottom_profile(mask)

    valid_cols = np.where(prof >= 0)[0]
    if len(valid_cols) == 0:
        return None

    global_bottom = int(prof[valid_cols].max())
    top_rows  = np.where(mask.any(axis=1))[0]
    obj_h     = global_bottom - int(top_rows.min()) if len(top_rows) else 100
    tolerance = max(8, int(obj_h * 0.03))

    candidates = valid_cols[prof[valid_cols] >= global_bottom - tolerance]
    if len(candidates) == 0:
        candidates = valid_cols

    ax = int(candidates.max())
    return ax, int(prof[ax])


# ── Compositing ────────────────────────────────────────────────────────────────

def composite(
    product_img:  Image.Image,
    shadow_img:   Image.Image,
    shadow_scale: float = 0.5,
    offset_x:     int   = 0,
    offset_y:     int   = 0,
    bg_tolerance: int   = 30,
) -> Image.Image:
    prod = product_img.convert("RGBA")
    if not has_transparency(prod):
        prod = remove_solid_background(prod, tolerance=bg_tolerance)

    prod = strip_baked_shadow(prod)
    shad = shadow_img.convert("RGBA")

    p_bbox = prod.getbbox()
    s_bbox = shad.getbbox()
    if p_bbox is None or s_bbox is None:
        return prod

    prod = prod.crop(p_bbox)
    shad = shad.crop(s_bbox)
    pw, ph = prod.size

    # Scale shadow
    target_w = max(1, int(pw * shadow_scale))
    target_h = max(1, int(target_w * shad.height / shad.width))
    shad = shad.resize((target_w, target_h), Image.Resampling.LANCZOS)
    sw, sh = shad.size

    # Anchors
    p_anchor = find_bottom_left_anchor(prod)  or (0, ph - 1)
    s_anchor = find_bottom_right_anchor(shad) or (sw - 1, sh - 1)
    pcx, pcy = p_anchor
    sax, say = s_anchor

    # Shadow placement
    spx = pcx - sax + offset_x
    spy = pcy - say + offset_y

    # Dynamic canvas
    cx0 = min(0, spx); cy0 = min(0, spy)
    cx1 = max(pw, spx + sw); cy1 = max(ph, spy + sh)
    canvas = Image.new("RGBA", (cx1 - cx0, cy1 - cy0), (0, 0, 0, 0))
    canvas.paste(shad, (spx - cx0, spy - cy0), shad)
    canvas.paste(prod, (-cx0, -cy0), prod)

    bbox = canvas.getbbox()
    return canvas.crop(bbox) if bbox else canvas


# ── Utilities ──────────────────────────────────────────────────────────────────

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_default_shadow() -> Optional[Image.Image]:
    if os.path.exists(DEFAULT_SHADOW_FILE):
        try:
            return Image.open(DEFAULT_SHADOW_FILE)
        except Exception:
            pass
    return None


@st.cache_data
def _logo_b64() -> str:
    if os.path.exists(LOGO_FILE):
        with open(LOGO_FILE, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    return ""


def _logo_img(height_px: int = 52, style: str = "") -> str:
    b64 = _logo_b64()
    if not b64:
        return ""
    return f'<img src="data:image/png;base64,{b64}" style="height:{height_px}px;width:auto;{style}" alt="Harvia Labs">'


# ── UI ─────────────────────────────────────────────────────────────────────────

def render_sidebar() -> Tuple[float, int, int, int, bool, str, int]:
    with st.sidebar:
        st.markdown(
            f'<div style="padding-top:1rem;margin-bottom:1.1rem;">{_logo_img(height_px=46)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Shadow Scale</div>', unsafe_allow_html=True)
        shadow_scale = st.slider("shadow_scale", 0.1, 1.5, 0.5, 0.025, label_visibility="collapsed")
        st.caption(f"{shadow_scale:.3f}× product width")

        st.markdown("---")
        st.markdown('<div class="section-label">BG Removal Tolerance</div>', unsafe_allow_html=True)
        bg_tolerance = st.slider("bg_tolerance", 5, 80, 30, 1, label_visibility="collapsed")
        st.caption(f"Tolerance: {bg_tolerance}")

        st.markdown("---")
        st.markdown('<div class="section-label">Anchor Offset (px)</div>', unsafe_allow_html=True)
        offset_x = st.slider("X offset", -300, 300, 0, 1, label_visibility="collapsed")
        offset_y = st.slider("Y offset", -300, 300, 0, 1, label_visibility="collapsed")
        st.caption(f"X: {offset_x:+d} px  ·  Y: {offset_y:+d} px")

        st.markdown("---")
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        add_bg = st.checkbox("Add background colour", value=False)
        bg_hex = "#EAE8E0"
        if add_bg:
            bg_hex = st.color_picker("Colour", value="#EAE8E0", label_visibility="collapsed")
        margin_px = st.slider("Margin (px)", 0, 300, 0, 5, label_visibility="collapsed")
        st.caption(f"Margin: {margin_px} px")

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem;color:var(--muted);line-height:1.5;">'
            f'Place <code>{DEFAULT_SHADOW_FILE}</code> in the app folder to auto-load it.'
            '</div>',
            unsafe_allow_html=True,
        )

    return shadow_scale, offset_x, offset_y, bg_tolerance, add_bg, bg_hex, margin_px


def render_header() -> None:
    logo = _logo_img(height_px=58, style="display:block;")
    st.markdown(
        f'<div class="header-bar">'
        f'  <div style="flex-shrink:0;">{logo}</div>'
        f'  <div style="border-left:2px solid var(--border);padding-left:1.1rem;">'
        f'    <p style="font-size:1.6rem;font-weight:700;margin:0;line-height:1.1;color:var(--text);">'
        f'      <span style="font-family:\'Gotham\',\'Barlow\',sans-serif;">Cast-a-shadow</span>'
        f'      <span style="font-family:\'Chevin Pro\',\'Barlow\',sans-serif;font-weight:400;'
        f'             font-size:1rem;color:var(--muted);letter-spacing:0.04em;"> beta</span></p>'
        f'    <p style="font-size:0.78rem;color:var(--muted);margin:0.2rem 0 0;">'
        f'      Automated perspective-shadow compositing for product images</p>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _mark_anchor(img: Image.Image, point: Tuple[int, int], on_dark: bool = False) -> Image.Image:
    if on_dark:
        tile = 12
        bg = Image.new("RGBA", img.size, (180, 180, 180, 255))
        for ty in range(0, img.height, tile):
            for tx in range(0, img.width, tile):
                if (tx // tile + ty // tile) % 2 == 0:
                    bg.paste((210, 210, 210, 255), (tx, ty, min(tx+tile, img.width), min(ty+tile, img.height)))
        vis = bg
        vis.paste(img, mask=img)
    else:
        vis = img.copy().convert("RGBA")

    draw = ImageDraw.Draw(vis)
    px, py = point
    r   = max(6, min(18, img.width // 50))
    arm = r * 3
    draw.line([(px - arm, py), (px + arm, py)], fill=(255, 40, 40, 255), width=2)
    draw.line([(px, py - arm), (px, py + arm)], fill=(255, 40, 40, 255), width=2)
    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 40, 40, 200))
    return vis


def render_debug(prod: Image.Image, shad: Image.Image,
                 p_anchor: Optional[Tuple[int, int]],
                 s_anchor: Optional[Tuple[int, int]]) -> None:
    with st.expander("Anchor debug — click to verify anchor placement"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Product — bottom-left anchor**")
            if p_anchor:
                st.image(_mark_anchor(prod, p_anchor), use_container_width=True)
                st.code(f"x={p_anchor[0]}, y={p_anchor[1]}")
            else:
                st.image(prod, use_container_width=True)
                st.warning("Anchor not found")
            st.caption(f"Image size: {prod.size}")
        with c2:
            st.markdown("**Shadow — bottom-right anchor**")
            if s_anchor:
                st.image(_mark_anchor(shad, s_anchor, on_dark=True), use_container_width=True)
                st.code(f"x={s_anchor[0]}, y={s_anchor[1]}")
            else:
                st.image(shad, use_container_width=True)
                st.warning("Anchor not found")
            st.caption(f"Image size: {shad.size}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    shadow_scale, offset_x, offset_y, bg_tolerance, add_bg, bg_hex, margin_px = render_sidebar()
    render_header()

    # Upload inputs
    col_prod, col_shad = st.columns(2, gap="medium")
    with col_prod:
        st.markdown('<div class="section-label">Product Image</div>', unsafe_allow_html=True)
        product_file = st.file_uploader("product", type=["png","jpg","jpeg"], key="product", label_visibility="collapsed")
    with col_shad:
        st.markdown(
            '<div class="section-label">Shadow Image <span class="chip">optional</span></div>',
            unsafe_allow_html=True,
        )
        shadow_file = st.file_uploader("shadow", type=["png"], key="shadow", label_visibility="collapsed")

    # Resolve shadow source
    shadow_img: Optional[Image.Image] = None
    shadow_label = ""
    if shadow_file:
        shadow_img   = Image.open(shadow_file)
        shadow_label = "Uploaded shadow"
    else:
        default = load_default_shadow()
        if default:
            shadow_img   = default
            shadow_label = f"Auto-loaded `{DEFAULT_SHADOW_FILE}`"

    if product_file is None:
        st.markdown('<p style="color:var(--muted);font-size:0.9rem;margin-top:0.5rem;">Upload a product image above to get started.</p>', unsafe_allow_html=True)
        return

    if shadow_img is None:
        st.markdown(
            f'<p style="color:var(--muted);font-size:0.9rem;margin-top:0.5rem;">'
            f'No shadow found. Upload a shadow PNG or place <code>{DEFAULT_SHADOW_FILE}</code> in the app folder.</p>',
            unsafe_allow_html=True,
        )
        return

    # Composite
    product_img = Image.open(product_file)
    spinner = st.empty()
    spinner.markdown('<div class="cas-status"><span class="cas-ring"></span>Compositing…</div>', unsafe_allow_html=True)

    result = composite(product_img, shadow_img, shadow_scale=shadow_scale,
                       offset_x=offset_x, offset_y=offset_y, bg_tolerance=bg_tolerance)
    spinner.empty()

    # Output options
    output = result
    hx = bg_hex.lstrip("#")
    bg_r, bg_g, bg_b = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)

    if add_bg:
        canvas = Image.new("RGBA", output.size, (bg_r, bg_g, bg_b, 255))
        canvas.paste(output, mask=output)
        output = canvas

    if margin_px > 0:
        ow, oh = output.size
        fill = (bg_r, bg_g, bg_b, 255) if add_bg else (0, 0, 0, 0)
        m = Image.new("RGBA", (ow + 2*margin_px, oh + 2*margin_px), fill)
        m.paste(output, (margin_px, margin_px))
        output = m

    # Result display
    parts = [shadow_label]
    if add_bg:   parts.append(f"bg {bg_hex}")
    if margin_px: parts.append(f"{margin_px}px margin")

    st.markdown(
        f'<div class="result-card"><div class="section-label">Result — {" · ".join(parts)}</div></div>',
        unsafe_allow_html=True,
    )
    st.image(output, use_container_width=True)

    # Debug expander — recompute anchor inputs to match composite() exactly
    _prod = product_img.convert("RGBA")
    if not has_transparency(_prod):
        _prod = remove_solid_background(_prod, tolerance=bg_tolerance)
    _prod = strip_baked_shadow(_prod)
    _pbbox = _prod.getbbox()
    if _pbbox: _prod = _prod.crop(_pbbox)

    _shad = shadow_img.convert("RGBA")
    _sbbox = _shad.getbbox()
    if _sbbox: _shad = _shad.crop(_sbbox)

    _tw = max(1, int(_prod.size[0] * shadow_scale))
    _th = max(1, int(_tw * _shad.height / _shad.width))
    _shad_scaled = _shad.resize((_tw, _th), Image.Resampling.LANCZOS)

    render_debug(_prod, _shad_scaled,
                 find_bottom_left_anchor(_prod),
                 find_bottom_right_anchor(_shad_scaled))

    # Download
    st.download_button("Download PNG", data=to_png_bytes(output),
                       file_name="product_with_shadow.png", mime="image/png",
                       use_container_width=True)


main()
