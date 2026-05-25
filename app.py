import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageColor
import io
import base64
import os
from collections import deque
import zipfile

# ── Anchor ray angles ─────────────────────────────────────────────────────────
# Degrees from vertical for the sweep lines used to locate floor-contact corners.
# Increase -> weights downward position more; decrease -> weights left/right more.
ANCHOR_RAY_ANGLE: float = 30.0   # product bottom-left  sweep — tune here
SHADOW_RAY_ANGLE: float = 10.0   # shadow  bottom-right sweep — tune here

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="Cast-a-shadow beta - Harvia Labs",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (only what config.toml cannot do) ──────────────────────────────────────
def inject_css() -> None:
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');
@font-face { font-family:'Chevin Pro'; src:local('Chevin Pro'),local('ChevinPro'); }
@font-face { font-family:'Gotham';     src:local('Gotham'),local('GothamBook'); }
html, body, [class*="css"] {
  font-family: 'Chevin Pro','Gotham','Barlow',sans-serif !important;
}
:root {
  --bg:       #EAE8E0;
  --bg-card:  #F5F4F0;
  --red:      #ED1C24;
  --red-h:    #C41520;
  --text:     #505045;
  --muted:    #727266;
  --border:   #D9D6C8;
  --chip-bg:  #EAE8E0;
}
/* ── Sidebar header gap ───────────────────────────────────── */
[data-testid="stSidebarHeader"] {
  height:0 !important; min-height:0 !important;
  padding:0 !important; margin:0 !important; overflow:hidden !important;
}
section[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] { padding-top:0 !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  margin-top:0 !important; padding-top:0 !important; gap:0.65rem !important;
}
section[data-testid="stSidebar"] hr { margin:0.3rem 0 !important; }
/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  background:var(--bg-card) !important;
  border:1.5px dashed var(--border) !important;
  border-radius:10px !important; min-height:110px !important; padding:1rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--red) !important; }
[data-testid="stFileUploaderDropzone"] button {
  background:transparent !important; border:1px solid var(--border) !important;
  border-radius:6px !important; box-shadow:none !important;
  color:var(--muted) !important; font-size:0.80rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
  color:var(--red) !important; border-color:var(--red) !important;
}
/* Rename "Upload" → "Upload image" */
[data-testid="stFileUploaderDropzone"] button p { font-size:0 !important; }
[data-testid="stFileUploaderDropzone"] button p::after { content:'Upload image'; font-size:13px !important; color:var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] small { color:var(--muted) !important; font-size:0.68rem !important; }
[data-testid="stFileUploaderDropzoneDragOver"],
[data-testid="stFileUploaderDropzone"]:focus-within {
  background-color:rgba(217,214,200,0.25) !important; border-color:var(--border) !important;
}
/* ── Slider: suppress last-child red box on hover ────────── */
.stSlider [data-baseweb="slider"] > div:last-child { background-color:transparent !important; }
.stSlider [role="slider"]:focus,
.stSlider [role="slider"]:focus-visible { outline:none !important; box-shadow:none !important; }
/* ── Custom spinner ───────────────────────────────────────── */
@keyframes cas-spin { to { transform:rotate(360deg); } }
.cas-status { display:flex; align-items:center; gap:0.55rem; color:#505045; font-size:0.90rem; margin:0.6rem 0; }
.cas-ring { flex-shrink:0; width:16px; height:16px; border:2px solid #D9D6C8; border-top-color:#727266; border-radius:50%; animation:cas-spin 0.75s linear infinite; }
/* ── Layout ───────────────────────────────────────────────── */
.block-container { padding-top:1rem !important; padding-left:1.2rem !important; padding-right:1.2rem !important; max-width:none !important; }
[data-testid="stHorizontalBlock"] { align-items:flex-start !important; }
/* ── Reusable components ──────────────────────────────────── */
.section-label { font-size:0.70rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:0.35rem; }
section[data-testid="stSidebar"] .section-label { margin-bottom:0.55rem; }
.result-card { background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:0.9rem 1.5rem; margin-top:2rem; margin-bottom:1.25rem; }
.header-bar { display:flex; align-items:center; gap:0.8rem; margin-bottom:1.4rem; }
.chip { display:inline-block; background:var(--chip-bg); border:1px solid var(--border); border-radius:20px; padding:0.15rem 0.6rem; font-size:0.74rem; font-weight:500; color:var(--text); }
/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] > details { background-color:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:8px !important; }
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { background-color:var(--bg-card) !important; color:var(--text) !important; }
/* ── st.code (anchor debug) ───────────────────────────────── */
[data-testid="stCode"] { background-color:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:6px !important; }
[data-testid="stCode"] code { color:var(--text) !important; background:transparent !important; }
/* ── Alert / notification boxes ───────────────────────────── */
[data-testid="stAlert"],
[data-baseweb="notification"] {
  background-color:var(--bg-card) !important;
  border:1px solid var(--border) !important;
  border-left:3px solid var(--muted) !important;
  border-radius:8px !important;
}
[data-testid="stAlert"] p,
[data-baseweb="notification"] p { color:var(--muted) !important; }
[data-testid="stAlert"] svg,
[data-baseweb="notification"] svg { display:none !important; }
/* ── Download button — Harvia red ─────────────────────────── */
[data-testid="stDownloadButton"] button {
  background-color:var(--red) !important;
  color:#FFFFFF !important;
  border-color:var(--red) !important;
}
[data-testid="stDownloadButton"] button:hover {
  background-color:var(--red-h) !important;
  border-color:var(--red-h) !important;
  color:#FFFFFF !important;
}
/* ── Sidebar text ─────────────────────────────────────────── */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
/* ── Chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility:hidden; }
</style>""", unsafe_allow_html=True)


# ── Logo helper ────────────────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def _logo_img(path: str = "harvia_labs_red.png") -> str:
    # Resolve relative paths against app.py's directory so the logo works
    # regardless of which working directory Streamlit was launched from.
    candidates = [path, os.path.join(_APP_DIR, path)]
    for p in candidates:
        try:
            with open(p, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(p)[1].lower().lstrip(".")
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f"data:{mime};base64,{data}"
        except OSError:
            continue
    return ""


# ── BFS flood fill ─────────────────────────────────────────────────────────────
def _bfs_background(fg_mask: np.ndarray) -> np.ndarray:
    """BFS from all four edges. fg_mask: True=foreground. Returns True=connected background."""
    h, w = fg_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    def _seed(r: int, c: int) -> None:
        if not fg_mask[r, c] and not visited[r, c]:
            visited[r, c] = True
            queue.append((r, c))

    for x in range(w):
        _seed(0, x)
        _seed(h - 1, x)
    for y in range(h):
        _seed(y, 0)
        _seed(y, w - 1)

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and not fg_mask[nr, nc]:
                visited[nr, nc] = True
                queue.append((nr, nc))
    return visited


# ── Background removal ─────────────────────────────────────────────────────────
def remove_solid_background(img: Image.Image, tolerance: int = 30) -> Image.Image:
    img = img.convert("RGBA")
    arr = np.array(img, dtype=np.float32)
    rgb = arr[:, :, :3]
    h, w = arr.shape[:2]

    # Sample corners for background color
    cpx = max(1, int(min(h, w) * 0.03))
    corners = np.concatenate([
        rgb[:cpx, :cpx].reshape(-1, 3),
        rgb[:cpx, -cpx:].reshape(-1, 3),
        rgb[-cpx:, :cpx].reshape(-1, 3),
        rgb[-cpx:, -cpx:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)

    # Full-resolution color-distance map
    dist = np.sqrt(np.sum((rgb - bg_color) ** 2, axis=2))

    # Downscale to max 400px on long edge for BFS
    long_edge = max(h, w)
    scale = min(1.0, 400.0 / long_edge)
    dw = max(1, int(w * scale))
    dh = max(1, int(h * scale))

    dist_small = np.array(
        Image.fromarray(dist.astype(np.float32)).resize((dw, dh), Image.BILINEAR)
    )
    fg_small = dist_small > tolerance  # True = foreground

    # Dilate background in lower half by eroding foreground with 11px MinFilter
    half_row = dh // 2
    fg_lower_pil = Image.fromarray(fg_small[half_row:, :].astype(np.uint8) * 255)
    fg_lower_eroded = np.array(fg_lower_pil.filter(ImageFilter.MinFilter(11))) > 127
    fg_small[half_row:, :] = fg_lower_eroded

    # BFS from edges to find connected background
    bg_small = _bfs_background(fg_small)

    # Upsample background mask to full resolution
    bg_full = np.array(
        Image.fromarray(bg_small.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
    ) > 127
    fg_full = ~bg_full

    # Transition zone: erode and dilate fg_mask by trans_px
    trans_px = int(max(h, w) * 0.015)
    trans_px = max(12, min(30, trans_px))
    ksize = trans_px * 2 + 1

    fg_pil = Image.fromarray(fg_full.astype(np.uint8) * 255)
    core = np.array(fg_pil.filter(ImageFilter.MinFilter(ksize))) > 127
    halo = np.array(fg_pil.filter(ImageFilter.MaxFilter(ksize))) > 127
    transition = halo & ~core

    # Build alpha: core=1, bg=0, transition=color-distance blend
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[core] = 1.0
    alpha[transition] = np.clip(dist[transition] / max(float(tolerance), 1.0), 0.0, 1.0)

    # Smooth staircase artifacts
    alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8))
    alpha = np.array(alpha_pil.filter(ImageFilter.GaussianBlur(radius=0.8))).astype(np.float32) / 255.0

    # Defringe: remove background color contamination from semi-transparent pixels
    semi = (alpha > 0.01) & (alpha < 0.99)
    if np.any(semi):
        a3 = alpha[:, :, np.newaxis]
        bg3 = bg_color[np.newaxis, np.newaxis, :]
        defringe = (rgb - (1.0 - a3) * bg3) / np.maximum(a3, 0.01)
        rgb = np.where(semi[:, :, np.newaxis], np.clip(defringe, 0.0, 255.0), rgb)

    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    result[:, :, 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(result, "RGBA")


# ── Baked shadow stripping ─────────────────────────────────────────────────────
def strip_baked_shadow(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]

    valid = (
        (alpha > 20)
        & ~np.all(rgb > 235, axis=2)
        & ~np.all(rgb < 20, axis=2)
    )

    row_counts = np.sum(valid, axis=1)
    top70 = max(1, int(h * 0.7))
    peak = max(1, int(np.max(row_counts[:top70])))

    solid_base_row = 0
    for r in range(h - 1, -1, -1):
        if row_counts[r] >= 0.08 * peak:
            solid_base_row = r
            break

    margin = max(20, int(h * 0.03))
    crop_row = min(solid_base_row + margin, h)
    return img.crop((0, 0, w, crop_row))


# ── Anchor detection ───────────────────────────────────────────────────────────
def find_bottom_left_anchor(img: Image.Image) -> tuple:
    """Ray-sweep from the lower-left at ANCHOR_RAY_ANGLE degrees from vertical.

    For each valid pixel (x, y) the signed projection onto the sweep direction is:
        proj = x * sin(theta) - y * cos(theta)

    Pixels further left OR further down both decrease this value, so the pixel
    with the minimum projection is the one hit first by the approaching sweep
    line — always the true floor-contact bottom-left corner:

      * Barrel sauna: foot corner beats the circular-arc fringe because the foot
        is lower (large y drives the -y*cos term well below the fringe value).
      * Perspective box: left face beats the centre-bottom pixel because it is
        further left; the slight height difference is outweighed at 30 deg.
      * Baked shadow extension: sits higher than the product base, so its
        projection is larger and is never selected.
    """
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha, rgb = arr[:, :, 3], arr[:, :, :3]

    valid = (alpha > 100) & ~np.all(rgb > 235, axis=2)
    if not np.any(valid):
        return (0, h - 1)

    theta = np.radians(ANCHOR_RAY_ANGLE)
    ys, xs = np.where(valid)
    proj = xs.astype(np.float64) * np.sin(theta) - ys.astype(np.float64) * np.cos(theta)
    idx = int(np.argmin(proj))
    return (int(xs[idx]), int(ys[idx]))


def find_bottom_right_anchor(img: Image.Image) -> tuple:
    """Mirror of find_bottom_left_anchor: ray-sweep from the lower-right.

    Projection: proj = x * sin(theta) + y * cos(theta)
    Pixels further right OR further down get a larger projection and are hit
    first, giving the shadow's floor-contact bottom-right corner.
    """
    arr = np.array(img.convert("RGBA"))
    h, w = arr.shape[:2]

    valid = arr[:, :, 3] >= 25  # shadow pixels: any with alpha >= 25
    if not np.any(valid):
        return (w - 1, h - 1)

    theta = np.radians(SHADOW_RAY_ANGLE)
    ys, xs = np.where(valid)
    proj = xs.astype(np.float64) * np.sin(theta) + ys.astype(np.float64) * np.cos(theta)
    idx = int(np.argmax(proj))
    return (int(xs[idx]), int(ys[idx]))


# ── Debug helpers ──────────────────────────────────────────────────────────────
def draw_crosshair(img: Image.Image, anchor: tuple, color: tuple = (255, 40, 40)) -> Image.Image:
    img = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img)
    x, y = anchor
    w = img.size[0]
    r = max(6, min(18, w // 50))
    arm = r * 3
    c = color + (255,)
    draw.line([(x - arm, y), (x + arm, y)], fill=c, width=2)
    draw.line([(x, y - arm), (x, y + arm)], fill=c, width=2)
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=c)
    return img


def make_checkerboard_bg(size: tuple, tile: int = 16) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    for row in range(0, h, tile):
        for col in range(0, w, tile):
            v = 220 if ((row // tile) + (col // tile)) % 2 == 0 else 180
            arr[row:row + tile, col:col + tile, :3] = v
    return Image.fromarray(arr, "RGBA")


# ── Compositing (two-stage: heavy prepare + light assemble) ────────────────────
def prepare_layers(
    product_img: Image.Image,
    shadow_img: Image.Image,
    shadow_scale: float,
    tolerance: int,
) -> tuple:
    """Heavy stage: background removal, baked-shadow strip, anchor detection.
    Output feeds assemble_composite, which is cheap and re-runnable for offset tuning.

    Returns: (prod, shad, prod_anchor, shad_anchor)
    """
    prod = product_img.convert("RGBA")

    arr = np.array(prod)
    if np.mean(arr[:, :, 3] < 200) < 0.005:
        prod = remove_solid_background(prod, tolerance)

    shad = shadow_img.convert("RGBA")
    prod = strip_baked_shadow(prod)

    prod_bbox = prod.getbbox()
    if prod_bbox is None:
        empty = product_img.convert("RGBA")
        return empty, shad, (0, 0), (0, 0)
    prod = prod.crop(prod_bbox)

    shad_bbox = shad.getbbox()
    if shad_bbox is None:
        return prod, shad, (0, prod.height - 1), (shadow_img.width - 1, shadow_img.height - 1)
    shad = shad.crop(shad_bbox)

    pw, _ = prod.size
    sw, sh = shad.size
    target_w = max(1, int(shadow_scale * pw))
    target_h = max(1, int(target_w * sh / max(sw, 1)))
    shad = shad.resize((target_w, target_h), Image.LANCZOS)

    prod_anchor = find_bottom_left_anchor(prod)
    shad_anchor = find_bottom_right_anchor(shad)
    return prod, shad, prod_anchor, shad_anchor


def assemble_composite(
    prod: Image.Image,
    shad: Image.Image,
    prod_anchor: tuple,
    shad_anchor: tuple,
    offset_x: int = 0,
    offset_y: int = 0,
    bg_color: str | None = None,
    margin: int = 0,
) -> Image.Image:
    """Light stage: paste pre-prepared layers, apply offset, bg fill, margin."""
    pw, ph = prod.size
    sw, sh = shad.size

    shadow_left = prod_anchor[0] - shad_anchor[0] + offset_x
    shadow_top = prod_anchor[1] - shad_anchor[1] + offset_y
    prod_left, prod_top = 0, 0

    if shadow_left < 0:
        prod_left = -shadow_left
        shadow_left = 0
    if shadow_top < 0:
        prod_top = -shadow_top
        shadow_top = 0

    canvas_w = max(prod_left + pw, shadow_left + sw)
    canvas_h = max(prod_top + ph, shadow_top + sh)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(shad, (shadow_left, shadow_top), shad)
    canvas.paste(prod, (prod_left, prod_top), prod)

    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)

    if bg_color:
        try:
            bg_rgb = ImageColor.getrgb(bg_color)[:3]
            bg_img = Image.new("RGBA", canvas.size, bg_rgb + (255,))
            bg_img.paste(canvas, (0, 0), canvas)
            canvas = bg_img
        except Exception:
            pass

    if margin > 0:
        mw = canvas.width + 2 * margin
        mh = canvas.height + 2 * margin
        if bg_color:
            try:
                bg_rgb = ImageColor.getrgb(bg_color)[:3]
                bg_canvas = Image.new("RGBA", (mw, mh), bg_rgb + (255,))
            except Exception:
                bg_canvas = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        else:
            bg_canvas = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
        bg_canvas.paste(canvas, (margin, margin), canvas)
        canvas = bg_canvas

    return canvas


@st.cache_data(show_spinner=False)
def _prepare_layers_cached(
    product_bytes: bytes,
    shadow_bytes: bytes,
    shadow_scale: float,
    tolerance: int,
) -> tuple:
    """Cached entry point — keyed by file bytes + scale + tolerance.
    Survives reruns; invalidated only when one of the inputs actually changes."""
    prod_img = Image.open(io.BytesIO(product_bytes))
    shad_img = Image.open(io.BytesIO(shadow_bytes))
    return prepare_layers(prod_img, shad_img, shadow_scale, tolerance)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple:
    with st.sidebar:
        logo_src = _logo_img("harvia_labs_red.png")
        if logo_src:
            st.markdown(
                f'<div style="padding-top:1rem;margin-bottom:1.1rem;">'
                f'<img src="{logo_src}" style="height:46px;" alt="Harvia Labs"/>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label" style="padding-top:1.1rem;">Shadow Scale</div>', unsafe_allow_html=True)
        shadow_scale = st.slider(
            "Shadow Scale", 0.10, 1.5, 0.50, 0.025,
            label_visibility="collapsed",
            help="Shadow width as multiple of product bbox width",
        )
        st.caption(f"Shadow Scale: {shadow_scale:.3f}")

        st.markdown("---")
        st.markdown('<div class="section-label">BG Removal Tolerance</div>', unsafe_allow_html=True)
        bg_tolerance = st.slider(
            "BG Removal Tolerance", 5, 80, 30, 1,
            label_visibility="collapsed",
            help="Max color distance from detected background",
        )
        st.caption(f"Tolerance: {bg_tolerance}")

        st.markdown("---")
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        add_bg = st.checkbox("Add background colour", value=False)
        bg_color_val: str | None = None
        if add_bg:
            bg_color_val = st.color_picker("Background colour", "#EAE8E0")

        out_margin = st.slider(
            "Margin (px)", 0, 300, 0, 5,
            label_visibility="collapsed",
            help="Padding around the final image",
        )
        st.caption(f"Margin: {out_margin} px")

        st.markdown("---")
        st.markdown(
            '<p style="font-size:0.68rem;">'
            'Place <code>default_shadow.png</code> in the working directory to auto-load it as the default shadow.'
            '</p>',
            unsafe_allow_html=True,
        )

    return shadow_scale, bg_tolerance, add_bg, bg_color_val, out_margin


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header() -> None:
    logo_src = _logo_img("harvia_labs_red.png")
    logo_html = (
        f'<img src="{logo_src}" style="height:58px;display:block;" alt="Harvia Labs"/>'
        if logo_src else ""
    )
    st.markdown(f"""<div class="header-bar">
  {logo_html}
  <div style="border-left:2px solid #D9D6C8;padding-left:0.8rem;">
    <div style="font-family:'Gotham','Barlow',sans-serif;font-weight:700;font-size:1.6rem;color:#505045;line-height:1.2;">
      Cast-a-shadow
      <span style="font-family:'Chevin Pro','Barlow',sans-serif;font-size:1rem;color:#727266;letter-spacing:0.04em;font-weight:400;"> beta</span>
    </div>
    <div style="font-size:0.78rem;color:#727266;margin-top:0.15rem;">Automated perspective-shadow compositing for product images</div>
  </div>
</div>""", unsafe_allow_html=True)

    with st.expander("How it works"):
        st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem 2rem;padding:0.25rem 0;">

<div>
<div class="section-label">1 · Background removal</div>
<p style="font-size:0.85rem;color:#505045;margin:0.3rem 0 0;">Corner pixels are sampled to detect the background colour. A BFS flood-fill from all four edges builds a mask, and a Gaussian-smoothed trimap blend removes the background while preserving semi-transparent edges.</p>
</div>

<div>
<div class="section-label">2 · Floor contact anchor</div>
<p style="font-size:0.85rem;color:#505045;margin:0.3rem 0 0;">A sweep line at 30° from vertical scans all solid product pixels. The pixel with the lowest projection value — furthest left <em>and</em> down — is chosen as the floor contact point. This reliably finds the bottom-left corner regardless of product shape.</p>
</div>

<div>
<div class="section-label">3 · Shadow anchor</div>
<p style="font-size:0.85rem;color:#505045;margin:0.3rem 0 0;">The same ray-sweep logic at 10° from vertical finds the shadow's bottom-right floor contact point — the pixel furthest right and down. Both angles are tunable constants in the code.</p>
</div>

<div>
<div class="section-label">4 · Compositing</div>
<p style="font-size:0.85rem;color:#505045;margin:0.3rem 0 0;">The two anchors are aligned on a shared canvas — the shadow behind, the product on top. Shadow width scales as a multiple of the product bounding-box width. Each result has its own X/Y micro-adjustment in the debug panel for fine-tuning.</p>
</div>

</div>
""", unsafe_allow_html=True)


# ── Utils ──────────────────────────────────────────────────────────────────────
def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_zip(items: list[tuple[str, Image.Image]]) -> bytes:
    """Pack a list of (filename, PIL image) pairs into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, img in items:
            zf.writestr(filename, img_to_bytes(img))
    return buf.getvalue()


# ── Image display helper ───────────────────────────────────────────────────────
def _show_image(img: Image.Image, max_h: int = 600) -> None:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-height:{max_h}px;max-width:100%;width:auto;height:auto;'
        f'display:block;margin:1.5rem auto;" />',
        unsafe_allow_html=True,
    )


# ── Per-image result renderer ──────────────────────────────────────────────────
def _render_result(
    key: str,
    stem: str,
    prod: Image.Image,
    shad: Image.Image,
    prod_anchor: tuple,
    shad_anchor: tuple,
    bg_color: str | None,
    margin: int,
    dl_key_suffix: str,
) -> Image.Image:
    """Render one composited result with its own micro-adjustment controls.
    Reads `(applied_x, applied_y)` from session state and applies them to the cached
    layers. Number inputs adjust pending values; only the Apply button commits."""
    offsets = st.session_state.setdefault("offsets", {})
    applied_x, applied_y = offsets.get(key, (0, 0))

    output_img = assemble_composite(
        prod, shad, prod_anchor, shad_anchor,
        applied_x, applied_y, bg_color, margin,
    )
    _show_image(output_img)

    with st.expander("Anchor debug & micro-adjustment — click to fine-tune"):
        # ── Adjustment controls ────────────────────────────────────────────────
        st.markdown('<div class="section-label">Micro-adjustment (per image)</div>', unsafe_allow_html=True)
        adj_cols = st.columns([1, 1, 1, 1])
        with adj_cols[0]:
            pending_x = st.number_input(
                "X offset (px)", value=applied_x, step=1,
                key=f"x_{key}", help="Nudge shadow horizontally",
            )
        with adj_cols[1]:
            pending_y = st.number_input(
                "Y offset (px)", value=applied_y, step=1,
                key=f"y_{key}", help="Nudge shadow vertically",
            )
        pending = (int(pending_x), int(pending_y))
        is_pending = pending != (applied_x, applied_y)
        with adj_cols[2]:
            st.markdown('<div style="height:1.65rem;"></div>', unsafe_allow_html=True)
            if st.button(
                "Apply", key=f"apply_{key}",
                use_container_width=True, disabled=not is_pending,
                type="primary" if is_pending else "secondary",
            ):
                offsets[key] = pending
                st.rerun()
        with adj_cols[3]:
            st.markdown('<div style="height:1.65rem;"></div>', unsafe_allow_html=True)
            if st.button(
                "Reset", key=f"reset_{key}",
                use_container_width=True, disabled=(applied_x == 0 and applied_y == 0),
            ):
                offsets[key] = (0, 0)
                st.rerun()

        # ── Live preview of pending change ────────────────────────────────────
        if is_pending:
            st.markdown(
                '<div class="section-label" style="margin-top:0.6rem;">'
                'Preview (click Apply to commit)</div>',
                unsafe_allow_html=True,
            )
            preview_img = assemble_composite(
                prod, shad, prod_anchor, shad_anchor,
                pending[0], pending[1], bg_color, margin,
            )
            _show_image(preview_img, max_h=380)

        # ── Anchor crosshairs ─────────────────────────────────────────────────
        st.markdown(
            '<div class="section-label" style="margin-top:0.8rem;">Anchor placement</div>',
            unsafe_allow_html=True,
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown('<div class="section-label">Product anchor</div>', unsafe_allow_html=True)
            st.image(draw_crosshair(prod, prod_anchor), use_container_width=True)
            st.code(f"x={prod_anchor[0]}, y={prod_anchor[1]}")
            st.caption(f"Image size: {prod.size[0]}×{prod.size[1]}")
        with dcol2:
            st.markdown('<div class="section-label">Shadow anchor</div>', unsafe_allow_html=True)
            checker = make_checkerboard_bg(shad.size)
            checker.paste(shad, (0, 0), shad)
            st.image(draw_crosshair(checker, shad_anchor), use_container_width=True)
            st.code(f"x={shad_anchor[0]}, y={shad_anchor[1]}")
            st.caption(f"Image size: {shad.size[0]}×{shad.size[1]}")

    st.download_button(
        "Download PNG",
        data=img_to_bytes(output_img),
        file_name=f"{stem}_shadow.png",
        mime="image/png",
        use_container_width=True,
        key=f"dl_{dl_key_suffix}",
    )
    return output_img


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    inject_css()

    shadow_scale, bg_tolerance, add_bg, bg_color_val, out_margin = render_sidebar()
    render_header()

    upload_row, _ = st.columns([3, 1])
    col1, col2 = upload_row.columns(2, gap="medium")
    with col1:
        st.markdown(
            '<div style="height:1.55rem;display:flex;align-items:center;margin-bottom:0.35rem;">'
            '<span class="section-label" style="margin-bottom:0;">Product Image</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        product_files = st.file_uploader(
            "Product Images", type=["png", "jpg", "jpeg"],
            label_visibility="collapsed", key="product_upload",
            accept_multiple_files=True,
        )
    with col2:
        st.markdown(
            '<div style="height:1.55rem;display:flex;align-items:center;margin-bottom:0.35rem;">'
            '<span class="section-label" style="margin-bottom:0;">Shadow Image</span>'
            '&nbsp;<span class="chip">optional</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        shadow_file = st.file_uploader(
            "Shadow Image", type=["png"],
            label_visibility="collapsed", key="shadow_upload",
        )

    # ── Load shadow bytes (cached prepare_layers needs bytes, not PIL Image) ───
    shadow_bytes: bytes | None = None
    shadow_label = "default shadow"
    if shadow_file:
        shadow_bytes = shadow_file.getvalue()
        shadow_label = shadow_file.name
    elif os.path.exists("default_shadow.png"):
        with open("default_shadow.png", "rb") as f:
            shadow_bytes = f.read()
        shadow_label = "default_shadow.png"

    if not product_files:
        st.info("Upload one or more product images to get started.")
        return

    if shadow_bytes is None:
        st.markdown(
            '<p style="color:#727266;font-size:0.9rem;margin-top:1.2rem;">'
            'No shadow uploaded and no <code>default_shadow.png</code> found. '
            'Upload a shadow PNG above or place <code>default_shadow.png</code> here.</p>',
            unsafe_allow_html=True,
        )
        return

    # ── Shared settings string ─────────────────────────────────────────────────
    options_parts = []
    if add_bg and bg_color_val:
        options_parts.append(f"bg {bg_color_val}")
    if out_margin:
        options_parts.append(f"margin {out_margin}px")
    options_str = " · ".join(options_parts) if options_parts else "default settings"

    bg_color = bg_color_val if add_bg else None

    # ── Run heavy stage for each file (cached) ─────────────────────────────────
    spinner_slot = st.empty()
    prepared: list[tuple] = []  # [(file, key, layers_or_None, err)]
    n = len(product_files)

    for i, f in enumerate(product_files):
        spinner_slot.markdown(
            f'<div class="cas-status"><span class="cas-ring"></span>'
            f'Preparing {i + 1} / {n} — {f.name}</div>',
            unsafe_allow_html=True,
        )
        key = getattr(f, "file_id", None) or f.name
        try:
            product_bytes = f.getvalue()
            layers = _prepare_layers_cached(product_bytes, shadow_bytes, shadow_scale, bg_tolerance)
            prepared.append((f, key, layers, ""))
        except Exception as exc:
            prepared.append((f, key, None, str(exc)))

    spinner_slot.empty()

    # ── Render results ─────────────────────────────────────────────────────────
    if n == 1:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="section-label">Result — {shadow_label} · {options_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        f, key, layers, err = prepared[0]
        if layers is None:
            st.error(f"Compositing failed: {err}")
            return
        prod, shad, prod_anchor, shad_anchor = layers
        stem = os.path.splitext(f.name)[0]
        _render_result(
            key, stem, prod, shad, prod_anchor, shad_anchor,
            bg_color, out_margin, dl_key_suffix="single",
        )
    else:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="section-label">Batch — {n} images · {shadow_label} · {options_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        outputs: list[tuple[str, Image.Image]] = []
        for i, (f, key, layers, err) in enumerate(prepared):
            stem = os.path.splitext(f.name)[0]
            st.markdown(
                f'<div class="section-label" style="margin-top:1rem;">{stem}</div>',
                unsafe_allow_html=True,
            )
            if layers is None:
                st.error(f"Failed: {err}")
                continue
            prod, shad, prod_anchor, shad_anchor = layers
            out_img = _render_result(
                key, stem, prod, shad, prod_anchor, shad_anchor,
                bg_color, out_margin, dl_key_suffix=f"batch_{i}",
            )
            outputs.append((f"{stem}_shadow.png", out_img))

        if outputs:
            st.markdown("---")
            st.download_button(
                f"Download all {len(outputs)} as ZIP",
                data=create_zip(outputs),
                file_name="cast_a_shadow_batch.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_zip",
            )


if __name__ == "__main__":
    main()
