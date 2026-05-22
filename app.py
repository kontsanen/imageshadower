"""
Harvia Labs — Image Shadow Compositor
======================================
Automated perspective-shadow compositing for e-commerce product images.

Core algorithm:
  1. Crop both images to their tight bounding boxes.
  2. Scale the shadow to match the desired fraction of the product width.
  3. Locate the product's bottom-left anchor and the shadow's bottom-right anchor
     using visual corner detection (ignores transparent / white / black pixels).
  4. Align anchor points and paste shadow behind product on a dynamic canvas.
  5. Trim transparent margins and return the final composite.
"""

import io
import os
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShadowDrop — Harvia Labs",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DESIGN TOKENS & GLOBAL CSS ───────────────────────────────────────────────
# Matches the Harvia Labs design mockup: beige background, red accent, Gotham/
# Chevin Pro font stack. Both fonts are commercial; a Google-hosted humanist
# sans (Barlow) is listed as the open-source fallback so the layout still looks
# clean on machines without the licensed fonts installed.
STYLE = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&display=swap');

  /* Let the browser pick up locally-installed Gotham / Chevin Pro first */
  @font-face { font-family:'Chevin Pro'; src:local('Chevin Pro'),local('ChevinPro'); }
  @font-face { font-family:'Gotham';     src:local('Gotham'),local('GothamBook');    }

  html, body, [class*="css"] {
    font-family: 'Chevin Pro','Gotham','Barlow',sans-serif !important;
  }

  :root {
    --bg:           #F0EDE6;
    --bg-sidebar:   #E4E0D8;
    --bg-card:      #FAF9F6;
    --red:          #CC0000;
    --red-h:        #AA0000;
    --text:         #1A1A1A;
    --muted:        #6B6560;
    --border:       #D4CFC7;
    --chip-bg:      #EDEBE5;
  }

  .stApp                                  { background-color: var(--bg); }
  section[data-testid="stSidebar"]        { background-color: var(--bg-sidebar);
                                            border-right: 1px solid var(--border); }

  /* Slider thumb + track */
  .stSlider > div > div > div > div       { background-color: var(--red) !important; }
  .stSlider [data-baseweb="slider"] > div:last-child { background-color: var(--red) !important; }

  /* Buttons & download */
  .stButton > button,
  .stDownloadButton > button {
    background-color: var(--red)   !important;
    color:            white         !important;
    border:           none          !important;
    border-radius:    6px           !important;
    font-family:      'Chevin Pro','Gotham','Barlow',sans-serif !important;
    font-weight:      600           !important;
    letter-spacing:   0.03em;
    padding:          0.45rem 1.2rem !important;
    transition:       background-color 0.15s;
  }
  .stButton > button:hover,
  .stDownloadButton > button:hover { background-color: var(--red-h) !important; }

  /* File uploader drop zone */
  [data-testid="stFileUploaderDropzone"] {
    background-color: var(--bg-card)   !important;
    border:           1.5px dashed var(--border) !important;
    border-radius:    10px             !important;
  }

  /* Result card wrapper */
  .result-card {
    background:    var(--bg-card);
    border:        1px solid var(--border);
    border-radius: 12px;
    padding:       1.5rem;
    margin-top:    1rem;
  }

  /* Reusable label style (uppercase track) */
  .section-label {
    font-size:      0.70rem;
    font-weight:    700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color:          var(--muted);
    margin-bottom:  0.35rem;
  }

  /* Header bar */
  .header-bar   { display:flex; align-items:center; gap:0.8rem; margin-bottom:1.4rem; }
  .logo-box     { background:var(--red); color:white; font-family:'Chevin Pro','Gotham','Barlow',sans-serif;
                  font-weight:700; font-size:1.05rem; padding:0.3rem 0.65rem;
                  border-radius:4px; letter-spacing:0.06em; line-height:1.1; }
  .logo-sub     { font-size:0.6rem; color:var(--muted); letter-spacing:0.14em;
                  text-transform:uppercase; margin-top:1px; }
  .app-title    { font-size:1.45rem; font-weight:700; color:var(--text); margin:0;
                  font-family:'Chevin Pro','Gotham','Barlow',sans-serif; }

  /* Inline chip / badge */
  .chip { display:inline-block; background:var(--chip-bg); border:1px solid var(--border);
          border-radius:20px; padding:0.15rem 0.6rem; font-size:0.74rem;
          font-weight:500; color:var(--text); }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility:hidden; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DEFAULT_SHADOW_FILE = "default_shadow.png"

# Pixel classification thresholds
ALPHA_THRESHOLD = 25       # below → transparent
WHITE_THRESHOLD = 235      # all channels above → near-white (background)
BLACK_THRESHOLD = 20       # all channels below → near-black (shadow artifact)

# What fraction of the bounding-box height counts as the "ground contact zone"
GROUND_ZONE_FRACTION = 0.25


# ─── PIXEL CLASSIFICATION ─────────────────────────────────────────────────────

def valid_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """
    Mask for PRODUCT images.

    Rejects: transparent canvas, near-white backgrounds, near-black baked shadows.
    Keeps: any mid-tone or coloured pixel that belongs to the actual object.
    """
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    is_transparent = a < ALPHA_THRESHOLD
    is_near_white  = (r > WHITE_THRESHOLD) & (g > WHITE_THRESHOLD) & (b > WHITE_THRESHOLD)
    is_near_black  = (r < BLACK_THRESHOLD) & (g < BLACK_THRESHOLD) & (b < BLACK_THRESHOLD)

    return ~(is_transparent | is_near_white | is_near_black)


def shadow_pixel_mask(arr: np.ndarray) -> np.ndarray:
    """
    Mask for SHADOW images.

    Standard Photoshop shadow exports store the shadow as black (or very dark)
    pixels with alpha controlling the density — so near-black IS the content.
    This mask only rejects fully-transparent pixels; it keeps everything else,
    including dark and semi-transparent pixels that form the shadow body.

    Using valid_pixel_mask here would wipe the entire shadow, leaving no anchor.
    """
    a = arr[:, :, 3]
    return a >= ALPHA_THRESHOLD


# ─── BACKGROUND DETECTION & REMOVAL ──────────────────────────────────────────

def has_transparency(img: Image.Image) -> bool:
    """
    Returns True when the image already has a meaningful transparent region.

    Threshold: > 0.5 % of pixels with alpha < 200. This catches fully-cut
    product PNGs while ignoring JPEGs and fully-opaque PNGs.
    """
    if img.mode not in ("RGBA", "LA", "PA"):
        return False
    arr = np.array(img.convert("RGBA"))
    return bool((arr[:, :, 3] < 200).mean() > 0.005)


def remove_solid_background(
    img: Image.Image,
    tolerance: int = 30,
    corner_sample: int = 10,
) -> Image.Image:
    """
    Removes a solid studio background using a trimap / alpha-matting approach.

    Why the previous erode→feather approach produced dirty borders
    --------------------------------------------------------------
    Eroding the binary BFS mask then Gaussian-blurring it replaces the genuine
    anti-aliased edge pixels (which contain a mix of product + background colour)
    with artificially semi-transparent *inner* pixels.  Those inner pixels have
    the right product colour but the wrong spatial position, producing a jagged,
    unnatural edge.

    Trimap pipeline (what this does instead)
    -----------------------------------------
    1. Sample corners → background colour.
    2. Full-resolution colour-distance map  dist(x,y) = |pixel − bg_color|.
    3. BFS flood-fill on a ≤400 px downscale → rough fg/bg split.
    4. Upsample BFS result (NEAREST) → definite fg mask.
    5. At full resolution define a transition zone:
         • erode  the fg mask by TRANS_PX → definite core (alpha = 1)
         • dilate the fg mask by TRANS_PX → outermost halo
         • everything between core and halo = transition ring
    6. In the transition ring: alpha = clip(dist / tolerance, 0, 1).
       This uses the *actual per-pixel colour* to decide how opaque each
       boundary pixel is — a pixel that is 50 % blended with the background
       naturally gets alpha ≈ 0.5, matching the original render.
    7. Light Gaussian smooth (radius 0.8) to remove any staircase artefacts
       from the NEAREST-upsampled BFS boundary.
    8. Defringe: for semi-transparent pixels, subtract background colour
       contamination using the matting equation  Fg = (C − (1−α)·Bg) / α.
    """
    from collections import deque

    img_rgba = img.convert("RGBA")
    arr = np.array(img_rgba, dtype=np.float32)
    H, W = arr.shape[:2]

    # ── 1. Background colour from corners ─────────────────────────────────
    # Sample at least 3 % of the smaller dimension so thin-border images still
    # get an accurate background estimate.
    c = min(max(corner_sample, int(min(H, W) * 0.03)), H // 4, W // 4)
    corners = np.vstack([
        arr[:c,  :c,  :3].reshape(-1, 3),
        arr[:c,  -c:, :3].reshape(-1, 3),
        arr[-c:, :c,  :3].reshape(-1, 3),
        arr[-c:, -c:, :3].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)   # (3,)

    # ── 2. Full-resolution colour-distance map ─────────────────────────────
    dist = np.sqrt(((arr[:, :, :3] - bg_color) ** 2).sum(axis=2))   # (H, W)
    is_bg_candidate = dist <= tolerance

    # ── 3. BFS flood-fill on downscaled mask ──────────────────────────────
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

    for y in range(sh):
        _seed(y, 0);      _seed(y, sw - 1)
    for x in range(sw):
        _seed(0, x);      _seed(sh - 1, x)

    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        y, x = queue.popleft()
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < sh and 0 <= nx < sw and not bg_small[ny, nx] and mask_small[ny, nx]:
                bg_small[ny, nx] = True
                queue.append((ny, nx))

    # ── 3b. Dilate background in lower half to absorb floor-shadow halos ────
    # The floor shadow (medium-grey pixels too dark for the BFS to flood) sits
    # below the product. Expanding the BG mask by ~5 px at this scale (≈15 px
    # at full res for a 1200 px image) removes it. Limiting the dilation to the
    # lower half keeps the upper product edges untouched.
    split_row   = sh // 2
    bg_lower    = bg_small.copy()
    bg_lower[:split_row, :] = False                                 # blank upper half
    bg_pil_low  = Image.fromarray((bg_lower * 255).astype(np.uint8), "L")
    bg_pil_low  = bg_pil_low.filter(ImageFilter.MaxFilter(11))     # 5 px dilation
    bg_small    = bg_small | (np.array(bg_pil_low) > 128)

    # ── 4. Upsample BFS → definite fg/bg at full resolution ───────────────
    # NEAREST preserves hard edges; the transition zone handles smoothness.
    fg_mask = np.array(
        Image.fromarray(((~bg_small) * 255).astype(np.uint8), "L")
        .resize((W, H), Image.Resampling.NEAREST)
    ) > 128

    # ── 5. Transition zone via full-resolution morphological ops ──────────
    # Width scales with image size (≈1.5 % of the longer edge, capped at 30 px).
    trans_px = min(30, max(12, int(max(H, W) * 0.015)))
    ksize    = 2 * trans_px + 1

    fg_pil   = Image.fromarray((fg_mask * 255).astype(np.uint8), "L")
    fg_core  = np.array(fg_pil.filter(ImageFilter.MinFilter(ksize))) > 128  # eroded
    fg_halo  = np.array(fg_pil.filter(ImageFilter.MaxFilter(ksize))) > 128  # dilated
    transition = fg_halo & ~fg_core   # the uncertain boundary ring

    # ── 6. Compose alpha ──────────────────────────────────────────────────
    # Core fg → 1.0 | transition → colour-distance ramp | bg → 0.0
    alpha_dist = np.clip(dist / tolerance, 0.0, 1.0)
    alpha = np.where(fg_core, 1.0,
            np.where(transition, alpha_dist,
            0.0))

    # ── 7. Light Gaussian smooth ───────────────────────────────────────────
    alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), "L")
    alpha_pil = alpha_pil.filter(ImageFilter.GaussianBlur(radius=0.8))
    alpha = np.array(alpha_pil, dtype=np.float32) / 255.0

    # ── 8. Defringe ────────────────────────────────────────────────────────
    # Matting equation: Fg = (C − (1−α)·Bg) / α
    rgb  = arr[:, :, :3].copy()
    semi = (alpha > 0.01) & (alpha < 0.99)
    if semi.any():
        a3  = alpha[:, :, np.newaxis]
        bg3 = bg_color[np.newaxis, np.newaxis, :]
        fg  = (rgb - (1.0 - a3) * bg3) / np.clip(a3, 0.01, 1.0)
        rgb = np.where(semi[:, :, np.newaxis], np.clip(fg, 0, 255), rgb)

    # ── Assemble RGBA ──────────────────────────────────────────────────────
    out          = np.zeros((H, W, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[:, :,  3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


# ─── VISUAL CORNER DETECTION ─────────────────────────────────────────────────

def find_solid_base_row(mask: np.ndarray) -> int:
    """
    Returns the y-coordinate (row) of the lowest row that still belongs to the
    SOLID object body, ignoring any baked-in ground shadow below it.

    Many product renders ship with a soft elliptical floor shadow already
    composited into the PNG. That shadow appears as rows of *sparse* valid pixels
    at the very bottom of the image — the shadow fades/tapers outward. The actual
    object feet sit above those sparse rows in dense, high-coverage rows.

    Algorithm
    ---------
    1. Count valid pixels per row.
    2. Determine the peak density inside the upper 70 % of the image (the body).
    3. Define "solid" as ≥ DENSITY_FRACTION × peak. Rows below that are sparse
       (ground shadow, taper, or anti-alias fringe).
    4. Scan from the bottom up; return the first row that qualifies as solid.

    This lets the anchor search ignore the baked-in shadow and land on the
    actual feet of the object.
    """
    DENSITY_FRACTION = 0.08   # a row must have ≥ 8 % of peak density to count

    H = mask.shape[0]
    row_counts = mask.sum(axis=1).astype(float)   # valid pixels per row

    # Reference peak: look only at the top 70 % to avoid the shadow region biasing it
    body_counts = row_counts[: max(1, int(H * 0.70))]
    peak = body_counts.max()

    if peak == 0:
        return H - 1  # no valid pixels at all

    threshold = peak * DENSITY_FRACTION

    for row in range(H - 1, -1, -1):
        if row_counts[row] >= threshold:
            return row

    return H - 1


def _bottom_profile(mask: np.ndarray) -> np.ndarray:
    """
    For every column x, returns the row index of the bottommost valid pixel,
    or -1 if the column has no valid pixels at all.

    This gives the "silhouette ground line" — how close each vertical strip of
    the object comes to the floor — which is the foundation for corner detection.
    """
    H = mask.shape[0]
    has_any = mask.any(axis=0)                           # (W,) bool
    # argmax on the vertically-flipped mask finds the first True from the bottom
    dist_from_bottom = np.argmax(mask[::-1, :], axis=0)  # (W,) int
    bottom_rows = H - 1 - dist_from_bottom
    return np.where(has_any, bottom_rows, -1).astype(int)


def find_bottom_left_anchor(img: Image.Image) -> Optional[Tuple[int, int]]:
    """
    Finds the visual bottom-left corner of the product — where the perspective
    shadow attaches to the object's front-left foot.

    Why the old "leftmost pixel in bottom zone" approach failed
    -----------------------------------------------------------
    On a perspective box (e.g. a sauna viewed from the front-right), the back-left
    wall extends furthest left in the image, but its bottom edge sits HIGH UP due
    to perspective foreshortening. "Leftmost pixel in bottom zone" can land on
    that distant wall — not the front foot.

    Correct Photoshop-artist reasoning
    -----------------------------------
    A retoucher asks: "Which is the leftmost column that still touches the ground?"
    The front-left foot IS leftward AND reaches the global base of the object.
    The back wall is leftward but does NOT reach that low.

    Algorithm (bottom-profile approach)
    ------------------------------------
    1. Build a per-column "bottom profile": bottommost valid pixel in each column.
    2. Find the global bottom row (max of the profile).
    3. Accept columns whose bottom pixel is within TOLERANCE of that global bottom.
       Tight tolerance (3 % of object height, min 8 px) rejects the back wall
       whose bottom sits noticeably higher due to perspective.
    4. The leftmost accepted column is the front-left foot.
    """
    arr  = np.array(img.convert("RGBA"))
    mask = valid_pixel_mask(arr)
    prof = _bottom_profile(mask)

    valid_cols = np.where(prof >= 0)[0]
    if len(valid_cols) == 0:
        return None

    # ── Column-density guard ───────────────────────────────────────────────
    # Sparse columns (floor-shadow remnants, anti-alias fringe) contain very
    # few valid pixels vertically. Real object columns span a meaningful
    # fraction of the image height. Filtering them out prevents the anchor
    # from landing on a shadow patch that extends below the product feet.
    col_counts    = mask.sum(axis=0)                       # valid px per column
    min_col_count = max(3, int(mask.shape[0] * 0.02))     # ≥ 2 % of image height
    valid_cols    = valid_cols[col_counts[valid_cols] >= min_col_count]
    if len(valid_cols) == 0:
        return None

    global_bottom = int(prof[valid_cols].max())

    # Tolerance: how far above global_bottom still counts as "on the ground"
    top_rows   = np.where(mask.any(axis=1))[0]
    obj_height = global_bottom - int(top_rows.min()) if len(top_rows) > 0 else 100
    tolerance  = max(8, int(obj_height * 0.03))   # 3 % of height, min 8 px

    candidates = valid_cols[prof[valid_cols] >= global_bottom - tolerance]
    if len(candidates) == 0:
        candidates = valid_cols

    anchor_x = int(candidates.min())
    anchor_y = int(prof[anchor_x])
    return anchor_x, anchor_y


def find_bottom_right_anchor(img: Image.Image) -> Optional[Tuple[int, int]]:
    """
    Finds the visual bottom-right corner of the shadow image — the tip that is
    pinned to the product's front-left foot.

    Mirror of find_bottom_left_anchor: we want the RIGHTMOST column that still
    reaches the global bottom of the shadow. This is the dense attachment edge of
    a perspective shadow (the far-left tip tapers away from the object and sits
    higher, so it is correctly rejected by the tolerance filter).

    Uses shadow_pixel_mask (not valid_pixel_mask) because shadow PNGs are
    typically black pixels with alpha — keeping only non-transparent pixels.
    """
    arr  = np.array(img.convert("RGBA"))
    mask = shadow_pixel_mask(arr)
    prof = _bottom_profile(mask)

    valid_cols = np.where(prof >= 0)[0]
    if len(valid_cols) == 0:
        return None

    global_bottom = int(prof[valid_cols].max())

    top_rows   = np.where(mask.any(axis=1))[0]
    obj_height = global_bottom - int(top_rows.min()) if len(top_rows) > 0 else 100
    tolerance  = max(8, int(obj_height * 0.03))

    candidates = valid_cols[prof[valid_cols] >= global_bottom - tolerance]
    if len(candidates) == 0:
        candidates = valid_cols

    anchor_x = int(candidates.max())
    anchor_y = int(prof[anchor_x])
    return anchor_x, anchor_y


def strip_baked_shadow(img: Image.Image) -> Image.Image:
    """
    Crops the product image so that any baked-in floor shadow below the solid
    object base is removed. The result is the product isolated to its body,
    ready for the compositor to attach a clean new shadow.

    Uses find_solid_base_row() to locate the crop line, then adds a small
    margin so anti-aliased pixels at the feet aren't clipped.
    """
    arr = np.array(img.convert("RGBA"))
    mask = valid_pixel_mask(arr)
    base_row = find_solid_base_row(mask)

    # Breathing-room margin so anti-aliased foot pixels aren't clipped
    margin = max(20, int(img.height * 0.03))
    crop_bottom = min(base_row + margin, img.height)
    return img.crop((0, 0, img.width, crop_bottom))


# ─── COMPOSITING ──────────────────────────────────────────────────────────────

def composite(
    product_img:  Image.Image,
    shadow_img:   Image.Image,
    shadow_scale: float = 0.8,
    offset_x:     int = 0,
    offset_y:     int = 0,
    strip_shadow: bool = True,
    bg_tolerance: int = 30,
) -> Image.Image:
    """
    Main compositing function.

    Steps
    -----
    0. If the product has no transparency, auto-remove its solid background.
    1. Convert to RGBA. Optionally strip the baked-in floor shadow.
    2. Crop both images to tight bounding boxes.
    3. Scale the shadow so its width = shadow_scale × product_width.
    4. Detect anchors (product bottom-left, shadow bottom-right).
    5. Compute where the shadow's top-left pixel must be placed so the anchors
       coincide. Apply offset_x / offset_y fine-tuning.
    6. Build a canvas large enough to contain both images without cropping.
    7. Paste shadow first (background layer), then product (foreground).
    8. Trim transparent borders and return.

    Parameters
    ----------
    product_img  : Source product image (any mode; converted to RGBA internally).
    shadow_img   : Source shadow image (PNG with alpha channel recommended).
    shadow_scale : shadow_width = shadow_scale × product_bounding_box_width.
    offset_x     : Pixel nudge for the shadow position (positive = right).
    offset_y     : Pixel nudge for the shadow position (positive = down).
    strip_shadow : When True, auto-remove any baked-in floor shadow from the
                   product before compositing.
    """

    # ── 0. Auto background removal ────────────────────────────────────────
    # If the product image has no transparent region it's a flat JPEG / shot-
    # on-white PNG. Remove the solid background first so the compositing and
    # anchor detection work on a clean cutout.
    prod = product_img.convert("RGBA")
    if not has_transparency(prod):
        prod = remove_solid_background(prod, tolerance=bg_tolerance)

    shad = shadow_img.convert("RGBA")

    # ── 1. Optionally strip the existing floor shadow ─────────────────────
    if strip_shadow:
        prod = strip_baked_shadow(prod)

    p_bbox = prod.getbbox()
    s_bbox = shad.getbbox()

    # If either image is entirely transparent, return the product as-is
    if p_bbox is None or s_bbox is None:
        return prod

    prod = prod.crop(p_bbox)
    shad = shad.crop(s_bbox)

    pw, ph = prod.size

    # ── 2. Scale shadow ────────────────────────────────────────────────────
    target_w = max(1, int(pw * shadow_scale))
    aspect   = shad.height / shad.width
    target_h = max(1, int(target_w * aspect))
    shad = shad.resize((target_w, target_h), Image.Resampling.LANCZOS)
    sw, sh = shad.size

    # ── 3. Detect anchors ──────────────────────────────────────────────────
    p_anchor = find_bottom_left_anchor(prod)
    s_anchor = find_bottom_right_anchor(shad)

    # Geometric fallbacks if detection finds no valid pixels
    if p_anchor is None:
        p_anchor = (0, ph - 1)
    if s_anchor is None:
        s_anchor = (sw - 1, sh - 1)

    pcx, pcy = p_anchor  # product corner (in cropped product space)
    sax, say = s_anchor  # shadow anchor  (in scaled shadow space)

    # ── 4. Shadow placement ────────────────────────────────────────────────
    # We want the shadow's anchor pixel to land exactly on the product's anchor
    # pixel. The shadow's top-left is therefore at (pcx - sax, pcy - say)
    # relative to the product's top-left, plus any fine-tune offsets.
    spx = pcx - sax + offset_x
    spy = pcy - say + offset_y

    # ── 5. Dynamic canvas ──────────────────────────────────────────────────
    # Product occupies [0, pw) × [0, ph) in its own coordinate system.
    # Shadow occupies  [spx, spx+sw) × [spy, spy+sh).
    cx0 = min(0, spx)
    cy0 = min(0, spy)
    cx1 = max(pw, spx + sw)
    cy1 = max(ph, spy + sh)

    canvas = Image.new("RGBA", (cx1 - cx0, cy1 - cy0), (0, 0, 0, 0))

    # Convert to canvas-relative coordinates
    p_x = -cx0
    p_y = -cy0
    s_x = spx - cx0
    s_y = spy - cy0

    # ── 6. Composite ───────────────────────────────────────────────────────
    canvas.paste(shad, (s_x, s_y), shad)  # shadow → background layer
    canvas.paste(prod, (p_x, p_y), prod)  # product → foreground layer

    # ── 7. Trim transparent borders ────────────────────────────────────────
    final_bbox = canvas.getbbox()
    if final_bbox:
        canvas = canvas.crop(final_bbox)

    return canvas


# ─── UTILITY ──────────────────────────────────────────────────────────────────

def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_default_shadow() -> Optional[Image.Image]:
    """Load default_shadow.png from the working directory, if present."""
    if os.path.exists(DEFAULT_SHADOW_FILE):
        try:
            return Image.open(DEFAULT_SHADOW_FILE)
        except Exception:
            pass
    return None


# ─── UI ───────────────────────────────────────────────────────────────────────

def render_sidebar() -> Tuple[float, int, int, int, bool, str, int]:
    """Render sidebar controls; return (shadow_scale, offset_x, offset_y, bg_tolerance, add_bg, bg_hex, margin_px)."""
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:1rem;">'
            '<div class="logo-box" style="font-size:0.85rem;padding:0.25rem 0.5rem;">HARVIA</div>'
            '<div class="logo-sub">Labs</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Shadow Scale</div>', unsafe_allow_html=True)
        shadow_scale = st.slider(
            "shadow_scale",
            min_value=0.1, max_value=1.5, value=0.7, step=0.025,
            help="Shadow width as a multiple of the product's bounding-box width.",
            label_visibility="collapsed",
        )
        st.caption(f"{shadow_scale:.3f}× product width")

        st.markdown("---")
        st.markdown('<div class="section-label">BG Removal Tolerance</div>', unsafe_allow_html=True)
        bg_tolerance = st.slider(
            "bg_tolerance",
            min_value=5, max_value=80, value=30, step=1,
            help="Max colour distance from detected background. Raise if BG isn't "
                 "fully removed; lower if product edges are being eaten.",
            label_visibility="collapsed",
        )
        st.caption(f"Tolerance: {bg_tolerance}")

        st.markdown("---")
        st.markdown('<div class="section-label">Anchor Offset (px)</div>', unsafe_allow_html=True)
        offset_x = st.slider(
            "X offset", min_value=-300, max_value=300, value=0, step=1,
            help="Horizontal nudge for shadow alignment.",
            label_visibility="collapsed",
        )
        offset_y = st.slider(
            "Y offset", min_value=-300, max_value=300, value=0, step=1,
            help="Vertical nudge for shadow alignment.",
            label_visibility="collapsed",
        )
        st.caption(f"X: {offset_x:+d} px  ·  Y: {offset_y:+d} px")

        st.markdown("---")
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        add_bg = st.checkbox(
            "Add background colour", value=False,
            help="Composite over a solid colour instead of exporting transparent PNG.",
        )
        bg_hex = "#F0EDE6"
        if add_bg:
            bg_hex = st.color_picker("Colour", value="#F0EDE6", label_visibility="collapsed")
        margin_px = st.slider(
            "Margin (px)", min_value=0, max_value=300, value=0, step=5,
            help="Padding added around the final image.",
            label_visibility="collapsed",
        )
        st.caption(f"Margin: {margin_px} px")

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.72rem;color:var(--muted);line-height:1.5;">'
            'Place <code>default_shadow.png</code> in the app folder to load it automatically.'
            '</div>',
            unsafe_allow_html=True,
        )

    return shadow_scale, offset_x, offset_y, bg_tolerance, add_bg, bg_hex, margin_px


def render_header() -> None:
    st.markdown(
        '<div class="header-bar" style="align-items:flex-end;gap:1.1rem;">'
        '  <div style="flex-shrink:0;">'
        '    <div class="logo-box" style="font-size:1.55rem;padding:0.35rem 0.75rem;'
        '         letter-spacing:0.08em;">HARVIA</div>'
        '    <div class="logo-sub" style="font-size:0.58rem;letter-spacing:0.18em;'
        '         margin-top:3px;">LABS</div>'
        '  </div>'
        '  <div style="border-left:2px solid var(--border);padding-left:1rem;">'
        '    <p class="app-title" style="font-size:1.65rem;margin:0;line-height:1.1;">'
        '      ShadowDrop</p>'
        '    <p style="font-size:0.78rem;color:var(--muted);margin:0.2rem 0 0;">'
        '      Automated perspective-shadow compositing for product images</p>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _mark_anchor(img: Image.Image, point: Tuple[int, int], on_dark: bool = False) -> Image.Image:
    """
    Returns a copy of img with a red crosshair drawn at point.
    on_dark=True composites the image over a grey checkerboard so transparent
    regions are visible (useful for the shadow whose background is empty).
    """
    # Checkerboard background makes alpha regions visible for the shadow panel
    if on_dark:
        tile = 12
        bg = Image.new("RGBA", img.size, (180, 180, 180, 255))
        for ty in range(0, img.height, tile):
            for tx in range(0, img.width, tile):
                if (tx // tile + ty // tile) % 2 == 0:
                    box = (tx, ty, min(tx + tile, img.width), min(ty + tile, img.height))
                    bg.paste((210, 210, 210, 255), box)
        vis = bg
        vis.paste(img, mask=img)
    else:
        vis = img.copy().convert("RGBA")

    draw = ImageDraw.Draw(vis)
    px, py = point
    r  = max(6, min(18, img.width // 50))   # dot radius scales with image size
    arm = r * 3                              # crosshair arm length

    # Cross arms
    draw.line([(px - arm, py), (px + arm, py)], fill=(255, 40, 40, 255), width=2)
    draw.line([(px, py - arm), (px, py + arm)], fill=(255, 40, 40, 255), width=2)
    # Filled circle
    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=(255, 40, 40, 200))

    return vis


def render_debug(
    prod_for_anchor: Image.Image,
    shad_for_anchor: Image.Image,
    p_anchor:        Optional[Tuple[int, int]],
    s_anchor:        Optional[Tuple[int, int]],
) -> None:
    """Expander with visual anchor markers so misalignments are easy to spot."""
    with st.expander("Anchor debug — click to verify anchor placement"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Product — bottom-left anchor**")
            if p_anchor:
                st.image(_mark_anchor(prod_for_anchor, p_anchor), use_container_width=True)
                st.code(f"x={p_anchor[0]}, y={p_anchor[1]}")
            else:
                st.image(prod_for_anchor, use_container_width=True)
                st.warning("Anchor not found")
            st.caption(f"Image size: {prod_for_anchor.size}")
        with c2:
            st.markdown("**Shadow — bottom-right anchor**")
            if s_anchor:
                st.image(_mark_anchor(shad_for_anchor, s_anchor, on_dark=True),
                         use_container_width=True)
                st.code(f"x={s_anchor[0]}, y={s_anchor[1]}")
            else:
                st.image(shad_for_anchor, use_container_width=True)
                st.warning("Anchor not found")
            st.caption(f"Image size: {shad_for_anchor.size}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    shadow_scale, offset_x, offset_y, bg_tolerance, add_bg, bg_hex, margin_px = render_sidebar()
    render_header()

    # ── Upload inputs ──────────────────────────────────────────────────────
    col_prod, col_shad = st.columns(2, gap="medium")

    with col_prod:
        st.markdown('<div class="section-label">Product Image</div>', unsafe_allow_html=True)
        product_file = st.file_uploader(
            "product",
            type=["png", "jpg", "jpeg"],
            key="product",
            label_visibility="collapsed",
        )

    with col_shad:
        st.markdown(
            '<div class="section-label">'
            'Shadow Image <span class="chip">optional</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        shadow_file = st.file_uploader(
            "shadow",
            type=["png"],
            key="shadow",
            label_visibility="collapsed",
        )

    # ── Resolve shadow source ──────────────────────────────────────────────
    shadow_img: Optional[Image.Image] = None
    shadow_label = ""

    if shadow_file is not None:
        shadow_img   = Image.open(shadow_file)
        shadow_label = "Uploaded shadow"
    else:
        default = load_default_shadow()
        if default is not None:
            shadow_img   = default
            shadow_label = f"Auto-loaded `{DEFAULT_SHADOW_FILE}`"

    # ── Early exits ────────────────────────────────────────────────────────
    if product_file is None:
        st.info("Upload a product image above to get started.")
        return

    if shadow_img is None:
        st.warning(
            f"No shadow image uploaded and `{DEFAULT_SHADOW_FILE}` was not found "
            "in the app directory. Please upload a shadow PNG or add a default file."
        )
        return

    # ── Load & composite ───────────────────────────────────────────────────
    product_img = Image.open(product_file)

    with st.spinner("Compositing…"):
        result = composite(
            product_img,
            shadow_img,
            shadow_scale=shadow_scale,
            offset_x=offset_x,
            offset_y=offset_y,
            bg_tolerance=bg_tolerance,
        )

    # ── Apply output options ───────────────────────────────────────────────
    # Work on a separate copy so the debug panel still sees the raw composite.
    output = result

    # Parse bg colour once so it's available for both the fill and margin steps
    hx = bg_hex.lstrip("#")
    bg_r, bg_g, bg_b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)

    if add_bg:
        bg_canvas = Image.new("RGBA", output.size, (bg_r, bg_g, bg_b, 255))
        bg_canvas.paste(output, mask=output)
        output = bg_canvas

    if margin_px > 0:
        ow, oh = output.size
        fill = (bg_r, bg_g, bg_b, 255) if add_bg else (0, 0, 0, 0)
        margin_canvas = Image.new("RGBA", (ow + 2 * margin_px, oh + 2 * margin_px), fill)
        margin_canvas.paste(output, (margin_px, margin_px))
        output = margin_canvas

    # ── Result display ─────────────────────────────────────────────────────
    label_parts = [shadow_label]
    if add_bg:
        label_parts.append(f"bg {bg_hex}")
    if margin_px:
        label_parts.append(f"{margin_px}px margin")

    st.markdown(
        f'<div class="result-card">'
        f'<div class="section-label">Result — {" · ".join(label_parts)}</div>',
        unsafe_allow_html=True,
    )
    st.image(output, use_container_width=True)

    # ── Recompute anchor inputs for the debug panel ───────────────────────
    # Must mirror exactly what composite() does so the preview matches reality.
    _prod = product_img.convert("RGBA")
    if not has_transparency(_prod):
        _prod = remove_solid_background(_prod, tolerance=bg_tolerance)
    _prod = strip_baked_shadow(_prod)
    _pbbox = _prod.getbbox()
    if _pbbox:
        _prod = _prod.crop(_pbbox)

    _shad = shadow_img.convert("RGBA")
    _sbbox = _shad.getbbox()
    if _sbbox:
        _shad = _shad.crop(_sbbox)

    _pw = _prod.size[0]
    _tw = max(1, int(_pw * shadow_scale))
    _th = max(1, int(_tw * (_shad.height / _shad.width)))
    _shad_scaled = _shad.resize((_tw, _th), Image.Resampling.LANCZOS)

    render_debug(
        _prod, _shad_scaled,
        find_bottom_left_anchor(_prod),
        find_bottom_right_anchor(_shad_scaled),
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────
    st.download_button(
        label="Download PNG",
        data=to_png_bytes(output),
        file_name="product_with_shadow.png",
        mime="image/png",
        use_container_width=True,
    )


main()
