import streamlit as st
import numpy as np
from PIL import Image
import io

# Asetetaan sivun otsikko ja leveys
st.set_page_config(page_title="Varjoautomaatio v2.0", page_icon="🌤️", layout="wide")

def find_bottom_left_corner_precise(image):
    """
    Etsii esineen vasemman alakulman rajaamalla tarkastelun esineen alimpaan 25 prosenttiin.
    Ennen tätä esine pitää croppata.
    """
    img_array = np.array(image.convert("RGBA"))
    
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    a = img_array[:, :, 3]
    
    is_transparent = a < 10
    # Tehdään väreistä tarkempi suodatus
    is_pure_white = (r == 255) & (g == 255) & (b == 255)
    is_pure_black = (r == 0) & (g == 0) & (b == 0)
    
    # Pikselit, jotka kuuluvat itse esineeseen
    is_object = ~(is_transparent | is_pure_white | is_pure_black)
    
    # Etsitään kaikkien esineeseen kuuluvien pikselien koordinaatit (y, x)
    valid_pixels = np.argwhere(is_object)
    
    if len(valid_pixels) == 0:
        return None
        
    # Selvitetään koko esineen ääripisteet (vaikka se on croppattu, varmistetaan)
    min_y = valid_pixels.min(axis=0)[0]
    max_y = valid_pixels.max(axis=0)[0]
    
    # Määritellään "alaosa" (esim. alin 25% esineen korkeudesta)
    height = max_y - min_y
    bottom_threshold = max_y - int(height * 0.25)
    
    # Suodatetaan vain ne pikselit, jotka ovat tämän kynnyksen alapuolella
    bottom_pixels = valid_pixels[valid_pixels[:, 0] >= bottom_threshold]
    
    # Etsitään alaosan pikseleistä se, jolla on pienin X (vasemmanpuoleisin)
    leftmost_idx = np.argmin(bottom_pixels[:, 1])
    target_y, target_x = bottom_pixels[leftmost_idx]
    
    return target_x, target_y

def prepare_canvas_and_composite(product_img, shadow_img, shadow_width_factor, anchor_x_percent, anchor_y_percent):
    """
    Rajaa tuotteen, skaalaa varjon ja yhdistää ne tarkan ankkuroinnin perusteella.
    """
    p_img = product_img.convert("RGBA")
    s_img = shadow_img.convert("RGBA")
    
    # --- STEP 1: Rajataan tuotekuva, jotta tyhjä tila ei haittaa
    p_bbox = p_img.getbbox()
    if p_bbox:
        p_img = p_img.crop(p_bbox)
    pw, ph = p_img.size
    
    # --- STEP 2: Etsitään saunan kulma (pöydän jalka) croppatusta kuvasta
    p_corner = find_bottom_left_corner_precise(p_img)
    if p_corner is None:
        return p_img 
        
    pcx, pcy = p_corner # Piste, mihin varjo liitetään
    
    # --- STEP 3: Skaalataan varjo tuotteen leveyden mukaan (Sliderssistä)
    # 1.0 = sama leveys, 2.0 = tuplaleveys jne.
    sw_target = int(pw * shadow_width_factor)
    aspect_ratio = s_img.height / s_img.width
    sh_target = int(sw_target * aspect_ratio)
    s_img_scaled = s_img.resize((sw_target, sh_target), Image.Resampling.LANCZOS)
    sw, sh = s_img_scaled.size
    
    # --- STEP 4: Määritellään varjon "ankkuri" (Piste varjon omassa kuvassa)
    # Tämä on se piste, joka koskettaa saunan jalkaa.
    # Ankkurin pitää olla perspektiivivarjossa takana-oikealla.
    sax = int(sw * anchor_x_percent)
    say = int(sh * anchor_y_percent)
    
    # --- STEP 5: Lasketaan varjon absoluuttinen sijainti
    # Varjon ankkuripisteen sax, say pitää osua tuotteen pcx, pcy pisteeseen.
    spx = pcx - sax # Varjon ylävasemman kulman x
    spy = pcy - say # Varjon ylävasemman kulman y
    
    # --- STEP 6: Lasketaan uusi kankaan koko
    # Lasketaan min- ja max-rajat kaikkien pikselien mukaan
    min_x = min(0, spx)
    max_x = max(pw, spx + sw)
    min_y = min(0, spy)
    max_y = max(ph, spy + sh)
    
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    
    # Luodaan tyhjä läpinäkyvä kangas
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    
    # Tuote ja varjo ovat nyt suhteessa kankaan (0,0) pisteeseen.
    # Negatiiviset offsetit muunnetaan positiivisiksi siirtymiksi kankaalla.
    p_paste_x = -min_x
    p_paste_y = -min_y
    
    s_paste_x = spx - min_x
    s_paste_y = spy - min_y
    
    # Varjo pohjalle (mask=s_img_scaled varmistaa läpinäkyvyyden säilymisen)
    canvas.paste(s_img_scaled, (s_paste_x, s_paste_y), s_img_scaled)
    # Tuote päälle
    canvas.paste(p_img, (p_paste_x, p_paste_y), p_img)
    
    # Cropataan lopuksi turha tyhjä tila pois kankaan reunoilta
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
        
    return canvas

# --- KÄYTTÖLIITTYMÄ (UI) v2 ---

st.title("Varjogeneraattori v2.0 🌤️")
st.markdown("Automatisoi perspektiivivarjojen sijoitus. Säädä varjo sopivaksi sivupaneelin ohjaimilla lennosta.")

# Sivupaneelin ohjaimet (Fine-tuning)
st.sidebar.header("Hienosäätö 🎛️")
st.sidebar.markdown("Säädä nämä ensin 'oikean' tuotteen (kuten saunan) kanssa ja käytä niitä samoja asetuksia muille saman kategorian tuotteille.")

# 1. Varjon Skaala (Suhteessa tuotteen leveyteen)
st.sidebar.subheader("1. Varjon Koko")
shadow_width_factor = st.sidebar.slider(
    "Varjon kerroin (1.0 = sama leveys kuin tuote)",
    min_value=0.5, max_value=4.0, value=1.0, step=0.05
)

# 2. Varjon "Ankkuri" (Missä kohtaa varjoa tuotteen kulma sijaitsee?)
st.sidebar.subheader("2. Varjon Ankkuri (Pöydän jalka)")
st.sidebar.markdown("Piste varjokuvassa (prosentteina), joka koskettaa tuotteen vasenta alakulmaa.")
anchor_x_percent = st.sidebar.slider(
    "X-ankkuri (0=Vasen reuna, 1=Oikea reuna)",
    min_value=0.0, max_value=1.0, value=0.95, step=0.01
)
anchor_y_percent = st.sidebar.slider(
    "Y-ankkuri (0=Ylä/Taka reuna, 1=Ala/Etu reuna)",
    min_value=0.0, max_value=1.0, value=0.05, step=0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Oletukset saunalle:** Kerroin 1.0-1.5, X: 0.95, Y: 0.05")

# Ladataan tiedostot
col_in1, col_in2 = st.columns(2)
with col_in1:
    shadow_file = st.file_uploader("1. Lataa varjokuva (PNG)", type=["png"], key="shadow")
with col_in2:
    product_file = st.file_uploader("2. Lataa tuotekuva", type=["png", "jpg", "jpeg"], key="product")

if shadow_file and product_file:
    try:
        shadow_image = Image.open(shadow_file)
        product_image = Image.open(product_file)
        
        with st.spinner("Prosessoidaan kuvaa lennosta..."):
            final_image = prepare_canvas_and_composite(
                product_image, 
                shadow_image, 
                shadow_width_factor, 
                anchor_x_percent, 
                anchor_y_percent
            )
            
            st.subheader("Valmis kuva (Hienosäädä sivupaneelista)")
            st.image(final_image, use_container_width=True)
            
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="⬇️ Lataa valmis kuva",
                data=byte_im,
                file_name="varjostettu_tuote.png",
                mime="image/png",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Virhe kuvan käsittelyssä: {e}")
