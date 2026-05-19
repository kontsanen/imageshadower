import streamlit as st
import numpy as np
from PIL import Image
import io

# Asetetaan sivun otsikko ja leveys
st.set_page_config(page_title="Varjoautomaatio", page_icon="🌤️", layout="centered")

def find_contact_point(image):
    """
    Etsii esineen alimman pisteen (y) ja sen x-keskikohdan.
    Sivuuttaa läpinäkyvät, täysin valkoiset ja täysin mustat pikselit.
    """
    # Muutetaan kuva RGBA-muotoon ja sitten Numpy-matriisiksi nopeaa laskentaa varten
    img_array = np.array(image.convert("RGBA"))
    
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    a = img_array[:, :, 3]
    
    # Määritellään värit, jotka haluamme sivuuttaa (toleranssit voi säätää tarvittaessa)
    is_transparent = a < 10
    is_white = (r > 240) & (g > 240) & (b > 240)
    is_black = (r < 15) & (g < 15) & (b < 15)
    
    # Pikselit, jotka kuuluvat itse esineeseen
    is_object = ~(is_transparent | is_white | is_black)
    
    # Katsotaan, millä riveillä (y) on esineen pikseleitä
    valid_rows = np.any(is_object, axis=1)
    
    if not np.any(valid_rows):
        return None  # Kuvasta ei löytynyt sopivia pikseleitä
        
    # Alin rivi, josta löytyi esine
    lowest_y = np.where(valid_rows)[0][-1]
    
    # Etsitään esineen leveys (x-akselilla), jotta varjo osataan keskittää
    valid_cols = np.any(is_object, axis=0)
    min_x = np.where(valid_cols)[0][0]
    max_x = np.where(valid_cols)[0][-1]
    center_x = (min_x + max_x) // 2
    
    return center_x, lowest_y

def apply_shadow(product_img, shadow_img):
    """
    Yhdistää varjon tuotekuvaan dynaamisesti lasketun alimman pisteen perusteella.
    """
    product_img = product_img.convert("RGBA")
    shadow_img = shadow_img.convert("RGBA")
    
    point = find_contact_point(product_img)
    if point is None:
        return product_img # Palautetaan alkuperäinen, jos kohdetta ei löydy
        
    center_x, lowest_y = point
    
    # Laitetaan varjon vaakasuuntainen keskikohta esineen keskikohdalle
    shadow_x = center_x - (shadow_img.width // 2)
    
    # Laitetaan varjon pystysuuntainen keskikohta esineen alimpaan pisteeseen
    shadow_y = lowest_y - (shadow_img.height // 2)
    
    # Koska varjo voi mennä alkuperäisen kuvan rajojen yli, lasketaan uudelle
    # yhdistetylle kuvalle riittävän suuri kangas (canvas)
    max_x = max(product_img.width, shadow_x + shadow_img.width)
    max_y = max(product_img.height, shadow_y + shadow_img.height)
    min_x = min(0, shadow_x)
    min_y = min(0, shadow_y)
    
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    # Luodaan tyhjä läpinäkyvä tausta
    canvas = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    
    # Lasketaan offset, jotta negatiiviset koordinaatit eivät leikkaudu pois
    offset_x = -min_x
    offset_y = -min_y
    
    # Liimataan ensin varjo alimmaksi tasoksi
    canvas.paste(shadow_img, (shadow_x + offset_x, shadow_y + offset_y), shadow_img)
    
    # Liimataan tuotekuva sen päälle
    canvas.paste(product_img, (offset_x, offset_y), product_img)
    
    # Lopuksi rajataan ylimääräinen tyhjä tila pois, jotta kuva pysyy siistinä
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
        
    return canvas

# --- KÄYTTÖLIITTYMÄ (UI) ---

st.title("Varjogeneraattori 🌤️")
st.markdown("Pudota puutteellinen tuotekuva alle. Järjestelmä etsii esineen alareunan ja sijoittaa vakiovarjon sen alle automaattisesti.")

# Ladataan tiedostot
col1, col2 = st.columns(2)
with col1:
    shadow_file = st.file_uploader("1. Lataa varjokuva (PNG)", type=["png"])
with col2:
    product_file = st.file_uploader("2. Lataa tuotekuva(t)", type=["png", "jpg", "jpeg"])

if shadow_file and product_file:
    try:
        shadow_image = Image.open(shadow_file)
        product_image = Image.open(product_file)
        
        with st.spinner("Prosessoidaan kuvaa..."):
            final_image = apply_shadow(product_image, shadow_image)
            
            # Näytetään tulos
            st.subheader("Valmis kuva")
            st.image(final_image, use_column_width=True)
            
            # Muutetaan kuva takaisin tavuiksi latausta varten
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
