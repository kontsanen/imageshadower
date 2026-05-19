import streamlit as st
import numpy as np
from PIL import Image
import io

# Asetetaan sivun otsikko ja leveys
st.set_page_config(page_title="Varjoautomaatio", page_icon="🌤️", layout="centered")

def find_bottom_left_corner(image):
    """
    Etsii esineen vasemman alakulman rajaamalla tarkastelun esineen alimpaan 20 prosenttiin
    ja etsimällä sieltä vasemmanpuoleisimman pikselin.
    """
    img_array = np.array(image.convert("RGBA"))
    
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    a = img_array[:, :, 3]
    
    is_transparent = a < 10
    is_white = (r > 240) & (g > 240) & (b > 240)
    is_black = (r < 15) & (g < 15) & (b < 15)
    
    is_object = ~(is_transparent | is_white | is_black)
    
    # Etsitään kaikkien esineeseen kuuluvien pikselien koordinaatit (y, x)
    valid_pixels = np.argwhere(is_object)
    
    if len(valid_pixels) == 0:
        return None
        
    # Selvitetään koko esineen ääripisteet
    min_y, min_x = valid_pixels.min(axis=0)
    max_y, max_x = valid_pixels.max(axis=0)
    
    # Määritellään "alaosa" (esim. alin 20% esineen korkeudesta)
    height = max_y - min_y
    bottom_threshold = max_y - int(height * 0.20)
    
    # Suodatetaan vain ne pikselit, jotka ovat tämän kynnyksen alapuolella
    bottom_pixels = valid_pixels[valid_pixels[:, 0] >= bottom_threshold]
    
    # Etsitään alaosan pikseleistä se, jolla on pienin X (vasemmanpuoleisin)
    leftmost_idx = np.argmin(bottom_pixels[:, 1])
    target_y, target_x = bottom_pixels[leftmost_idx]
    
    return target_x, target_y

def apply_shadow(product_img, shadow_img):
    """
    Yhdistää varjon siten, että varjon oikea reuna kiinnittyy tuotteen vasempaan alakulmaan.
    """
    product_img = product_img.convert("RGBA")
    shadow_img = shadow_img.convert("RGBA")
    
    corner_point = find_bottom_left_corner(product_img)
    if corner_point is None:
        return product_img 
        
    target_x, target_y = corner_point
    
    # SÄÄDÖT: Voit muuttaa näitä prosentteja, jos varjo näyttää olevan liikaa irti esineestä.
    overlap_x_percent = 0.15 
    overlap_y_percent = 0.30
    
    # Lasketaan varjon sijainti. 
    shadow_x = target_x - shadow_img.width + int(shadow_img.width * overlap_x_percent)
    shadow_y = target_y - shadow_img.height + int(shadow_img.height * overlap_y_percent)
    
    # Lasketaan kankaan koko ja offset
    max_x = max(product_img.width, shadow_x + shadow_img.width)
    max_y = max(product_img.height, shadow_y + shadow_img.height)
    min_x = min(0, shadow_x)
    min_y = min(0, shadow_y)
    
    new_width = max_x - min_x
    new_height = max_y - min_y
    
    canvas = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
    
    offset_x = -min_x
    offset_y = -min_y
    
    # Varjo pohjalle, esine päälle
    canvas.paste(shadow_img, (shadow_x + offset_x, shadow_y + offset_y), shadow_img)
    canvas.paste(product_img, (offset_x, offset_y), product_img)
    
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
        
    return canvas

# --- KÄYTTÖLIITTYMÄ (UI) ---

st.title("Varjogeneraattori 🌤️")
st.markdown("Pudota puutteellinen tuotekuva alle. Järjestelmä etsii esineen vasemman alakulman ja sijoittaa vakiovarjon sen alle automaattisesti.")

# Ladataan tiedostot
col1, col2 = st.columns(2)
with col1:
    shadow_file = st.file_uploader("1. Lataa varjokuva (PNG)", type=["png"])
with col2:
    product_file = st.file_uploader("2. Lataa tuotekuva", type=["png", "jpg", "jpeg"])

if shadow_file and product_file:
    try:
        shadow_image = Image.open(shadow_file)
        product_image = Image.open(product_file)
        
        with st.spinner("Prosessoidaan kuvaa..."):
            final_image = apply_shadow(product_image, shadow_image)
            
            st.subheader("Valmis kuva")
            st.image(final_image, use_container_width=True)
            
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
