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
    # Esimerkiksi 0.1 tarkoittaa, että varjo limittyy 10% leveydestään esineen alle.
    overlap_x_percent = 0.15 
    overlap_y_percent = 0.30
    
    # Lasketaan varjon sijainti. Oletuksena varjo sijoitetaan esineen vasemmalle puolelle
    # siten, että varjon oikea reuna koskettaa/limittyy löydettyyn kulmaan.
    shadow_x = target_x - shadow_img.width + int(shadow_img.width * overlap_x_percent)
    
    # Varjon pystysuuntainen sijoitus. Säädetään niin, että varjon alaosa on linjassa kulman kanssa.
    shadow_y = target_y - shadow_img.height + int(shadow_img.height * overlap_y_percent)
    
    # Lasketaan kankaan koko ja offset samalla tavalla kuin aiemmin
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
