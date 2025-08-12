import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64

def create_logo():
    """
    Создает логотип FairTender.kz и сохраняет его как файл
    Возвращает путь к файлу
    """
    # Создаем изображение
    width, height = 220, 50
    image = Image.new('RGBA', (width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    # Пробуем использовать системный шрифт
    try:
        # Возможные пути к шрифтам
        font_paths = [
            '/System/Library/Fonts/Supplemental/Arial Bold.ttf',  # MacOS
            '/Library/Fonts/Arial Bold.ttf',  # MacOS alternative
            '/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf',  # Ubuntu
            'C:\\Windows\\Fonts\\arialbd.ttf',  # Windows
            '/System/Library/Fonts/Helvetica.ttc'  # Fallback
        ]
        
        font_path = None
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
                
        if font_path:
            font = ImageFont.truetype(font_path, 22)
        else:
            # Если шрифт не найден, используем дефолтный
            font = ImageFont.load_default()
            
    except Exception:
        # В случае ошибки используем дефолтный шрифт
        font = ImageFont.load_default()
    
    # Рисуем текст
    text = "FairTender.kz"
    
    # Цвета
    fair_color = (0, 102, 204)  # Синий
    tender_color = (0, 153, 51)  # Зеленый
    
    # Рисуем Fair синим цветом
    draw.text((10, 10), "Fair", font=font, fill=fair_color)
    
    # Измеряем ширину "Fair"
    fair_width = draw.textlength("Fair", font=font)
    
    # Рисуем Tender зеленым цветом
    draw.text((10 + fair_width, 10), "Tender", font=font, fill=tender_color)
    
    # Измеряем ширину "Tender"
    tender_width = draw.textlength("Tender", font=font)
    
    # Рисуем .kz синим цветом
    draw.text((10 + fair_width + tender_width, 10), ".kz", font=font, fill=fair_color)
    
    # Сохраняем изображение
    logo_path = "static/logo.png"
    image.save(logo_path)
    
    return logo_path

def get_logo_as_base64():
    """
    Возвращает логотип в виде строки base64 для встраивания в HTML
    """
    logo_path = create_logo()
    
    # Если файл существует, читаем его и кодируем в base64
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    return None

# Создаем логотип при импорте модуля
if __name__ == "__main__":
    create_logo() 