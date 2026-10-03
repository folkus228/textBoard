# %%

from PIL import Image, ImageDraw, ImageFont
from random import randint as rint
import string
import json
import os
from os import listdir
from os.path import join

def render_text_line(text, font_path, font_size=36):
    font = ImageFont.truetype(font_path, font_size)

    # Создаем временный холст для замера
    dummy_img = Image.new('L', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    # Получаем точные координаты: x0, y0 (левый верхний угол), x1, y1 (правый нижний)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    x0, y0, x1, y1 = bbox

    # Считаем чистую ширину и высоту символов
    w = x1 - x0
    h = y1 - y0

    # Добавляем поля (отступы) со всех сторон, например, по 20 пикселей
    padding = 20
    img_w = w + padding * 2
    img_h = h + padding * 2

    # Создаем финальное изображение
    img = Image.new('L', (img_w, img_h), color=255) # white background
    draw = ImageDraw.Draw(img)

    # ГЛАВНОЕ: вычитаем x0 и y0, чтобы компенсировать внутренние сдвиги шрифта,
    # и добавляем padding для безопасного отступа от краев холста
    draw.text((padding - x0, padding - y0), text, font=font, fill=0) # black text
    return img

def generate_data(font_paths, symbols, data_dir_name='generate_data', size_range=(36, 240), len_range=(4, 19), count=100):
    tags_data={}
    os.mkdir(data_dir_name)
    os.mkdir(f'{data_dir_name}/images')
    for font_path in font_paths:
        font = font_path.split("/")[-1]
        tags_data[font]={}
        font_dir = font.split(".")[0]
        os.mkdir(f'{data_dir_name}/images/{font_dir}')
        for i in range(count):
            text = ''
            for _ in range(rint(*len_range)):
                text+=symbols[rint(0, len(symbols)-1)]
            img = render_text_line(text, font_path, rint(*size_range))
            img.save(f'{data_dir_name}/images/{font_dir}/{i}.png')
            tags_data[font][i]=text
    with open(f'{data_dir_name}/tags_data.json', 'w', encoding='utf-8') as f:
        json.dump(tags_data, f, ensure_ascii=False, indent=4)


def generate_data_from_json(fonts_path, jfile_local_path, data_dir, size_range=(108, 108), len_range=(4, 19)):
    # os.mkdir(data_dir)
    with open(join(fonts_path, jfile_local_path), 'r', encoding='utf-8') as jf:
        j = json.load(jf)

        # creating dirs structure
        all_fonts = list(j.keys())
        all_languages = []
        for font_file_name in j.keys():
            all_languages += list(j[font_file_name][0].keys())
        all_languages = set(all_languages)
        for l in all_languages:
            os.mkdir(f'{data_dir}/{l}')
            for f in all_fonts:
                os.mkdir(f'{data_dir}/{l}/{f}')

        # generating data
        for font_file_name in j.keys(): # select font
            metadata = j[font_file_name][1]
            languages = list(metadata.keys())
            languages.remove('type')
            for language in languages: # select language
                vacab = j[font_file_name][0][language]
                font_path = join(fonts_path, font_file_name)
                for i in range(metadata[language]): # generated data unit
                    text = ''
                    for _ in range(rint(*len_range)):
                        text+=vacab[rint(0, len(vacab)-1)]
                    img = render_text_line(text, font_path, rint(*size_range))
                    img.save(f'{data_dir}/{language}/{font_file_name}/{i}.png')
                    with open(f'{data_dir}/{language}/{font_file_name}/{i}.txt', 'w', encoding='utf-8') as tf:
                        tf.write(text)


generate_data_from_json('fonts', 'symbol_supports.json', 'fonts_data')


# FONTS_DIR = 'fonts'
# fonts = listdir(FONTS_DIR)
# paths = [join(FONTS_DIR, f) for f in fonts]

# generate_data(
#     paths,
#     string.ascii_letters + string.digits + string.punctuation,
#     data_dir_name='train_data',
#     count=2000
# )
# generate_data(
#     paths,
#     string.ascii_letters + string.digits + string.punctuation,
#     len_range=(1, 100),
#     data_dir_name='test_data',
#     count=1000
# )
# generate_data(
#     paths,
#     string.ascii_letters + string.digits + string.punctuation,
#     len_range=(3, 30),
#     data_dir_name='validation_data',
#     count=100
# )
