# %%

from PIL import Image, ImageDraw, ImageFont
import random
from os import listdir
from os.path import join

from torchvision.transforms import v2

tfm = v2.ToImage()

def render_text_line(text, font_path, font_size=48):
    font = ImageFont.truetype(font_path, font_size)
    dummy_img = Image.new('L', (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    bbox = dummy_draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2]-bbox[0], bbox[3] - bbox[1]

    img = Image.new('L', (w+20, h+20), color=255) # white background
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, font=font, fill=0) # black text
    return img

img = tfm(render_text_line('The quick brown fox jumps over the lazy dog.', 'fonts/Zull_Wettis_Cyrillic_Script.otf'))


# %%

from matplotlib import pyplot as plt

print(img.shape)

plt.figure(figsize=(10, 6))
plt.imshow(img.permute(1, 2, 0))
plt.show()
