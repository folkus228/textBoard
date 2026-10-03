from matplotlib import pyplot as plt
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torch.nn import functional as F
from os import listdir
from os.path import join
from PIL import Image

class HTR_Tokenizer():
    def __init__(self, tokens: tuple, blank_idx:int=0):
        super().__init__()
        self.char_to_idx = {ch: i for i, ch in enumerate(tokens)}
        self.idx_to_char = {i: ch for i, ch in enumerate(tokens)}
        self.blank_idx=blank_idx
        self.vocab_size = len(tokens)

    def encode(self, text: str) -> torch.Tensor:
            # Конвертирует строку в тензор индексов (LongTensor)
            ids = [self.char_to_idx.get(ch, 0) for ch in text]
            return torch.tensor(ids, dtype=torch.long)

    def decode(self, tensor: torch.Tensor) -> str:
        # Конвертирует тензор/список индексов обратно в строку
        if tensor.ndim > 1:
            tensor = tensor.squeeze(0)
        chars = [self.idx_to_char.get(idx.item(), '') for idx in tensor]
        return ''.join(chars)
tokenizer = HTR_Tokenizer(
    ('<|blank|>',
    ' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
)

def cut_zero_border(img):
    # [C, H, W]
    mask = img.any(dim=0)
    nonzero_coords = torch.nonzero(mask)
    from_h = nonzero_coords[:, 0].min()
    to_h = nonzero_coords[:, 0].max() + 1
    from_w = nonzero_coords[:, 1].min()
    to_w = nonzero_coords[:, 1].max() + 1
    return img[:, from_h:to_h, from_w:to_w]

tfm_val = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.Lambda(lambda x: (x < 0.8).float()),
    v2.Lambda(cut_zero_border),
    v2.Resize(size=100),
    v2.Normalize(mean=[0.0426], std=[0.2020])
])

tfm_train_IAM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.Lambda(lambda x: (x < 0.8).float()),
    v2.RandomRotation(
        degrees=(-13, 13),
        expand=True,
        fill=0),
    v2.RandomAffine(
        degrees=(0, 0),
        # translate=(0.1, 0.1),    # Случайное смещение до 10% по горизонтали и вертикали
        scale=(0.6, 0.9),        # Случайное масштабирование (зум) от 90% до 110%
        shear=(-10, 10),         # Сдвиг (дисторсия/растяжение по осям) от -10 до +10 градусов
        fill=0                   # Чем заполнять пустые зоны, образовавшиеся при сдвиге (0 - черный)
    ),
    v2.Lambda(cut_zero_border),
    v2.Normalize(mean=[0.0426], std=[0.2020])
])

tfm_train_font = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float, scale=True),
    v2.Lambda(lambda x: (x < 0.8).float()),
    v2.RandomRotation(
        degrees=(-15, 15),
        expand=True,
        fill=0),
    v2.RandomAffine(
        degrees=(0, 0),
        translate=(0.1, 0.1),    # Случайное смещение до 10% по горизонтали и вертикали
        scale=(0.6, 1.4),        # Случайное масштабирование (зум) от 90% до 110%
        shear=(-10, 10),         # Сдвиг (дисторсия/растяжение по осям) от -10 до +10 градусов
        fill=0                   # Чем заполнять пустые зоны, образовавшиеся при сдвиге (0 - черный)
    ),
    v2.Lambda(cut_zero_border),
    v2.Normalize(mean=[0.0426], std=[0.2020])
])

# load_from_disk("Teklia/IAM-line")
# dataset
def get_IAM():
    return load_dataset("Teklia/IAM-line")

class HTR_Full_Dataset(Dataset):
    def __init__(self, dataset1, dataset2) -> None:
        super().__init__()
        self.ds1 = dataset1
        self.ds2 = dataset2

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, index):
        return self.ds1[index] if index < len(self.ds1) else self.ds2[index - len(self.ds1)]


# for downloading dataset
class HTR_Dataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data=data
        self.transform=transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return (self.transform(self.data[index]['image']), self.data[index]['text'])

class HTR_Fonts_Dataset(Dataset):
    def __init__(self, data_path, tfm, languages=['en']) -> None:
        super().__init__()
        self.data_path=data_path
        self.languages=languages
        self.tfm=tfm

        self.lens = {}
        for l in languages:
            self.lens[l] = {}
            for f in listdir(join(data_path, l)):
                self.lens[l][f] = len(listdir(join(data_path, l, f))) // 2

    def __len__(self):
        len = 0
        for l in self.languages:
            for f in self.lens[l].keys():
                len += self.lens[l][f]
        return len

    def __getitem__(self, index):
        for l in self.languages:
            for f in self.lens[l].keys():
                index -= self.lens[l][f]
                if index < 0:
                    index = index + self.lens[l][f]
                    file = f'{join(self.data_path, l, f)}/{index}'
                    img = Image.open(f'{file}.png')
                    text = ''
                    with open(f'{file}.txt', 'r', encoding='utf-8') as t:
                        text = t.readline()
                    return self.tfm(img), text
        return None

def htr_collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]

    target_text = torch.cat([tokenizer.encode(t) for t in texts])
    target_len = torch.tensor([len(t) for t in texts])

    max_h = max(img.shape[-2] for img in images)
    max_w = max(img.shape[-1] for img in images)

    padded_images = []
    images_w = []
    for img in images:
        current_h = img.shape[-2]
        current_w = img.shape[-1]

        # Считаем общий паддинг
        total_pad_h = max_h - current_h
        total_pad_w = max_w - current_w

        # Формат F.pad: (left, right, top, bottom)
        img_padded = F.pad(img, (0, total_pad_w, total_pad_h, 0), value=0.0)
        padded_images.append(img_padded)
        images_w.append(current_w)
    images_tensor = torch.stack(padded_images)
    images_len = torch.tensor(images_w) // 8 # 8 - coefficient rescailing of CNN
    return images_tensor, images_len, target_text, target_len
