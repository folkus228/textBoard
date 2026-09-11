# %%

from PIL import Image
import torch
from tools import ctc_greedy_decode
from matplotlib import pyplot as plt
from data import dataset, tfm_val, tokenizer
from torchvision.transforms import v2
from model import CRNN

device='cpu'

img = Image.open("models/image3.png")
img = tfm_val(img)

model = CRNN(tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load('models/epoch(14_20)_cel0.31_cer0.09.pth', map_location=torch.device('cpu'))['model'])
model.eval()

print(tfm_val(dataset['train']['image'][962]).shape)

pred = model(img[None, None, 0, :, :].to(device))

plt.imshow(img[0, :, :, None])
plt.title('i want you')
plt.show()

print(tokenizer.decode(torch.tensor(ctc_greedy_decode(pred))))
