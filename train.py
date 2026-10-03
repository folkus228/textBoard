# %%
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR #, ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

# from test import dataset
from tools import ctc_greedy_decode
from data import tokenizer, HTR_Dataset, htr_collate_fn, tfm_train_IAM, tfm_val, get_IAM
from model import CRNN

from tqdm import tqdm
from jiwer import cer

# USE_CHECKPOINT = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
# device = 'cpu'
print(f'Select device ({device})')

def validate(model:nn.Module, loss_fn, validation_loader, device, batch_max:int=70):
    with torch.no_grad():
        model.eval()
        batch_count=0
        losses=[]
        total_cer = 0.0
        total_chars = 0
        for x, img_l, y, text_l in validation_loader: # l-[B, count]
            batch_count+=1
            x, y = x.to(device), y.to(device) # [B, toks]
            y_pred = model(x) # [B, W, C]
            losses.append(loss_fn(y_pred.permute(1, 0, 2).to('cpu'), y.to('cpu'), img_l.to('cpu'), text_l.to('cpu')).to('cpu').item())

            decoded_tokens = ctc_greedy_decode(y_pred) # [B, T(idx)] - расшифровка предикта
            texts_true=[]
            offset=0

            for l in text_l.tolist():
                chunk = y[offset:offset+l]
                texts_true.append(tokenizer.decode(chunk))
                offset+=l
            for pred_ids, true_text in zip(decoded_tokens, texts_true):
                pred_text = tokenizer.decode(torch.tensor(pred_ids)) if pred_ids else ''
                err = cer(true_text, pred_text) if len(true_text) else 0.0
                total_cer += err * len(true_text)
                total_chars += len(true_text)

            if batch_count==batch_max:
                break
        return { 'cel': (sum(losses) / len(losses)), 'cer': (total_cer / total_chars) }

def train(model:nn.Module, loss_fn, optimizer, sheduler, train_loader, validation_loader, device, current_epochs:int=1, epochs:int=10, acum_steps=8):
    for epoch in range(current_epochs, epochs+1):
        model.train()
        for i, (x, img_l, y, text_l) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred.permute(1, 0, 2).to('cpu'), y.to('cpu'), img_l.to('cpu'), text_l.to('cpu'))
            loss.backward()
            if ((i+1) % acum_steps == 0) or i==len(train_loader)-1:
                optimizer.step()
                sheduler.step()
                optimizer.zero_grad()

        metrics_val=validate(model, loss_fn, validation_loader, device)
        metrics_train=validate(model, loss_fn, train_loader, device)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sheduler': sheduler.state_dict(), 'current_epoch': epoch, 'max_epoch': epochs},
                   f'/content/drive/MyDrive/models_CRNN_V2/epoch({epoch}|{epochs})_cel{metrics_val["cel"]:.2f}_cer{metrics_val["cer"]:.2f}.pth')
        print(f'''
            epoch ({epoch}):
                validation:
                    cel - {metrics_val['cel']:.4f}
                    cer - {(100*metrics_val['cer']):.2f}
                train:
                    cel - {metrics_train['cel']:.4f}
                    cer - {(100*metrics_train['cer']):.2f}
            ''')

# ----------------------------------------------| DATA |--------------------------------------------
dataset = get_IAM()
train_dataset = HTR_Dataset(dataset['train'], transform=tfm_train_IAM)
validation_dataset = HTR_Dataset(dataset['validation'], transform=tfm_val)
test_dataset = HTR_Dataset(dataset['test'], transform=tfm_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=htr_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=htr_collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, collate_fn=htr_collate_fn)

# ----------------------------------------------| INIT ALL |----------------------------------------
model = CRNN(tokenizer.vocab_size, 0.2).to(device)
loss_fn = nn.CTCLoss(tokenizer.blank_idx)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
# sheduler = CosineAnnealingLR(optimizer, T_max=(len(validation_loader) // 8), eta_min=0)
sheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=((len(train_loader)*20) // 8)+1, pct_start=0.1, div_factor=25, final_div_factor=1e4)

# ----------------------------------------------| TRAIN |-------------------------------------------
model_name = 'models/IAM_epoch(14_20)_cel0.31_cer0.09.pth'
model.load_state_dict(torch.load(model_name, map_location=device)['model'])
# optimizer.load_state_dict(torch.load(model_name)['optimizer'])
# sheduler.load_state_dict(torch.load(model_name)['sheduler'])
# current_epoch = torch.load(model_name)['current_epoch']
# max_epoch = torch.load(model_name)['max_epoch']
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# train(model, loss_fn, optimizer, sheduler, train_loader, validation_loader, device, current_epochs=1, epochs=20, acum_steps=8)
# train(model, loss_fn, optimizer, sheduler, train_loader, validation_loader, device, current_epochs=current_epoch+1, epochs=max_epoch, acum_steps=8)
# validate(model, loss_fn, test_loader, device, batch_max=-1)
# torch.save(model.state_dict(), 'models/final.pth')


# ----------------------------------------------| DATA |--------------------------------------------
test_metrics = validate(model, loss_fn, test_loader, device, -1)
print(f'''
    metrics:
        cel - {test_metrics['cel']:.4f}
        cer - {(100*test_metrics['cer']):.4f}
    ''')
