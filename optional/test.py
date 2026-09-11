# %%

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR #, ReduceLROnPlateau

model = nn.Linear(128, 128)
optimizer = AdamW(model.parameters(), lr=1e-4)
sheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=int(1234*4 // 4), pct_start=0.1, div_factor=25, final_div_factor=1e4)

torch.save({ 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sheduler': sheduler.state_dict(), 'test_value': 12}, 'DEL_MODEL.pth')
model.load_state_dict(torch.load('DEL_MODEL.pth')['model'])
optimizer.load_state_dict(torch.load('DEL_MODEL.pth')['optimizer'])
sheduler.load_state_dict(torch.load('DEL_MODEL.pth')['sheduler'])
a=0
a = torch.load('DEL_MODEL.pth')['test_value']
print(a)
