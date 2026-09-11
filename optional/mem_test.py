# %%

from pandas.core.missing import F
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
from model import create_model
from data import HTR_Dataset, dataset, tfm_max_train, tokenizer
from tqdm import tqdm

# device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu')
device='cpu'

USE_CHECKPOINT = True


gpt_params = {
    'vocab_size': tokenizer.vocab_size,
    'emb_size': 128,
    'heads': 8,
    'hidden': 192,
    'drop_rate': 0.1,
    't_layers': 1,
    'max_trained_len': 10_000
}
vit_params = {
    'emb_size': 128,
    'heads': 8,
    'hidden': 384,
    'drop_rate': 0.1,
    'conv_layers': [1, 32, 48, 64, 128],
    't_layers': 6,
    'max_trained_h': 10_000,
    'max_trained_w': 10_000
}
model_params = {
    'vocab_size': tokenizer.vocab_size,
    'emb_size': 128,
    'heads': 8,
    'hidden': 384,
    'drop_rate': 0.1,
    'max_trained_len': 10_000,
    't_layers': 3
}

train_dataset = HTR_Dataset(dataset['train'], transform=tfm_max_train, device=device)
max_h=[0, 0]
max_h_idx=1594
max_w=[0, 0]
max_w_idx=1594
max_hw=[0, 0, 0]
max_hw_idx=1594
idx = 0
# for x, y in train_dataset:
#     _, H, W = x.shape

#     if max_h[0] < H:
#         max_h_idx=idx
#         max_h=[H, W]
#     if max_w[1] < W:
#         max_w_idx=idx
#         max_w=[H, W]
#     if max_hw[2] < H*W:
#         max_hw_idx=idx
#         max_hw=[H, W, H*W]
#     idx+=1

# print(f'max H - ({max_h}) with idx {max_h_idx}')
# print(f'max W - ({max_w}) with idx {max_w_idx}')
# print(f'max H * W - ({max_hw[:2]}) with idx {max_hw_idx}')

model = create_model(vit_params, gpt_params, model_params).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_idx.get(tokenizer.pad_idx, 0))
model.train()
with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
    x, y = train_dataset[max_h_idx]
    y = tokenizer.encode(y).unsqueeze(0)
    x = x.unsqueeze(0)
    y_pred = model(x, y[:, :-1])
    loss = loss_fn(y_pred.permute(0,2,1), y)
    loss.backward()

print()
print(prof.key_averages().table(sort_by='self_cpu_memory_usage', row_limit=15))
