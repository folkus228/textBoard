# %%

def print_memory_size(bits):
    bytes = bits // 8
    bits %= 8
    Kbytes = bytes // 1024
    bytes %= 1024
    Mbytes = Kbytes // 1024
    Kbytes %= 1024
    Gbytes = Mbytes // 1024
    Mbytes %= 1024
    print(f'''
bits:\t{bits}
bytes:\t{bytes}
Kbytes:\t{Kbytes}
Mbytes:\t{Mbytes}
Gbytes:\t{Gbytes}
        ''')

param_count = 1_200_000
params_per_weight = 2 # AdamW - 2 параметра на вес

weights = param_count * 4 # 4 - sizeof(float32)
gradients = weights
optimizer = weights * params_per_weight

model_weights = weights + gradients + optimizer
print(model_weights)
print_memory_size(model_weights)
