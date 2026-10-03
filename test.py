# %%
from data import HTR_Dataset, tfm_train
from matplotlib import pyplot as plt

dataset = HTR_Dataset('fonts_data', tfm_train, languages=['en'])
plt.imshow(dataset[123][0].squeeze(0))
plt.show()
