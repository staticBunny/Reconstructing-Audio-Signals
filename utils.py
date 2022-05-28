import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import pandas as pd
import torch.nn.functional as F
import torchaudio
from torchmetrics import SignalNoiseRatio as SNR
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE

class AudioDataset(Dataset):
	def __init__(self, filename):
		self.data = pd.read_pickle(filename)
		self.clean_signals = self.data[0]
		self.noisy_signals = self.data[1]

	def __len__(self):
		return len(self.clean_signals)

	def __getitem__(self, index):
		noisy_signal = self.noisy_signals[index]
		clean_signal = self.clean_signals[index]

		noisy_signal = noisy_signal.repeat(2)

		noisy_signal = np.resize(noisy_signal, (1, noisy_signal.shape[0]))
		clean_signal = np.resize(clean_signal, (1, clean_signal.shape[0]))

		return noisy_signal, clean_signal


def get_loaders(filename,
		batch_size,
		num_workers=4,
		pin_memory=True,
		):
		
	dataset = AudioDataset(filename)
	train_set, val_set = random_split(dataset, [9000, 600])

	train_loader = DataLoader(train_set,
				batch_size=batch_size,
				num_workers=num_workers,
				pin_memory=pin_memory,
				)
	
	val_loader = DataLoader(val_set,
				batch_size=batch_size,
				num_workers=num_workers,
				pin_memory=pin_memory,
				)

	return train_loader, val_loader

def save_checkpoint(state, filename):
	print("=> Saving chekpoint")
	torch.save(state, filename)

def load_checkpoint(checkpoint, model):
	print("=> Loading checkpoint")
	model.load_state_dict(checkpoint["state_dict"])

def check_accuracy(loader, model, device='CUDA'):
	model.eval()
	
	snr = SNR().to(device)
	mse = MSE().to(device)
	mae = MAE().to(device)

	snr_score = 0
	mse_score = 0
	mae_score = 0
	with torch.no_grad():
		for x, y in loader:
			x = x.to(device, dtype=torch.float)
			y = y.to(device, dtype=torch.float)
			preds = model(x)

			snr_score += snr(preds, y)
			mse_score += mse(preds, y)
			mae_score += mae(preds, y)

	print('Signal to Noise Ratio: {}'.format(snr_score/len(loader)))
	print('Mean Squared Error: {}'.format(mse_score/len(loader)))
	print('Mean Absolute Error: {}'.format(mae_score/len(loader)))
	model.train()

def save_audio_predictions(loader, model, folder, device='cuda'):
	model.eval()
	for idx, (x, y) in enumerate(loader):
		x = x.to(device=device, dtype=torch.float)
		with torch.no_grad():
			preds = model(x)

		preds = preds.to(device='cpu')
		y = y.to(torch.float)
		x = x.to(device='cpu')
		torchaudio.save(f'{folder}/pred_{idx}.wav', preds[0], preds.shape[2])
		torchaudio.save(f'{folder}/clean_{idx}.wav', y[0], y.shape[2])
		torchaudio.save(f'{folder}/noisy_{idx}.wav', x[0], x.shape[2])

	model.train()

def test():
	train, val = get_loaders('denoise_dataset.pkl', 64, 0, False)
	noisy, og = next(iter(train))
	print(noisy.shape, og.shape)

if __name__ == '__main__':
	test()
