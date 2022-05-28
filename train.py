import torch
from tqdm import tqdm
import torch.optim as optim
from model.loss import mse_loss
from model.unet_basic import Model
from utils import get_loaders, save_checkpoint, load_checkpoint, check_accuracy, save_audio_predictions


#Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_EPOCHS = 100 
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False

FILENAME = 'denoise_dataset.pkl'
CHECKPOINT = 'my_checkpoint.pth.tar'
SAVE_AUDIO = 'model_predictions'

def train_fn(loader, model, optimizer, loss_fn):
	loop = tqdm(loader)

	for idx, (data, targets) in enumerate(loop):
		data = data.to(device=DEVICE, dtype=torch.float)
		targets = targets.to(device=DEVICE, dtype=torch.float)

		predictions = model(data)
		loss = loss_fn(predictions, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		loop.set_postfix(loss=loss.item())


def main():
	model = Model().to(DEVICE)

	loss_fn = mse_loss()
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	train_loader, val_loader = get_loaders(filename=FILENAME, 
						batch_size=BATCH_SIZE, 
						num_workers=NUM_WORKERS, 
						pin_memory=PIN_MEMORY,
						)

	if LOAD_MODEL:
		load_checkpoint(torch.load(CHECKPOINT), model)

	for epoch in range(NUM_EPOCHS):
		train_fn(train_loader, model, optimizer, loss_fn)

		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}
		save_checkpoint(checkpoint, CHECKPOINT)

		check_accuracy(val_loader, model, device=DEVICE)
		save_audio_predictions(val_loader, model, SAVE_AUDIO, device=DEVICE)
	
if __name__ == "__main__":
	main()
