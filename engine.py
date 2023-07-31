import torch
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from data.dataset import label_to_text
import editdistance
import Levenshtein
import numpy as np


class Trainer():
    def __init__(self, lr_scheduler, patience=10, save_path='checkpoints/best_model.pth', best_val_loss=float('inf')):
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.save_path = save_path
        self.best_val_loss = best_val_loss
        self.counter = 0
        self.min_delta = 1e-3
        self.stop = False

    def __call__(self, current_valid_loss, model, epoch, optimizer):
        if self.best_val_loss - current_valid_loss > self.min_delta:
            print(f'Validation loss improved from {self.best_val_loss} to {current_valid_loss}!')
            self.best_val_loss = current_valid_loss
            self.counter = 0

            print('Saving best model ...')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, self.save_path)

        else:
            self.counter += 1
            print(
                f'Validation loss did not improve from {self.best_val_loss}! Counter {self.counter} of {self.patience}.')
            if self.counter < self.patience:
                self.lr_scheduler.step(current_valid_loss)

            else:
                self.stop = True


def train_model(model, device, dataset, dataloader, optimizer):
    model = model.to(device)
    model.train()
    train_loss = 0.0
    for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        optimizer.zero_grad()

        images = data[0].to(device)
        targets = data[1].to(device)
        target_lengths = data[2].to(device)

        _, loss = model(images, targets, target_lengths)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    return train_loss / len(dataloader)


def valid_model(model, device, dataset, dataloader):
    model = model.to(device)
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            images = data[0].to(device)
            targets = data[1].to(device)
            target_lengths = data[2].to(device)

            _, loss = model(images, targets, target_lengths)
            valid_loss += loss.item()

    return valid_loss / len(dataloader)


def inference(model, device, dataset, mode, decoder):
    model.eval()
    subset_indices = torch.randint(size=(100,), low=0, high=len(dataset))   # chọn random 100 label

    subset = Subset(dataset, indices=subset_indices)
    dataloader = DataLoader(subset, batch_size=1)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            images = data[0].to(device)
            labels = data[1].to(device)

            log_probs, _ = model(images)

            decoded_seqs = decoder(log_probs)

            for seq in decoded_seqs:
                all_preds.append(seq)


            for label in labels:
                all_labels.append(label_to_text(label))

    mean_norm_ed = 0.0
    total_wer = 0.0
    total_cer = 0.0
    accu = 0.0
    count = 0

    for i in range(len(all_labels)):
        print("Label: {0:70} Prediction: {1}".format(all_labels[i], all_preds[i]))
        mean_norm_ed += editdistance.eval(all_preds[i], all_labels[i])
        mean_norm_ed /= len(all_labels[i])

        wer = calculate_wer(all_labels[i], all_preds[i])
        cer = calculate_cer(all_labels[i], all_preds[i])
        total_wer += wer
        total_cer += cer
        count += 1

    wer_score = total_wer / count
    cer_score = total_cer / count

    print(f'Accuracy of the prediction: {accu}')
    print(f'Mean Normalized Edit Distance: {mean_norm_ed}')
    print(f'WER: {wer_score:.4f}')
    print(f'CER: {cer_score:.4f}')


def calculate_cer(reference, hypothesis):
    reference = reference.lower()
    hypothesis = hypothesis.lower()

    distance = Levenshtein.distance(reference, hypothesis)
    cer = distance / len(reference.split())

    return cer


def calculate_wer(reference, hypothesis):
    reference_words = reference.lower().split()
    hypothesis_words = hypothesis.lower().split()

    distance = Levenshtein.distance(reference_words, hypothesis_words)
    wer = distance / len(reference_words)

    return wer
