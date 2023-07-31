import os
from decoder.decoder import GreedySearchDecoder, BeamSearchDecoder
from data.dataset import VietOCR, my_collate_fn, num_letters, label_dict
from network.model import VietOCRVGG16
from network.model import VietOCRResNet50, OCRMobile, OCRDenseNet
from engine import train_model, valid_model, inference, Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import torch
import argparse
import matplotlib.pyplot as plt
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import optuna

def train_model_tune(config):
    # Define device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define network
    checkpoint = torch.load('checkpoints/best_model.pth')
    model = VietOCRVGG16(num_letters=num_letters, finetune=config['ft'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Define dataset and dataloader
    img_list = os.listdir(config['img_path'])
    shuffle(img_list)

    train_img_list = img_list[:int(len(img_list) * 0.8)]
    valid_img_list = img_list[int(len(img_list) * 0.8):]

    train_dataset = VietOCR(
        image_path=config['img_path'],
        image_list=train_img_list,
        label_path=config['label_path'],
        img_w=2560,
        img_h=160,
        phase='train'
    )
    valid_dataset = VietOCR(
        image_path=config['img_path'],
        image_list=valid_img_list,
        label_path=config['label_path'],
        img_w=2560,
        img_h=160,
        phase='valid'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=my_collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=my_collate_fn
    )

    # Define optimizer and scheduler
    optimizer = Adam(params=model.parameters(), lr=config['lr'], weight_decay=0.005)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)

    # Define trainer
    trainer = Trainer(lr_scheduler)

    # Define decoder
    if config['mode'] == 'greedy':
        decoder = GreedySearchDecoder(labels=label_dict)
    elif config['mode'] == 'beam':
        decoder = BeamSearchDecoder(labels=label_dict)

    for epoch in range(config['epoch']):
        print(f"Epoch {epoch + 1} / {config['epoch']}")
        train_loss = train_model(model, device, train_dataset, train_dataloader, optimizer)
        print(f'Training loss: {train_loss}')
        val_loss = valid_model(model, device, valid_dataset, valid_dataloader)
        print(f'Validation loss: {val_loss}')
        inference(model, device, valid_dataset, config['mode'], decoder)
        trainer(val_loss, model, epoch, optimizer)
        if trainer.stop:
            break

        tune.report(train_loss=train_loss, val_loss=val_loss)


def objective(trial):
    config = {
        'epoch': trial.suggest_int('epoch', 1, args.max_epochs),
        'lr': trial.suggest_loguniform('lr', 1e-6, 1e-3),
        'batch_size': trial.suggest_categorical('batch_size', [3, 6, 9]),
        'img_path': args.img_path,
        'label_path': args.label_path,
        'ft': True,
        'mode': 'greedy'
    }

    # Call the training function with the config
    result = train_model_tune(config)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--img_path", default="/root/intern/write/Train", type=str)
    parser.add_argument("--label_path", default="/root/intern/write/labels_train.json", type=str)

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ray.init(configure_logging=False)

    scheduler = ASHAScheduler(
        metric='val_loss',
        mode='min',
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=['epoch', 'lr', 'batch_size'],
        metric_columns=['train_loss', 'val_loss', 'training_iteration']
    )

    analysis = tune.run(
        objective,
        resources_per_trial={'cpu': 1, 'gpu': args.gpus},
        config={},  # Leave it as an empty dictionary
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        search_alg=optuna.OptunaSearch(metric="val_loss", mode="min")
    )

