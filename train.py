import os
from decoder.decoder import GreedySearchDecoder, BeamSearchDecoder
from data.dataset import VietOCR, my_collate_fn, num_letters, label_dict
from network.model import VietOCRVGG16, OCRDenseNet, OCRMobile
from engine import train_model, valid_model, inference, Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle
import torch
import argparse
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_path", default="Train", type=str)
    parser.add_argument("--label_path", default="labels_train.json", type=str)
    parser.add_argument("--ft", type=bool, default=True)
    parser.add_argument("--mode", type=str, default='greedy')
    parser.add_argument("--model", type=str, default='densnet')
    parser.add_argument("--save_path", type=str, default='/home/htp/code/write/checkpoints/best_densnet_model.pth')

    args = parser.parse_args()

    # Define device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define network
    if args.model == 'mobilenet':
        model = OCRMobile(num_letters=num_letters, finetune=args.ft).to(device)
        args.save_path = 'checkpoints/best_mobilenet_model.pth'
    elif args.model == 'densnet':
        model = OCRDenseNet(num_letters=num_letters, finetune=args.ft).to(device)
        args.save_path = 'checkpoints/best_densnet_model.pth'
    elif args.model == 'vgg16':
        model = VietOCRVGG16(num_letters=num_letters, finetune=args.ft).to(device)
        args.save_path = 'checkpoints/best_vgg16_model.pth'

    # Define dataset and dataloader
    img_list = os.listdir(args.img_path)
    shuffle(img_list)

    train_img_list = img_list[:int(len(img_list) * 0.8)]
    valid_img_list = img_list[int(len(img_list) * 0.8):]

    train_dataset = VietOCR(
        image_path=args.img_path,
        image_list=train_img_list,
        label_path=args.label_path,
        img_w=5120,
        img_h=160,
        phase='train'
    )
    valid_dataset = VietOCR(
        image_path=args.img_path,
        image_list=valid_img_list,
        label_path=args.label_path,
        img_w=5120,
        img_h=160,
        phase='valid'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=my_collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=my_collate_fn
    )

    # Define optimizer and scheduler
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=0.005)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Define trainer
    trainer = Trainer(lr_scheduler, save_path=args.save_path)

    # Define decoder
    if args.mode == 'greedy':
        decoder = GreedySearchDecoder(labels=label_dict)
    elif args.mode == 'beam':
        decoder = BeamSearchDecoder(labels=label_dict)

    # Define lists for storing losses
    train_losses = []
    valid_losses = []

    for epoch in range(args.epoch):
        print(f"Epoch {epoch + 1} / {args.epoch}")
        train_loss = train_model(model, device, train_dataset, train_dataloader, optimizer)
        print(f'Training loss: {train_loss}')
        val_loss = valid_model(model, device, valid_dataset, valid_dataloader)
        print(f'Validation loss: {val_loss}')
        inference(model, device, valid_dataset, args.mode, decoder)
        trainer(val_loss, model, epoch, optimizer)
        if trainer.stop:
            break

        # Append losses to the lists
        train_losses.append(train_loss)
        valid_losses.append(val_loss)

    # Plot benchmark
    fig, ax = plt.subplots()
    ax.plot(range(1, args.epoch + 1), train_losses, label='Training Loss')
    ax.plot(range(1, args.epoch + 1), valid_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.legend()
    plt.savefig('benchmark.png')  # Lưu biểu đồ vào file 'benchmark.png'
    plt.show()
