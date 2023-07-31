import os
import argparse
import torch
from torch.utils.data import DataLoader
from data.dataset import VietOCR, label_dict, num_letters, my_collate_fn
from network.model import OCRMobile, OCRDenseNet, VietOCRVGG16
from decoder.decoder import GreedySearchDecoder
from engine import inference

def load_model(model_path, model_type, num_letters):
    if model_type == 'vgg16':
        model = VietOCRVGG16(num_letters=num_letters).to(device)
    elif model_type == 'dense':
        model = OCRDenseNet(num_letters=num_letters).to(device)
    elif model_type == 'mobile':
        model = OCRMobile(num_letters=num_letters).to(device)
    else:
        raise ValueError(f"Invalid model type '{model_type}'. Please choose 'vgg16', 'dense', or 'mobile'.")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=['vgg16', 'dense', 'mobile'], default='vgg16',
                        help="Type of OCR model: 'vgg16', 'dense', or 'mobile'. Default is 'vgg16'.")
    parser.add_argument("--model_path", type=str, default="/home/htp/code/write/checkpoints/best_model.pth",
                        help="Path to the pre-trained OCR model.")
    parser.add_argument("--img_path", default="Test", type=str, help="/home/htp/code/write/test.py.")
    parser.add_argument("--label_path", default="labels_test.json", type=str, help="/home/htp/code/write/labels_test.json.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to predict.")
    args = parser.parse_args()

    

    # Load the OCR model
    def load_model(model_path, num_letters):
        model = VietOCRVGG16(num_letters=num_letters)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    # Define device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_letters = num_letters  # Replace this with the actual number of letters in your dataset
    model = load_model(args.model_path, num_letters=num_letters).to(device)
    
    # Define dataset and dataloader
    img_list = os.listdir(args.img_path)
    dataset = VietOCR(args.img_path, image_list=img_list, label_path=args.label_path, img_w=2560, img_h=160, phase='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=my_collate_fn)

    # Define decoder
    decoder = GreedySearchDecoder(labels=label_dict)

    # Perform inference
    predictions = inference(model, device, dataset, 'greedy', decoder)


