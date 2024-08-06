import numpy as np
import argparse 
import os 
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
from src import Generator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])

def main(args):
    generator =  Generator().to(device)
    generator.load_state_dict(torch.load(args.model_path))
    print('Model loaded')
    print(f'Inference using {device}')
    images = get_images_batch(args.image_path)

    for image in images:
        with torch.no_grad():
            generated = generator(image[0])
        save_image(generated.detach(), args.save_path + image[1].split('.')[0] + '_generated.png')

def get_images_batch(image_path):
    image_paths = os.listdir(image_path)
    images = []
    for path in image_paths:
        img = Image.open(os.path.join(image_path, path))
        img = transform(img).unsqueeze(0).to(device)
        images.append([img, path])
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with SRGAN model')
    parser.add_argument('--model_path', type=str, help='Path to the generator model', default='SRGAN.pth')
    parser.add_argument('--image_path', type=str, help='Path to the images', default='inputs/')
    parser.add_argument('--save_path', type=str, help='Path to save the generated images', default='outputs/')
    args = parser.parse_args()
    main(args)