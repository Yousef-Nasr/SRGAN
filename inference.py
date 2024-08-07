import numpy as np
import argparse 
import os 
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from PIL import Image
import time
from src import Generator


transform = transforms.Compose([
    transforms.ToTensor(),
])

def main(args):
    ## Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Switching to CPU')
        args.device = 'cpu'
    device = torch.device(args.device)
    generator =  Generator().to(device)
    generator.load_state_dict(torch.load(args.model_path, map_location=device))
    print('Model loaded successfully')
    print(f'Inference using {str(device).upper()}')
    images = get_images_batch(args.image_path, device)

    for image in images:
        start = time.time()
        with torch.no_grad():
            generated = generator(image[0])
        save_image(generated.detach(), args.save_path + image[1].split('.')[0] + '_generated.png')
        end = time.time()
        print(f'Generated {image[1]} done in {end-start} seconds')

def get_images_batch(image_path, device):
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
    parser.add_argument('--device', type=str, help='Device to use for inference', default='cuda')
    args = parser.parse_args()
    main(args)