import argparse
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from src import Generator, Discriminator, FeatureExtractor, SRDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_data(args):
    # Load the dataset
    dataset = SRDataset(lr_dir=args.lr_dir , hr_dir=args.hr_dir, crop_size=args.crop_size, augment=args.augment)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader

def prepare_model(args):
    # Initialize the generator, discriminator, and feature extractor
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Load the saved state dictionaries if the paths are provided
    if args.generator_path != None:
        generator.load_state_dict(torch.load(args.generator_path))
    if args.discriminator_path != None:
        discriminator.load_state_dict(torch.load(args.discriminator_path))

    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(0.9, 0.999))

    adversarial_loss = nn.BCEWithLogitsLoss().to(device)
    content_loss = nn.MSELoss().to(device)

    return generator, discriminator, feature_extractor, optimizer_G, optimizer_D, adversarial_loss, content_loss

def train_generator(args, generator, discriminator, feature_extractor, optimizer_G, adversarial_loss, content_loss, images_batch, valid, fake):
    # Train the Generator
    lr_imgs, hr_imgs = images_batch
    optimizer_G.zero_grad()
    generated_images = generator(lr_imgs.to(device))
    pixel_loss = content_loss(generated_images, hr_imgs)
    content_Loss = content_loss(feature_extractor(generated_images), feature_extractor(hr_imgs)) * args.content_weight
    adversarial_Loss = adversarial_loss(discriminator(generated_images), valid) * args.adversarial_weight
    Loss_G = content_Loss + adversarial_Loss + pixel_loss
    Loss_G.backward()
    optimizer_G.step()

    return generated_images, Loss_G


def train_discriminator(discriminator, optimizer_D, adversarial_loss, images_batch, generated_images, valid, fake):
    # Train the Discriminator
    _, hr_imgs = images_batch
    optimizer_D.zero_grad()
    real_loss = adversarial_loss(discriminator(hr_imgs), valid)
    fake_loss = adversarial_loss(discriminator(generated_images.detach()), fake)
    Loss_D = (real_loss + fake_loss) * 0.5
    Loss_D.backward()
    optimizer_D.step()

    return Loss_D 

def train_model(args):
    # Prepare the data
    dataloader = prepare_data(args)

    # Prepare the model
    generator, discriminator, feature_extractor, optimizer_G, optimizer_D, adversarial_loss, content_loss = prepare_model(args)

    # Adversarial ground truths
    valid = torch.ones((args.batch_size, 1, args.crop_size//4, args.crop_size//4), requires_grad=False).to(device)
    fake = torch.zeros((args.batch_size, 1, args.crop_size//4, args.crop_size//4), requires_grad=False).to(device)

    epochs = args.epochs
    # Training loop
    for epoch in range(epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train the generator
            generated_images, Loss_G = train_generator(args, generator, discriminator, feature_extractor, optimizer_G, adversarial_loss, content_loss, (lr_imgs, hr_imgs), valid, fake)
            # Train the discriminator
            Loss_D = train_discriminator(discriminator, optimizer_D, adversarial_loss, (lr_imgs, hr_imgs), generated_images, valid, fake)
            print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, Loss_G: {Loss_G}, Loss_D: {Loss_D}", end='\r')
        print(f"Epoch {epoch+1}/{epochs}, Loss_G: {Loss_G}, Loss_D: {Loss_D}")

        # Save the model weights
        if not os.path.exists("model_weights/"):
            os.mkdir("model_weights")  
        torch.save(generator.state_dict(), f"{args.checkpoint_name}.pth")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SRGAN model')
    parser.add_argument('--lr_dir', type=str, help='Path to the low resolution images', default='data/lr')
    parser.add_argument('--hr_dir', type=str, help='Path to the high resolution images', default='data/hr')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--crop_size', type=int, help='Crop size', default=24)
    parser.add_argument('--augment', type=bool, help='Augment the data', default=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
    parser.add_argument('--lr_g', type=float, help='Learning rate for the generator', default=1e-4)
    parser.add_argument('--lr_d', type=float, help='Learning rate for the discriminator', default=1e-4)
    parser.add_argument('--content_weight', type=float, help='Weight for the content loss', default=0.006)
    parser.add_argument('--adversarial_weight', type=float, help='Weight for the adversarial loss', default=1e-3)
    parser.add_argument('--generator_path', type=str, help='Path to the generator model', default=None)
    parser.add_argument('--discriminator_path', type=str, help='Path to the discriminator model', default=None)
    parser.add_argument('--checkpoint_name', type=str, help='Name of the checkpoint', default='model_weights/SRGAN_train')
    args = parser.parse_args()
    train_model(args)