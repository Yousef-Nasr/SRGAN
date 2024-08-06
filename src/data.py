import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from albumentations.augmentations.crops import functional as albF
import albumentations as A
from PIL import Image
from glob import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import random
 
transform = transforms.Compose([
    transforms.ToTensor(),
])

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=24, transform=transform, augment=False):
        """
        Parameters:
        - lr_dir: Path to the directory containing the low resolution images.
        - hr_dir: Path to the directory containing the high resolution images.
        - crop_size: Size of the crops for the lr images.
        - transform: Transformations to apply to the images.
        """
        self.lr_images = sorted(glob(lr_dir + "/*"))
        self.hr_images = sorted(glob(hr_dir + "/*"))
        self.transform = transform
        self.crop_size = crop_size
        self.augment_ = augment

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_images[idx])
        hr_img = Image.open(self.hr_images[idx])

        lr_img, hr_img = self.paired_crop([lr_img, hr_img], crops_sizes=[self.crop_size, self.crop_size*4])

        # imgs = {}
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        # imgs = {'lr': lr_img, 'hr': hr_img}
        return lr_img, hr_img

    def paired_crop(self, images, crops_sizes):
        '''
        images: list of images [lr, hr]
        crops_sizes: list of crop sizes [lr_size, hr_size]

        return: list of cropped images [lr_patches, hr_patches]
        '''
        xmin_hr = random.randint(0, images[1].size[0] - crops_sizes[1])
        ymin_hr = random.randint(0, images[1].size[1] - crops_sizes[1])

        scale_factor = crops_sizes[1] // crops_sizes[0]
        xmin_lr = xmin_hr // scale_factor
        ymin_lr = ymin_hr // scale_factor
        lr, hr = images

        lr_patch = A.Crop(xmin_lr, ymin_lr, x_max=xmin_lr+crops_sizes[0], y_max=ymin_lr+crops_sizes[0])(image=np.array(lr))['image']
        hr_patch = A.Crop(xmin_hr, ymin_hr, x_max=xmin_hr+crops_sizes[1], y_max=ymin_hr+crops_sizes[1])(image=np.array(hr))['image']
        if self.augment_:
            if random.random() > 0.1:
                angel = random.choice([90, -90])
                lr_patch, hr_patch = self.augment([lr_patch, hr_patch], angel)

        return lr_patch, hr_patch
    
    def augment(self, images, angle):
        '''
        images: list of images [lr, hr]
        angle: rotation angle

        return: list of rotated images [lr_rotated, hr_rotated]
        '''
        lr, hr = images
        choice = random.randint(0, 2)
        ## Rotate the images
        if choice == 0:
            lr_rotated = A.Rotate(limit=angle, p=1)(image=np.array(lr))['image']
            hr_rotated = A.Rotate(limit=angle, p=1)(image=np.array(hr))['image']

        ## HorizontalFlip the images
        elif choice == 1:
            lr_rotated = A.HorizontalFlip(p=1)(image=np.array(lr))['image']
            hr_rotated = A.HorizontalFlip(p=1)(image=np.array(hr))['image']

        ## VerticalFlip the images
        else:
            lr_rotated = A.VerticalFlip(p=1)(image=np.array(lr))['image']
            hr_rotated = A.VerticalFlip(p=1)(image=np.array(hr))['image']
    
        return lr_rotated, hr_rotated

