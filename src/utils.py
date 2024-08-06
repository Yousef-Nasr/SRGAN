import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.ToTensor(),
])   

## Define the ScaleHR class that scales the image to the range [-1, 1] and vice versa
class ScaleHR(object):
    def __call__(self, img, reverse=False):
        if reverse:
            return img / 2.0 + 0.5
        return img * 2.0 - 1.0
    

## Define the InterpolateAndConcatenate class that resizes the LR image to the target size and concatenates the LR, generated, and HR images
class InterpolateAndConcatenate(object):
    def __init__(self, target_size):
        self.target_size = target_size  # Desired output size for LR images
    
    def __call__(self, lr_img, hr_img, gen_img):
        # Resize LR image to target size
        lr_resized = F.interpolate(lr_img.unsqueeze(0), size=self.target_size, mode='bicubic', align_corners=False).squeeze(0)
        
        # Concatenate resized LR, generated, and HR images
        concatenated_img = torch.cat((lr_resized, gen_img, hr_img), dim=-2)
        
        return concatenated_img
    

def plot_examples(gen, dataloader=None, n=1, lr=None, hr=None, save=False, target_size=(96,96), name="test.png"):
    """
    Plot n examples of low resolution, high resolution, and generated images.

    Parameters:
    - gen: Generator model.
    - dataloader: DataLoader object if randomness is required.
    - n: Number of examples to plot.
    - lr: Low resolution images.
    - hr: High resolution images.
    - save: Boolean indicating whether to save the image.
    - target_size: Tuple indicating the target size for LR images.
    - name: Name of the saved image.

    Returns:
    - None
    
    """
    if lr is None or hr is None:
      batch = iter(dataloader)
      lr, hr = next(batch)
    else:
      lr, hr = lr, hr

    for i in range(n):
      fig, axs = plt.subplots(1, 3, figsize=(8, 4))
      axs[0].set_axis_off()
      axs[0].imshow(lr[i].permute(1, 2, 0))
      axs[0].set_title("low res")

      with torch.no_grad():
          upscaled_img = gen(lr[i].unsqueeze(0).to(device))

      axs[1].set_axis_off()
      axs[1].imshow(upscaled_img.cpu().squeeze(0).permute(1, 2, 0))
      axs[1].set_title("predicted")

      scale = ScaleHR()
    #   hr_img = hr[i]
    #   hr_img = scale(hr_img.unsqueeze(0), reverse=True).squeeze(0)
      axs[2].set_axis_off()
      axs[2].imshow(hr[i].permute(1, 2, 0))
      axs[2].set_title("high res")

      if save:
        lr_img_save = lr[:n].cpu().detach()
        hr_img_save = hr[:n].cpu().detach()
        generated_img = gen(lr_img_save.to(device)).cpu().detach()
        # Apply the transformation to each image in the batch
        interpolate_transform = InterpolateAndConcatenate(target_size=target_size)
        concatenated_images = [interpolate_transform(lr, gen, hr) for lr, hr, gen in zip(lr_img_save, generated_img, hr_img_save)]
        concatenated_images = torch.stack(concatenated_images)
        save_image(concatenated_images.data, name)
      plt.tight_layout()
      plt.show()


## Define the CropAndCombine class that crops and combines the image pieces
class CropAndCombine:
    def __init__(self, image, crop_size):
        self.image = image
        self.crop_size = crop_size
        self.padded_image = self.pad_image(self.image, self.crop_size)
    def pad_image(self, image, crop_size):
        """
        Pad the image so that its dimensions are divisible by the crop size.
        
        Parameters:
        - image: Input image as a NumPy array.
        - crop_size: Tuple indicating the size of the crops (height, width).
        
        Returns:
        - Padded image as a NumPy array.
        """
        h, w = image.shape[:2]
        ch, cw = crop_size
        
        pad_h = (ch - h % ch) % ch
        pad_w = (cw - w % cw) % cw
        
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
        return padded_image
    

    def crop_image(self, image, crop_size):
        """
        Crop the image into small pieces.
        
        Parameters:
        - image: Input image as a NumPy array.
        - crop_size: Tuple indicating the size of the crops (height, width).
        
        Returns:
        - List of cropped pieces.
        """
        pieces = []
        h, w = image.shape[:2]
        ch, cw = crop_size
        
        for i in range(0, h, ch):
            for j in range(0, w, cw):
                piece = image[i:i+ch, j:j+cw]
                pieces.append(piece)
        
        return pieces

    def get_image_pieces(self):
        """
        Get the pieces of the image.
        
        Returns:
        - List of image pieces.
        """
        pieces = self.crop_image(self.padded_image, self.crop_size)
        return pieces
    
    def combine_pieces(self, pieces, original_size, crop_size, upsample_factor=4, delete_padded=False):
        """
        Combine the pieces back into a single image.
        
        Parameters:
        - pieces: List of processed pieces.
        - original_size: Tuple indicating the size of the original image (height, width).
        - crop_size: Tuple indicating the size of the crops (height, width).
        - upsample_factor: Factor by which each piece was upsampled.
        
        Returns:
        - Combined image as a NumPy array.
        """
        h, w = original_size
        ch, cw = crop_size
        combined_image = np.zeros((h * upsample_factor, w * upsample_factor, 3), dtype=np.uint8)
        idx = 0
        for i in range(0, h * upsample_factor, ch * upsample_factor):
            for j in range(0, w * upsample_factor, cw * upsample_factor):
                combined_image[i:i+ch*upsample_factor, j:j+cw*upsample_factor] = pieces[idx]
                idx += 1
        if delete_padded:
            h, w = self.image.shape[:2]
            combined_image = combined_image[:h * upsample_factor, :w * upsample_factor]
        
        return combined_image
    
 