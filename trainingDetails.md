# Argument List for SRGAN Training Script

The following arguments are available for configuring the SRGAN training script:

## Arguments

- `--lr_dir`
  - **Type**: `str`
  - **Default**: `data/lr`
  - **Description**: Path to the directory containing low-resolution images.

- `--hr_dir`
  - **Type**: `str`
  - **Default**: `data/hr`
  - **Description**: Path to the directory containing high-resolution images.

- `--batch_size`
  - **Type**: `int`
  - **Default**: `16`
  - **Description**: Number of images per batch during training.

- `--crop_size`
  - **Type**: `int`
  - **Default**: `24`
  - **Description**: Size of the crop for data augmentation.

- `--augment`
  - **Type**: `bool`
  - **Default**: `False`
  - **Description**: Whether to apply data augmentation.

- `--epochs`
  - **Type**: `int`
  - **Default**: `50`
  - **Description**: Number of training epochs.

- `--lr_g`
  - **Type**: `float`
  - **Default**: `0.0001`
  - **Description**: Learning rate for the generator.

- `--lr_d`
  - **Type**: `float`
  - **Default**: `0.0001`
  - **Description**: Learning rate for the discriminator.

- `--content_weight`
  - **Type**: `float`
  - **Default**: `0.006`
  - **Description**: Weight for the content loss in the loss function.

- `--adversarial_weight`
  - **Type**: `float`
  - **Default**: `0.001`
  - **Description**: Weight for the adversarial loss in the loss function.

- `--generator_path`
  - **Type**: `str`
  - **Default**: `None`
  - **Description**: Path to the pre-trained generator model. If not specified, the generator will be initialized randomly.

- `--discriminator_path`
  - **Type**: `str`
  - **Default**: `None`
  - **Description**: Path to the pre-trained discriminator model. If not specified, the discriminator will be initialized randomly.

- `--checkpoint_name`
  - **Type**: `str`
  - **Default**: `model_weights/SRGAN_train`
  - **Description**: Name of the file to save the model checkpoints.

## Usage Example

```bash
python train.py --lr_dir data/lr --hr_dir data/hr --batch_size 16 --crop_size 24 --augment True --epochs 50 --lr_g 0.0001 --lr_d 0.0001 --content_weight 0.006 --adversarial_weight 0.001 --generator_path models/generator.pth --discriminator_path models/discriminator.pth --checkpoint_name model_weights/SRGAN_train
