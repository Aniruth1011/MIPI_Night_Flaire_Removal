import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, input_folder, output_folder, transform=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.transform = transform
        self.image_list = os.listdir(input_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_folder, self.image_list[idx])
        output_path = os.path.join(self.output_folder, self.image_list[idx])

        with Image.open(input_path) as input_image, Image.open(output_path) as output_image:
            input_image = input_image.convert("RGB")
            output_image = output_image.convert("RGB")

            #resize_transform = transforms.Resize((64, 64))
            #input_image = resize_transform(input_image)
            #output_image = resize_transform(output_image)

            # Normalize pixel values to the range [0, 1]
            input_image = transforms.ToTensor()(input_image) / 255.0
            output_image = transforms.ToTensor()(output_image) / 255.0

            """if self.transform:
                input_image = self.transform(input_image)
                output_image = self.transform(output_image)"""

        return input_image, output_image
