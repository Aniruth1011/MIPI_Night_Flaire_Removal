import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import UNet , unet 
from tqdm import tqdm 
from dataloader_ import ImageDataset
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToPILImage


transform = transforms.Compose([
    transforms.ToTensor(),
])

to_pil = ToPILImage()


def process_images(input_folder, output_folder, model_path):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    model = unet(3, 3)
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    model.eval()

    # Assuming your CustomDataset takes a folder path as an argument
    dataset = ImageDataset(input_folder , input_folder)

    for idx in tqdm(range(len(dataset))):

        original_image, _ = dataset[idx]

        #input_tensor = transform(original_image)
        input_tensor =  original_image
        input_tensor = input_tensor.unsqueeze(0) 

        with torch.no_grad():
            model_output = model(input_tensor)


        model_output = to_pil(model_output.squeeze(0))

        #model_output = transforms.ToPILImage()(model_output/255.0)
        # Resize the output image to the shape of the original image
        #resized_output = transforms.ToPILImage()(model_output)
        #resized_output = resized_output.resize(original_image.size, Image.ANTIALIAS)

        # Save the resized output image as a PNG file in the output folder
        output_image_path = os.path.join(output_folder, f"output_{idx}.png")
        model_output.save(output_image_path, format='PNG')
        print(f"Processed and saved output image at: {output_image_path}")

# Example usage
if __name__ == "__main__":

    input_folder = "validation"
    output_folder = "output"
    model_path = "unet_model_20240217130442_epochs_15.pth"  

    process_images(input_folder, output_folder, model_path)
