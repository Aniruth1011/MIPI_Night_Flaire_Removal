import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloader_ import ImageDataset
from model import UNet , UNetTransformer , UNetwithprompts
from tqdm import tqdm 
from datetime import datetime
from loss import DiceLoss , SoftDiceLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import unet
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 1
learning_rate = 3e-5
num_epochs = 20

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset and dataloader
input_folder = r"train_input_2k/train_input_2k"
output_folder =  r"train_gt_2k/train_gt_2k"

dataset = ImageDataset(input_folder, output_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create U-Net model
model = UNet(in_channels=3, out_channels=3).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate , weight_decay = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

pretrained_model_path = "unet_model_20240214050731_epochs_50.pth"

#model.load_state_dict(torch.load(pretrained_model_path))
#print(f"Pre-trained weights loaded from {pretrained_model_path}")


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

model.apply(init_weights)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        #print(inputs.shape)
        #print(outputs.shape)

        if (outputs.shape!=inputs.shape):
            continue

        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    scheduler.step(average_loss)


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f"unet_model_{timestamp}_epochs_{num_epochs}.pth"
torch.save(model.state_dict(), model_name)
