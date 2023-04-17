# Training - To Train the entire CNN model and produce a model .pt file
import torch
import pandas as pd
import torchvision.transforms as transforms
from torchvision import datasets

from cnn.cnn import CNN
from cnn.train_cnn import train_net

# Setting manual seed to 0 for reproducibility
torch.manual_seed(0)

# Define directory to load image patches from and transformation to apply to them
PATCH_DIR = "patches_nc_with_rot/" 
transform = transforms.Compose([transforms.ToTensor()])

# Loading the image patches into a PyTorch dataset
data = datasets.ImageFolder(root=PATCH_DIR, transform=transform)  

# Determine device to use (CPU or CUDA if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate a CNN and move it to the selected device
if str(device) == "cuda:0":
    print("Cuda Enabled")
    cnn = CNN().cuda()
else:
    print("no cuda")
    cnn = CNN()

# Training the CNN using the dataset
epoch_loss, epoch_accuracy = train_net(cnn, data, n_epochs=250, learning_rate=0.0001, batch_size=128)

# Save the loss and accuracy for each epoch to a CSV file
pd.DataFrame(epoch_loss).to_csv('SRM_loss.csv')
pd.DataFrame(epoch_accuracy).to_csv('SRM_accuracy.csv')

# Save the trained CNN's state dictionary to a file
torch.save(cnn.state_dict(), 'Cnn_nc16.pt')

