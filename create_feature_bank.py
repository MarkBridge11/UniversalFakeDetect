import torch
import clip
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
from dataset_paths import DATASET_PATHS
import numpy as np
import random
from tqdm import tqdm

##################################################

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

###################################################


class BankDataset(Dataset):
    def __init__(self,image_dir):
        self.image_dir=image_dir

        self.image_paths = []
        for fname in os.listdir(image_dir):            
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):  
                self.image_paths.append(os.path.join(image_dir, fname))    

        stat_from = "clip"
        self.transform = transforms.Compose([
                #transforms.Resize(224,), in CLIP model authors do also this, but UnivFD do not. I prefered to remove it because resizing is an harmful operation for AIGD task
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
        

def extract_features(image_batch, model, device):
    image_batch = image_batch.to(device)
    with torch.no_grad():
        features = model.encode_image(image_batch)  # shape [batch_size, 768]
        features = features / features.norm(dim=-1, keepdim=True) 
        # This ensures cosine distance is equivalent to Euclidean in normalized space.
        # Cosine similarity vectors must be normalized to not let the magnitude influence the results.
    return features.cpu() # image_features are tensors in the GPU, saving them in the features array will cause problems. Moreover save() cannot be used with GPU tensors.

if __name__ == '__main__':
    set_seed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "ViT-L/14"
    print(f"Loading {model_name} on {device}...")
    model, preprocess = clip.load(model_name,device)
    model.eval()
    print("Model loaded...")

    fake_dataset = BankDataset(image_dir="dataset/1_fake")
    real_dataset = BankDataset(image_dir="dataset/0_real")

    fake_loader = DataLoader(fake_dataset,batch_size=8,num_workers=6)
    real_loader = DataLoader(real_dataset,batch_size=8,num_workers=6)

    fake_features = []
    real_features = []

    print("Datasets and Dataloaders loaded.")

    for i, fake_batch in enumerate(tqdm(fake_loader, desc="Processing fake batches")):
        features = extract_features(fake_batch, model, device) # [batch_size,768]
        fake_features.append(features) #len(fake_features) increases by 1, because it is a list of tensors with dim [batch_size,768]

    for i, real_batch in enumerate(tqdm(real_loader, desc="Processing real batches")):
        features = extract_features(real_batch, model, device)
        real_features.append(features)

    fake_features_tensor = torch.cat(fake_features, dim=0)  # shape: [N_fake_images, 768], here we take all the array in the tensors and put in a single vector
    real_features_tensor = torch.cat(real_features, dim=0)  # shape: [N_real_images, 768]

    torch.save({"real": real_features_tensor, "fake": fake_features_tensor}, "feature_bank.pt") # this saves as a dictionary



