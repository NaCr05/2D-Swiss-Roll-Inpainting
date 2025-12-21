import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_optimal_num_workers():
    # Detect CPU cores and use N-1 (min 1)
    cores = os.cpu_count() or 1
    return max(1, cores - 1)

class OxfordPetLoader:
    def __init__(self, root='./data', batch_size=8, image_size=256, download=True, cat_only=True):
        self.root = root
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        full_dataset = datasets.OxfordIIITPet(
            root=root, 
            split='trainval', 
            target_types='category', 
            download=download,
            transform=self.transform
        )
        
        if cat_only:
            # 1. Define cat breeds
            cat_breeds = [
                "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", 
                "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", 
                "Siamese", "Sphynx"
            ]
            
            # 2. Get corresponding label IDs
            # full_dataset.class_to_idx 是一個 dict {'Abyssinian': 0, ...}
            cat_ids = set()
            for cat in cat_breeds:
                if cat in full_dataset.class_to_idx:
                    cat_ids.add(full_dataset.class_to_idx[cat])
            
            print(f"Cat Label IDs: {cat_ids}")

            # 3. Filter dataset for cat images only
            # OxfordIIITPet does not provide direct access to labels, so we access the private attribute
            # full_dataset._labels is a list of labels corresponding to each image
            all_labels = full_dataset._labels 
            
            cat_indices = [i for i, label in enumerate(all_labels) if label in cat_ids]
            
            self.dataset = torch.utils.data.Subset(full_dataset, cat_indices)
            print(f"Filtered Oxford-Pet for Cats. Total Images: {len(self.dataset)}")
        else:
            self.dataset = full_dataset


    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle for training
            num_workers=get_optimal_num_workers(), # Get CPU cores - 1
            pin_memory=True # Pin memory for faster transfers
        )
