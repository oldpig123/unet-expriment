import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class VerSeDataset(Dataset):
    """
    VerSe '19 / '20 Spinal CT Dataset.
    Loads pre-extracted 2D sagittal CT PNG slices from preprocessed directories,
    normalizes intensity, and provides binary labels:
      - 0: Background
      - 1: Vertebrae
    """
    def __init__(self, data_dir, target_size=(512, 512), file_list=None):
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        self.target_size = target_size
        
        if file_list is not None:
            self.filenames = file_list
        else:
            if os.path.exists(self.images_dir):
                self.filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
            else:
                self.filenames = []
                
        print(f"[INFO] Initializing VerSeDataset with {len(self.filenames)} files.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        lbl_path = os.path.join(self.labels_dir, fname)
        
        img_pil = Image.open(img_path).convert("L")
        lbl_pil = Image.open(lbl_path)
        
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)
        lbl_resized = lbl_pil.resize(self.target_size, Image.NEAREST)
        
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        lbl_np = np.array(lbl_resized, dtype=np.uint8)
        
        # Already binary 0/1 from preprocessing
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # (1, 512, 512)
        seg_tensor = torch.tensor(lbl_np, dtype=torch.long)  # (512, 512)
        
        return img_tensor, seg_tensor


class LumbarMriDataset(Dataset):
    """
    Mendeley Lumbar Spine MRI Dataset.
    Loads pre-extracted/resized 2D sagittal MRI PNG slices, normalizes intensity,
    and maps multiclass labels:
      - 100 -> 1 (Vertebrae)
      - 50  -> 2 (Intervertebral Discs)
      - Other values (e.g. 250 background) -> 0 (Background)
    """
    def __init__(self, data_dir, target_size=(512, 512), file_list=None):
        self.images_dir = os.path.join(data_dir, "lumbar_mri", "images")
        self.labels_dir = os.path.join(data_dir, "lumbar_mri", "labels")
        self.target_size = target_size
        
        # Get matching image and label file names
        if file_list is not None:
            self.filenames = file_list
        else:
            if os.path.exists(self.images_dir):
                self.filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])
            else:
                self.filenames = []
                
        print(f"[INFO] Initializing LumbarMriDataset with {len(self.filenames)} files.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.images_dir, fname)
        lbl_path = os.path.join(self.labels_dir, fname)
        
        # Load images
        # The images are stored in RGB, convert to grayscale (L)
        img_pil = Image.open(img_path).convert("L")
        lbl_pil = Image.open(lbl_path)
        
        # Resize
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)
        lbl_resized = lbl_pil.resize(self.target_size, Image.NEAREST)
        
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        lbl_np = np.array(lbl_resized, dtype=np.uint8)
        
        # Map label pixels:
        # Value 100 is Vertebrae -> 1
        # Value 50 is Discs -> 2
        # Others (250 background, 150/200 other tissues) -> 0
        mapped_lbl = np.zeros_like(lbl_np)
        mapped_lbl[lbl_np == 100] = 1
        mapped_lbl[lbl_np == 50] = 2
        
        # Convert to PyTorch tensors
        img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # (1, 512, 512)
        seg_tensor = torch.tensor(mapped_lbl, dtype=torch.long)  # (512, 512)
        
        return img_tensor, seg_tensor


def get_dataloaders(dataset_name, data_dir, batch_size=2, train_val_split=0.8):
    """
    Helper function to instantiate the datasets and return train and validation Dataloaders.
    """
    dataset_name_lower = dataset_name.lower()
    
    if dataset_name_lower in ["verse19", "verse20", "verse"]:
        # Map "verse" to "verse19" for legacy support
        folder_name = "verse19" if dataset_name_lower in ["verse19", "verse"] else "verse20"
        dataset_path = os.path.join(data_dir, folder_name)
        images_dir = os.path.join(dataset_path, "images")
        
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"VerSe images directory not found: {images_dir}")
            
        all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        if len(all_files) == 0:
            raise FileNotFoundError(f"No VerSe images found in {images_dir}")
            
        # Get unique patient IDs
        # Filename format: sub-verse004_slice085.png or similar
        patient_ids = sorted(list(set([f.split("_")[0] for f in all_files])))
        num_patients = len(patient_ids)
        
        np.random.seed(42)
        indices = np.random.permutation(num_patients)
        split_idx = int(num_patients * train_val_split)
        
        train_pids = set([patient_ids[i] for i in indices[:split_idx]])
        val_pids = set([patient_ids[i] for i in indices[split_idx:]])
        
        train_files = [f for f in all_files if f.split("_")[0] in train_pids]
        val_files = [f for f in all_files if f.split("_")[0] in val_pids]
        
        train_dataset = VerSeDataset(dataset_path, file_list=train_files)
        val_dataset = VerSeDataset(dataset_path, file_list=val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
        
    elif dataset_name_lower == "lumbar_mri":
        # Mendeley dataset: split file list randomly
        images_dir = os.path.join(data_dir, "lumbar_mri", "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Mendeley MRI images directory not found: {images_dir}")
            
        all_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".png")])
        num_files = len(all_files)
        
        if num_files == 0:
            raise FileNotFoundError(f"No MRI images found in {images_dir}")
            
        np.random.seed(42)
        indices = np.random.permutation(num_files)
        split_idx = int(num_files * train_val_split)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_files = [all_files[i] for i in train_indices]
        val_files = [all_files[i] for i in val_indices]
        
        train_dataset = LumbarMriDataset(data_dir, file_list=train_files)
        val_dataset = LumbarMriDataset(data_dir, file_list=val_files)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


if __name__ == "__main__":
    # Self-test code
    print("Testing Dataset configurations...")
    data_dir = "./data"
    
    for ds_name in ["verse19", "verse20", "lumbar_mri"]:
        try:
            print(f"\n--- Testing {ds_name} ---")
            train_loader, val_loader = get_dataloaders(ds_name, data_dir, batch_size=1)
            print(f"{ds_name} Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
            img, seg = next(iter(train_loader))
            print(f"{ds_name} Sample Shapes - Image: {img.shape}, Seg: {seg.shape}")
            print(f"{ds_name} Seg Labels: {torch.unique(seg).tolist()}")
        except Exception as e:
            print(f"{ds_name} load failed: {e}")
