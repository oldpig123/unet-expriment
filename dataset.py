import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import nibabel as nib

class VerSeDataset(Dataset):
    """
    VerSe '19 Spinal CT Dataset.
    Loads 3D CT volumes (NIfTI format), extracts 2D sagittal slices (along Axis 2 / Left-Right),
    filters out slices without vertebrae annotations, normalizes Hounsfield Units (HU),
    and resizes to 512x512.
    """
    def __init__(self, data_dir, patient_ids, target_size=(512, 512)):
        self.data_dir = os.path.join(data_dir, "verse")
        self.patient_ids = patient_ids
        self.target_size = target_size
        self.slices = []

        print(f"[INFO] Initializing VerSeDataset for patients: {patient_ids}")
        for pid in patient_ids:
            ct_path = os.path.join(self.data_dir, f"{pid}_ct.nii.gz")
            seg_path = os.path.join(self.data_dir, f"{pid}_seg.nii.gz")
            
            if not os.path.exists(ct_path) or not os.path.exists(seg_path):
                print(f"[WARN] VerSe files not found for patient {pid}, skipping.")
                continue
                
            # Load NIfTI volumes
            ct_img = nib.load(ct_path)
            seg_img = nib.load(seg_path)
            
            ct_data = ct_img.get_fdata()
            seg_data = seg_img.get_fdata()
            
            # Verify orientation (typically Left-Right axis is Axis 2 for sagittal slicing)
            # In RAS/PIR orientations, the coronal/sagittal mapping is consistent.
            # Slice along axis 2 (z-axis in PIR is Right-Left)
            num_slices = ct_data.shape[2]
            
            # Normalize entire CT volume HU to [0, 1] using standard clipping for bone window
            # Bone window: Level=400, Width=1800 -> min=-500, max=1300
            ct_normalized = np.clip(ct_data, -500, 1300)
            ct_normalized = (ct_normalized - (-500)) / (1300 - (-500))
            
            for s_idx in range(num_slices):
                slice_seg = seg_data[:, :, s_idx]
                
                # Check if slice contains any labeled vertebrae (non-zero)
                if np.sum(slice_seg > 0) > 10:  # Threshold to ignore minor artifacts
                    slice_ct = ct_normalized[:, :, s_idx]
                    self.slices.append((slice_ct, slice_seg))
                    
        print(f"[INFO] Loaded {len(self.slices)} sagittal slices for VerSe dataset.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slice_ct, slice_seg = self.slices[idx]
        
        # Resize using PIL
        # Convert to PIL Image for high-quality resizing
        img_pil = Image.fromarray((slice_ct * 255).astype(np.uint8))
        # For segmentation labels, we map to class labels: 0=background, 1=vertebrae
        # Map any non-zero vertebra label in VerSe (e.g. 16, 17, ...) to class 1
        seg_mapped = (slice_seg > 0).astype(np.uint8)
        seg_pil = Image.fromarray(seg_mapped)
        
        img_resized = img_pil.resize(self.target_size, Image.BILINEAR)
        seg_resized = seg_pil.resize(self.target_size, Image.NEAREST)
        
        # Convert to PyTorch tensors
        img_tensor = torch.tensor(np.array(img_resized), dtype=torch.float32).unsqueeze(0) / 255.0  # (1, 512, 512)
        seg_tensor = torch.tensor(np.array(seg_resized), dtype=torch.long)  # (512, 512)
        
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
    if dataset_name.lower() == "verse":
        # VerSe dataset: split by patient (patient 004 for train, patient 005 for validation)
        train_dataset = VerSeDataset(data_dir, patient_ids=["sub-verse004"])
        val_dataset = VerSeDataset(data_dir, patient_ids=["sub-verse005"])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
        
    elif dataset_name.lower() == "lumbar_mri":
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
    import sys
    data_dir = "./data"
    
    try:
        train_loader, val_loader = get_dataloaders("verse", data_dir, batch_size=1)
        print(f"VerSe Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        img, seg = next(iter(train_loader))
        print(f"VerSe Sample Shapes - Image: {img.shape}, Seg: {seg.shape}")
        print(f"VerSe Seg Labels: {torch.unique(seg).tolist()}")
    except Exception as e:
        print(f"VerSe load failed: {e}")
        
    try:
        train_loader, val_loader = get_dataloaders("lumbar_mri", data_dir, batch_size=1)
        print(f"Lumbar MRI Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        img, seg = next(iter(train_loader))
        print(f"Lumbar MRI Sample Shapes - Image: {img.shape}, Seg: {seg.shape}")
        print(f"Lumbar MRI Seg Labels: {torch.unique(seg).tolist()}")
    except Exception as e:
        print(f"Lumbar MRI load failed: {e}")
