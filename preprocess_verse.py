import os
import glob
import numpy as np
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
from PIL import Image
from tqdm import tqdm

def reorient_to(img, axcodes_to=('P', 'I', 'R')):
    """
    Reorients the nifti from its original orientation to another specified orientation.
    """
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    return newimg

def find_segmentation(ct_path):
    """
    Finds the corresponding segmentation file for a given CT path.
    Looks for BIDS-compliant derivatives paths and suffixes.
    """
    # Directory mapping
    dir_name = os.path.dirname(ct_path)
    base_name = os.path.basename(ct_path)
    
    # Precise match prefix (e.g. sub-verse414_split-verse273)
    prefix = base_name.replace("_ct.nii.gz", "").replace("_ct.nii", "")
    
    # Subject ID fallback (e.g. sub-verse004)
    if "sub-" in base_name:
        sub_id = base_name.split("_")[0]
    else:
        sub_id = os.path.basename(dir_name)

    # Candidates for suffixes
    seg_suffixes = ["_seg-vert_msk.nii.gz", "_seg.nii.gz", "_seg-vert_msk.nii", "_seg.nii"]
    
    # Candidate directories
    # 1. derivatives folder (standard BIDS structure)
    # 2. same folder as CT
    # Let's search recursively under the parent of parent folder or derivatives
    parent_dir = os.path.dirname(dir_name)
    grandparent_dir = os.path.dirname(parent_dir)
    
    search_dirs = [dir_name, parent_dir, grandparent_dir]
    if "rawdata" in ct_path:
        search_dirs.append(ct_path.split("rawdata")[0] + "derivatives")
        
    for sd in search_dirs:
        if not os.path.exists(sd):
            continue
            
        # Strategy A: Try precise prefix matching first to avoid cross-matching split files
        for suffix in seg_suffixes:
            candidate_pattern = os.path.join(sd, "**", f"{prefix}{suffix}")
            candidates = glob.glob(candidate_pattern, recursive=True)
            if candidates:
                return candidates[0]
                
        # Strategy B: Fallback to subject-level matching if precise prefix doesn't match
        for suffix in seg_suffixes:
            candidate_pattern = os.path.join(sd, "**", f"{sub_id}_*{suffix.lstrip('_')}")
            candidates = glob.glob(candidate_pattern, recursive=True)
            if candidates:
                return candidates[0]
                
    return None

def preprocess_single(ct_path, output_dir):
    """
    Helper function to preprocess a single patient CT scan and extract slices.
    Must be at global module level for multiprocessing serialization.
    """
    # 1. Find matching segmentation
    seg_path = find_segmentation(ct_path)
    if not seg_path or not os.path.exists(seg_path):
        return False, 0, f"No segmentation found for {ct_path}"
        
    base_name = os.path.basename(ct_path)
    pid = base_name.split("_")[0]
    
    try:
        # 2. Load volumes
        ct_img = nib.load(ct_path)
        seg_img = nib.load(seg_path)
        
        # 3. Reorient both to PIR
        ct_img_pir = reorient_to(ct_img, axcodes_to=('P', 'I', 'R'))
        seg_img_pir = reorient_to(seg_img, axcodes_to=('P', 'I', 'R'))
        
        # 4. Resample to 1.0mm isotropic resolution
        ct_res = nip.resample_to_output(ct_img_pir, voxel_sizes=(1.0, 1.0, 1.0), order=3)
        seg_res = nip.resample_to_output(seg_img_pir, voxel_sizes=(1.0, 1.0, 1.0), order=0)
        
        # 5. Extract data arrays
        ct_data = ct_res.get_fdata()
        seg_data = seg_res.get_fdata()
        
        assert ct_data.shape == seg_data.shape, f"Shape mismatch: CT {ct_data.shape} vs Seg {seg_data.shape}"
        
        # 6. Normalize HU using bone window [-500, 1300]
        ct_norm = np.clip(ct_data, -500, 1300)
        ct_norm = (ct_norm - (-500)) / (1300 - (-500))
        
        # 7. Extract slices along Axis 2 (Right-Left dimension in PIR orientation)
        num_slices = ct_data.shape[2]
        patient_slices = 0
        
        # First, find min and max slice index containing at least 10 mask pixels
        valid_indices = []
        for s_idx in range(num_slices):
            if np.sum(seg_data[:, :, s_idx] > 0) >= 10:
                valid_indices.append(s_idx)
                
        if not valid_indices:
            print(f"No valid slices found for {pid}")
            return True, 0, None
            
        min_idx = min(valid_indices)
        max_idx = max(valid_indices)
        
        # Keep all slices between min_idx and max_idx to maintain 3D Z-axis continuity
        for s_idx in range(min_idx, max_idx + 1):
            slice_ct = ct_norm[:, :, s_idx]
            slice_seg = seg_data[:, :, s_idx]
            
            slice_seg_binary = (slice_seg > 0).astype(np.uint8)
            slice_ct_8bit = (slice_ct * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(slice_ct_8bit)
            lbl_pil = Image.fromarray(slice_seg_binary)
            
            slice_name = f"{pid}_slice{s_idx:03d}.png"
            img_pil.save(os.path.join(output_dir, "images", slice_name))
            lbl_pil.save(os.path.join(output_dir, "labels", slice_name))
            
            patient_slices += 1
                
        return True, patient_slices, None
    except Exception as e:
        return False, 0, f"Failed processing patient {pid}: {e}"

def preprocess_dataset(raw_dir, output_dir):
    """
    Main function to preprocess raw VerSe NIfTI dataset and output pre-extracted 2D sagittal PNG slices.
    """
    print(f"\n=== Preprocessing dataset in: {raw_dir} ===")
    print(f"Saving preprocessed slices to: {output_dir}")
    
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Recursively find all CT scans
    ct_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*_ct.nii.gz"), recursive=True))
    if not ct_paths:
        ct_paths = sorted(glob.glob(os.path.join(raw_dir, "**", "*_ct.nii"), recursive=True))
        
    print(f"Found {len(ct_paths)} CT scan candidates.")
    
    # Check already processed patients to support resumption
    processed_pids = set()
    images_dir = os.path.join(output_dir, "images")
    if os.path.exists(images_dir):
        for f in os.listdir(images_dir):
            if f.endswith(".png"):
                processed_pids.add(f.split("_")[0])
                
    # Filter out already processed scans
    ct_paths_to_process = []
    skipped_count = 0
    for ct_path in ct_paths:
        base_name = os.path.basename(ct_path)
        pid = base_name.split("_")[0]
        if pid in processed_pids:
            skipped_count += 1
        else:
            ct_paths_to_process.append(ct_path)
            
    print(f"Skipped {skipped_count} already processed patients.")
    print(f"Scans remaining to process: {len(ct_paths_to_process)}")
    
    if not ct_paths_to_process:
        print("All scans already processed.")
        return
        
    success_count = 0
    total_slices_extracted = 0
    
    # Use ProcessPoolExecutor to parallelize preprocessing
    # Using 4 workers to prevent memory limits / OOM issues
    num_workers = min(4, os.cpu_count() or 1)
    print(f"Using {num_workers} parallel process workers.")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(preprocess_single, ct_path, output_dir): ct_path for ct_path in ct_paths_to_process}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            ct_path = futures[future]
            try:
                success, num_slices, err_msg = future.result()
                if success:
                    success_count += 1
                    total_slices_extracted += num_slices
                else:
                    if "No segmentation found" in err_msg:
                        print(f"[WARN] {err_msg}")
                    else:
                        print(f"[ERROR] {err_msg}")
            except Exception as exc:
                print(f"[ERROR] {ct_path} generated an exception: {exc}")
                
    print(f"\nSuccessfully preprocessed {success_count}/{len(ct_paths)} patients.")
    print(f"Extracted {total_slices_extracted} total slices.")

if __name__ == "__main__":
    data_dir = "./data"
    
    # Preprocess VerSe 19
    verse19_raw = os.path.join(data_dir, "verse19_raw")
    verse19_out = os.path.join(data_dir, "verse19")
    if os.path.exists(verse19_raw):
        preprocess_dataset(verse19_raw, verse19_out)
    else:
        print(f"Directory {verse19_raw} does not exist, skipping VerSe 19 preprocessing.")
        
    # Preprocess VerSe 20
    verse20_raw = os.path.join(data_dir, "verse20_raw")
    verse20_out = os.path.join(data_dir, "verse20")
    if os.path.exists(verse20_raw):
        preprocess_dataset(verse20_raw, verse20_out)
    else:
        print(f"Directory {verse20_raw} does not exist, skipping VerSe 20 preprocessing.")
