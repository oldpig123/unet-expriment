import os
import sys
import time
from tqdm import tqdm
from osfclient.api import OSF
from osfclient.models.session import OSFSession

# Monkey-patch OSFSession to retry on transient 502/503/504 errors
original_get = OSFSession.get

def retry_get(self, url, *args, **kwargs):
    retries = 5
    backoff = 2
    for attempt in range(retries):
        try:
            response = original_get(self, url, *args, **kwargs)
            if response.status_code in [502, 503, 504]:
                print(f"\n[HTTP {response.status_code}] server error on {url}, retrying in {backoff}s (attempt {attempt+1}/{retries})...")
                time.sleep(backoff)
                backoff *= 2
                continue
            return response
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"\n[Error] {e} on {url}, retrying in {backoff}s (attempt {attempt+1}/{retries})...")
            time.sleep(backoff)
            backoff *= 2
    # One last try if retries exhausted
    return original_get(self, url, *args, **kwargs)

OSFSession.get = retry_get

def download_osf_project(project_id, output_dir):
    print(f"\n=== Connecting to OSF Project {project_id} ===")
    osf = OSF()
    project = osf.project(project_id)
    storage = project.storage('osfstorage')
    
    print("Fetching list of remote files...")
    files_to_download = []
    
    for file_ in storage.files:
        path = file_.path
        # Keep only NIfTI images/labels, JSON annotations, and preview PNGs
        if not (path.endswith('.nii.gz') or path.endswith('.nii') or path.endswith('.json') or path.endswith('.png')):
            continue
        files_to_download.append(file_)
        
    print(f"Found {len(files_to_download)} candidate files to sync.")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for file_ in tqdm(files_to_download, desc=f"Syncing {project_id}"):
        rel_path = file_.path.lstrip('/')
        local_path = os.path.join(output_dir, rel_path)
        local_dir = os.path.dirname(local_path)
        
        # Skip if already fully downloaded
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            skip_count += 1
            continue
            
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            with open(local_path, 'wb') as f:
                file_.write_to(f)
            success_count += 1
        except Exception as e:
            # Clean up partial download
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            print(f"\n[WARN] Failed to download {file_.path}: {e}")
            fail_count += 1
            
    print(f"Finished project {project_id}:")
    print(f"  - Downloaded: {success_count}")
    print(f"  - Skipped (already exists): {skip_count}")
    print(f"  - Failed/Forbidden: {fail_count}")

if __name__ == "__main__":
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Sync VerSe '19 (jtfa5) - already completed
    # download_osf_project("jtfa5", os.path.join(data_dir, "verse19_raw"))
    
    # 2. Sync VerSe '20 (4skx2)
    download_osf_project("4skx2", os.path.join(data_dir, "verse20_raw"))
