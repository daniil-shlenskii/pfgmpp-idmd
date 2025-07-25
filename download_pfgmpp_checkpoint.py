import argparse
import os
import pickle
from urllib.parse import parse_qs, urlparse

import gdown
import torch

CKPTS_DIR = "downloads/pgmpp_ckpts"


def extract_file_id_from_url(url):
    """Extract file ID from Google Drive share link"""
    parsed = urlparse(url)
    if 'drive.google.com' in parsed.netloc:
        if 'file/d/' in url:
            # Format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            file_id = url.split('/file/d/')[1].split('/')[0]
        else:
            # Format: https://drive.google.com/open?id=FILE_ID
            file_id = parse_qs(parsed.query)['id'][0]
        return file_id
    return None

def download_and_convert(gdrive_url, output_path):
    """Download torch file from Google Drive and convert to pickle"""
    # Extract file ID from URL
    file_id = extract_file_id_from_url(gdrive_url)
    if not file_id:
        raise ValueError("Invalid Google Drive URL")
    
    # Create temporary directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Temporary file path for the downloaded file
    temp_file = os.path.join(os.path.dirname(output_path), f"temp_{file_id}.pth")
    
    try:
        # Download the file using gdown
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, temp_file, quiet=False)
        
        # Load the torch file
        torch_data = torch.load(temp_file, map_location=torch.device('cpu'))
        
        # Save as pickle
        with open(output_path, 'wb') as f:
            pickle.dump(torch_data, f)
            
        print(f"Successfully converted and saved to {output_path}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download PyTorch file from Google Drive and convert to pickle')
    parser.add_argument('gdrive_url', type=str, help='Google Drive share link')
    parser.add_argument('filename', type=str, help='Output path for the pickle file')
    
    args = parser.parse_args()
    
    os.makedirs(CKPTS_DIR, exist_ok=True)
    download_and_convert(args.gdrive_url, f"{CKPTS_DIR}/{args.filename}.pkl")
