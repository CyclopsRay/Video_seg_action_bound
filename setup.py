import os
import argparse
import requests
import zipfile
import json
import pandas as pd
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file with progress bar
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), 
                          total=total_size//block_size, 
                          unit='KB', unit_scale=True):
            f.write(data)
    
    return destination

def setup_youcook2(data_dir='./youcook2_data'):
    """
    Setup YoucookII dataset
    """
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # URLs for different components of the dataset
    # Note: These URLs might need to be updated based on the current availability
    dataset_urls = {
        'annotations': 'http://youcook2.eecs.umich.edu/static/YouCookII/youcookii_annotations_trainval.json',
        'train_split': 'http://youcook2.eecs.umich.edu/static/YouCookII/splits/train.txt',
        'val_split': 'http://youcook2.eecs.umich.edu/static/YouCookII/splits/val.txt',
        'test_split': 'http://youcook2.eecs.umich.edu/static/YouCookII/splits/test.txt',
        # You may need to get the latest URLs for features, segments, etc.
    }
    
    # Download each component
    downloaded_files = {}
    for name, url in dataset_urls.items():
        try:
            dest_path = os.path.join(data_dir, f"{name}{os.path.splitext(url)[1]}")
            print(f"Downloading {name} from {url}...")
            download_file(url, dest_path)
            downloaded_files[name] = dest_path
            print(f"Successfully downloaded {name} to {dest_path}")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
    
    # For demonstration, access the downloaded annotations
    if 'annotations' in downloaded_files and os.path.exists(downloaded_files['annotations']):
        try:
            with open(downloaded_files['annotations'], 'r') as f:
                annotations = json.load(f)
            print(f"Loaded annotations with {len(annotations)} entries")
            # Display some basic statistics
            print_dataset_stats(annotations)
        except Exception as e:
            print(f"Error loading annotations: {e}")
    
    print("\nNote: For the complete dataset including video features, please visit:")
    print("  - Official website: http://youcook2.eecs.umich.edu/")
    print("  - GitHub repo: https://github.com/LuoweiZhou/YouCook2-Leaderboard")
    print("  - ProcNets repo: https://github.com/LuoweiZhou/ProcNets-YouCook2")

def print_dataset_stats(annotations):
    """
    Print basic statistics about the dataset
    """
    recipes = set()
    total_videos = 0
    total_segments = 0
    
    for annotation in annotations:
        recipes.add(annotation.get('recipe_type', ''))
        total_videos += 1
        total_segments += len(annotation.get('segments', []))
    
    print(f"Dataset Statistics:")
    print(f"  - Total recipes: {len(recipes)}")
    print(f"  - Total videos: {total_videos}")
    print(f"  - Total segments: {total_segments}")
    print(f"  - Average segments per video: {total_segments/total_videos:.2f}")

def run_demo(data_dir='./youcook2_data'):
    """
    Run a simple demo using the YoucookII dataset
    """
    annotations_file = os.path.join(data_dir, 'annotations.json')
    
    if not os.path.exists(annotations_file):
        print(f"Annotations file not found at {annotations_file}")
        print("Please run the setup function first to download the dataset.")
        return
    
    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Display a sample of the dataset
    print("Sample from YoucookII dataset:")
    for i, annotation in enumerate(annotations[:3]):
        print(f"\nVideo {i+1}: {annotation.get('video_id', 'Unknown')}")
        print(f"Recipe: {annotation.get('recipe_type', 'Unknown')}")
        print("Segments:")
        for j, segment in enumerate(annotation.get('segments', [])[:3]):
            print(f"  {j+1}. {segment.get('sentence', 'No description')}")
            print(f"     Time: {segment.get('start_time', 0):.2f}s - {segment.get('end_time', 0):.2f}s")
    
    print("\nDemo complete! This is a basic demonstration of accessing the YoucookII dataset.")
    print("For more advanced usage, you might want to explore the feature extraction and model training code.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YoucookII Dataset Downloader and Demo')
    parser.add_argument('--data_dir', type=str, default='./youcook2_data',
                        help='Directory to store the dataset')
    parser.add_argument('--run_demo', action='store_true',
                        help='Run a simple demo after downloading')
    
    args = parser.parse_args()
    
    # Setup the dataset
    setup_youcook2(args.data_dir)
    
    # Run the demo if requested
    if args.run_demo:
        run_demo(args.data_dir)


# Run the script
# python setup.py --data_dir ./youcook2_data --run_demo

# Dependencies:
# pip install requests tqdm pandas
# Note: The URLs for downloading the dataset components may change over time.
# Ensure you have the latest URLs from the official YoucookII website.
# The script is designed to be run from the command line.
# Example command to run the script:
# python setup.py --data_dir ./youcook2_data --run_demo
# The script includes a demo function that prints out a sample of the dataset.
