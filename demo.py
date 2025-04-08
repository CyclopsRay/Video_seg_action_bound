import os
import json
import argparse
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd
import numpy as np

def download_youcook2(data_dir='./youcook2_data'):
    """
    Download the YouCook2 dataset using HuggingFace Datasets
    """
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    print("Loading YouCook2 dataset from HuggingFace...")
    try:
        # Try to load from merve/YouCook2
        dataset = load_dataset("merve/YouCook2")
        print("Successfully loaded YouCook2 dataset from merve/YouCook2")
    except Exception as e:
        print(f"Error loading from merve/YouCook2: {e}")
        try:
            # Try alternative source: lmms-lab/YouCook2
            dataset = load_dataset("lmms-lab/YouCook2")
            print("Successfully loaded YouCook2 dataset from lmms-lab/YouCook2")
        except Exception as e2:
            print(f"Error loading from lmms-lab/YouCook2: {e2}")
            print("\nFalling back to manual annotation loading...")
            # If HuggingFace datasets don't work, try direct download from official website
            return None
    
    # Save the dataset information
    try:
        for split in dataset.keys():
            # Convert to pandas DataFrame for easier manipulation
            df = dataset[split].to_pandas()
            
            # Save as JSON
            json_path = os.path.join(data_dir, f"{split}_annotations.json")
            df.to_json(json_path, orient='records')
            print(f"Saved {split} annotations to {json_path}")
            
            # Save as CSV for easier viewing
            csv_path = os.path.join(data_dir, f"{split}_annotations.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {split} annotations to {csv_path}")
    except Exception as e:
        print(f"Error saving dataset: {e}")
    
    return dataset

def create_manual_dataset(data_dir='./youcook2_data'):
    """
    Create a minimal YouCook2 dataset manually for demo purposes
    when the download fails
    """
    # Sample data based on YouCook2 structure
    sample_data = [
        {
            "video_id": "sample_video_1",
            "recipe_type": "Hamburger",
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 15.5,
                    "sentence": "Place the ground beef in a bowl and add salt and pepper."
                },
                {
                    "start_time": 16.0,
                    "end_time": 30.0,
                    "sentence": "Form the meat into patties and cook on the grill."
                },
                {
                    "start_time": 31.0,
                    "end_time": 45.5,
                    "sentence": "Place the patty on a bun with lettuce, tomato, and ketchup."
                }
            ]
        },
        {
            "video_id": "sample_video_2",
            "recipe_type": "Pasta",
            "segments": [
                {
                    "start_time": 0.0,
                    "end_time": 12.5,
                    "sentence": "Boil water in a large pot and add salt."
                },
                {
                    "start_time": 13.0,
                    "end_time": 25.0,
                    "sentence": "Add pasta to the boiling water and cook for 8-10 minutes."
                },
                {
                    "start_time": 26.0,
                    "end_time": 40.5,
                    "sentence": "Drain the pasta and add sauce."
                }
            ]
        }
    ]
    
    # Create and save the annotations
    annotations_path = os.path.join(data_dir, "annotations.json")
    with open(annotations_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    print(f"Created sample annotations file at {annotations_path}")
    
    # Create a simple train/test split
    train_path = os.path.join(data_dir, "train.txt")
    with open(train_path, 'w') as f:
        f.write("sample_video_1")
    
    test_path = os.path.join(data_dir, "test.txt")
    with open(test_path, 'w') as f:
        f.write("sample_video_2")
    
    print("Created sample train/test split files")
    
    # Create a DataFrame for easier manipulation
    segments = []
    for video in sample_data:
        for segment in video['segments']:
            segments.append({
                'video_id': video['video_id'],
                'recipe_type': video['recipe_type'],
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'sentence': segment['sentence']
            })
    
    df = pd.DataFrame(segments)
    return {'train': df[df['video_id'] == 'sample_video_1'], 
            'test': df[df['video_id'] == 'sample_video_2']}

def visualize_segments(dataset, split='train', sample_idx=0, data_dir='./youcook2_data'):
    """
    Visualize the temporal segments of a video
    """
    try:
        # Get the dataset as a DataFrame
        if isinstance(dataset, dict):
            df = dataset[split]
        else:
            df = dataset[split].to_pandas()
        
        # Map column names based on what's available
        id_column = 'video_id' if 'video_id' in df.columns else ('youtube_id' if 'youtube_id' in df.columns else 'id')
        print(f"Using '{id_column}' as the video identifier column")
        print(f"Available columns: {df.columns.tolist()}")
        
        # Get unique videos
        videos = df[id_column].unique()
        if sample_idx >= len(videos):
            print(f"Video index {sample_idx} out of range. Max index is {len(videos)-1}")
            return
        
        # Get video data
        video_id = videos[sample_idx]
        video_data = df[df[id_column] == video_id]
        recipe_type = video_data['recipe_type'].iloc[0] if 'recipe_type' in video_data.columns else "Unknown"
        
        print(f"Video ID: {video_id}")
        print(f"Recipe: {recipe_type}")
        
        # Handle segment data which might be in different formats
        if 'segment' in video_data.columns:
            # For lmms-lab/YouCook2 format where segment info is in a single column
            segments = []
            for _, row in video_data.iterrows():
                segment_info = row['segment']
                
                # Check if segment is a string that needs parsing
                if isinstance(segment_info, str) and ('[' in segment_info or '{' in segment_info):
                    try:
                        # Try to parse as JSON/dict
                        if isinstance(segment_info, str):
                            segment_info = json.loads(segment_info.replace("'", '"'))
                    except:
                        print(f"Warning: Could not parse segment: {segment_info}")
                
                # Extract start and end times from segment info
                if isinstance(segment_info, dict):
                    start_time = segment_info.get('start_time', 0)
                    end_time = segment_info.get('end_time', 0)
                elif isinstance(segment_info, list) and len(segment_info) >= 2:
                    start_time = segment_info[0]
                    end_time = segment_info[1]
                else:
                    # Default if we can't parse
                    start_time = 0
                    end_time = 10
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'sentence': row['sentence']
                })
            
            # Convert segments to DataFrame for easier processing
            segments_df = pd.DataFrame(segments)
            
        elif 'start_time' in video_data.columns and 'end_time' in video_data.columns:
            # For datasets with explicit start_time and end_time columns
            segments_df = video_data[['start_time', 'end_time', 'sentence']]
        else:
            print("Could not find segment information in the dataset")
            return
        
        # Sort segments by start time
        segments_df = segments_df.sort_values('start_time')
        print(f"Number of segments: {len(segments_df)}")
        
        # Visualize segments on a timeline
        if len(segments_df) > 0:
            # Find the duration of the video (assume the last segment's end time is the end)
            duration = segments_df['end_time'].max()
            
            # Create a figure
            plt.figure(figsize=(12, 6))
            plt.title(f"Video segments for {recipe_type} (Video ID: {video_id})")
            plt.xlabel("Time (seconds)")
            plt.xlim(0, duration * 1.1)  # Add some padding
            plt.ylim(0, len(segments_df) + 1)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot each segment
            for i, (_, segment) in enumerate(segments_df.iterrows()):
                start = segment['start_time']
                end = segment['end_time']
                text = segment['sentence']
                
                # Plot segment as a horizontal bar
                plt.barh(i+1, end-start, left=start, height=0.5, 
                         color=plt.cm.viridis(i/len(segments)), alpha=0.8)
                
                # Add text label (truncate if too long)
                short_text = text[:40] + "..." if len(text) > 40 else text
                plt.text(start, i+1, f" {short_text}", va='center', fontsize=8)
            
            plt.tight_layout()
            output_path = os.path.join(data_dir, f"{video_id}_segments.png")
            plt.savefig(output_path)
            plt.show()
            print(f"Segment visualization saved to {output_path}")
        else:
            print("No segments found for this video.")
    except Exception as e:
        print(f"Error visualizing segments: {e}")

def analyze_text_descriptions(dataset, split='train', num_samples=10):
    """
    Analyze text descriptions from a sample of video segments
    """
    try:
        # Get the dataset as a DataFrame
        if isinstance(dataset, dict):
            df = dataset[split]
        else:
            df = dataset[split].to_pandas()
        
        # Get text descriptions
        if num_samples and num_samples < len(df):
            if hasattr(df, 'sample'):
                df = df.sample(num_samples)
            else:
                # If sample method is not available, use simple slicing
                df = df.iloc[:num_samples]
        
        descriptions = df['sentence'].tolist()
        
        # Basic analysis
        if descriptions:
            print(f"Analyzing {len(descriptions)} text descriptions")
            
            # Word counts
            all_words = ' '.join(descriptions).lower().split()
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Find top words
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 20 most common words:")
            for word, count in sorted_words[:20]:
                print(f"  {word}: {count}")
            
            # Average description length
            avg_length = sum(len(desc.split()) for desc in descriptions) / len(descriptions)
            print(f"Average description length: {avg_length:.2f} words")
        else:
            print("No descriptions found for analysis.")
    except Exception as e:
        print(f"Error analyzing text descriptions: {e}")

def run_demo(dataset, data_dir='./youcook2_data'):
    """
    Run a demo using the YouCook2 dataset
    """
    print("=" * 50)
    print("YouCook2 Dataset Demo")
    print("=" * 50)
    
    # Check if we have a valid dataset
    if dataset is None:
        print("No dataset available. Creating mock dataset for demo...")
        dataset = create_manual_dataset(data_dir)
    
    # Visualize segments for a random video
    try:
        splits = list(dataset.keys())
        selected_split = splits[0]  # Usually 'train'
        print(f"\nSelected split: {selected_split}")
        
        # Visualize segments for the first video
        visualize_segments(dataset, split=selected_split, sample_idx=0, data_dir=data_dir)
        
        # Analyze text descriptions
        analyze_text_descriptions(dataset, split=selected_split)
        
        print("\nDemo complete! This demonstrates basic analysis capabilities with the YouCook2 dataset.")
        print("For more advanced usage including video feature extraction and model training,")
        print("refer to the official repositories and documentation.")
    except Exception as e:
        print(f"Error running demo: {e}")
        print("\nPlease check that you have valid annotation files in your data directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YouCook2 Dataset Demo via HuggingFace')
    parser.add_argument('--data_dir', type=str, default='./youcook2_data',
                        help='Directory to store the dataset')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading and use existing files')
    # parser.add_argument('--mock_data', action='store_true',
    #                     help='Use mock data instead of downloading')
    
    args = parser.parse_args()
    
    # Download or load the dataset
    dataset = None
    if args.mock_data:
        print("Using mock YouCook2 data for demo...")
        dataset = create_manual_dataset(args.data_dir)
    elif not args.skip_download:
        dataset = download_youcook2(args.data_dir)
    
    # Run the demo
    # run_demo(dataset, args.data_dir)