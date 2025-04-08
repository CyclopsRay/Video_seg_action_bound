# dataset.py - Dataset handling

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import json
import torchfile


class YouCook2Dataset(Dataset):
    def __init__(self, split='train', data_root='/home/leiy4/video-segmentation/data', 
                 clip_length=16, stride=8, fps=30.0, is_test=False):
        """
        YouCook2 dataset for video segmentation
        
        Args:
            split: 'train', 'val', or 'test'
            data_root: Root directory of the dataset
            clip_length: Number of frames in each clip
            stride: Stride for sliding window
            fps: Frames per second
            is_test: Whether in test mode
        """
        self.split = split
        self.data_root = data_root
        self.clip_length = clip_length
        self.stride = stride
        self.fps = fps
        self.is_test = is_test
        
        # Set paths
        self.annotations_dir = os.path.join(data_root, 'annotations')
        self.splits_dir = os.path.join(data_root, 'splits')
        self.raw_videos_dir = os.path.join(data_root, 'raw_videos')
        
        # Load split file
        split_file = os.path.join(self.splits_dir, f"{split}_list.txt")
        with open(split_file, 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]
        
        # Load annotations - different file based on split
        if split in ['train', 'val']:
            annotation_file = os.path.join(self.annotations_dir, "youcookii_annotations_trainval.json")
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:  # test
            annotation_file = os.path.join(self.annotations_dir, "youcookii_annotations_test_segments_only.json")
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
            
        # Create mapping from video_id to video path
        self.video_paths = {}
        self.samples = []
        
        # Process videos and annotations
        self._process_data()
    
    def _process_data(self):
        """Process the data and create samples"""
        for video_id in self.video_ids:
            # Extract youtube ID (last part of the video_id)
            youtube_id = video_id.split('/')[-1]
            
            # Get information about video subset (training/testing/validation) from annotations
            video_subset = None
            video_recipe_type = None
            if youtube_id in self.annotations['database']:
                video_data = self.annotations['database'][youtube_id]
                if 'subset' in video_data:
                    video_subset = video_data['subset']
                if 'recipe_type' in video_data:
                    video_recipe_type = video_data['recipe_type']
            
            # Find video path
            video_path = None
            if video_subset and video_recipe_type:
                # Construct path like /home/leiy4/video-segmentation/data/raw_videos/testing/101/YSes0R7EksY.mp4
                video_path = os.path.join(self.raw_videos_dir, video_subset, video_recipe_type, f"{youtube_id}.mp4")
            
            # If path doesn't exist or couldn't be constructed, try to find it
            if not video_path or not os.path.exists(video_path):
                video_path = self._find_video_path(youtube_id)
                if not video_path:
                    print(f"Warning: Video not found for {youtube_id}: {video_path}")
                    continue
            
            self.video_paths[video_id] = video_path
            
            # Get video duration and number of frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_path}")
                continue
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / self.fps
            cap.release()
            
            # Get boundaries for this video
            boundaries = []
            if youtube_id in self.annotations['database']:
                video_data = self.annotations['database'][youtube_id]
                
                # Extract segment boundaries
                if 'annotations' in video_data:
                    for annotation in video_data['annotations']:
                        if 'segment' in annotation and len(annotation['segment']) == 2:
                            boundary_time = annotation['segment'][0]  # Use start of segment as boundary
                            boundaries.append(boundary_time)
            
            # Create samples with sliding window
            for start_frame in range(0, frame_count - self.clip_length + 1, self.stride):
                start_time = start_frame / self.fps
                end_time = (start_frame + self.clip_length) / self.fps
                
                # Check if boundary falls within this clip
                has_boundary = 0
                for boundary in boundaries:
                    if start_time <= boundary < end_time:
                        has_boundary = 1
                        break
                
                self.samples.append({
                    'video_id': video_id,
                    'video_path': video_path,
                    'start_frame': start_frame,
                    'has_boundary': has_boundary
                })
        print(f"Total samples: {len(self.samples)}")
    
    def _find_video_path(self, youtube_id):
        """Find video path by searching through raw_videos directory"""
        # First check common locations based on known structure
        subsets = ['training', 'testing', 'validation']
        recipe_types = [f"{i:03d}" for i in range(1, 150)]  # Assuming recipe types from 001-149
        
        for subset in subsets:
            for recipe_type in recipe_types:
                path = os.path.join(self.raw_videos_dir, subset, recipe_type, f"{youtube_id}.mp4")
                if os.path.exists(path):
                    return path
        
        # If not found in common locations, do a full search
        for root, dirs, files in os.walk(self.raw_videos_dir):
            for file in files:
                if file.startswith(youtube_id) and file.endswith('.mp4'):
                    return os.path.join(root, file)
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        start_frame = sample['start_frame']
        has_boundary = sample['has_boundary']
        
        # Load video clip
        frames = self._load_clip(video_path, start_frame)
        
        # Convert to tensor
        frames_tensor = torch.FloatTensor(frames)
        # Normalize to [0, 1] and then apply standard normalization
        frames_tensor = frames_tensor / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
        
        # Standard normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        if self.is_test:
            return {
                'frames': frames_tensor,
                'video_id': sample['video_id'],
                'start_frame': start_frame
            }
        else:
            label = torch.FloatTensor([has_boundary])
            return {
                'frames': frames_tensor,
                'labels': label  # Changed 'label' to 'labels' to match trainer.py
            }
    
    def _load_clip(self, video_path, start_frame):
        """Load clip from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return np.zeros((self.clip_length, 224, 224, 3))
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                # If we run out of frames, pad with zeros
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
                
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
class YouCook2FeatureDataset(Dataset):
    def __init__(self, split='train', data_root='/home/leiy4/video-segmentation/data', 
                 clip_length=16, stride=8, is_test=False):
        """
        YouCook2 dataset using precomputed features
        
        Args:
            split: 'train', 'val', or 'test'
            data_root: Root directory of the dataset
            clip_length: Number of frames in each clip
            stride: Stride for sliding window
            is_test: Whether in test mode
        """
        self.split = split
        self.data_root = data_root
        self.clip_length = clip_length
        self.stride = stride
        self.is_test = is_test
        
        # Set paths
        self.annotations_dir = os.path.join(data_root, 'annotations')
        self.splits_dir = os.path.join(data_root, 'splits')
        
        # Path to the feat_dat directory with the correct structure
        self.features_dir = os.path.join(data_root, 'feat_dat', f'{split}_frame_feat')
        
        # Load split file
        split_file = os.path.join(self.splits_dir, f"{split}_list.txt")
        with open(split_file, 'r') as f:
            self.video_ids = [line.strip() for line in f.readlines()]
        
        # Load annotations - different file based on split
        if split in ['train', 'val']:
            annotation_file = os.path.join(self.annotations_dir, "youcookii_annotations_trainval.json")
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
        else:  # test
            annotation_file = os.path.join(self.annotations_dir, "youcookii_annotations_test_segments_only.json")
            with open(annotation_file, 'r') as f:
                self.annotations = json.load(f)
            
        # Process features and annotations
        self.samples = []
        self._process_data()
    
    def _process_data(self):
        """Process the data and create samples"""
        for video_id in self.video_ids:
            # Parse the recipe_type and youtube_id from the video_id
            recipe_type, youtube_id = video_id.split('/')
            
            # Construct feature file path based on the structure
            # Expected path: /home/leiy4/video-segmentation/data/feat_dat/train_frame_feat/101/0O4bxhpFX9o/0001/resnet_34_feat_mscoco.dat
            feature_dir = os.path.join(self.features_dir, recipe_type, youtube_id)
            
            # Check if directory exists
            if not os.path.exists(feature_dir):
                print(f"Warning: Feature directory not found for {video_id}")
                continue
            
            # Find the subdirectories (like '0001')
            subdirs = [d for d in os.listdir(feature_dir) if os.path.isdir(os.path.join(feature_dir, d))]
            
            if not subdirs:
                print(f"Warning: No subdirectories found in {feature_dir}")
                continue
                
            # Use the first subdir (usually should be '0001')
            subdir = subdirs[0]
            feature_file = os.path.join(feature_dir, subdir, "resnet_34_feat_mscoco.dat")
            
            if not os.path.exists(feature_file):
                print(f"Warning: Feature file not found: {feature_file}")
                continue
            
            # Get feature dimensions and number of frames by reading the first frames
            # Note: We'll need a function to read .dat files
            try:
                features = self._read_torch_binary(feature_file)
                num_frames = features.shape[0]
            except Exception as e:
                print(f"Error loading features for {video_id}: {e}")
                continue
            
            # Get boundaries for this video
            boundaries = []
            if youtube_id in self.annotations['database']:
                video_data = self.annotations['database'][youtube_id]
                
                # Extract segment boundaries
                if 'annotations' in video_data:
                    for annotation in video_data['annotations']:
                        if 'segment' in annotation and len(annotation['segment']) == 2:
                            boundary_time = annotation['segment'][0]  # Use start of segment as boundary
                            # Convert time to frame index (assuming 1.58 fps from readme)
                            boundary_frame = int(boundary_time * 1.58)
                            boundaries.append(boundary_frame)
            
            # Create samples with sliding window
            for start_frame in range(0, num_frames - self.clip_length + 1, self.stride):
                end_frame = start_frame + self.clip_length
                
                # Check if boundary falls within this clip
                has_boundary = 0
                for boundary in boundaries:
                    if start_frame <= boundary < end_frame:
                        has_boundary = 1
                        break
                
                self.samples.append({
                    'video_id': video_id,
                    'youtube_id': youtube_id,
                    'feature_file': feature_file,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'has_boundary': has_boundary
                })
        print(f"Total samples: {len(self.samples)}")
    
    def _read_torch_binary(self, filepath):
        """
        Read a Torch binary .dat file using torchfile library
        
        Note: You'll need to install torchfile: pip install torchfile
        """
        
        try:
            # Load the Torch file
            data = torchfile.load(filepath)
            
            # Convert to numpy array if it's not already
            if not isinstance(data, np.ndarray):
                data = np.array(data)
                
            return data
        except Exception as e:
            print(f"Error reading Torch file {filepath}: {e}")
            # Return a small empty array as fallback
            return np.zeros((10, 512))  # Assuming ResNet-34 features with 512 dimensions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature_file = sample['feature_file']
        start_frame = sample['start_frame']
        end_frame = sample['end_frame']
        has_boundary = sample['has_boundary']
        
        # Load features
        features = self._read_torch_binary(feature_file)
        clip_features = features[start_frame:end_frame]
        
        # Ensure we have the correct number of frames
        if clip_features.shape[0] < self.clip_length:
            # Pad with zeros if needed
            padding = np.zeros((self.clip_length - clip_features.shape[0], clip_features.shape[1]))
            clip_features = np.vstack([clip_features, padding])
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(clip_features)
        
        if self.is_test:
            return {
                'features': features_tensor,
                'video_id': sample['video_id'],
                'youtube_id': sample['youtube_id'],
                'start_frame': start_frame
            }
        else:
            label = torch.FloatTensor([has_boundary])
            return {
                'features': features_tensor,
                'labels': label
            }