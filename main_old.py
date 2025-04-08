import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import pandas as pd
import struct
from torch.utils.data import Dataset, DataLoader
import glob
# import cv2

# Function to read .dat feature files
def read_dat_file(file_path):
    with open(file_path, 'rb') as f:
        # Read the feature dimension and number of frames
        feat_dim = struct.unpack('i', f.read(4))[0]
        num_frames = struct.unpack('i', f.read(4))[0]
        
        # Read the features
        data = np.zeros((num_frames, feat_dim), dtype=np.float32)
        for i in range(num_frames):
            for j in range(feat_dim):
                data[i, j] = struct.unpack('f', f.read(4))[0]
    
    return data

# Parse CSV annotations
def parse_csv_annotations(annotation_file):
    annotations = pd.read_csv(annotation_file)
    video_segments = {}
    
    for _, row in annotations.iterrows():
        video_id = row['youtube_id']
        recipe_type = str(row['recipe_type'])
        segment_str = row['segment']
        
        # Parse segment string like [41. 56.]
        segment_str = segment_str.strip('[]')
        start_time, end_time = map(float, segment_str.split('.'))
        
        if video_id not in video_segments:
            video_segments[video_id] = {'recipe_type': recipe_type, 'segments': []}
        
        video_segments[video_id]['segments'].append({
            'start_time': start_time,
            'end_time': end_time
        })
    
    return video_segments

# Custom dataset for YouCook2
class YouCook2Dataset(Dataset):
    def __init__(self, annotation_file, feature_dir, fps=30, window_size=64, stride=16):
        # Load annotations
        self.annotations = parse_csv_annotations(annotation_file)
        self.feature_dir = feature_dir
        self.fps = fps  # Frames per second, for converting time to frame
        self.window_size = window_size
        self.stride = stride
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        
        for video_id, video_data in self.annotations.items():
            recipe_type = video_data['recipe_type']
            
            # Find all feature files for this video
            feature_pattern = os.path.join(
                self.feature_dir, 
                recipe_type, 
                video_id, 
                "*/resnet_34_feat_mscoco.dat"
            )
            feature_files = glob.glob(feature_pattern)
            
            if not feature_files:
                print(f"No features found for video {video_id}")
                continue
            
            # Load and concatenate all features for this video
            all_features = []
            for feature_file in sorted(feature_files):
                features = read_dat_file(feature_file)
                all_features.append(features)
            
            if not all_features:
                continue
                
            features = np.concatenate(all_features, axis=0)
            total_frames = features.shape[0]
            
            # Convert time-based segments to frame indices
            boundaries = []
            for segment in video_data['segments']:
                start_frame = int(segment['start_time'] * self.fps)
                end_frame = int(segment['end_time'] * self.fps)
                
                # Ensure frames are within video length
                start_frame = min(start_frame, total_frames - 1)
                end_frame = min(end_frame, total_frames - 1)
                
                # Mark the frame after each segment as a boundary
                if end_frame < total_frames - 1:
                    boundaries.append(end_frame + 1)
            
            # Create sliding windows
            for start_idx in range(0, total_frames - self.window_size, self.stride):
                end_idx = start_idx + self.window_size
                window_features = features[start_idx:end_idx]
                
                # Create boundary labels for this window
                labels = torch.zeros(self.window_size)
                for boundary in boundaries:
                    if start_idx <= boundary < end_idx:
                        # Set the boundary frame to 1
                        labels[boundary - start_idx] = 1.0
                
                samples.append((window_features, labels, video_id, start_idx))
        
        print(f"Total samples prepared: {len(samples)}")
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        features, labels, video_id, start_idx = self.samples[idx]
        return torch.FloatTensor(features), labels, video_id, start_idx

# Define a temporal action segmentation model
class ActionSegmentationModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_layers=2, dropout=0.5):
        super(ActionSegmentationModel, self).__init__()
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Boundary detection head
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        lstm_out, _ = self.lstm(x)
        boundary_scores = self.boundary_detector(lstm_out).squeeze(-1)
        return boundary_scores

# Training function with early stopping
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        
        for batch_idx, (features, labels, _, _) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss /= len(train_loader)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels, _, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_model_checkpoint.pth')
            print(f"Saved best model checkpoint with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

# Function to post-process predictions and generate final segments
def generate_segments(model, test_loader, threshold=0.5, fps=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Store predictions by video
    video_predictions = {}
    
    with torch.no_grad():
        for features, _, video_ids, start_indices in test_loader:
            features = features.to(device)
            
            outputs = model(features)
            predictions = (outputs > threshold).float().cpu().numpy()
            
            # Group predictions by video
            for i in range(len(video_ids)):
                video_id = video_ids[i]
                start_idx = start_indices[i].item()
                
                if video_id not in video_predictions:
                    video_predictions[video_id] = []
                
                # Store (frame_idx, boundary_score) pairs
                for j in range(len(predictions[i])):
                    frame_idx = start_idx + j
                    score = float(outputs[i, j].cpu().numpy())
                    video_predictions[video_id].append((frame_idx, score))
    
    # Convert boundary predictions to segments
    video_segments = {}
    
    for video_id, predictions in video_predictions.items():
        # Sort and deduplicate predictions by frame index
        predictions_dict = {}
        for frame_idx, score in predictions:
            predictions_dict[frame_idx] = max(score, predictions_dict.get(frame_idx, 0))
        
        predictions = [(frame_idx, score) for frame_idx, score in predictions_dict.items()]
        predictions.sort()
        
        # Find all boundary frames (scores above threshold)
        boundary_frames = [idx for idx, score in predictions if score >= threshold]
        
        # Convert boundary frames to time-based segments
        segments = []
        if boundary_frames:
            start_frame = 0
            
            for boundary in boundary_frames:
                # Convert frames to time
                start_time = start_frame / fps
                end_time = boundary / fps
                
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time
                })
                
                start_frame = boundary + 1
            
            # Add final segment if needed
            if start_frame < predictions[-1][0]:
                start_time = start_frame / fps
                end_time = predictions[-1][0] / fps
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time
                })
        
        video_segments[video_id] = segments
    
    return video_segments

# For evaluation: calculate IoU score
def calculate_iou(pred_segments, gt_segments):
    """
    Calculate IoU between predicted and ground truth segments
    """
    total_iou = 0
    matched_segments = 0
    
    for pred_segment in pred_segments:
        pred_start = pred_segment["start_time"]
        pred_end = pred_segment["end_time"]
        
        best_iou = 0
        for gt_segment in gt_segments:
            gt_start = gt_segment["start_time"]
            gt_end = gt_segment["end_time"]
            
            # Calculate intersection and union
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            
            if union > 0:
                iou = intersection / union
                best_iou = max(best_iou, iou)
        
        if best_iou > 0:
            total_iou += best_iou
            matched_segments += 1
    
    # Average IoU
    avg_iou = total_iou / max(1, matched_segments)
    return avg_iou

# Main function to tie everything together
def main():
    # Path configurations based on your examples
    train_annotation_file = '/home/leiy4/video-segmentation/data/annotations/youcookii_annotations_trainval.json'
    val_annotation_file = '/home/leiy4/video-segmentation/data/annotations/val_annotations.csv'
    test_annotation_file = '/home/leiy4/video-segmentation/data/annotations/test_annotations.csv'
    
    feature_dir = '/home/leiy4/video-segmentation/data/feat_dat'
    raw_video_dir = '/home/leiy4/video-segmentation/data/raw_videos'
    
    # Parameters
    fps = 30
    window_size = 64
    stride = 32
    batch_size = 16
    epochs = 50
    lr = 0.001
    
    # Create datasets
    print("Creating training dataset...")
    train_dataset = YouCook2Dataset(train_annotation_file, feature_dir, fps, window_size, stride)
    print("Creating validation dataset...")
    val_dataset = YouCook2Dataset(val_annotation_file, feature_dir, fps, window_size, stride)
    print("Creating test dataset...")
    test_dataset = YouCook2Dataset(test_annotation_file, feature_dir, fps, window_size, stride)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Check if we have samples
    if len(train_dataset) == 0:
        print("Error: No training samples found. Check paths and data loading!")
        return
    
    # Initialize model
    input_dim = train_dataset.samples[0][0].shape[1] if train_dataset.samples else 2048
    print(f"Feature dimension: {input_dim}")
    model = ActionSegmentationModel(input_dim=input_dim)
    
    # Train model
    model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
    
    # Generate segments
    video_segments = generate_segments(model, test_loader, threshold=0.5, fps=fps)
    
    # Save results
    with open('predicted_segments.json', 'w') as f:
        json.dump(video_segments, f, indent=2)
    
    print(f"Predictions saved to predicted_segments.json")
    
    # Load ground truth and evaluate
    gt_annotations = parse_csv_annotations(test_annotation_file)
    
    # Calculate IoU for each video
    total_iou = 0
    video_count = 0
    
    for video_id, segments in video_segments.items():
        if video_id in gt_annotations:
            gt_segments = gt_annotations[video_id]['segments']
            iou = calculate_iou(segments, gt_segments)
            print(f"Video {video_id}: IoU = {iou:.4f}")
            total_iou += iou
            video_count += 1
    
    if video_count > 0:
        avg_iou = total_iou / video_count
        print(f"Average IoU across all videos: {avg_iou:.4f}")

if __name__ == "__main__":
    main()