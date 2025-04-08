# utils.py - Utility functions

import random
import logging
import torch
import numpy as np
import os
import json
import cv2
from sklearn.metrics import precision_recall_fscore_support

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(name):
    """Setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'{name}.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def get_video_info(video_path):
    """Get video information"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': frame_count / fps
    }

def evaluate_iou(pred_segments, gt_segments, iou_threshold=0.5):
    """
    Calculate IoU metrics between predicted and ground truth segments
    
    Args:
        pred_segments: List of predicted segments (dict with start_time, end_time)
        gt_segments: List of ground truth segments ([start_time, end_time])
        iou_threshold: IoU threshold for positive detection
    
    Returns:
        metrics: Dict with precision, recall, F1, and mIoU
    """
    # Handle empty cases
    if not pred_segments and not gt_segments:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'miou': 1.0
        }
    
    if not pred_segments:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'miou': 0.0
        }
    
    if not gt_segments:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'miou': 0.0
        }
    
    # Calculate IoU for each prediction
    ious = []
    matched_gt = set()
    
    for pred in pred_segments:
        pred_start = pred['start_time']
        pred_end = pred['end_time']
        
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_segments):
            gt_start = gt[0]
            gt_end = gt[1]
            
            # Calculate intersection and union
            intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
            union = max(pred_end, gt_end) - min(pred_start, gt_start)
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        ious.append(best_iou)
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
    
    # Calculate metrics
    tp = len(matched_gt)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    miou = sum(ious) / max(len(ious), 1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou
    }

def evaluate_boundaries(pred_boundaries, gt_boundaries, tolerance=5):
    """
    Evaluate boundary detection with a temporal tolerance
    
    Args:
        pred_boundaries: List of predicted boundary frames
        gt_boundaries: List of ground truth boundary frames
        tolerance: Number of frames tolerance for correct detection
    
    Returns:
        metrics: Dict with precision, recall, F1
    """
    # Handle empty cases
    if not pred_boundaries and not gt_boundaries:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        }
    
    if not pred_boundaries:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    if not gt_boundaries:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    # Find matches
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(pred_boundaries):
        for j, gt in enumerate(gt_boundaries):
            if abs(pred - gt) <= tolerance and j not in matched_gt:
                matched_gt.add(j)
                matched_pred.add(i)
                break
    
    # Calculate metrics
    tp = len(matched_pred)
    fp = len(pred_boundaries) - tp
    fn = len(gt_boundaries) - tp
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def visualize_predictions(video_path, pred_segments, gt_segments, output_path, fps=30):
    """
    Create a visualization of predicted and ground truth segments
    
    Args:
        video_path: Path to the video
        pred_segments: List of predicted segments (dict with start_time, end_time)
        gt_segments: List of ground truth segments ([start_time, end_time])
        output_path: Path to save the visualization
        fps: Frames per second
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 100))
    
    # Convert segments to frame ranges
    pred_ranges = []
    for segment in pred_segments:
        start_frame = int(segment['start_time'] * fps)
        end_frame = int(segment['end_time'] * fps)
        pred_ranges.append((start_frame, end_frame))
    
    gt_ranges = []
    for segment in gt_segments:
        start_frame = int(segment[0] * fps)
        end_frame = int(segment[1] * fps)
        gt_ranges.append((start_frame, end_frame))
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create visualization area
        vis_area = np.zeros((100, width, 3), dtype=np.uint8)
        
        # Draw timeline
        cv2.line(vis_area, (0, 25), (width, 25), (200, 200, 200), 2)
        cv2.line(vis_area, (0, 75), (width, 75), (200, 200, 200), 2)
        
        # Draw predicted segments
        for start, end in pred_ranges:
            if start <= frame_idx <= end:
                # Current segment - bright color
                color = (0, 255, 0)
            else:
                # Other segments - darker color
                color = (0, 100, 0)
            
            # Map frame indices to pixel positions
            start_pos = int(start / total_frames * width)
            end_pos = int(end / total_frames * width)
            cv2.rectangle(vis_area, (start_pos, 15), (end_pos, 35), color, -1)
        
        # Draw ground truth segments
        for start, end in gt_ranges:
            if start <= frame_idx <= end:
                # Current segment - bright color
                color = (255, 0, 0)
            else:
                # Other segments - darker color
                color = (100, 0, 0)
            
            # Map frame indices to pixel positions
            start_pos = int(start / total_frames * width)
            end_pos = int(end / total_frames * width)
            cv2.rectangle(vis_area, (start_pos, 65), (end_pos, 85), color, -1)
        
        # Add current frame marker
        pos = int(frame_idx / total_frames * width)
        cv2.line(vis_area, (pos, 0), (pos, 100), (255, 255, 0), 2)
        
        # Add text
        cv2.putText(vis_area, "Predictions", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_area, "Ground Truth", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add timestamp
        time_str = f"Time: {frame_idx/fps:.2f}s"
        cv2.putText(vis_area, time_str, (width-150, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine frame and visualization
        combined = np.vstack([frame, vis_area])
        
        # Write to output
        out.write(combined)
    
    # Release resources
    cap.release()
    out.release()
    
    return True

def smooth_predictions(scores, window_size=5):
    """
    Apply temporal smoothing to boundary scores
    
    Args:
        scores: List of (frame_idx, score) tuples
        window_size: Size of the smoothing window
    
    Returns:
        smoothed_scores: List of (frame_idx, smoothed_score) tuples
    """
    # Sort by frame index
    scores.sort(key=lambda x: x[0])
    
    # Extract frame indices and scores
    frame_indices, score_values = zip(*scores)
    
    # Convert to numpy array for easier processing
    score_array = np.array(score_values)
    
    # Apply smoothing
    smoothed = np.zeros_like(score_array)
    for i in range(len(score_array)):
        start = max(0, i - window_size // 2)
        end = min(len(score_array), i + window_size // 2 + 1)
        smoothed[i] = np.mean(score_array[start:end])
    
    # Combine with frame indices
    smoothed_scores = list(zip(frame_indices, smoothed))
    
    return smoothed_scores

def save_evaluation_results(metrics, output_file):
    """
    Save evaluation metrics to file
    
    Args:
        metrics: Dictionary of metrics
        output_file: Path to save results
    """
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_groundtruth(csv_file, video_id=None):
    """
    Load ground truth segments from CSV file
    
    Args:
        csv_file: Path to CSV file
        video_id: Optional filter for specific video
    
    Returns:
        Dictionary mapping video IDs to segment lists
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file)
    result = {}
    
    for _, row in df.iterrows():
        if video_id and row['youtube_id'] != video_id:
            continue
        
        vid = row['youtube_id']
        segment_str = row['segment'].strip('[]')
        start_time, end_time = map(float, segment_str.split('.'))
        
        if vid not in result:
            result[vid] = []
        
        result[vid].append([start_time, end_time])
    
    return result

def create_submission_file(video_segments, output_file):
    """
    Create submission file in the required format
    
    Args:
        video_segments: Dictionary mapping video IDs to segment lists
        output_file: Path to save submission
    """
    with open(output_file, 'w') as f:
        f.write("video_id,segment\n")
        
        for video_id, segments in video_segments.items():
            for i, segment in enumerate(segments):
                start_time = segment['start_time']
                end_time = segment['end_time']
                f.write(f"{video_id}_{i},[{start_time:.1f}. {end_time:.1f}.]\n")