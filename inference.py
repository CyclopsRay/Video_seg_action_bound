# inference.py - Inference functions

import os
import json
import torch
import numpy as np
from tqdm import tqdm

def predict_boundaries(model, test_loader, test_dataset, boundary_threshold, output_dir, logger):
    """Run inference on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Store predictions by video
    video_predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            frames = batch['frames'].to(device)
            video_ids = batch['video_id']
            start_frames = batch['start_frame']
            
            # Get predictions
            outputs = model(frames)
            
            # Store predictions
            for i in range(len(video_ids)):
                video_id = video_ids[i]
                start_frame = start_frames[i].item()
                
                if video_id not in video_predictions:
                    video_predictions[video_id] = []
                
                # Store (frame_idx, score) for each frame
                for j in range(outputs.size(1)):
                    frame_idx = start_frame + j
                    score = outputs[i, j].item()
                    video_predictions[video_id].append((frame_idx, score))
    
    # Process predictions to get final segments
    video_segments = {}
    for video_id, predictions in video_predictions.items():
        # Sort by frame index and handle duplicates
        frame_scores = {}
        for frame_idx, score in predictions:
            if frame_idx not in frame_scores:
                frame_scores[frame_idx] = []
            frame_scores[frame_idx].append(score)
        
        # Average scores for frames with multiple predictions
        frame_scores = {
            frame_idx: np.mean(scores) 
            for frame_idx, scores in frame_scores.items()
        }
        
        # Sort by frame index
        sorted_scores = sorted(frame_scores.items())
        
        # Find peaks (local maxima) above threshold
        boundaries = []
        for i in range(1, len(sorted_scores)-1):
            curr_idx, curr_score = sorted_scores[i]
            prev_idx, prev_score = sorted_scores[i-1]
            next_idx, next_score = sorted_scores[i+1]
            
            # Check if it's a peak and above threshold
            if (curr_score > boundary_threshold and 
                curr_score > prev_score and 
                curr_score >= next_score):
                boundaries.append(curr_idx)
        
        # Convert boundaries to segments
        segments = []
        fps = test_dataset.fps
        
        if boundaries:
            # First segment starts at frame 0
            start_frame = 0
            
            for boundary in boundaries:
                # Convert to seconds
                start_time = start_frame / fps
                end_time = boundary / fps
                
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time
                })
                
                # Update start frame for next segment
                start_frame = boundary + 1
            
            # Add final segment if needed
            if sorted_scores:
                last_frame = sorted_scores[-1][0]
                if start_frame < last_frame:
                    segments.append({
                        'start_time': start_frame / fps,
                        'end_time': last_frame / fps
                    })
        
        video_segments[video_id] = segments
    
    # Save predictions
    output_file = os.path.join(output_dir, 'predicted_segments.json')
    with open(output_file, 'w') as f:
        json.dump(video_segments, f, indent=2)
    
    logger.info(f"Saved predictions to {output_file}")
    
    return video_segments