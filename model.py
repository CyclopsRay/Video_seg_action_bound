# model.py - TimeSformer model for boundary detection

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange

class PatchEmbed(nn.Module):
    """Convert images into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        x = rearrange(x, '(b t) e h w -> b t e (h w)', b=B, t=T)
        return x

class TimeSformerBoundary(nn.Module):
    """TimeSformer model for boundary detection"""
    def __init__(self, num_frames=16, img_size=224, patch_size=16, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., pretrained=True, dropout=0.1):
        super().__init__()
        
        # Use a pre-trained ViT as backbone
        if pretrained:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            self.vit.head = nn.Identity()  # Remove classification head
        else:
            self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
            self.vit.head = nn.Identity()
        
        # Parameters
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Temporal embedding
        self.time_embed = nn.Parameter(
            torch.zeros(1, num_frames, 1, embed_dim)
        )
        
        # Simplify the model - use fewer temporal attention layers
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(2)  # Reduced from depth//3 to just 2
        ])
        
        # Simplified boundary detection head
        self.boundary_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
            # Remove sigmoid - we'll apply it after aggregation
        )
        
        # Global pooling and final classification
        self.classifier = nn.Sequential(
            nn.Linear(num_frames, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        
        # Process each frame with ViT (frozen)
        with torch.no_grad():  # No gradients needed for the backbone
            frame_features = []
            for t in range(T):
                frame = x[:, :, t, :, :]  # [B, C, H, W]
                features = self.vit.forward_features(frame)  # [B, num_patches+1, embed_dim]
                cls_token = features[:, 0]  # [B, embed_dim]
                frame_features.append(cls_token)
            
            # Stack frame features
            temporal_features = torch.stack(frame_features, dim=1)  # [B, T, embed_dim]
        
        # Add temporal embedding
        temporal_features = temporal_features + self.time_embed.squeeze(2)
        
        # Apply temporal attention
        for temporal_layer in self.temporal_attn:
            # Transpose for attention: [T, B, embed_dim]
            temp_feat = temporal_features.permute(1, 0, 2)
            
            # Self-attention
            temp_feat, _ = temporal_layer(temp_feat, temp_feat, temp_feat)
            
            # Add residual and transpose back: [B, T, embed_dim]
            temporal_features = temporal_features + temp_feat.permute(1, 0, 2)
        
        # Apply boundary detection head to each frame's features
        frame_scores = []
        for t in range(T):
            frame_feat = temporal_features[:, t, :]  # [B, embed_dim]
            score = self.boundary_head(frame_feat)  # [B, 1]
            frame_scores.append(score)
        
        # Stack frame scores
        frame_scores = torch.cat(frame_scores, dim=1)  # [B, T]
        
        # Process sequence information to get final clip-level prediction
        clip_score = self.classifier(frame_scores)
        clip_score = torch.sigmoid(clip_score)  # Apply sigmoid at the end
        
        return clip_score
    
class FeatureBoundaryDetector(nn.Module):
    """Boundary detector using precomputed features"""
    def __init__(self, feature_dim=512, hidden_dim=256, num_frames=16, dropout=0.1):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Temporal embedding
        self.time_embed = nn.Parameter(
            torch.zeros(1, num_frames, hidden_dim)
        )
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Temporal attention modules
        self.temporal_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, 8, dropout=dropout)
            for _ in range(2)
        ])
        
        # Boundary detection head
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_frames, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: [B, T, feature_dim]
        B, T, _ = x.shape
        
        # Project features to hidden dimension
        features = self.feature_proj(x)  # [B, T, hidden_dim]
        
        # Add temporal embedding
        features = features + self.time_embed
        
        # Apply temporal attention
        for temporal_layer in self.temporal_attn:
            # Transpose for attention: [T, B, hidden_dim]
            temp_feat = features.permute(1, 0, 2)
            
            # Self-attention
            temp_feat, _ = temporal_layer(temp_feat, temp_feat, temp_feat)
            
            # Add residual and transpose back: [B, T, hidden_dim]
            features = features + temp_feat.permute(1, 0, 2)
        
        # Apply boundary detection head to each frame
        frame_scores = []
        for t in range(T):
            frame_feat = features[:, t, :]  # [B, hidden_dim]
            score = self.boundary_head(frame_feat)  # [B, 1]
            frame_scores.append(score)
        
        # Stack frame scores
        frame_scores = torch.cat(frame_scores, dim=1)  # [B, T]
        
        # Process sequence information to get clip-level prediction
        clip_score = self.classifier(frame_scores)
        clip_score = torch.sigmoid(clip_score)  # Apply sigmoid at the end
        
        return clip_score