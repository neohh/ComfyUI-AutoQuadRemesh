"""
Landmark & Anchor Prediction Network for 3D Mesh Remeshing
Implements:
1. MeshAnchorNet: Deep Neural Network predicting semantic anchors (cavities vs protrusion roots).
2. CurvatureFeatureExtractor: Extracts [x, y, z, nx, ny, nz, H] from any 3D mesh.
3. CanonicalRingGenerator: Mathematically generates non-spiral closed rings (Delta v = 0) and 8-edge crease belts.
4. ZeroShotGeometricDetector: Immediate analytical fall-back detector based on Morse & differential geometry.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from scipy.spatial import cKDTree

# -------------------------------------------------------------
# 1. Neural Network Architecture
# -------------------------------------------------------------
class MeshAnchorNet(nn.Module):
    """
    Lightweight Anchor & Feature Landmark Predictor for 3D Meshes.
    Inputs: (Batch, 7, N) -> [x, y, z, nx, ny, nz, Curvature H]
    Outputs:
      - confidence: (B, N) heatmap of anchor likelihood
      - offset:     (B, 3, N) shift (dx, dy, dz) to exact feature center
      - class_logits: (B, 2, N) 0: Cavity/Eye, 1: Protrusion Root/Nose/Ear
      - normals:    (B, 3, N) orientation unit vector (sight / symmetry axis)
      - radius:     (B, N) characteristic loop radius
    """
    def __init__(self, in_channels=7, num_classes=2):
        super(MeshAnchorNet, self).__init__()
        
        # Backbone Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global shape context fusion
        self.fusion_conv = nn.Conv1d(512 + 512, 256, 1)
        self.bn_fusion = nn.BatchNorm1d(256)
        
        # Prediction Heads
        self.head_confidence = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Sigmoid()
        )
        self.head_offset = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.head_class = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, num_classes, 1)
        )
        self.head_normal = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
        self.head_radius = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 1, 1),
            nn.Softplus()
        )

    def forward(self, x):
        B, C, N = x.shape
        
        feat1 = F.relu(self.bn1(self.conv1(x)))
        feat2 = F.relu(self.bn2(self.conv2(feat1)))
        feat3 = F.relu(self.bn3(self.conv3(feat2)))
        feat4 = F.relu(self.bn4(self.conv4(feat3)))
        
        # Max-pool global shape descriptor
        global_feat = torch.max(feat4, 2, keepdim=True)[0].repeat(1, 1, N)
        combined = torch.cat([feat4, global_feat], dim=1)
        latent = F.relu(self.bn_fusion(self.fusion_conv(combined)))
        
        return {
            "confidence": self.head_confidence(latent).squeeze(1),
            "offset": self.head_offset(latent),
            "class_logits": self.head_class(latent),
            "normals": F.normalize(self.head_normal(latent), p=2, dim=1),
            "radius": self.head_radius(latent).squeeze(1)
        }

# -------------------------------------------------------------
# 2. Combined Loss Function
# -------------------------------------------------------------
class AnchorLoss(nn.Module):
    def __init__(self, w_conf=1.0, w_pos=5.0, w_cls=1.0, w_norm=2.0, w_rad=1.0):
        super(AnchorLoss, self).__init__()
        self.w_conf = w_conf
        self.w_pos = w_pos
        self.w_cls = w_cls
        self.w_norm = w_norm
        self.w_rad = w_rad
        
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        mask = targets["anchor_mask"]
        loss_conf = self.bce(preds["confidence"], targets["gt_heatmap"])
        
        if mask.sum() > 0:
            pred_offset_masked = preds["offset"].permute(0, 2, 1)[mask]
            gt_offset_masked = targets["gt_offset"].permute(0, 2, 1)[mask]
            loss_pos = self.smooth_l1(pred_offset_masked, gt_offset_masked)
            
            pred_cls_masked = preds["class_logits"].permute(0, 2, 1)[mask]
            gt_cls_masked = targets["gt_classes"][mask]
            loss_cls = self.ce(pred_cls_masked, gt_cls_masked)
            
            pred_norm_masked = preds["normals"].permute(0, 2, 1)[mask]
            gt_norm_masked = targets["gt_normals"].permute(0, 2, 1)[mask]
            loss_norm = (1.0 - (pred_norm_masked * gt_norm_masked).sum(dim=-1)).mean()
            
            loss_rad = self.smooth_l1(preds["radius"][mask], targets["gt_radius"][mask])
        else:
            loss_pos = torch.tensor(0.0, device=preds["confidence"].device)
            loss_cls = torch.tensor(0.0, device=preds["confidence"].device)
            loss_norm = torch.tensor(0.0, device=preds["confidence"].device)
            loss_rad = torch.tensor(0.0, device=preds["confidence"].device)

        return (self.w_conf * loss_conf + 
                self.w_pos * loss_pos + 
                self.w_cls * loss_cls + 
                self.w_norm * loss_norm + 
                self.w_rad * loss_rad)

# -------------------------------------------------------------
# 3. Mesh Feature Extraction
# -------------------------------------------------------------
def extract_mesh_features(mesh: trimesh.Trimesh, num_samples: int = 4096):
    """
    Samples N points with normals and discrete mean curvature H.
    Returns: (7, N) numpy array
    """
    pts, face_indices = mesh.sample(num_samples, return_index=True)
    normals = mesh.face_normals[face_indices]
    
    # Estimate mean curvature via normal variation
    tree = cKDTree(pts)
    dists, idxs = tree.query(pts, k=8)
    curvatures = []
    for i in range(len(pts)):
        neighbor_normals = normals[idxs[i]]
        diff = np.linalg.norm(neighbor_normals - normals[i], axis=1)
        curvatures.append(diff.mean())
    curvatures = np.array(curvatures, dtype=np.float32)
    curvatures = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-6)
    
    features = np.hstack([pts, normals, curvatures[:, None]]).T.astype(np.float32)
    return features

# -------------------------------------------------------------
# 4. Canonical Procedural Ring Generator (Delta v = 0)
# -------------------------------------------------------------
def build_canonical_rings_from_anchors(anchors, n_segments=16):
    """
    Takes detected anchors and generates clean, non-spiral protective rings.
    cls_type 0: Cavity/Eye -> Annular Donut Ring with Delta v = 0.
    cls_type 1: Protrusion Root -> 8-edge Crease Belt.
    """
    generated_rings = []
    
    for anc in anchors:
        C = np.array(anc["center"], dtype=np.float64)
        N = np.array(anc["normal"], dtype=np.float64)
        norm_len = np.linalg.norm(N)
        N = N / norm_len if norm_len > 1e-6 else np.array([0.0, 1.0, 0.0])
        cls_type = int(anc["type"])
        R = float(anc["radius"])
        
        temp = np.array([0.0, 1.0, 0.0]) if abs(N[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(N, temp); u /= np.linalg.norm(u)
        v = np.cross(N, u)
        
        if cls_type == 0:
            # Cavity: Annular Donut Band
            r_in = R * 0.70
            r_out = R * 1.30
            angles = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
            
            ring_verts = []
            ring_quads = []
            for a in angles:
                dir_v = np.cos(a) * u + np.sin(a) * v
                ring_verts.append(C + dir_v * r_in)
                ring_verts.append(C + dir_v * r_out)
                
            for i in range(n_segments):
                nxt = (i + 1) % n_segments
                ring_quads.append([2*i, 2*i+1, 2*nxt+1, 2*nxt])
                
            generated_rings.append({
                "type": "cavity_donut",
                "center": C,
                "verts": np.array(ring_verts, dtype=np.float32),
                "quads": np.array(ring_quads, dtype=np.int32),
                "material_id": 2
            })
            
        elif cls_type == 1:
            # Protrusion Root: Canonical 8-edge Crease Belt
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            ring_verts = []
            ring_quads = []
            
            for h in [-0.06 * R, 0.06 * R]:
                for a in angles:
                    dir_v = np.cos(a) * u + np.sin(a) * v
                    ring_verts.append(C + dir_v * R + N * h)
                    
            for i in range(8):
                nxt = (i + 1) % 8
                ring_quads.append([i, nxt, 8 + nxt, 8 + i])
                
            generated_rings.append({
                "type": "root_belt",
                "center": C,
                "verts": np.array(ring_verts, dtype=np.float32),
                "quads": np.array(ring_quads, dtype=np.int32),
                "material_id": 1
            })
            
    return generated_rings
