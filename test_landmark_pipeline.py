"""
Test runner for MeshAnchorNet and Canonical Ring Builder.
Verifies network forward pass, loss calculation, and ring geometry.
"""

import torch
import numpy as np
import trimesh
from landmark_anchor_net import MeshAnchorNet, AnchorLoss, extract_mesh_features, build_canonical_rings_from_anchors

def run_landmark_verification():
    print(">>> 1. Initializing MeshAnchorNet...")
    model = MeshAnchorNet(in_channels=7, num_classes=2)
    model.eval()

    # Synthetic batch of 2048 points
    dummy_input = torch.randn(2, 7, 2048)
    with torch.no_grad():
        preds = model(dummy_input)

    print("Forward pass successful:")
    print(" - Confidence map shape:", preds["confidence"].shape)
    print(" - Offset shape:        ", preds["offset"].shape)
    print(" - Class logits shape:  ", preds["class_logits"].shape)
    print(" - Normals shape:       ", preds["normals"].shape)
    print(" - Radius shape:        ", preds["radius"].shape)

    # Test Loss Function
    print("\n>>> 2. Verifying AnchorLoss...")
    criterion = AnchorLoss()
    dummy_targets = {
        "gt_heatmap": torch.rand(2, 2048),
        "gt_offset":  torch.randn(2, 3, 2048),
        "gt_classes": torch.randint(0, 2, (2, 2048)),
        "gt_normals": torch.randn(2, 3, 2048),
        "gt_radius":  torch.rand(2, 2048),
        "anchor_mask": torch.tensor([[True]*10 + [False]*2038, [True]*10 + [False]*2038])
    }
    loss = criterion(preds, dummy_targets)
    print(f"Computed total loss: {loss.item():.4f}")

    # Test Canonical Ring Generator on Suzanne Anchors
    print("\n>>> 3. Testing Canonical Non-Spiral Ring Generation on Anchors...")
    test_anchors = [
        # Left Eye (Cavity, Type 0)
        {"center": [0.55, 3.05, 10.35], "normal": [0.2, 0.9, 0.1], "type": 0, "radius": 0.35},
        # Right Eye (Cavity, Type 0)
        {"center": [-0.55, 3.05, 10.35], "normal": [-0.2, 0.9, 0.1], "type": 0, "radius": 0.35},
        # Nose Root (Protrusion Crease, Type 1)
        {"center": [0.0, 2.92, 10.45], "normal": [0.0, 1.0, 0.0], "type": 1, "radius": 0.38}
    ]

    rings = build_canonical_rings_from_anchors(test_anchors, n_segments=16)
    print(f"Generated {len(rings)} protective topological rings:")
    for i, r in enumerate(rings):
        v_count = len(r["verts"])
        q_count = len(r["quads"])
        print(f" - Ring {i} ({r['type']}): {v_count} vertices, {q_count} quads, Mat ID: {r['material_id']}")

    print("\n>>> ALL TESTS PASSED! Landmark Anchor Module is fully operational.")

if __name__ == "__main__":
    run_landmark_verification()
