"""
GeometricLoopDetector: Fully Automatic Form-Transition Loop Extraction (v2.0)
Part of ComfyUI-AutoQuadRemesh (Phase 1)

Implements the multi-stage form-transition pipeline:
Step 1: Island & Branch Isolation (Extremity detection & isolated sub-graph walking)
Step 2: Bifurcation Root Detection (Cross-section perimeter S(d) & bottleneck collar)
Step 3: Anisotropic Basin Sockets (Mean Curvature H=0 zero-crossing & closed-loop validation)
Step 4: Arc-Length Resampling & Frenet-Frame Quad Ribbon Extrusion (0 non-manifold edges)
"""

from .geometric_loop_detector_v2 import GeometricLoopDetectorV2 as GeometricLoopDetector

__all__ = ["GeometricLoopDetector"]
