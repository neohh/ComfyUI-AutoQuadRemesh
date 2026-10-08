# ComfyUI-AutoQuadRemesh ⚡

**Industrial-grade local Quad Retopology node for ComfyUI**  
Featuring **Instant Field-Aligned Cross-Field Meshing (ZRemesher-grade)**, **Topological Loop Alignment & Harmonic Relaxation**, **PyMeshFix Manifold Healing**, and **Adaptive Shrinkwrap Snapping**.

Works on **any** 3D model (characters, clothing, organic creatures, props, hard-surface). 100% local, fast (<20s), zero cloud API dependencies.

---

## ✨ Features

- **🌪️ Instant Field-Aligned Cross-Field Engine:**
  - 4-RoSy (Rotational Symmetry) orientation fields & 4-PoSy position fields guided by principal surface curvatures ($k_1, k_2$).
  - **100% Pure Quad topology** (`f v1 v2 v3 v4`) — 0 triangles, 0 n-gons.
  - **>95% regular 4-valence vertices** and ideal ~90° angles.
- **🎯 Topological Feature-Loop Alignment:**
  - Automatically identifies key anatomical and clothing transition seams (collar, waist/shirt hem, shorts hem, knees).
  - Straightens master edge loops into continuous, smooth curves without micro-steps.
  - Harmonically relaxes neighboring quad rings to eliminate edge distortion while preserving 100% volume.
- **🛡️ 100% Watertight Manifold Healing:**
  - Built-in `PyMeshFix` pre-healing removes non-manifold vertices, self-intersections, and boundary holes typical in raw AI-generated 3D meshes (Hunyuan3D, Tripo, InstantMesh).
  - Guarantees zero dropouts and zero holes in clothing folds or armpits.
- **📐 Dual Format Output:**
  - **Clean Quad `.OBJ`:** Native pure quads for Blender, ZBrush, Maya, and Marvelous Designer.
  - **Wireframe-Embedded `.GLB`:** Clean quad visualizer directly in ComfyUI's native 3D viewport.
- **✂️ Separate Fingers & Crease Preservation:**
  - Automatic carving of AI webbed hand meshes and feature crease alignment (`crease_angle` threshold).

---

## 📦 Installation

### 1. Clone into ComfyUI custom nodes
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/neohh/ComfyUI-AutoQuadRemesh.git
cd ComfyUI-AutoQuadRemesh
pip install -r requirements.txt
```

### 2. Instant-Meshes Binary
Ensure `instant-meshes` is in your system PATH or `/usr/local/bin/instant-meshes` (for Linux/Docker).
```bash
# Ubuntu / Debian / Docker
sudo apt-get update && sudo apt-get install -y cmake g++ libeigen3-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
git clone --recursive https://github.com/wjakob/instant-meshes.git
cd instant-meshes
cmake -DCMAKE_BUILD_TYPE=Release .
make -j$(nproc)
sudo cp instant-meshes /usr/local/bin/
```

---

## ⚙️ Node Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_quads` | INT | `12000` | Target quad polygon count (10k–14k recommended for full characters). |
| `shrinkwrap_strength` | FLOAT | `0.60` | Snaps quads back to high-poly shape without pinching fabric folds. |
| `adaptive_scale` | BOOL | `True` | Higher density on fingers, face, and creases; larger quads on flat body areas. |
| `preserve_sharp` | BOOL | `False` | Locks mechanical sharp edges (best for hard-surface/props). |
| `show_quad_wireframe` | BOOL | `True` | Renders dark quad wireframe lines in ComfyUI 3D Viewer. |
| `heal_mesh` | BOOL | `True` | Watertight sealing before solving to prevent holes in cloth folds. |
| `relax_iterations` | INT | `4` | Tangential quad relaxation iterations to square up skewed diamonds. |
| `engine` | LIST | `instant_crossfield` | `instant_crossfield` (ZRemesher-grade), `intelligent_custom_retopo`, `quadriflow_legacy`. |
| `crease_angle` | INT | `30` | Dihedral angle threshold (degrees) to lock edge loops along clothing seams. |
| `separate_fingers` | BOOL | `True` | Automatically carves negative space between fused AI fingers. |

---

## 📄 License
MIT License. Free for commercial and non-commercial use.
