import os
import sys
import time
import cv2
import torch
import trimesh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.append("/opt/comfyui")
sys.path.append("/opt/comfyui/custom_nodes/ComfyUI-AutoQuadRemesh")
from custom_quad_engine import generate_structured_quad_character
import custom_nodes.comfyui_controlnet_aux as aux

t0 = time.time()
print(">>> Running Custom Quad Retopology Generator Test (v3)...")

raw_path = "/opt/comfyui/output/stages_data/02_raw_mesh/asset_raw_00005_.glb"
target_mesh = trimesh.load(raw_path, force="mesh")

img_np = cv2.imread("/opt/comfyui/output/test_dwpose_in.png")
t_img = torch.from_numpy(img_np[:, :, ::-1].copy()).float() / 255.0
t_img = t_img.unsqueeze(0)

dw = aux.NODE_CLASS_MAPPINGS["DWPreprocessor"]()
res = dw.estimate_pose(
    image=t_img,
    detect_hand="enable",
    detect_body="enable",
    detect_face="enable",
    resolution=512,
    bbox_detector="None",
    pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt"
)
p_info = res["result"][1][0]["people"][0]
dwpose_kps = {
    "body": np.array(p_info["pose_keypoints_2d"]).reshape(-1, 3),
    "hand_left": np.array(p_info["hand_left_keypoints_2d"]).reshape(-1, 3),
    "hand_right": np.array(p_info["hand_right_keypoints_2d"]).reshape(-1, 3)
}

clothing_loops = {
    "Collar": 0.582,
    "Shirt Hem": 0.073,
    "Shorts Waistband": -0.002,
    "Shorts Hem": -0.189
}

snap_verts, final_quads, part_ranges = generate_structured_quad_character(target_mesh, dwpose_kps, clothing_loops)
print(f"\n>>> Generation finished in {round(time.time() - t0, 2)}s!")
print(f"    Total Quads:    {len(final_quads)} (100% PURE QUADS)")
print(f"    Total Vertices: {len(snap_verts)}")

out_obj = "/opt/comfyui/output/stages_data/03_quad_mesh/asset_quad_custom_engine.obj"
os.makedirs(os.path.dirname(out_obj), exist_ok=True)
with open(out_obj, "w") as f:
    f.write("# Intelligent Pure Quad Retopology Engine\n")
    for v in snap_verts:
        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for q in final_quads:
        f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
print(f"Saved pure quad OBJ: {out_obj}")

# Diagnostic wireframe render
fig, axes = plt.subplots(1, 3, figsize=(22, 10), dpi=150)
fig.patch.set_facecolor("#111115")

def get_quad_lines(verts, quads):
    lines = []
    for q in quads:
        for i in range(4):
            v0 = verts[q[i]]
            v1 = verts[q[(i+1)%4]]
            lines.append([[v0[0], v0[1]], [v1[0], v1[1]]])
    return lines

# View 1: Full Body Front with Anatomical Color Coding
ax1 = axes[0]
ax1.set_facecolor("#16161a")

color_map = {
    "Torso": "#00ffcc",
    "Waist_Transition": "#33ccff",
    "Shorts_Pelvis": "#ffaa00",
    "Left_Shorts_Cuff": "#ffcc00",
    "Right_Shorts_Cuff": "#ffcc00",
    "Left_Leg": "#00ff88",
    "Right_Leg": "#00ff88",
    "Neck": "#cc66ff",
    "Head": "#aa44ff",
    "Left_Arm": "#ff6699",
    "Right_Arm": "#ff6699",
}

for name, (s, e) in part_ranges.items():
    col = color_map.get(name, "#ff3333") if not "Left_" in name and not "Right_" in name else color_map.get(name, "#ff5555")
    if "Thumb" in name or "Index" in name or "Middle" in name or "Ring" in name or "Pinky" in name:
        col = "#ffff33"
    lines = get_quad_lines(snap_verts, final_quads[s:e])
    lc = LineCollection(lines, colors=col, linewidths=0.6, alpha=0.9)
    ax1.add_collection(lc)

ax1.set_title("Full Body Quad Retopology (100% Pure Quads)", color="#ffffff", fontsize=14, pad=10)
ax1.set_aspect("equal")
ax1.set_xlim(snap_verts[:, 0].min() - 0.05, snap_verts[:, 0].max() + 0.05)
ax1.set_ylim(snap_verts[:, 1].min() - 0.05, snap_verts[:, 1].max() + 0.05)
ax1.axis("off")

# View 2: Shorts & Torso Seams Zoom
ax2 = axes[1]
ax2.set_facecolor("#16161a")
for p_name in ["Torso", "Waist_Transition", "Shorts_Pelvis", "Left_Shorts_Cuff", "Right_Shorts_Cuff"]:
    if p_name in part_ranges:
        s, e = part_ranges[p_name]
        col = color_map.get(p_name, "#ffffff")
        lines = get_quad_lines(snap_verts, final_quads[s:e])
        lc = LineCollection(lines, colors=col, linewidths=0.8, alpha=0.95)
        ax2.add_collection(lc)

ax2.axhline(0.582, color="#ff3366", linestyle="--", linewidth=1.5, label="Collar (Y=0.582)")
ax2.axhline(0.073, color="#ff3366", linestyle="--", linewidth=1.5, label="Shirt Hem (Y=0.073)")
ax2.axhline(-0.002, color="#33ccff", linestyle="--", linewidth=1.5, label="Shorts Waist (Y=-0.002)")
ax2.axhline(-0.189, color="#33ccff", linestyle="--", linewidth=1.5, label="Shorts Hem (Y=-0.189)")
ax2.legend(loc="upper right", facecolor="#222228", edgecolor="none", labelcolor="#ffffff", fontsize=9)
ax2.set_title("Clothing Boundary Loops (Collar, Waist, Hems)", color="#ffffff", fontsize=14, pad=10)
ax2.set_aspect("equal")
ax2.set_xlim(-0.30, 0.30)
ax2.set_ylim(-0.25, 0.65)
ax2.axis("off")

# View 3: Left Hand Zoom (Only 5 Finger Cylinders)
ax3 = axes[2]
ax3.set_facecolor("#16161a")
finger_colors = ["#ff3333", "#ff9933", "#ffff33", "#33ff33", "#33ccff"]
f_names = ["Left_Thumb", "Left_Index", "Left_Middle", "Left_Ring", "Left_Pinky"]
for f_name, f_col in zip(f_names, finger_colors):
    if f_name in part_ranges:
        s, e = part_ranges[f_name]
        lines = get_quad_lines(snap_verts, final_quads[s:e])
        lc = LineCollection(lines, colors=f_col, linewidths=1.2, alpha=0.95, label=f_name.replace("Left_", ""))
        ax3.add_collection(lc)

ax3.legend(loc="upper right", facecolor="#222228", edgecolor="none", labelcolor="#ffffff", fontsize=9)
ax3.set_title("Left Hand: 5 Separated Quad Finger Cylinders", color="#ffffff", fontsize=14, pad=10)
ax3.set_aspect("equal")
ax3.set_xlim(0.24, 0.33)
ax3.set_ylim(-0.15, 0.03)
ax3.axis("off")

plt.tight_layout()
out_vis = "/opt/comfyui/output/custom_engine_retopo_vis.png"
plt.savefig(out_vis, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"Diagnostic render saved: {out_vis}")
