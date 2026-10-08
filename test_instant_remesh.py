import os
import subprocess
import time
import trimesh

print(">>> Testing Instant Meshes Cross-Field Quadrangulator...")
t0 = time.time()

# 1. Load raw mesh and export to PLY
raw_path = "/opt/comfyui/output/stages_data/02_raw_mesh/asset_raw_00005_.glb"
target_mesh = trimesh.load(raw_path, force="mesh")
ply_in = "/tmp/raw_for_instant.ply"
target_mesh.export(ply_in)
print(f"Exported {ply_in}: {len(target_mesh.vertices)} verts, {len(target_mesh.faces)} faces.")

# 2. Run instant-meshes CLI
out_obj = "/opt/comfyui/output/test_instant_mesh_12k.obj"
cmd = [
    "/usr/local/bin/instant-meshes",
    "-r", "4",
    "-p", "4",
    "-f", "12000",
    "-c", "40",
    "-S", "2",
    "-d",
    "-o", out_obj,
    ply_in
]
print(f"Running command: {' '.join(cmd)}")
res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print("STDOUT:\n", res.stdout)
if res.stderr:
    print("STDERR:\n", res.stderr)

elapsed = round(time.time() - t0, 2)
print(f"Finished in {elapsed}s! Output: {out_obj}")
