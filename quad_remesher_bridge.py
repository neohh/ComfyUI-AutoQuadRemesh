"""
QuadRemesher Standalone Engine Bridge & Canonical Loop Guides
Module supporting closed circular loops, canonical 8-ring protrusions, and headless xremesh execution.
"""

import os
import subprocess
import numpy as np

def run_quad_remesher(
    input_fbx_path: str,
    output_fbx_path: str,
    engine_dir: str = r"C:\ProgramData\Exoside\QuadRemesher\Datas_Blender\QuadRemesherEngine_1.2",
    target_quad_count: int = 2800,
    curvature_adaptiveness: int = 0,
    use_materials: bool = True,
    sym_axis: str = "X"
) -> bool:
    """
    Executes Quad Remesher standalone engine (xremesh.exe) with guided parameters.
    """
    exe_path = os.path.join(engine_dir, "xremesh.exe" if os.name == "nt" else "xremesh")
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"Quad Remesher engine not found at: {exe_path}")

    prog_file = output_fbx_path + ".prog.txt"
    settings_file = output_fbx_path + ".settings.txt"

    settings = f"""HostApp=Blender
FileIn="{input_fbx_path}"
FileOut="{output_fbx_path}"
ProgressFile="{prog_file}"
TargetQuadCount={target_quad_count}
CurvatureAdaptivness={curvature_adaptiveness}
ExactQuadCount=1
UseMaterialIds={1 if use_materials else 0}
"""
    if sym_axis:
        settings += f"SymAxis={sym_axis}\nSymLocal=1\n"

    with open(settings_file, "w", encoding="utf-8") as f:
        f.write(settings)

    res = subprocess.run([exe_path, "-s", settings_file], cwd=engine_dir)
    return res.returncode == 0 and os.path.exists(output_fbx_path)

def create_canonical_8edge_cylinder(
    base_center: np.ndarray,
    tip_center: np.ndarray,
    radius_base: float,
    radius_tip: float,
    num_rings: int = 6
):
    """
    Constructs a canonical 8-edge cylinder (octagonal cross-section)
    with a closed 4-quad cap at the tip, matching facial & finger topology standards.
    """
    vec = tip_center - base_center
    length = np.linalg.norm(vec)
    d = vec / length

    # Orthonormal basis
    arbitrary = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)

    angles = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
    radii = np.linspace(radius_base, radius_tip, num_rings)

    verts = []
    quads = []
    ring_indices = []

    for r in range(num_rings):
        frac = r / (num_rings - 1) * 0.88
        c = base_center + frac * vec
        cur_ring = []
        for a in angles:
            pt = c + (np.cos(a) * u + np.sin(a) * v) * radii[r]
            cur_ring.append(len(verts))
            verts.append(pt)
        ring_indices.append(cur_ring)

        if r > 0:
            for i in range(8):
                quads.append([
                    ring_indices[r-1][i],
                    ring_indices[r-1][(i+1)%8],
                    ring_indices[r][(i+1)%8],
                    ring_indices[r][i]
                ])

    # Tip quad dome & 4-quad cap
    c_dome = base_center + 0.94 * vec
    rad_dome = radius_tip * 0.65
    dome_ring = []
    for a in angles:
        pt = c_dome + (np.cos(a) * u + np.sin(a) * v) * rad_dome
        dome_ring.append(len(verts))
        verts.append(pt)

    last_shaft = ring_indices[-1]
    for i in range(8):
        quads.append([last_shaft[i], last_shaft[(i+1)%8], dome_ring[(i+1)%8], dome_ring[i]])

    c_cap = tip_center - d * (length * 0.02)
    r_cap = radius_tip * 0.32
    cap_center = []
    for a in [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
        pt = c_cap + (np.cos(a) * u + np.sin(a) * v) * r_cap
        cap_center.append(len(verts))
        verts.append(pt)

    quads.append([cap_center[0], cap_center[1], cap_center[2], cap_center[3]])
    quads.append([dome_ring[0], dome_ring[1], cap_center[0], dome_ring[7]])
    quads.append([dome_ring[1], dome_ring[2], cap_center[1], cap_center[0]])
    quads.append([dome_ring[2], dome_ring[3], dome_ring[4], cap_center[1]])
    quads.append([dome_ring[4], dome_ring[5], cap_center[2], cap_center[1]])
    quads.append([dome_ring[5], dome_ring[6], cap_center[3], cap_center[2]])
    quads.append([dome_ring[6], dome_ring[7], dome_ring[0], cap_center[3]])

    return np.array(verts, dtype=np.float32), np.array(quads, dtype=np.int32)
