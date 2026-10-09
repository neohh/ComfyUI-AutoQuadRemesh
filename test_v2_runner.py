import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from geometric_loop_detector_v2 import GeometricLoopDetectorV2

detector = GeometricLoopDetectorV2("/opt/comfyui/output/stages_data/03_quad_mesh/сюзанна.obj")
loops = detector.generate_all_loops()
verts, quads = detector.build_quad_mesh(loops)

# Export OBJ
out_obj = "/opt/comfyui/output/stages_data/03_quad_mesh/suzanne_v2_detector_loops.obj"
with open(out_obj, 'w') as f:
    f.write('g suzanne_base\n')
    for v in detector.vertices:
        f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
    for face in detector.faces:
        f.write(f'f {face[0]+1} {face[1]+1} {face[2]+1}\n')
    f.write('g quad_loops_v2\n')
    for v in verts[len(detector.vertices):]:
        f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
    for q in quads:
        f.write(f'f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n')

print(f"Exported to {out_obj}")

# Render shaded image
fig, axes = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': '3d'}, facecolor='#0d1117')
views = [
    (35, -125, '3/4 Слева (Изолированные ухо и глаз)'),
    (35, -90, 'Фронтальный Фас (v2.0 Раздельные маски)'),
    (35, -55, '3/4 Справа (Изолированные ухо и глаз)')
]

light = np.array([0.1, -0.85, 0.75])
light /= np.linalg.norm(light)

def shade(pts, base_color):
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm > 1e-6: n /= norm
    intensity = 0.45 + 0.55 * max(0, np.dot(n, light))
    return np.array(base_color) * intensity

b_polys = [detector.vertices[f] for f in detector.faces[::3]]
b_cols = [shade(p, [0.22, 0.28, 0.36]) for p in b_polys]

r_polys = [verts[q] for q in quads]
r_cols = [shade(p, [1.0, 0.08, 0.18]) for p in r_polys]

c = detector.vertices.mean(axis=0)

for ax, (elev, azim, title) in zip(axes, views):
    ax.view_init(elev=elev, azim=azim)
    pc_b = Poly3DCollection(b_polys, facecolors=b_cols, edgecolors='#141a22', linewidths=0.2, alpha=0.88)
    ax.add_collection3d(pc_b)
    pc_r = Poly3DCollection(r_polys, facecolors=r_cols, edgecolors='#ffffff', linewidths=0.8, alpha=1.0)
    ax.add_collection3d(pc_r)
    ax.set_xlim(c[0]-1.25, c[0]+1.25)
    ax.set_ylim(c[1]-1.0, c[1]+1.0)
    ax.set_zlim(c[2]-1.0, c[2]+1.0)
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=12, pad=10, fontweight='bold')

fig.suptitle('GEOMETRIC LOOP DETECTOR v2.0 (Спецификация: Изолированные зоны и шейки)\nРаздельные маски: левый и правый глаз не перехлестываются, уши отсечены в точке бифуркации',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
render_path = '/tmp/suzanne_v2_spec_showcase.png'
plt.savefig(render_path, dpi=160, facecolor='#0d1117')
print(f"Render successfully saved to {render_path}")
