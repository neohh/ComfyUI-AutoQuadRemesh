import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

mesh_path = '/opt/comfyui/output/stages_data/03_quad_mesh/suzanne_geodesic_zbrush_rings.obj'
v = []; tris = []; quads = []
with open(mesh_path, 'r', encoding='utf-8') as f:
    for line in f:
        p = line.strip().split()
        if not p: continue
        if p[0] == 'v': v.append([float(x) for x in p[1:4]])
        elif p[0] == 'f':
            face = [int(x.split('/')[0]) - 1 for x in p[1:]]
            if len(face) == 4: quads.append(face)
            elif len(face) == 3: tris.append(face)

v = np.array(v)
tris = np.array(tris)
quads = np.array(quads)

print('Loaded:', len(v), 'verts,', len(tris), 'tris,', len(quads), 'quads')

fig, axes = plt.subplots(1, 3, figsize=(21, 7), subplot_kw={'projection': '3d'}, facecolor='#0d1117')
views = [
    (35, -125, '3/4 Слева (Глаз, Ухо, Морда)'),
    (35, -90, 'Фронтальный Фас (Анатомические кольца)'),
    (35, -55, '3/4 Справа (Глаз, Ухо, Морда)')
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

b_polys = [v[f] for f in tris[::3]]
b_cols = [shade(p, [0.22, 0.28, 0.36]) for p in b_polys]

r_polys = [v[q] for q in quads]
r_cols = [shade(p, [1.0, 0.08, 0.18]) for p in r_polys]

c = v[:26760].mean(axis=0)

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
    ax.set_title(title, fontsize=13, color='white', weight='bold', pad=12)
    ax.set_facecolor('#0d1117')

plt.suptitle('ГЕОДЕЗИЧЕСКИЕ ИЗОЛИНИИ (ZBrush / Level-Sets Метод)\nИдеальные замкнутые кольца вокруг глазниц, ушей, морды и шеи',
             fontsize=15, color='white', weight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/tmp/suzanne_geodesic_zbrush_showcase.png', dpi=180, facecolor='#0d1117')
print('Geodesic render successfully saved to /tmp/suzanne_geodesic_zbrush_showcase.png!')
