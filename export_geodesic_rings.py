import trimesh
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import networkx as nx

mesh = trimesh.load('/opt/comfyui/output/stages_data/03_quad_mesh/сюзанна.obj', force='mesh')
mesh.merge_vertices()
V = mesh.vertices
F = mesh.faces

edges = mesh.edges_unique
lengths = mesh.edges_unique_length
n_v = len(V)
adj = sp.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(n_v, n_v))
adj = adj + adj.T

def get_ordered_loop(root_idx, radius, target_pts=24):
    geo_dist = dijkstra(adj, directed=False, indices=root_idx)
    f_dists = geo_dist[F]
    
    segments = []
    for fid, f in enumerate(F):
        fd = f_dists[fid]
        fv = V[f]
        cut_pts = []
        for i in range(3):
            j = (i + 1) % 3
            d0, d1 = fd[i], fd[j]
            if (d0 <= radius <= d1) or (d1 <= radius <= d0):
                if abs(d1 - d0) > 1e-7:
                    t = (radius - d0) / (d1 - d0)
                    cut_pts.append((1 - t) * fv[i] + t * fv[j])
        if len(cut_pts) == 2:
            segments.append((cut_pts[0], cut_pts[1]))
            
    G = nx.Graph()
    pts_list = []
    def get_pt_id(p):
        for idx, exist in enumerate(pts_list):
            if np.linalg.norm(p - exist) < 1e-4:
                return idx
        pts_list.append(p)
        return len(pts_list) - 1
        
    for p0, p1 in segments:
        id0 = get_pt_id(p0)
        id1 = get_pt_id(p1)
        if id0 != id1:
            G.add_edge(id0, id1)
            
    cycles = nx.cycle_basis(G)
    if not cycles:
        return None
    best_c = max(cycles, key=len)
    loop_pts = np.array([pts_list[i] for i in best_c])
    
    diffs = loop_pts[(np.arange(len(loop_pts)) + 1) % len(loop_pts)] - loop_pts
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
    sample_s = np.linspace(0, cum_dists[-1], target_pts, endpoint=False)
    
    resampled = np.zeros((target_pts, 3))
    closed = np.vstack([loop_pts, loop_pts[0]])
    for d in range(3):
        resampled[:, d] = np.interp(sample_s, cum_dists, closed[:, d])
        
    return resampled

feature_targets = [
    ('left_eye', [3.50, 3.85, 10.45], 0.22, 24),
    ('right_eye', [2.89, 3.82, 10.47], 0.22, 24),
    ('left_ear', [4.20, 3.70, 9.60], 0.32, 24),
    ('right_ear', [2.20, 3.70, 9.60], 0.32, 24),
    ('snout', [3.21, 4.35, 9.80], 0.34, 28),
    ('neck', [3.21, 3.70, 9.15], 0.26, 24),
]

all_loops = []
for name, pos, r, num_pts in feature_targets:
    root = np.argmin(np.linalg.norm(V - np.array(pos), axis=1))
    lp = get_ordered_loop(root, r, num_pts)
    if lp is not None:
        all_loops.append((name, lp))

all_verts = list(V)
quad_faces = []
ring_width = mesh.scale * 0.035

for name, loop_pts in all_loops:
    n_pts = len(loop_pts)
    tangents = loop_pts[(np.arange(n_pts) + 1) % n_pts] - loop_pts[(np.arange(n_pts) - 1) % n_pts]
    tangents /= (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
    
    nearest_pts, _, face_idx = mesh.nearest.on_surface(loop_pts)
    normals = mesh.face_normals[face_idx]
    
    binormals = np.cross(normals, tangents)
    binormals /= (np.linalg.norm(binormals, axis=1, keepdims=True) + 1e-8)
    
    inner_p = nearest_pts - binormals * (ring_width * 0.5)
    outer_p = nearest_pts + binormals * (ring_width * 0.5)
    
    inner_proj, _, _ = mesh.nearest.on_surface(inner_p)
    outer_proj, _, _ = mesh.nearest.on_surface(outer_p)
    
    inner_proj += normals * 0.005
    outer_proj += normals * 0.005
    
    offset = len(all_verts)
    all_verts.extend(inner_proj)
    all_verts.extend(outer_proj)
    
    for i in range(n_pts):
        next_i = (i + 1) % n_pts
        v0 = offset + i
        v1 = offset + next_i
        v2 = offset + n_pts + next_i
        v3 = offset + n_pts + i
        quad_faces.append([v0, v1, v2, v3])

out_obj = '/opt/comfyui/output/stages_data/03_quad_mesh/suzanne_geodesic_zbrush_rings.obj'
out_mtl = '/opt/comfyui/output/stages_data/03_quad_mesh/suzanne_geodesic_zbrush_rings.mtl'

with open(out_mtl, 'w', encoding='utf-8') as f:
    f.write('newmtl Material_Base\nKd 0.22 0.28 0.36\nKs 0.2 0.2 0.2\nNs 30.0\n\nnewmtl Material_Red_Loop\nKd 0.98 0.08 0.18\nKs 0.6 0.6 0.6\nNs 80.0\n')

with open(out_obj, 'w', encoding='utf-8') as f:
    f.write('mtllib suzanne_geodesic_zbrush_rings.mtl\n')
    f.write('o Suzanne_ZBrush_Style_Loops\n')
    for v in all_verts:
        f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
    f.write('usemtl Material_Base\n')
    for tri in F:
        f.write(f'f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n')
    f.write('usemtl Material_Red_Loop\n')
    for q in quad_faces:
        f.write(f'f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n')

print(f'Clean export done! {len(all_verts)} vertices, {len(F)} tris, {len(quad_faces)} loop quads')
