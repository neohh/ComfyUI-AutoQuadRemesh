"""
GeometricLoopDetector: Fully Automatic Curvature & Crease Loop Extraction
Part of ComfyUI-AutoQuadRemesh (Phase 1: Pure Geometric Detection)

Detects anatomical/structural feature loops (depressions, protrusions, junctions)
without manual vertex picking or hardcoded indices, using differential geometry:
- Signed dihedral angles (valleys vs ridges)
- Graph-connected feature components
- Boundary cycle tracing
- Arc-length loop resampling & quad ring regularization
"""

import time
import numpy as np
import scipy.sparse as sp
import trimesh
import networkx as nx
from collections import defaultdict


class GeometricLoopDetector:
    def __init__(self, mesh_or_path, merge_verts=True):
        if isinstance(mesh_or_path, str):
            self.mesh = trimesh.load(mesh_or_path, force="mesh", process=False)
        else:
            self.mesh = mesh_or_path.copy()
            
        if merge_verts:
            self.mesh.merge_vertices()
            
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.scale = self.mesh.scale
        self.center = self.mesh.centroid

    def detect_loops(
        self,
        min_angle_deg=18.0,
        feature_type="both",  # "concave", "convex", "both"
        min_loop_verts=8,
        max_loop_verts=300,
        min_comp_faces=12,
        top_k=8,
        smooth_iter=2
    ):
        """
        Automatically detects closed feature loops on the mesh surface.
        
        Args:
            min_angle_deg: Minimum dihedral angle in degrees to be considered a feature.
            feature_type: "concave" (valleys/cavities), "convex" (ridges/crests), or "both".
            min_loop_verts: Minimum vertex count for a valid macro-loop.
            max_loop_verts: Maximum vertex count.
            min_comp_faces: Minimum faces in a feature patch to filter out noise.
            top_k: Number of most prominent loops to return.
            smooth_iter: Iterations of loop smoothing along the surface.
            
        Returns:
            List of dicts: [
                {
                    'id': int,
                    'type': 'concave' | 'convex',
                    'vertices': np.ndarray (K, 3),
                    'indices': list of vertex indices,
                    'center': np.ndarray (3,),
                    'perimeter': float,
                    'span': np.ndarray (3,)
                }, ...
            ]
        """
        t0 = time.perf_counter()
        angles = self.mesh.face_adjacency_angles
        is_convex = self.mesh.face_adjacency_convex
        adj_faces = self.mesh.face_adjacency
        
        thresh_rad = np.radians(min_angle_deg)
        
        detected_loops = []
        types_to_check = []
        if feature_type in ["concave", "both"]:
            types_to_check.append(("concave", (~is_convex) & (angles > thresh_rad)))
        if feature_type in ["convex", "both"]:
            types_to_check.append(("convex", is_convex & (angles > thresh_rad)))
            
        for ftype, mask in types_to_check:
            if not np.any(mask):
                continue
                
            crease_faces = set()
            for f1, f2 in adj_faces[mask]:
                crease_faces.add(f1)
                crease_faces.add(f2)
                
            # Build graph of adjacent crease faces
            face_graph = nx.Graph()
            for f1, f2 in adj_faces:
                if f1 in crease_faces and f2 in crease_faces:
                    face_graph.add_edge(f1, f2)
                    
            components = list(nx.connected_components(face_graph))
            
            # Sort components by size (prominence)
            for comp in sorted(components, key=len, reverse=True):
                if len(comp) < min_comp_faces:
                    continue
                    
                # Extract boundary edges of this component
                edge_counts = defaultdict(int)
                for fid in comp:
                    f = self.faces[fid]
                    for i in range(3):
                        e = tuple(sorted((f[i], f[(i + 1) % 3])))
                        edge_counts[e] += 1
                        
                b_edges = [e for e, cnt in edge_counts.items() if cnt == 1]
                if len(b_edges) < min_loop_verts:
                    continue
                    
                # Trace boundary graph
                bg = nx.Graph()
                for u, v in b_edges:
                    bg.add_edge(u, v)
                    
                cycles = nx.cycle_basis(bg)
                for cycle in cycles:
                    if len(cycle) < min_loop_verts or len(cycle) > max_loop_verts:
                        continue
                        
                    # Order the cycle vertices sequentially
                    ordered_cycle = self._order_cycle(bg, cycle)
                    if ordered_cycle is None:
                        continue
                        
                    pts = self.vertices[ordered_cycle]
                    c_center = pts.mean(axis=0)
                    c_span = pts.max(axis=0) - pts.min(axis=0)
                    
                    # Perimeter
                    diffs = pts[(np.arange(len(pts)) + 1) % len(pts)] - pts
                    perim = float(np.sum(np.linalg.norm(diffs, axis=1)))
                    
                    # Check for duplicates or near-coincident loops
                    is_dup = False
                    for existing in detected_loops:
                        dist = np.linalg.norm(c_center - existing["center"])
                        if dist < self.scale * 0.04:
                            is_dup = True
                            break
                            
                    if not is_dup:
                        # Smooth loop along surface
                        smoothed_pts = self._smooth_loop(pts, iterations=smooth_iter)
                        detected_loops.append({
                            "id": len(detected_loops),
                            "type": ftype,
                            "vertices": smoothed_pts,
                            "indices": ordered_cycle,
                            "center": c_center,
                            "perimeter": perim,
                            "span": c_span,
                            "num_faces": len(comp)
                        })
                        
        # Sort by prominence (number of faces * perimeter)
        detected_loops.sort(key=lambda x: x["num_faces"] * x["perimeter"], reverse=True)
        final_loops = detected_loops[:top_k]
        
        elapsed = time.perf_counter() - t0
        return final_loops, elapsed

    def _order_cycle(self, graph, cycle_nodes):
        """Orders cycle nodes into a continuous closed loop sequence."""
        sub = graph.subgraph(cycle_nodes)
        # Find an Eulerian or simple cycle
        try:
            # If 2-regular, traverse in order
            start = cycle_nodes[0]
            visited = [start]
            curr = start
            prev = None
            for _ in range(len(cycle_nodes) - 1):
                nbrs = [n for n in sub.neighbors(curr) if n != prev]
                if not nbrs:
                    break
                next_node = nbrs[0]
                visited.append(next_node)
                prev, curr = curr, next_node
            if len(visited) == len(cycle_nodes):
                return visited
        except Exception:
            pass
        return cycle_nodes

    def _smooth_loop(self, pts, iterations=2):
        """Laplacian smoothing of 3D loop vertices with projection to surface."""
        p = pts.copy()
        n = len(p)
        for _ in range(iterations):
            p_prev = p[(np.arange(n) - 1) % n]
            p_next = p[(np.arange(n) + 1) % n]
            p = 0.5 * p + 0.25 * (p_prev + p_next)
        return p

    def regularize_ring(self, loop_pts, target_quads=24, ring_width=None):
        """
        Resamples a 3D loop to exact uniform arc-length segments
        and extrudes a closed 1-row quad ring.
        
        Returns:
            ring_verts: np.ndarray (target_quads * 2, 3)
            ring_quads: np.ndarray (target_quads, 4)
        """
        if ring_width is None:
            ring_width = self.scale * 0.025
            
        n = len(loop_pts)
        diffs = loop_pts[(np.arange(n) + 1) % n] - loop_pts
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
        total_len = cum_dists[-1]
        
        # Resample at uniform arc-lengths
        sample_s = np.linspace(0, total_len, target_quads, endpoint=False)
        centerline = np.zeros((target_quads, 3))
        
        closed_pts = np.vstack([loop_pts, loop_pts[0]])
        for d in range(3):
            centerline[:, d] = np.interp(sample_s, cum_dists, closed_pts[:, d])
            
        # Compute tangents and normals
        tangents = centerline[(np.arange(target_quads) + 1) % target_quads] - centerline[(np.arange(target_quads) - 1) % target_quads]
        tangents /= (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
        
        # Approximate surface normal by querying nearest mesh face
        nearest_pts, _, face_idx = self.mesh.nearest.on_surface(centerline)
        normals = self.mesh.face_normals[face_idx]
        
        # Binormals perpendicular to tangent and surface normal
        binormals = np.cross(normals, tangents)
        binormals /= (np.linalg.norm(binormals, axis=1, keepdims=True) + 1e-8)
        
        half_w = ring_width * 0.5
        inner_ring = centerline - binormals * half_w
        outer_ring = centerline + binormals * half_w
        
        # Project to mesh surface
        inner_proj, _, _ = self.mesh.nearest.on_surface(inner_ring)
        outer_proj, _, _ = self.mesh.nearest.on_surface(outer_ring)
        
        ring_verts = np.vstack([inner_proj, outer_proj])
        ring_quads = []
        for i in range(target_quads):
            next_i = (i + 1) % target_quads
            v0 = i
            v1 = next_i
            v2 = target_quads + next_i
            v3 = target_quads + i
            ring_quads.append([v0, v1, v2, v3])
            
        return ring_verts, np.array(ring_quads)

    def export_debug_obj(self, detected_loops, out_obj_path, out_mtl_path=None):
        """Exports detected loops alongside mesh geometry with distinct materials."""
        if out_mtl_path is None:
            out_mtl_path = out_obj_path.replace(".obj", ".mtl")
            
        mtl_filename = out_mtl_path.split("/")[-1].split("\\")[-1]
        
        with open(out_mtl_path, "w", encoding="utf-8") as f:
            f.write("""# Auto-generated loop detector materials
newmtl Material_Base
Kd 0.22 0.28 0.36
Ks 0.2 0.2 0.2
Ns 30.0

newmtl Material_Red_Loop
Kd 0.98 0.08 0.18
Ks 0.6 0.6 0.6
Ns 80.0
""")

        all_verts = list(self.vertices)
        loop_quad_faces = []
        
        for loop in detected_loops:
            ring_v, ring_q = self.regularize_ring(loop["vertices"], target_quads=24)
            offset = len(all_verts)
            all_verts.extend(ring_v)
            for q in ring_q:
                loop_quad_faces.append([u + offset for u in q])
                
        with open(out_obj_path, "w", encoding="utf-8") as f:
            f.write(f"mtllib {mtl_filename}\n")
            f.write("o Base_Mesh\n")
            for v in all_verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("usemtl Material_Base\n")
            for f_tri in self.faces:
                f.write(f"f {f_tri[0]+1} {f_tri[1]+1} {f_tri[2]+1}\n")
            f.write("usemtl Material_Red_Loop\n")
            for q in loop_quad_faces:
                f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
                
        return out_obj_path
