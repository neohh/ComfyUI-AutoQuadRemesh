"""
GeometricLoopDetector v2.0: Isolated Branch & Anatomical Basin Detection
ComfyUI-AutoQuadRemesh (Phase 1, Specification v2.0)

Implements the multi-stage form-transition pipeline:
Step 1: Island & Branch Isolation (Extremity detection & isolated sub-graph walking)
Step 2: Bifurcation Root Detection (Cross-section perimeter S(d) & bottleneck collar)
Step 3: Anisotropic Basin Sockets (Mean Curvature H=0 zero-crossing & closed-loop validation)
Step 4: Arc-Length Resampling & Frenet-Frame Quad Ribbon Extrusion (0 non-manifold edges)
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.signal import argrelextrema
import networkx as nx
import trimesh
from trimesh.curvature import discrete_mean_curvature_measure, discrete_gaussian_curvature_measure


class GeometricLoopDetectorV2:
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
        self.centroid = self.mesh.centroid
        self.n_v = len(self.vertices)
        
        # Build weighted mesh edge graph for geodesics
        edges = self.mesh.edges_unique
        lengths = self.mesh.edges_unique_length
        self.adj = sp.coo_matrix((lengths, (edges[:, 0], edges[:, 1])), shape=(self.n_v, self.n_v))
        self.adj = self.adj + self.adj.T
        
        self.G_mesh = nx.Graph()
        self.G_mesh.add_edges_from(edges)
        
        # Curvature fields cache
        self.H = None
        self.K = None

    def compute_curvature(self, radius_factor=0.03):
        """Computes discrete mean and Gaussian curvature measures."""
        r = self.scale * radius_factor
        self.H = discrete_mean_curvature_measure(self.mesh, self.vertices, r)
        self.K = discrete_gaussian_curvature_measure(self.mesh, self.vertices, r)
        return self.H, self.K

    def extract_level_set(self, scalar_field, target_val=0.0, face_subset=None):
        """
        Extracts continuous 1-manifold level-set contours for scalar_field = target_val.
        Returns list of ordered cycles as (N, 3) arrays.
        """
        if face_subset is None:
            F_eval = self.faces
        else:
            F_eval = self.faces[face_subset]
            
        f_vals = scalar_field[F_eval]
        segments = []
        
        for fid in range(len(F_eval)):
            fv = f_vals[fid]
            pts = self.vertices[F_eval[fid]]
            cut_pts = []
            for i in range(3):
                j = (i + 1) % 3
                d0, d1 = fv[i], fv[j]
                if (d0 <= target_val <= d1) or (d1 <= target_val <= d0):
                    if abs(d1 - d0) > 1e-7:
                        t = (target_val - d0) / (d1 - d0)
                        cut_pts.append((1.0 - t) * pts[i] + t * pts[j])
            if len(cut_pts) == 2:
                segments.append((cut_pts[0], cut_pts[1]))
                
        if not segments:
            return []
            
        # Build segment connectivity graph
        G_cut = nx.Graph()
        node_map = {}
        pts_arr = []
        
        for p0, p1 in segments:
            k0 = tuple(np.round(p0, 4))
            k1 = tuple(np.round(p1, 4))
            if k0 not in node_map:
                node_map[k0] = len(pts_arr)
                pts_arr.append(p0)
            if k1 not in node_map:
                node_map[k1] = len(pts_arr)
                pts_arr.append(p1)
            G_cut.add_edge(node_map[k0], node_map[k1])
            
        pts_arr = np.array(pts_arr)
        cycles = nx.cycle_basis(G_cut)
        
        ordered_loops = []
        for c in cycles:
            if len(c) >= 8:
                ordered_loops.append(pts_arr[c])
        return ordered_loops

    @staticmethod
    def resample_loop(loop_pts, target_pts=24):
        """Resamples loop to uniform arc-length parameterization."""
        diffs = loop_pts[(np.arange(len(loop_pts)) + 1) % len(loop_pts)] - loop_pts
        dists = np.linalg.norm(diffs, axis=1)
        cum_dists = np.concatenate([[0.0], np.cumsum(dists)])
        if cum_dists[-1] < 1e-6:
            return None
        sample_s = np.linspace(0, cum_dists[-1], target_pts, endpoint=False)
        resampled = np.zeros((target_pts, 3))
        closed = np.vstack([loop_pts, loop_pts[0]])
        for d in range(3):
            resampled[:, d] = np.interp(sample_s, cum_dists, closed[:, d])
        return resampled

    def detect_branches_and_bottlenecks(self, target_pts=24):
        """
        Step 1 & 2: Island & Branch Isolation.
        Detects branch extremities (e.g. ear tips, snout, neck) and finds
        the root bifurcation / bottleneck before merging into the main body.
        """
        V = self.vertices
        # Find extreme tips along principal axes
        extremities = [
            ('left_branch', np.argmax(V[:, 0])),
            ('right_branch', np.argmin(V[:, 0])),
            ('bottom_branch', np.argmin(V[:, 2])),
        ]
        
        branch_loops = []
        for name, tip_idx in extremities:
            dists = dijkstra(self.adj, directed=False, indices=tip_idx)
            # Scan cross-sectional perimeter as a function of distance
            d_vals = np.linspace(0.15 * self.scale, 0.65 * self.scale, 35)
            best_loop = None
            min_perim = 1e9
            
            for d in d_vals:
                loops = self.extract_level_set(dists, target_val=d)
                for lp in loops:
                    diffs = lp[(np.arange(len(lp)) + 1) % len(lp)] - lp
                    perim = np.sum(np.linalg.norm(diffs, axis=1))
                    # Bottleneck criterion: local minimum perimeter before body expansion
                    if 0.5 < perim < min_perim and perim < 2.5:
                        min_perim = perim
                        best_loop = lp
                        
            if best_loop is not None:
                resampled = self.resample_loop(best_loop, target_pts)
                if resampled is not None:
                    branch_loops.append({
                        'name': name,
                        'type': 'branch_root_bottleneck',
                        'points': resampled,
                        'center': np.mean(resampled, axis=0),
                        'perimeter': min_perim
                    })
        return branch_loops

    def detect_cavity_basins(self, concave_thresh=-0.04, target_pts=24):
        """
        Step 3: Anisotropic Basin Sockets.
        Isolates individual concave basins (eye sockets, mouth) and extracts
        the exact zero-crossing (H = 0) boundary loop within each basin's mask.
        """
        if self.H is None:
            self.compute_curvature()
            
        concave_v = set(np.where(self.H < concave_thresh)[0])
        subG = self.G_mesh.subgraph(concave_v)
        components = [list(c) for c in nx.connected_components(subG) if len(c) > 60]
        
        basin_loops = []
        all_H0_loops = self.extract_level_set(self.H, target_val=0.0)
        
        mid_x = self.centroid[0]
        
        for b_idx, comp in enumerate(components):
            b_pts = self.vertices[comp]
            b_center = np.mean(b_pts, axis=0)
            
            # Find loop that best bounds this specific basin
            best_c = None
            best_dist = 1e9
            
            for lp in all_H0_loops:
                lp_center = np.mean(lp, axis=0)
                diffs = lp[(np.arange(len(lp)) + 1) % len(lp)] - lp
                perim = np.sum(np.linalg.norm(diffs, axis=1))
                
                # Closed-Loop Validation:
                # 1. Perimeter within anatomical bounds
                if not (1.0 < perim < 3.2):
                    continue
                    
                # 2. Loop center must match basin center
                dist = np.linalg.norm(lp_center - b_center)
                if dist < 0.35 and dist < best_dist:
                    # 3. Symmetry check: left socket must not cross to right side
                    if b_center[0] > mid_x + 0.05 and lp_center[0] < mid_x:
                        continue
                    if b_center[0] < mid_x - 0.05 and lp_center[0] > mid_x:
                        continue
                        
                    best_dist = dist
                    best_c = (lp, perim, lp_center)
                    
            if best_c is not None:
                lp, perim, lp_center = best_c
                # Determine tag
                if lp_center[0] > mid_x + 0.1:
                    tag = 'left_eye_socket_rim'
                elif lp_center[0] < mid_x - 0.1:
                    tag = 'right_eye_socket_rim'
                else:
                    tag = 'muzzle_cavity_rim'
                    
                resampled = self.resample_loop(lp, target_pts)
                if resampled is not None:
                    basin_loops.append({
                        'name': tag,
                        'type': 'zero_crossing_rim',
                        'points': resampled,
                        'center': lp_center,
                        'perimeter': perim
                    })
        return basin_loops

    def generate_all_loops(self):
        """Runs full v2.0 pipeline: branches + cavity basins."""
        t0 = time.perf_counter()
        self.compute_curvature()
        branches = self.detect_branches_and_bottlenecks(target_pts=24)
        basins = self.detect_cavity_basins(target_pts=24)
        
        all_loops = branches + basins
        elapsed = time.perf_counter() - t0
        print(f"GeometricLoopDetector v2.0 completed in {elapsed:.3f}s: found {len(all_loops)} loops.")
        return all_loops

    def build_quad_mesh(self, detected_loops, ribbon_width=None):
        """
        Step 4: Extrudes regular quad ribbons along the surface using the Frenet frame.
        Guarantees 100% pure quads with 0 non-manifold edges.
        """
        if ribbon_width is None:
            ribbon_width = self.scale * 0.035
            
        all_verts = list(self.vertices)
        quad_faces = []
        
        for item in detected_loops:
            loop_pts = item['points']
            n_pts = len(loop_pts)
            
            tangents = loop_pts[(np.arange(n_pts) + 1) % n_pts] - loop_pts[(np.arange(n_pts) - 1) % n_pts]
            tangents /= (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8)
            
            _, _, face_idx = self.mesh.nearest.on_surface(loop_pts)
            normals = self.mesh.face_normals[face_idx]
            
            binormals = np.cross(normals, tangents)
            binormals /= (np.linalg.norm(binormals, axis=1, keepdims=True) + 1e-8)
            
            v_out, _, _ = self.mesh.nearest.on_surface(loop_pts + 0.5 * ribbon_width * binormals)
            v_in, _, _ = self.mesh.nearest.on_surface(loop_pts - 0.5 * ribbon_width * binormals)
            
            start_v = len(all_verts)
            all_verts.extend(v_out)
            all_verts.extend(v_in)
            
            for i in range(n_pts):
                next_i = (i + 1) % n_pts
                o1 = start_v + i
                o2 = start_v + next_i
                i1 = start_v + n_pts + i
                i2 = start_v + n_pts + next_i
                quad_faces.append([o1, o2, i2, i1])
                
        return np.array(all_verts), quad_faces
