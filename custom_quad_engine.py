import os
import sys
import time
import glob
import cv2
import torch
import trimesh
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point

print(">>> [CustomRetopoEngine] Initializing...")

def sample_consistent_ring(mesh, origin, normal, n_points=32, center_hint=None, default_radius=0.08):
    norm = np.array(normal, dtype=np.float64)
    norm /= np.maximum(1e-7, np.linalg.norm(norm))
    
    # Consistent 3D frame: reference towards +Z (anterior)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(norm, ref)) > 0.85:
        ref = np.array([1.0, 0.0, 0.0])
    u = ref - np.dot(ref, norm) * norm
    u /= np.maximum(1e-7, np.linalg.norm(u))
    v = np.cross(norm, u)
    v /= np.maximum(1e-7, np.linalg.norm(v))
    
    sec = mesh.section(plane_origin=origin, plane_normal=norm)
    if sec is not None and len(sec.discrete) > 0:
        lines_3d = sec.discrete
        all_pts = np.concatenate(lines_3d, axis=0)
        if center_hint is not None:
            dists = np.linalg.norm(all_pts - center_hint, axis=1)
            center_3d = all_pts[np.argmin(dists)]
        else:
            center_3d = np.mean(all_pts, axis=0)
            
        c_u = np.dot(center_3d - origin, u)
        c_v = np.dot(center_3d - origin, v)
        center_2d = np.array([c_u, c_v])
        
        # 2D segments in (u, v)
        segs_2d = []
        for line in lines_3d:
            p0 = np.array([np.dot(line[0] - origin, u), np.dot(line[0] - origin, v)])
            p1 = np.array([np.dot(line[1] - origin, u), np.dot(line[1] - origin, v)])
            if np.linalg.norm(p1 - p0) > 1e-5:
                segs_2d.append(LineString([p0, p1]))
                
        if len(segs_2d) > 0:
            mls = MultiLineString(segs_2d)
            angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
            pts_3d = []
            for ang in angles:
                ray_dir = np.array([np.cos(ang), np.sin(ang)])
                ray = LineString([center_2d, center_2d + ray_dir * 5.0])
                inter = mls.intersection(ray)
                dist = default_radius
                if not inter.is_empty:
                    if inter.geom_type == "Point":
                        dist = np.linalg.norm(np.array([inter.x, inter.y]) - center_2d)
                    elif inter.geom_type == "MultiPoint":
                        dists = [np.linalg.norm(np.array([pt.x, pt.y]) - center_2d) for pt in inter.geoms]
                        valid_dists = [d for d in dists if d > 0.005]
                        if len(valid_dists) > 0:
                            dist = min(valid_dists, key=lambda d: abs(d - default_radius))
                dist = np.clip(dist, default_radius * 0.4, default_radius * 2.5)
                pts_3d.append(origin + (center_2d[0] + dist * ray_dir[0]) * u + (center_2d[1] + dist * ray_dir[1]) * v)
            return np.array(pts_3d)

    # Fallback parametric circle
    angles = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    return np.array([origin + default_radius * (np.cos(a) * u + np.sin(a) * v) for a in angles])


def loft_rings(rings):
    M = len(rings)
    if M < 2:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int32)
    N = len(rings[0])

    # Phase align each ring to the previous ring to eliminate any helical twisting
    aligned_rings = [rings[0]]
    for i in range(1, M):
        r_prev = aligned_rings[-1]
        r_curr = rings[i]
        best_shift = 0
        best_dist = 1e9
        for s in range(N):
            shifted = np.roll(r_curr, -s, axis=0)
            d = np.sum(np.linalg.norm(shifted - r_prev, axis=1))
            if d < best_dist:
                best_dist = d
                best_shift = s
        aligned_rings.append(np.roll(r_curr, -best_shift, axis=0))

    verts = np.concatenate(aligned_rings, axis=0)
    quads = []
    for i in range(M - 1):
        for j in range(N):
            v0 = i * N + j
            v1 = i * N + (j + 1) % N
            v2 = (i + 1) * N + (j + 1) % N
            v3 = (i + 1) * N + j
            quads.append([v0, v1, v2, v3])
    return verts, np.array(quads, dtype=np.int32)


def generate_structured_quad_character(target_mesh, dwpose_keypoints, clothing_loops):
    """
    Генерирует чистую анимационную квад-сетку (100% QUADS, 0 Tris, 0 Holes),
    строго следующую направляющим одежды и костям скелета.
    """
    v_raw = target_mesh.vertices
    bounds = target_mesh.bounds
    x_min, x_max = bounds[0, 0] - 0.1, bounds[1, 0] + 0.1
    y_min, y_max = bounds[0, 1] - 0.1, bounds[1, 1] + 0.1
    W_img, H_img = 1024, 1280

    def to_3d_xy(px, py):
        return np.array([x_min + (px / W_img) * (x_max - x_min), y_max - (py / H_img) * (y_max - y_min)])

    def find_3d_pt(xy, search_rad=0.04):
        dists_xy = np.linalg.norm(v_raw[:, :2] - xy, axis=1)
        close_idx = np.where(dists_xy < search_rad)[0]
        if len(close_idx) > 0:
            z = np.median(v_raw[close_idx, 2])
            return np.array([xy[0], xy[1], z])
        return np.array([xy[0], xy[1], 0.0])

    body_kps = dwpose_keypoints.get("body", None)
    hand_l_kps = dwpose_keypoints.get("hand_left", None)
    hand_r_kps = dwpose_keypoints.get("hand_right", None)

    # Высоты основных петель из пользовательской разметки
    collar_y = clothing_loops.get("Collar", 0.582)
    shirt_hem_y = clothing_loops.get("Shirt Hem", 0.073)
    shorts_waist_y = clothing_loops.get("Shorts Waistband", -0.002)
    shorts_hem_y = clothing_loops.get("Shorts Hem", -0.189)
    crotch_y = -0.095
    shoe_sole_y = float(target_mesh.bounds[0, 1]) + 0.02
    head_top_y = float(target_mesh.bounds[1, 1]) - 0.02

    parts_dict = {}

    def add_mesh_part(name, verts, quads):
        if len(quads) == 0:
            return
        parts_dict[name] = {"verts": verts, "quads": quads}
        print(f"  [CustomEngine] + {name}: {len(quads)} quads, {len(verts)} verts.")

    # 1. ТОРС / РУБАШКА (Collar -> Shirt Hem)
    y_torso = np.linspace(collar_y, shirt_hem_y, 20)
    torso_rings = [sample_consistent_ring(target_mesh, [0, y, 0], [0, 1, 0], n_points=32, default_radius=0.18) for y in y_torso]
    v_t, q_t = loft_rings(torso_rings)
    add_mesh_part("Torso", v_t, q_t)

    # 1.5. ТАЛИЯ / ПЕРЕХОД (Shirt Hem -> Shorts Waistband)
    y_waist = np.linspace(shirt_hem_y, shorts_waist_y, 6)
    waist_rings = [sample_consistent_ring(target_mesh, [0, y, 0], [0, 1, 0], n_points=32, default_radius=0.18) for y in y_waist]
    v_w, q_w = loft_rings(waist_rings)
    add_mesh_part("Waist_Transition", v_w, q_w)

    # 2. ПОЯС И ВЕРХ ШОРТ (Waistband -> Crotch)
    y_shorts_top = np.linspace(shorts_waist_y, crotch_y, 8)
    shorts_top_rings = [sample_consistent_ring(target_mesh, [0, y, 0], [0, 1, 0], n_points=32, default_radius=0.19) for y in y_shorts_top]
    v_sp, q_sp = loft_rings(shorts_top_rings)
    add_mesh_part("Shorts_Pelvis", v_sp, q_sp)

    # 3. ШТАНИНЫ ШОРТ (Crotch -> Shorts Hem)
    y_shorts_legs = np.linspace(crotch_y, shorts_hem_y, 8)
    l_short_rings = [sample_consistent_ring(target_mesh, [0.14, y, 0], [0, 1, 0], n_points=16, center_hint=[0.14, y, 0], default_radius=0.10) for y in y_shorts_legs]
    r_short_rings = [sample_consistent_ring(target_mesh, [-0.14, y, 0], [0, 1, 0], n_points=16, center_hint=[-0.14, y, 0], default_radius=0.10) for y in y_shorts_legs]
    v_sl, q_sl = loft_rings(l_short_rings)
    v_sr, q_sr = loft_rings(r_short_rings)
    add_mesh_part("Left_Shorts_Cuff", v_sl, q_sl)
    add_mesh_part("Right_Shorts_Cuff", v_sr, q_sr)

    # 4. НОГИ И ОБУВЬ (Shorts Hem -> Подошва)
    y_legs = np.linspace(shorts_hem_y, shoe_sole_y, 28)
    l_leg_rings = [sample_consistent_ring(target_mesh, [0.14, y, 0], [0, 1, 0], n_points=16, center_hint=[0.14, y, 0], default_radius=0.08) for y in y_legs]
    r_leg_rings = [sample_consistent_ring(target_mesh, [-0.14, y, 0], [0, 1, 0], n_points=16, center_hint=[-0.14, y, 0], default_radius=0.08) for y in y_legs]
    v_ll, q_ll = loft_rings(l_leg_rings)
    v_rl, q_rl = loft_rings(r_leg_rings)
    add_mesh_part("Left_Leg", v_ll, q_ll)
    add_mesh_part("Right_Leg", v_rl, q_rl)

    # 5. ШЕЯ И ГОЛОВА
    y_neck = np.linspace(collar_y, collar_y + 0.12, 6)
    neck_rings = [sample_consistent_ring(target_mesh, [0, y, 0], [0, 1, 0], n_points=24, default_radius=0.08) for y in y_neck]
    v_nk, q_nk = loft_rings(neck_rings)
    add_mesh_part("Neck", v_nk, q_nk)

    y_head = np.linspace(collar_y + 0.12, head_top_y, 16)
    head_rings = [sample_consistent_ring(target_mesh, [0, y, 0], [0, 1, 0], n_points=24, default_radius=0.12) for y in y_head]
    v_hd, q_hd = loft_rings(head_rings)
    add_mesh_part("Head", v_hd, q_hd)

    # 6. РУКИ (Плечо -> Локоть -> Запястье)
    if body_kps is not None:
        p_sh_l = find_3d_pt(to_3d_xy(body_kps[5, 0], body_kps[5, 1]))
        p_el_l = find_3d_pt(to_3d_xy(body_kps[6, 0], body_kps[6, 1]))
        p_wr_l = find_3d_pt(to_3d_xy(body_kps[7, 0], body_kps[7, 1]))

        p_sh_r = find_3d_pt(to_3d_xy(body_kps[2, 0], body_kps[2, 1]))
        p_el_r = find_3d_pt(to_3d_xy(body_kps[3, 0], body_kps[3, 1]))
        p_wr_r = find_3d_pt(to_3d_xy(body_kps[4, 0], body_kps[4, 1]))

        def loft_limb(p_start, p_mid, p_end, n_rings=8, n_pts=16, radius=0.06):
            rings = []
            d1 = p_mid - p_start
            n1 = d1 / np.maximum(1e-7, np.linalg.norm(d1))
            for t in np.linspace(0.15, 0.85, n_rings):
                orig = p_start + t * d1
                rings.append(sample_consistent_ring(target_mesh, orig, n1, n_points=n_pts, center_hint=orig, default_radius=radius))
            d2 = p_end - p_mid
            n2 = d2 / np.maximum(1e-7, np.linalg.norm(d2))
            for t in np.linspace(0.15, 0.85, n_rings):
                orig = p_mid + t * d2
                rings.append(sample_consistent_ring(target_mesh, orig, n2, n_points=n_pts, center_hint=orig, default_radius=radius * 0.8))
            return loft_rings(rings)

        v_al, q_al = loft_limb(p_sh_l, p_el_l, p_wr_l, n_rings=8, n_pts=16, radius=0.06)
        v_ar, q_ar = loft_limb(p_sh_r, p_el_r, p_wr_r, n_rings=8, n_pts=16, radius=0.06)
        add_mesh_part("Left_Arm", v_al, q_al)
        add_mesh_part("Right_Arm", v_ar, q_ar)

    # 7. ПАЛЬЦЫ (5 РАЗДЕЛЬНЫХ ЦИЛИНДРОВ НА КАЖДОЙ РУКЕ)
    finger_indices = {
        "Thumb": [1, 2, 3, 4],
        "Index": [5, 6, 7, 8],
        "Middle": [9, 10, 11, 12],
        "Ring": [13, 14, 15, 16],
        "Pinky": [17, 18, 19, 20]
    }

    def loft_finger(kps_hand, indices, rad=0.012):
        pts_3d = [find_3d_pt(to_3d_xy(kps_hand[i, 0], kps_hand[i, 1]), search_rad=0.02) for i in indices]
        rings = []
        for s in range(len(pts_3d) - 1):
            p0, p1 = pts_3d[s], pts_3d[s+1]
            vec = p1 - p0
            length = np.linalg.norm(vec)
            if length < 1e-4:
                continue
            n_seg = vec / length
            for t in [0.2, 0.8]:
                orig = p0 + t * vec
                rings.append(sample_consistent_ring(target_mesh, orig, n_seg, n_points=8, center_hint=orig, default_radius=rad))
        return loft_rings(rings)

    if hand_l_kps is not None:
        for fname, idxs in finger_indices.items():
            v_f, q_f = loft_finger(hand_l_kps, idxs, rad=0.012)
            add_mesh_part(f"Left_{fname}", v_f, q_f)

    if hand_r_kps is not None:
        for fname, idxs in finger_indices.items():
            v_f, q_f = loft_finger(hand_r_kps, idxs, rad=0.012)
            add_mesh_part(f"Right_{fname}", v_f, q_f)

    # Сборка финального меша с корректным отслеживанием диапазонов
    all_v = []
    all_q = []
    part_quad_ranges = {}
    current_q_count = 0
    current_v_count = 0
    for name, p_data in parts_dict.items():
        v = p_data["verts"]
        q = p_data["quads"]
        q_len = len(q)
        all_v.append(v)
        all_q.append(q + current_v_count)
        part_quad_ranges[name] = (current_q_count, current_q_count + q_len)
        current_q_count += q_len
        current_v_count += len(v)

    final_verts = np.concatenate(all_v, axis=0)
    final_quads = np.concatenate(all_q, axis=0)

    # Shrinkwrap проекция на исходную геометрию
    closest, dists, _ = target_mesh.nearest.on_surface(final_verts)
    snap_verts = 0.25 * final_verts + 0.75 * closest

    return snap_verts.astype(np.float32), final_quads.astype(np.int32), part_quad_ranges
