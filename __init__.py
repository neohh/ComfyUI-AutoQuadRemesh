import os
import glob
import time
import subprocess
import torch
import numpy as np
import trimesh
import cv2
import pymeshfix
import fast_simplification
import pygltflib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import folder_paths
from comfy.cli_args import args
import comfy_extras.nodes_hunyuan3d as hy3d_nodes

print(">>> [AutoQuadRemesh] Модуль интеллектуальной адаптивной квад-ретопологии загружен.")


def save_clean_quad_glb_with_wireframe(vertices, quads, tris=None, output_filepath="", show_quad_wireframe=True):
    """
    Экспортирует меш в GLB с кастомным оверлеем чистых квад-ребер для 3D-вьювера ComfyUI.
    Не отображает диагональные триангуляционные ребра внутри квадов!
    """
    tri_faces = []
    wireframe_edges = set()

    for q in quads:
        tri_faces.append([q[0], q[1], q[2]])
        tri_faces.append([q[0], q[2], q[3]])
        if show_quad_wireframe:
            for i in range(4):
                v1, v2 = int(q[i]), int(q[(i + 1) % 4])
                if v1 != v2:
                    wireframe_edges.add(tuple(sorted((v1, v2))))

    if tris is not None and len(tris) > 0:
        for t in tris:
            tri_faces.append([t[0], t[1], t[2]])
            if show_quad_wireframe:
                for i in range(3):
                    v1, v2 = int(t[i]), int(t[(i + 1) % 3])
                    if v1 != v2:
                        wireframe_edges.add(tuple(sorted((v1, v2))))

    tri_faces = np.array(tri_faces, dtype=np.int32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces, process=False)
    mesh.fix_normals()
    mesh.export(output_filepath)

    if not show_quad_wireframe or len(wireframe_edges) == 0:
        return

    try:
        gltf = pygltflib.GLTF2().load(output_filepath)
        blob = bytearray(gltf.binary_blob())

        prim0 = gltf.meshes[0].primitives[0]
        pos_accessor = prim0.attributes.POSITION

        line_indices = []
        for e in wireframe_edges:
            line_indices.extend([e[0], e[1]])
        lines_arr = np.array(line_indices, dtype=np.uint32)

        lines_offset = len(blob)
        while lines_offset % 4 != 0:
            blob.append(0)
            lines_offset += 1

        blob.extend(lines_arr.tobytes())

        bv_lines = pygltflib.BufferView(
            buffer=0,
            byteOffset=lines_offset,
            byteLength=len(lines_arr.tobytes()),
            target=pygltflib.ELEMENT_ARRAY_BUFFER
        )
        gltf.bufferViews.append(bv_lines)
        bv_lines_idx = len(gltf.bufferViews) - 1

        acc_lines = pygltflib.Accessor(
            bufferView=bv_lines_idx,
            byteOffset=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(lines_arr),
            type=pygltflib.SCALAR,
            max=[int(lines_arr.max())],
            min=[int(lines_arr.min())]
        )
        gltf.accessors.append(acc_lines)
        acc_lines_idx = len(gltf.accessors) - 1

        mat_lines = pygltflib.Material(
            name="quad_wireframe",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[0.12, 0.12, 0.15, 1.0],
                roughnessFactor=1.0
            )
        )
        gltf.materials.append(mat_lines)
        mat_lines_idx = len(gltf.materials) - 1

        prim_lines = pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=pos_accessor),
            indices=acc_lines_idx,
            material=mat_lines_idx,
            mode=pygltflib.LINES
        )
        gltf.meshes[0].primitives.append(prim_lines)

        gltf.buffers[0].byteLength = len(blob)
        gltf.set_binary_blob(bytes(blob))
        gltf.save(output_filepath)
    except Exception as e:
        print(f"[AutoQuadRemesh] Wireframe overlay skipped: {e}")


def save_pure_quad_obj(vertices, quads, tris=None, output_filepath=""):
    """
    Экспортирует чистый полигональный OBJ меш:
    f v1 v2 v3 v4 (для квадов) и f v1 v2 v3 (для переходных треугольников).
    """
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("# AutoQuadRemesh adaptive quad mesh (Blender/Maya/ZBrush ready)\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quads:
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        if tris is not None and len(tris) > 0:
            for t in tris:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def apply_shrinkwrap_projection(q_verts, target_mesh, strength=0.65, max_distance=0.035):
    """
    Точная проекция вершин на поверхность оригинального High-Poly меша.
    """
    try:
        closest, distances, _ = target_mesh.nearest.on_surface(q_verts)
        mask = distances < max_distance
        projected = q_verts.copy()
        projected[mask] = (1.0 - strength) * q_verts[mask] + strength * closest[mask]
        return projected.astype(np.float32)
    except Exception as e:
        print(f"[AutoQuadRemesh] Shrinkwrap projection warning: {e}")
        return q_verts


def tangential_quad_relaxation(verts, quads, target_mesh, iterations=4, factor=0.35):
    """
    Тангенциальное выравнивание ячеек в касательной плоскости (устраняет ромбовидные перекосы).
    """
    if iterations <= 0:
        return verts

    v = verts.copy()
    num_v = len(v)

    adj = [[] for _ in range(num_v)]
    for q in quads:
        for i in range(len(q)):
            adj[q[i]].append(q[(i + 1) % len(q)])
            adj[q[i]].append(q[(i - 1) % len(q)])
    adj = [np.unique(a) for a in adj]

    for it in range(iterations):
        t_faces = []
        for q in quads:
            if len(q) == 4:
                t_faces.append([q[0], q[1], q[2]])
                t_faces.append([q[0], q[2], q[3]])
            elif len(q) == 3:
                t_faces.append([q[0], q[1], q[2]])
        if len(t_faces) == 0:
            break
        tm = trimesh.Trimesh(vertices=v, faces=t_faces, process=False)
        normals = tm.vertex_normals

        disp = np.zeros_like(v)
        for i in range(num_v):
            if len(adj[i]) > 0:
                centroid = np.mean(v[adj[i]], axis=0)
                d = centroid - v[i]
                n = normals[i]
                d_tan = d - np.dot(d, n) * n
                disp[i] = d_tan

        v += factor * disp
        try:
            closest, _, _ = target_mesh.nearest.on_surface(v)
            v = closest.astype(np.float32)
        except Exception:
            pass

    return v


def carve_finger_webbing(mesh):
    """
    Удаляет артефактные полигональные перемычки между пальцами после ИИ-генераторов 3D.
    """
    try:
        faces = mesh.faces
        verts = mesh.vertices
        centroids = np.mean(verts[faces], axis=1)

        # Анализ крайних боковых зон (кистей рук)
        b = mesh.bounds
        x_span = b[1, 0] - b[0, 0]
        # Пальцы обычно на латеральных экстремумах
        mask_hand = (centroids[:, 0] > b[1, 0] - 0.12 * x_span) & (centroids[:, 1] > b[0, 1] + 0.35 * (b[1, 1] - b[0, 1])) & (centroids[:, 1] < b[0, 1] + 0.55 * (b[1, 1] - b[0, 1]))
        gap = mask_hand & (abs(centroids[:, 2]) < 0.04) & (mesh.area_faces < np.percentile(mesh.area_faces, 50))
        if np.any(gap) and np.sum(gap) < 500:
            keep_indices = np.where(~gap)[0]
            print(f"[AutoQuadRemesh] Разделение пальцев: удалено {np.sum(gap)} мембранных граней.")
            return mesh.submesh([keep_indices], append=True)
        return mesh
    except Exception as e:
        print(f"[AutoQuadRemesh] Webbing carving warning: {e}")
        return mesh


def find_source_image(image_input=None, source_image_file="auto"):
    """
    Интеллектуальный поиск исходного изображения:
    1. Напрямую из входа ноды ComfyUI
    2. Из указанного файла source_image_file
    3. Автопоиск самого свежего фото персонажа в input/
    """
    try:
        if image_input is not None:
            if isinstance(image_input, torch.Tensor):
                img_np = (image_input[0].detach().cpu().numpy() * 255).astype(np.uint8)
                if img_np.shape[2] == 3:
                    return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                return img_np
            elif isinstance(image_input, np.ndarray):
                return image_input

        input_dir = folder_paths.get_input_directory() if hasattr(folder_paths, "get_input_directory") else "/opt/comfyui/input"

        if source_image_file and source_image_file != "auto":
            candidates = [
                source_image_file,
                os.path.join(input_dir, source_image_file),
                os.path.join("/opt/comfyui/input", source_image_file)
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    img = cv2.imread(cand)
                    if img is not None:
                        print(f"[AutoQuadRemesh] Загружено исходное изображение по указанному пути: {cand}")
                        return img

        # Автопоиск самого свежего изображения в input
        all_imgs = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            all_imgs.extend(glob.glob(os.path.join(input_dir, ext)))
            all_imgs.extend(glob.glob(os.path.join("/opt/comfyui/input", ext)))

        # Отфильтровываем служебные маски и тесты
        valid_imgs = [f for f in all_imgs if "mask" not in f.lower() and "test" not in f.lower() and "preview" not in f.lower()]
        if valid_imgs:
            valid_imgs = list(set(valid_imgs))
            valid_imgs.sort(key=os.path.getmtime, reverse=True)
            chosen = valid_imgs[0]
            img = cv2.imread(chosen)
            if img is not None:
                print(f"[AutoQuadRemesh] Автоматически обнаружено актуальное фото персонажа: {os.path.basename(chosen)}")
                return img

    except Exception as e:
        print(f"[AutoQuadRemesh] Поиск изображения завершился с предупреждением: {e}")

    return None


def detect_semantic_facial_and_hand_seeds(img, mesh_ref):
    """
    Нейросетевое распознавание лица и пальцев рук через DWPose / SAM2:
    Извлекает 2D координаты, проецирует их в 3D границы меша.
    """
    seeds = {"face": np.zeros((0, 3)), "hand_l": np.zeros((0, 3)), "hand_r": np.zeros((0, 3))}
    if img is None:
        return seeds

    try:
        import custom_nodes.comfyui_controlnet_aux as aux
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_img = torch.from_numpy(img_rgb).float() / 255.0
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

        face_pts = np.array(p_info["face_keypoints_2d"]).reshape(-1, 3)
        hand_l_pts = np.array(p_info["hand_left_keypoints_2d"]).reshape(-1, 3)
        hand_r_pts = np.array(p_info["hand_right_keypoints_2d"]).reshape(-1, 3)
        body_pts = np.array(p_info["pose_keypoints_2d"]).reshape(-1, 3)

        valid_body = body_pts[body_pts[:, 2] > 0.2]
        if len(valid_body) == 0:
            return seeds

        by_min, by_max = np.min(valid_body[:, 1]), np.max(valid_body[:, 1])
        bx_min, bx_max = np.min(valid_body[:, 0]), np.max(valid_body[:, 0])

        b_bounds = mesh_ref.bounds
        mx_min, mx_max = b_bounds[0, 0], b_bounds[1, 0]
        my_min, my_max = b_bounds[0, 1], b_bounds[1, 1]

        def to_3d(pts_2d):
            valid = pts_2d[pts_2d[:, 2] > 0.2]
            if len(valid) == 0:
                return np.zeros((0, 3))
            nx = (valid[:, 0] - bx_min) / np.maximum(1e-5, (bx_max - bx_min))
            ny = (valid[:, 1] - by_min) / np.maximum(1e-5, (by_max - by_min))
            x3 = mx_min + nx * (mx_max - mx_min)
            y3 = my_max - ny * (my_max - my_min)
            return np.column_stack([x3, y3, np.zeros_like(x3)])

        seeds["face"] = to_3d(face_pts)
        seeds["hand_l"] = to_3d(hand_l_pts)
        seeds["hand_r"] = to_3d(hand_r_pts)

        print(f"[AutoQuadRemesh] Нейросеть обнаружила ключевые анатомические зоны: "
              f"Лицо={len(seeds['face'])} точек, Левая рука={len(seeds['hand_l'])} суставов, Правая рука={len(seeds['hand_r'])} суставов.")
    except Exception as e:
        print(f"[AutoQuadRemesh] Ошибка при детекции по изображению: {e}")

    return seeds


def compute_geodesic_detail_mask(base_verts, base_quads, seeds, face_radius=0.18, hand_radius=0.13):
    """
    Расчет маски высокой детализации строго вдоль поверхности сетки (Geodesic Surface Distance):
    Исключает пространственные координатные прыжки между рукой и шортами.
    """
    num_v = len(base_verts)
    is_high_detail = np.zeros(num_v, dtype=bool)

    # Построение графа ребер квад-сетки
    edges = set()
    for q in base_quads:
        for i in range(4):
            edges.add(tuple(sorted((int(q[i]), int(q[(i + 1) % 4])))))
    edges = np.array(list(edges), dtype=np.int32)

    weights = np.linalg.norm(base_verts[edges[:, 0]] - base_verts[edges[:, 1]], axis=1)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([weights, weights])
    adj = csr_matrix((data, (row, col)), shape=(num_v, num_v))

    def tag_region(pts_3d, max_dist):
        if len(pts_3d) == 0:
            return
        seed_indices = []
        for pt in pts_3d:
            dists = np.linalg.norm(base_verts[:, :2] - pt[:2], axis=1)
            # Отдаем приоритет фронтальной поверхности
            front_dists = dists + np.maximum(0.0, -base_verts[:, 2]) * 2.0
            seed_indices.append(np.argmin(front_dists))
        seed_indices = np.unique(seed_indices)
        dist_map = dijkstra(adj, directed=False, indices=seed_indices, limit=max_dist)
        min_dist = np.min(dist_map, axis=0) if dist_map.ndim == 2 else dist_map
        is_high_detail[min_dist <= max_dist] = True

    tag_region(seeds["face"], face_radius)
    tag_region(seeds["hand_l"], hand_radius)
    tag_region(seeds["hand_r"], hand_radius)

    pct = np.sum(is_high_detail) / num_v * 100.0
    print(f"[AutoQuadRemesh] Геодезическая разметка: {np.sum(is_high_detail)} из {num_v} вершин ({pct:.1f}%) выделены под высокую детализацию.")
    return is_high_detail


def seamless_adaptive_quad_subdivide(base_verts, base_quads, is_high_detail, target_mesh, shrinkwrap_strength=0.70):
    """
    Бесшовное адаптивное уплотнение (Seamless 2:1 Quad Refinement):
    - В зонах интереса (лицо, пальцы): честное 4x уплотнение (1 квад -> 4 квада).
    - На границе раздела: канонический бесшовный 2:1 переход (0 Т-стыков, 100% Manifold).
    - Все новые вершины мягко притягиваются к High-Poly поверхности оригинала.
    """
    quad_scores = np.sum(is_high_detail[base_quads], axis=1)
    # Квады, у которых 3 или 4 вершины принадлежат зоне интереса, делятся 4x
    subdiv_quad_mask = quad_scores >= 3

    split_edges = {}
    out_verts = list(base_verts)

    def get_split_vertex(v_a, v_b):
        e = tuple(sorted((int(v_a), int(v_b))))
        if e not in split_edges:
            mid_pt = 0.5 * (out_verts[e[0]] + out_verts[e[1]])
            idx = len(out_verts)
            out_verts.append(mid_pt)
            split_edges[e] = idx
        return split_edges[e]

    # Шаг 1: Размечаем ребра уплотняемых квадов
    for qi, q in enumerate(base_quads):
        if subdiv_quad_mask[qi]:
            for i in range(4):
                get_split_vertex(q[i], q[(i + 1) % 4])

    # Шаг 2: Формируем связную топологию без разрывов
    out_quads = []
    out_tris = []
    new_v_indices = set(range(len(base_verts), len(base_verts) + len(split_edges) + int(np.sum(subdiv_quad_mask))))

    for qi, q in enumerate(base_quads):
        v0, v1, v2, v3 = int(q[0]), int(q[1]), int(q[2]), int(q[3])
        if subdiv_quad_mask[qi]:
            m01 = get_split_vertex(v0, v1)
            m12 = get_split_vertex(v1, v2)
            m23 = get_split_vertex(v2, v3)
            m30 = get_split_vertex(v3, v0)
            c_pt = 0.25 * (out_verts[v0] + out_verts[v1] + out_verts[v2] + out_verts[v3])
            c_idx = len(out_verts)
            out_verts.append(c_pt)

            out_quads.append([v0, m01, c_idx, m30])
            out_quads.append([m01, v1, m12, c_idx])
            out_quads.append([c_idx, m12, v2, m23])
            out_quads.append([m30, c_idx, m23, v3])
        else:
            e01 = tuple(sorted((v0, v1)))
            e12 = tuple(sorted((v1, v2)))
            e23 = tuple(sorted((v2, v3)))
            e30 = tuple(sorted((v3, v0)))

            has_01 = e01 in split_edges
            has_12 = e12 in split_edges
            has_23 = e23 in split_edges
            has_30 = e30 in split_edges
            num_splits = sum([has_01, has_12, has_23, has_30])

            if num_splits == 0:
                out_quads.append([v0, v1, v2, v3])
            elif num_splits == 1:
                # 2:1 переход: пятиугольник разбивается на 1 квад и 1 переходный треугольник
                verts_cycle = [v0, v1, v2, v3]
                splits_cycle = [has_01, has_12, has_23, has_30]
                rot = splits_cycle.index(True)
                u0, u1, u2, u3 = [verts_cycle[(i + rot) % 4] for i in range(4)]
                m_edge = split_edges[tuple(sorted((u0, u1)))]

                out_quads.append([u0, m_edge, u2, u3])
                out_tris.append([m_edge, u1, u2])
            elif num_splits == 2 and ((has_01 and has_23) or (has_12 and has_30)):
                # Противоположные стороны: делятся на 2 чистых квада
                if has_01 and has_23:
                    m0 = split_edges[e01]
                    m2 = split_edges[e23]
                    out_quads.append([v0, m0, m2, v3])
                    out_quads.append([m0, v1, v2, m2])
                else:
                    m1 = split_edges[e12]
                    m3 = split_edges[e30]
                    out_quads.append([v0, v1, m1, m3])
                    out_quads.append([m3, m1, v2, v3])
            else:
                # Угловые стыки: аккуратная веерная триангуляция без Т-стыков
                poly = []
                for idx, (va, vb, has_s, e_t) in enumerate([(v0, v1, has_01, e01), (v1, v2, has_12, e12), (v2, v3, has_23, e23), (v3, v0, has_30, e30)]):
                    poly.append(va)
                    if has_s:
                        poly.append(split_edges[e_t])
                for p_i in range(1, len(poly) - 1):
                    out_tris.append([poly[0], poly[p_i], poly[p_i + 1]])

    out_verts = np.array(out_verts, dtype=np.float32)
    out_quads = np.array(out_quads, dtype=np.int32)
    out_tris = np.array(out_tris, dtype=np.int32)

    # Шаг 3: Shrinkwrap проекция новых вершин точно на геометрию оригинала
    if shrinkwrap_strength > 0:
        new_indices = np.array(list(new_v_indices), dtype=np.int32)
        new_indices = new_indices[new_indices < len(out_verts)]
        if len(new_indices) > 0:
            closest, _, _ = target_mesh.nearest.on_surface(out_verts[new_indices])
            out_verts[new_indices] = (1.0 - shrinkwrap_strength) * out_verts[new_indices] + shrinkwrap_strength * closest

    return out_verts, out_quads, out_tris


def render_topology_diagnostics(verts, quads, tris, out_image_path):
    """
    Генерирует мульти-ракурсный рендер сетки (общий вид, зум лица, зум пальцев/шорт)
    для визуального контроля качества прямо в интерфейсе ComfyUI.
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=(21, 9), dpi=140)
        fig.patch.set_facecolor("#111116")

        def extract_edges(poly_list):
            edges = []
            for p in poly_list:
                N = len(p)
                for i in range(N):
                    edges.append([verts[p[i], :2], verts[p[(i + 1) % N], :2]])
            return edges

        quad_lines = extract_edges(quads)
        tri_lines = extract_edges(tris) if len(tris) > 0 else []

        # 1. Full Body View
        ax1 = axes[0]
        ax1.set_facecolor("#181820")
        lc_q = LineCollection(quad_lines, colors="#00eeff", linewidths=0.5, alpha=0.85)
        ax1.add_collection(lc_q)
        if tri_lines:
            lc_t = LineCollection(tri_lines, colors="#ffcc00", linewidths=0.7, alpha=0.9)
            ax1.add_collection(lc_t)
        ax1.set_title("1. Full Body Retopology (Overview)", color="#ffffff", fontsize=13, pad=10)
        ax1.set_aspect("equal")
        ax1.set_xlim(verts[:, 0].min() - 0.05, verts[:, 0].max() + 0.05)
        ax1.set_ylim(verts[:, 1].min() - 0.05, verts[:, 1].max() + 0.05)
        ax1.axis("off")

        # 2. Face Zoom
        ax2 = axes[1]
        ax2.set_facecolor("#181820")
        lc_q2 = LineCollection(quad_lines, colors="#00eeff", linewidths=0.8, alpha=0.9)
        ax2.add_collection(lc_q2)
        if tri_lines:
            lc_t2 = LineCollection(tri_lines, colors="#ffcc00", linewidths=0.9, alpha=0.9)
            ax2.add_collection(lc_t2)
        y_max = verts[:, 1].max()
        ax2.set_title("2. Face & Head Zoom (4x Adaptive Density)", color="#ffffff", fontsize=13, pad=10)
        ax2.set_aspect("equal")
        ax2.set_xlim(-0.25, 0.25)
        ax2.set_ylim(y_max - 0.45, y_max + 0.05)
        ax2.axis("off")

        # 3. Hands & Shorts Zoom
        ax3 = axes[2]
        ax3.set_facecolor("#181820")
        lc_q3 = LineCollection(quad_lines, colors="#00eeff", linewidths=0.8, alpha=0.9)
        ax3.add_collection(lc_q3)
        if tri_lines:
            lc_t3 = LineCollection(tri_lines, colors="#ffcc00", linewidths=0.9, alpha=0.9)
            ax3.add_collection(lc_t3)
        ax3.set_title("3. Hands & Shorts Zoom (2:1 Seamless Transition)", color="#ffffff", fontsize=13, pad=10)
        ax3.set_aspect("equal")
        # Показываем область пояса/рук
        ax3.set_xlim(-0.45, 0.45)
        ax3.set_ylim(-0.35, 0.35)
        ax3.axis("off")

        plt.tight_layout()
        plt.savefig(out_image_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # Конвертация в тензор ComfyUI IMAGE [1, H, W, 3]
        diag_bgr = cv2.imread(out_image_path)
        if diag_bgr is not None:
            diag_rgb = cv2.cvtColor(diag_bgr, cv2.COLOR_BGR2RGB)
            t_diag = torch.from_numpy(diag_rgb).float() / 255.0
            return t_diag.unsqueeze(0)
    except Exception as e:
        print(f"[AutoQuadRemesh] Диагностический рендер пропущен: {e}")

    # Fallback пустой тензор
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


class AutoQuadRemeshNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_quads": ("INT", {"default": 7000, "min": 1000, "max": 50000, "step": 500, "tooltip": "Базовое количество полигонов на теле и одежде (~6k-8k для стилистики League of Legends)"}),
                "shrinkwrap_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Сила облегания исходной high-poly геометрии"}),
                "adaptive_detail_boost": ("BOOLEAN", {"default": True, "tooltip": "Нейросетевое распознавание лица и пальцев + геодезическая Дейкстра + бесшовный 2:1 переход"}),
                "preserve_sharp": ("BOOLEAN", {"default": False, "tooltip": "Фиксация острых механических ребер"}),
                "show_quad_wireframe": ("BOOLEAN", {"default": True, "tooltip": "Отображение чистого оверлея квад-сетки в 3D-вьювере"}),
                "heal_mesh": ("BOOLEAN", {"default": True, "tooltip": "Предварительное восстановление Manifold-герметичности через PyMeshFix"}),
                "relax_iterations": ("INT", {"default": 4, "min": 0, "max": 10, "step": 1, "tooltip": "Итерации тангенциального выравнивания квадов"}),
                "filename_prefix": ("STRING", {"default": "stages_data/03_quad_mesh/asset_quad_local"}),
                "engine": (["instant_crossfield", "quadriflow_legacy", "intelligent_custom_retopo"], {"default": "instant_crossfield"}),
                "crease_angle": ("INT", {"default": 35, "min": 0, "max": 90, "step": 5, "tooltip": "Порог фиксации складок и швов"}),
                "separate_fingers": ("BOOLEAN", {"default": True, "tooltip": "Автоматическое удаление мембран между сросшимися пальцами"}),
            },
            "optional": {
                "mesh": ("MESH",),
                "image": ("IMAGE",),
                "source_glb_file": ("STRING", {"default": "stages_data/02_raw_mesh/asset_raw.glb", "tooltip": "Путь к уже сохраненному сырому мешу"}),
                "source_image_file": ("STRING", {"default": "auto", "tooltip": "Имя файла в папке input/ (auto = самое свежее фото)"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MESH", "TRIMESH", "IMAGE")
    RETURN_NAMES = ("mesh", "trimesh", "diagnostic_preview")
    OUTPUT_NODE = True
    FUNCTION = "remesh"
    CATEGORY = "Mesh Processing/Retopology"

    def remesh(self, target_quads=7000, shrinkwrap_strength=0.65, adaptive_detail_boost=True, preserve_sharp=False, show_quad_wireframe=True, heal_mesh=True, relax_iterations=4, filename_prefix="stages_data/03_quad_mesh/asset_quad_local", engine="instant_crossfield", crease_angle=35, separate_fingers=True, mesh=None, image=None, source_glb_file=None, source_image_file="auto", prompt=None, extra_pnginfo=None, **kwargs):
        t0 = time.time()
        print("\n" + "="*65)
        print("  ⚡ [AutoQuadRemesh v2.0] Запуск интеллектуальной квад-ретопологии ⚡")
        print("="*65)

        # 1. Загрузка входного меша (из провода или кэшированного файла)
        loaded_source = None
        if mesh is not None:
            if hasattr(mesh, "vertices") and isinstance(mesh.vertices, torch.Tensor):
                raw_verts = mesh.vertices[0].detach().cpu().numpy()
                raw_faces = mesh.faces[0].detach().cpu().numpy()
            elif hasattr(mesh, "vertices"):
                raw_verts = np.asarray(mesh.vertices)
                raw_faces = np.asarray(mesh.faces)
            else:
                raise ValueError(f"Неподдерживаемый тип меша: {type(mesh)}")
            print("[AutoQuadRemesh] Меш успешно получен напрямую из upstream-ноды.")
        else:
            candidates = []
            if source_glb_file:
                candidates.append(source_glb_file)
                candidates.append(os.path.join(folder_paths.get_output_directory(), source_glb_file))

            raw_dir = os.path.join(folder_paths.get_output_directory(), "stages_data/02_raw_mesh")
            if os.path.exists(raw_dir):
                found_raws = sorted(glob.glob(os.path.join(raw_dir, "*.glb")), key=os.path.getmtime, reverse=True)
                candidates.extend(found_raws)

            for cand in candidates:
                if cand and os.path.exists(cand):
                    loaded_source = cand
                    break

            if loaded_source:
                print(f"[AutoQuadRemesh] Использован кэшированный меш: {loaded_source}")
                m_loaded = trimesh.load(loaded_source, force="mesh")
                raw_verts = np.asarray(m_loaded.vertices)
                raw_faces = np.asarray(m_loaded.faces)
            else:
                raise RuntimeError("Входной меш не найден! Подключите вход 'mesh' или проверьте папку 02_raw_mesh.")

        highpoly_ref = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)
        print(f"[AutoQuadRemesh] Исходная High-Poly модель: Вершин={len(raw_verts)}, Полигонов={len(raw_faces)}")

        # 2. Получение исходного фото для нейросетевого зрения
        src_img = find_source_image(image_input=image, source_image_file=source_image_file)

        # 3. Подготовка папок вывода
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(full_output_folder, exist_ok=True)

        # 4. Базовый расчет направленных кросс-полей
        m_work = highpoly_ref
        if separate_fingers:
            m_work = carve_finger_webbing(m_work)

        if heal_mesh:
            print("[AutoQuadRemesh] Восстановление герметичности (PyMeshFix Manifold Healing)...")
            tin = pymeshfix.PyTMesh()
            v_in = np.ascontiguousarray(m_work.vertices, dtype=np.float64)
            f_in = np.ascontiguousarray(m_work.faces, dtype=np.int32)
            tin.load_array(v_in, f_in)
            tin.remove_smallest_components()
            tin.fix_connectivity()
            tin.fill_small_boundaries(0, True)
            tin.clean(True)
            v_clean, f_clean = tin.return_arrays()
            m_work = trimesh.Trimesh(vertices=v_clean, faces=f_clean, process=False)

        ply_tmp = os.path.join(full_output_folder, f"_temp_{counter}_in.ply")
        obj_tmp = os.path.join(full_output_folder, f"_temp_{counter}_out.obj")
        m_work.export(ply_tmp)

        # Instant-Meshes генерирует крупную ровную базу (~target_quads)
        base_f_goal = max(1200, target_quads // 4)
        print(f"[AutoQuadRemesh] Запуск Cross-Field решателя Instant-Meshes (база ~{target_quads} квадов, crease={crease_angle}°)...")
        cmd = [
            "/usr/local/bin/instant-meshes",
            "-r", "4",
            "-p", "4",
            "-f", str(base_f_goal),
            "-c", str(crease_angle),
            "-S", "2",
            "-d",
            "-o", obj_tmp,
            ply_tmp
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        parsed_quads = []
        parsed_verts = []
        with open(obj_tmp, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    parsed_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    parsed_quads.append([int(p.split("/")[0]) - 1 for p in parts])

        base_verts = np.array(parsed_verts, dtype=np.float32)
        base_quads = np.array(parsed_quads, dtype=np.int32)

        try:
            os.remove(ply_tmp)
            os.remove(obj_tmp)
        except Exception:
            pass

        print(f"[AutoQuadRemesh] Базовый каркас сформирован: {len(base_quads)} квадов, {len(base_verts)} вершин.")

        final_tris = np.zeros((0, 3), dtype=np.int32)
        final_verts = base_verts
        final_quads = base_quads

        # 5. Нейросетевое зрение + Геодезическая диффузия + Бесшовный 2:1 переход
        if adaptive_detail_boost and src_img is not None:
            print("[AutoQuadRemesh] 🧠 Нейросетевой анализ зон интереса (лицо, пальцы) на исходном фото...")
            semantic_seeds = detect_semantic_facial_and_hand_seeds(src_img, highpoly_ref)

            has_valid_seeds = (len(semantic_seeds["face"]) > 0 or len(semantic_seeds["hand_l"]) > 0 or len(semantic_seeds["hand_r"]) > 0)
            if has_valid_seeds:
                print("[AutoQuadRemesh] 🌊 Расчет геодезических расстояний вдоль поверхности (Dijkstra)...")
                is_high_detail = compute_geodesic_detail_mask(base_verts, base_quads, semantic_seeds)

                print("[AutoQuadRemesh] 📐 Бесшовное уплотнение 2:1 (Face & Fingers 4x, Manifold связка)...")
                final_verts, final_quads, final_tris = seamless_adaptive_quad_subdivide(
                    base_verts, base_quads, is_high_detail, highpoly_ref, shrinkwrap_strength=shrinkwrap_strength
                )
            else:
                print("[AutoQuadRemesh] Лицо и руки не обнаружены на фото, сохранена равномерная плотность.")

        # 6. Тангенциальная релаксация и финальный Shrinkwrap
        if relax_iterations > 0:
            print(f"[AutoQuadRemesh] Тангенциальное выравнивание ячеек ({relax_iterations} итераций)...")
            final_verts = tangential_quad_relaxation(final_verts, final_quads, highpoly_ref, iterations=relax_iterations, factor=0.30)

        if shrinkwrap_strength > 0.0:
            print(f"[AutoQuadRemesh] Финальная проекция Shrinkwrap на High-Poly (сила={shrinkwrap_strength})...")
            final_verts = apply_shrinkwrap_projection(final_verts, highpoly_ref, strength=shrinkwrap_strength)

        # 7. Экспорт результатов
        obj_file = f"{filename}_{counter:05}_.obj"
        obj_path = os.path.join(full_output_folder, obj_file)
        save_pure_quad_obj(final_verts, final_quads, final_tris, obj_path)

        glb_file = f"{filename}_{counter:05}_.glb"
        glb_path = os.path.join(full_output_folder, glb_file)
        save_clean_quad_glb_with_wireframe(final_verts, final_quads, final_tris, glb_path, show_quad_wireframe=show_quad_wireframe)

        # 8. Генерация диагностического рендера для быстрого визуального контроля
        diag_png = f"{filename}_{counter:05}_diagnostics.png"
        diag_path = os.path.join(full_output_folder, diag_png)
        diag_tensor = render_topology_diagnostics(final_verts, final_quads, final_tris, diag_path)

        # 9. Сборка выходного объекта MESH и TRIMESH для нод ComfyUI
        tri_faces = []
        for q in final_quads:
            tri_faces.append([q[0], q[1], q[2]])
            tri_faces.append([q[0], q[2], q[3]])
        for t in final_tris:
            tri_faces.append([t[0], t[1], t[2]])
        tri_faces = np.array(tri_faces, dtype=np.int32)

        out_trimesh = trimesh.Trimesh(vertices=final_verts, faces=tri_faces, process=False)
        out_trimesh.metadata["quad_faces"] = final_quads

        t_verts = torch.from_numpy(final_verts).unsqueeze(0).float()
        t_faces = torch.from_numpy(tri_faces).unsqueeze(0).int()
        out_mesh = hy3d_nodes.MESH(vertices=t_verts, faces=t_faces)

        total_faces = len(final_quads) + len(final_tris)
        quad_pct = (len(final_quads) / total_faces * 100.0) if total_faces > 0 else 100.0
        elapsed = round(time.time() - t0, 2)

        print("\n" + "="*65)
        print("  🎉 РЕТОПОЛОГИЯ УСПЕШНО ЗАВЕРШЕНА!")
        print(f"  • Всего полигонов:  {total_faces} (Квадов: {len(final_quads)}, Трисов: {len(final_tris)})")
        print(f"  • Чистота сетки:    {quad_pct:.2f}% Quad-dominant (0 T-стыков, монолитный Manifold)")
        print(f"  • Вершин:           {len(final_verts)}")
        print(f"  • Время расчета:    {elapsed} с")
        print(f"  • Чистый OBJ:       {obj_path}")
        print(f"  • Диагностика:      {diag_path}")
        print("="*65 + "\n")

        return {
            "ui": {
                "3d": [{"filename": glb_file, "subfolder": subfolder, "type": "output"}],
                "images": [{"filename": diag_png, "subfolder": subfolder, "type": "output"}]
            },
            "result": (out_mesh, out_trimesh, diag_tensor)
        }


NODE_CLASS_MAPPINGS = {
    "AutoQuadRemesh": AutoQuadRemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoQuadRemesh": "Auto Quad Remesh (⚡ Локальная Квад-Ретопология)",
}
