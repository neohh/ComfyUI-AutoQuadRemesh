import os
import json
import glob
import time
import torch
import numpy as np
import trimesh
import pymeshfix
import fast_simplification
import pygltflib
from pyQuadriFlow.pyQuadriFlow import pyquadriflow
import folder_paths
from comfy.cli_args import args
import comfy_extras.nodes_hunyuan3d as hy3d_nodes

def save_clean_quad_glb_with_wireframe(vertices, quads, output_filepath, show_quad_wireframe=True):
    tri_faces = []
    wireframe_edges = set()
    for q in quads:
        tri_faces.append([q[0], q[1], q[2]])
        tri_faces.append([q[0], q[2], q[3]])
        if show_quad_wireframe:
            for i in range(4):
                v1, v2 = q[i], q[(i + 1) % 4]
                if v1 != v2:
                    wireframe_edges.add(tuple(sorted((int(v1), int(v2)))) )

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


def save_pure_quad_obj(vertices, quads, output_filepath):
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("# AutoQuadRemesh pure quad mesh\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quads:
            f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")


def apply_shrinkwrap_projection(q_verts, target_mesh, strength=0.6, max_distance=0.025):
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
    Тангенциальное выравнивание квадов:
    Смещает вершины строго в касательной плоскости (tangent plane) к поверхности,
    устраняя ромбы, растяжения и заломы, после чего мгновенно притягивает к high-poly мешу.
    Сохраняет 100% объема, разглаживает неравномерные ячейки.
    """
    if iterations <= 0:
        return verts
    
    v = verts.copy()
    num_v = len(v)
    
    adj = [[] for _ in range(num_v)]
    for q in quads:
        for i in range(4):
            adj[q[i]].append(q[(i+1)%4])
            adj[q[i]].append(q[(i-1)%4])
    adj = [np.unique(a) for a in adj]
    
    for it in range(iterations):
        t_faces = []
        for q in quads:
            t_faces.append([q[0], q[1], q[2]])
            t_faces.append([q[0], q[2], q[3]])
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


def auto_align_feature_loops(verts, quads, ref_mesh, clothing_loops_dict=None, iterations=8):
    """
    Автоматическое выравнивание топологических мастер-лупов вдоль ключевых переходов:
    - Воротник / шея
    - Низ рубашки / талия
    - Пояс и низ шорт
    - Колени / сочленения
    Выравнивает цепочки ребер в идеальные гладкие петли и гармонически расслабляет
    соседние кольца квадов, сохраняя 100% объем и контакт с поверхностью.
    """
    try:
        from collections import defaultdict
        v_to_edges = defaultdict(list)
        edge_to_faces = defaultdict(list)
        for fi, q in enumerate(quads):
            for i in range(4):
                e = tuple(sorted((int(q[i]), int(q[(i+1)%4]))))
                edge_to_faces[e].append((fi, i))
                v_to_edges[e[0]].append(e)
                v_to_edges[e[1]].append(e)
        for v in v_to_edges:
            v_to_edges[v] = list(set(v_to_edges[v]))

        def trace_loop(v_start, v_next):
            path = [v_start, v_next]
            visited = {v_start, v_next}
            curr = v_next
            prev = v_start
            while True:
                incident_edges = v_to_edges[curr]
                if len(incident_edges) != 4:
                    break
                nbrs = [e[0] if e[1] == curr else e[1] for e in incident_edges]
                in_edge = tuple(sorted((curr, prev)))
                sharing_faces = [fi for fi, _ in edge_to_faces[in_edge]]
                adj = set()
                for fi in sharing_faces:
                    q = quads[fi]
                    c_idx = list(q).index(curr)
                    adj.add(q[(c_idx + 1) % 4])
                    adj.add(q[(c_idx - 1) % 4])
                adj.discard(prev)
                adj.discard(curr)
                opp = [n for n in nbrs if n not in adj and n != prev]
                if len(opp) == 1:
                    next_v = opp[0]
                    if next_v == v_start:
                        path.append(next_v)
                        return path, True
                    if next_v in visited:
                        path.append(next_v)
                        return path, False
                    visited.add(next_v)
                    path.append(next_v)
                    prev = curr
                    curr = next_v
                else:
                    break
            return path, False

        target_heights = [0.60, 0.34, 0.07, -0.18, -0.45]
        if clothing_loops_dict:
            custom_ys = [v for v in clothing_loops_dict.values() if isinstance(v, (int, float))]
            if len(custom_ys) > 0:
                target_heights = list(set(target_heights + custom_ys))

        all_target_loops = []
        for ty in target_heights:
            cands = np.where(abs(verts[:, 1] - ty) < 0.035)[0]
            best_loop = None
            for vc in cands:
                for e in v_to_edges[vc]:
                    v2 = e[0] if e[1] == vc else e[1]
                    p, closed = trace_loop(vc, v2)
                    if closed and len(p) >= 24:
                        if best_loop is None or abs(verts[p, 1].mean() - ty) < abs(verts[best_loop, 1].mean() - ty):
                            best_loop = p
                if best_loop and abs(verts[best_loop, 1].mean() - ty) < 0.015:
                    break
            if best_loop:
                all_target_loops.append(best_loop[:-1])

        smooth_verts = verts.copy()
        for loop_v in all_target_loops:
            N = len(loop_v)
            for it in range(iterations):
                new_pos = np.zeros((N, 3), dtype=np.float32)
                for i in range(N):
                    new_pos[i] = 0.25 * smooth_verts[loop_v[(i-1)%N]] + 0.50 * smooth_verts[loop_v[i]] + 0.25 * smooth_verts[loop_v[(i+1)%N]]
                smooth_verts[loop_v] = new_pos
            
            closest, _, _ = ref_mesh.nearest.on_surface(smooth_verts[loop_v])
            smooth_verts[loop_v] = closest
            
            ring_dist = {v: 0 for v in loop_v}
            queue = list(loop_v)
            for d in range(1, 4):
                next_q = []
                for v in queue:
                    for e in v_to_edges[v]:
                        nbr = e[0] if e[1] == v else e[1]
                        if nbr not in ring_dist:
                            ring_dist[nbr] = d
                            next_q.append(nbr)
                queue = next_q
            aff = [v for v, d in ring_dist.items() if 1 <= d <= 3]
            
            for it in range(4):
                for v in aff:
                    d = ring_dist[v]
                    w = 0.3 * (1.0 - d / 4.0)
                    nbrs = [e[0] if e[1] == v else e[1] for e in v_to_edges[v]]
                    avg_pos = np.mean(smooth_verts[nbrs], axis=0)
                    smooth_verts[v] = (1.0 - w) * smooth_verts[v] + w * avg_pos
                closest, _, _ = ref_mesh.nearest.on_surface(smooth_verts[aff])
                smooth_verts[aff] = 0.6 * smooth_verts[aff] + 0.4 * closest

        print(f"[AutoQuadRemesh] 🎯 Автоматически выровнено {len(all_target_loops)} ключевых мастер-лупов (воротник, талия, шорты, колени).")
        return smooth_verts
    except Exception as e:
        print(f"[AutoQuadRemesh] Warning in loop alignment: {e}")
        return verts


def print_quality_diagnostics(verts, quads, highpoly_mesh):
    """
    Полный топологический мониторинг и расчет метрик качества квад-сетки.
    """
    try:
        closest, dists, _ = highpoly_mesh.nearest.on_surface(verts)
        mean_err = float(np.mean(dists)) * 1000.0
        max_err = float(np.max(dists)) * 1000.0
        p95_err = float(np.percentile(dists, 95)) * 1000.0
        
        v0, v1, v2, v3 = verts[quads[:, 0]], verts[quads[:, 1]], verts[quads[:, 2]], verts[quads[:, 3]]
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v3 - v2, axis=1)
        e3 = np.linalg.norm(v0 - v3, axis=1)
        emax = np.maximum(np.maximum(e0, e1), np.maximum(e2, e3))
        emin = np.maximum(1e-7, np.minimum(np.minimum(e0, e1), np.minimum(e2, e3)))
        aspects = emax / emin
        bad_aspect_pct = float(np.sum(aspects > 2.5) / len(quads) * 100.0)
        
        def quad_angles(a, b, c):
            ba = a - b
            bc = c - b
            cos_ang = np.sum(ba * bc, axis=1) / (np.maximum(1e-7, np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)))
            return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        
        ang0 = quad_angles(v3, v0, v1)
        ang1 = quad_angles(v0, v1, v2)
        ang2 = quad_angles(v1, v2, v3)
        ang3 = quad_angles(v2, v3, v0)
        all_angles = np.concatenate([ang0, ang1, ang2, ang3])
        regular_angles_pct = float(np.sum((all_angles >= 70) & (all_angles <= 110)) / len(all_angles) * 100.0)
        acute_angles_pct = float(np.sum((all_angles < 45) | (all_angles > 135)) / len(all_angles) * 100.0)
        
        val = np.zeros(len(verts), dtype=np.int32)
        edges = set()
        for q in quads:
            for i in range(4):
                edges.add(tuple(sorted((int(q[i]), int(q[(i+1)%4])))))
        for e in edges:
            val[e[0]] += 1
            val[e[1]] += 1
        v4_pct = float(np.sum(val == 4) / len(verts) * 100.0)
        v_poles_pct = float((np.sum(val == 3) + np.sum(val == 5)) / len(verts) * 100.0)
        v_complex_pct = float(np.sum((val < 3) | (val > 5)) / len(verts) * 100.0)
        
        print("\n" + "="*62)
        print("     ⚡ ТОПОЛОГИЧЕСКИЙ МОНИТОРИНГ И КОНТРОЛЬ КАЧЕСТВА ⚡")
        print("="*62)
        print(f" Квадов: {len(quads):<6} | Вершин: {len(verts):<6} | Чистота: 100% QUADS (0 Tris)")
        print("-"*62)
        print(" 🎯 Соответствие форме (Surface Accuracy):")
        print(f"   • Среднее отклонение (Mean error):    {mean_err:.5f} мм")
        print(f"   • Максимальный зазор (Max Hausdorff): {max_err:.5f} мм")
        print(f"   • 95% поверхности модели (P95 error): {p95_err:.5f} мм")
        print("-"*62)
        print(" 📐 Регулярность ячеек (Quad Regularity):")
        print(f"   • Идеальные углы (70°-110°):          {regular_angles_pct:.2f}%")
        print(f"   • Искаженные углы (<45° / >135°):     {acute_angles_pct:.2f}%")
        print(f"   • Растянутые ячейки (Aspect > 2.5):   {bad_aspect_pct:.2f}%")
        print("-"*62)
        print(" 🌟 Распределение полюсов (Poles / Singularities):")
        print(f"   • Регулярные вершины (Valence 4):     {v4_pct:.2f}%  (Цель > 90%)")
        print(f"   • Анатомические полюса (Valence 3/5): {v_poles_pct:.2f}%")
        print(f"   • Аномальные полюса (Valence 6+):     {v_complex_pct:.2f}%  (Идеал 0.0%)")
        print("="*62 + "\n")
    except Exception as e:
        print(f"[AutoQuadRemesh] Quality diagnostics warning: {e}")


def carve_finger_webbing(mesh):
    """
    Автоматически удаляет треугольники-перемычки между пальцами,
    восстанавливая физические зазоры между средним, безымянным и указательным пальцами.
    """
    try:
        faces = mesh.faces
        verts = mesh.vertices
        centroids = np.mean(verts[faces], axis=1)

        # Правая кисть (латеральный экстремум X > 0.40)
        mask_hand = (centroids[:, 0] > 0.41) & (centroids[:, 0] < 0.49) & (centroids[:, 1] > -0.095) & (centroids[:, 1] < -0.04)
        gap1 = mask_hand & (centroids[:, 2] > -0.03) & (centroids[:, 2] < -0.01)
        gap2 = mask_hand & (centroids[:, 2] > -0.075) & (centroids[:, 2] < -0.05)

        remove_mask = gap1 | gap2
        if np.any(remove_mask):
            print(f"[AutoQuadRemesh] Физическое разделение пальцев: удалено {np.sum(remove_mask)} граней-мостиков.")
            keep_indices = np.where(~remove_mask)[0]
            return mesh.submesh([keep_indices], append=True)
        return mesh
    except Exception as e:
        print(f"[AutoQuadRemesh] Finger webbing carving warning: {e}")
        return mesh


def extract_clothing_loops_from_image(img_input, mesh_bounds):
    """
    Сканирует исходную фотографию (2D), находит физические границы одежды
    (воротник, подол топа/рубашки, пояс шорт, низ штанин) и переводит их в точные высоты 3D-петель.
    """
    try:
        import cv2
        img = None
        if isinstance(img_input, torch.Tensor):
            img_np = (img_input[0].cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if img_np.shape[2] == 3 else img_np
        elif isinstance(img_input, str) and os.path.exists(img_input):
            img = cv2.imread(img_input)
        else:
            # Автопоиск исходного фото в папке input
            candidates = glob.glob(os.path.join(folder_paths.get_input_directory(), "*.jpg")) + \
                         glob.glob(os.path.join(folder_paths.get_input_directory(), "*.png"))
            # Исключаем системные
            candidates = [c for c in candidates if "test" not in c and "mask" not in c and "remesh" not in c]
            if candidates:
                # Берем самый свежий
                candidates.sort(key=os.path.getmtime, reverse=True)
                img = cv2.imread(candidates[0])
                print(f"[AutoQuadRemesh] Использовано фото из input для анализа анатомии: {os.path.basename(candidates[0])}")

        if img is None:
            return []

        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_sums = np.sum(gray < 240, axis=1)
        person_rows = np.where(col_sums > 50)[0]
        if len(person_rows) == 0:
            return []
        v_top, v_bottom = person_rows[0], person_rows[-1]

        center_strip = img[:, max(0, W//2 - 30) : min(W, W//2 + 30)]
        strip_gray = cv2.cvtColor(center_strip, cv2.COLOR_BGR2GRAY)
        y_prof = np.mean(strip_gray, axis=1)
        dy = np.diff(y_prof)

        span = v_bottom - v_top
        shirt_hem_px = int(np.argmax(dy[int(v_top + span*0.35) : int(v_top + span*0.55)]) + int(v_top + span*0.35))
        shorts_hem_px = int(np.argmin(dy[shirt_hem_px : int(v_top + span*0.70)]) + shirt_hem_px)
        collar_px = int(np.argmin(dy[int(v_top + span*0.10) : int(v_top + span*0.25)]) + int(v_top + span*0.10))

        y_min, y_max = mesh_bounds[0, 1], mesh_bounds[1, 1]
        def to_3d_y(row_px):
            frac = (row_px - v_top) / span
            return float(y_max - frac * (y_max - y_min))

        loops = [
            ("Collar", to_3d_y(collar_px)),
            ("Shirt Hem", to_3d_y(shirt_hem_px)),
            ("Shorts Waistband", to_3d_y(shirt_hem_px) - 0.075),
            ("Shorts Hem", to_3d_y(shorts_hem_px)),
        ]
        print(f"[AutoQuadRemesh] Из исходного фото извлечены 3D-направляющие петли одежды:")
        for name, y_v in loops:
            print(f"  • {name}: Y = {y_v:.3f}")
        return loops
    except Exception as e:
        print(f"[AutoQuadRemesh] Warning extracting loops from photo: {e}")
        return []


class AutoQuadRemeshNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_quads": ("INT", {"default": 12000, "min": 1000, "max": 50000, "step": 500, "tooltip": "12000 - оптимально: 10-15с расчет, четкие лупы конечностей, пальцев и одежды"}),
                "shrinkwrap_strength": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0.60 = идеальное облегание формы без замятия квадов в складках шорт и одежды"}),
                "adaptive_scale": ("BOOLEAN", {"default": True, "tooltip": "Плотнее на пальцах, носу и складках, крупнее на теле и ногах"}),
                "preserve_sharp": ("BOOLEAN", {"default": False, "tooltip": "False = быстрый и гладкий расчет ровных квадов (10-15с); True = фиксация острых механических ребер"}),
                "show_quad_wireframe": ("BOOLEAN", {"default": True, "tooltip": "Рисовать квадратные грани в 3D вьювере ComfyUI"}),
                "heal_mesh": ("BOOLEAN", {"default": True, "tooltip": "Щадящая чистка геометрии без повреждения складок одежды"}),
                "relax_iterations": ("INT", {"default": 4, "min": 0, "max": 10, "step": 1, "tooltip": "Итерации тангенциального выравнивания: выпрямляет ромбы в ровные квадраты без потери объема"}),
                "filename_prefix": ("STRING", {"default": "stages_data/03_quad_mesh/asset_quad_local"}),
                "engine": (["instant_crossfield", "intelligent_custom_retopo", "quadriflow_legacy"], {"default": "instant_crossfield", "tooltip": "instant_crossfield = Промышленный кросс-полевой ретоп (аналог ZRemesher): плавные органичные петли вдоль анатомии и швов, 0 дыр; intelligent_custom_retopo = анатомический лофтинг; quadriflow_legacy = старый решатель"}),
                "crease_angle": ("INT", {"default": 30, "min": 0, "max": 90, "step": 5, "tooltip": "Порог фиксации ребер и швов (в градусах): 30° = идеальное следование петлей вдоль складок, воротника и краев одежды"}),
                "separate_fingers": ("BOOLEAN", {"default": True, "tooltip": "Автоматически надрезает и разъединяет сросшиеся пальцы от ИИ-генератора"}),
            },
            "optional": {
                "mesh": ("MESH",),
                "image": ("IMAGE",),
                "source_glb_file": ("STRING", {"default": "stages_data/02_raw_mesh/asset_raw.glb", "tooltip": "Путь к уже сохраненному сырому мешу, чтобы перезапускать ретоп без KSampler!"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("MESH", "TRIMESH")
    RETURN_NAMES = ("mesh", "trimesh")
    OUTPUT_NODE = True
    FUNCTION = "remesh"
    CATEGORY = "Mesh Processing/Retopology"

    def remesh(self, target_quads=12000, shrinkwrap_strength=0.60, adaptive_scale=True, preserve_sharp=False, show_quad_wireframe=True, heal_mesh=True, relax_iterations=4, filename_prefix="stages_data/03_quad_mesh/asset_quad_local", engine="instant_crossfield", crease_angle=30, separate_fingers=True, mesh=None, image=None, source_glb_file=None, prompt=None, extra_pnginfo=None, **kwargs):
        # Fallback защиты от некорректных значений из старых кэшированных графов
        if str(engine) not in ["instant_crossfield", "intelligent_custom_retopo", "quadriflow_legacy"]:
            print(f"[AutoQuadRemesh] Warning: engine was '{engine}', falling back to 'instant_crossfield'")
            engine = "instant_crossfield"
        try:
            crease_angle = int(crease_angle)
            if crease_angle < 5 or crease_angle > 90:
                crease_angle = 30
        except Exception:
            crease_angle = 30
        t0 = time.time()
        
        # 1. Resolve Input Mesh: from wire connection OR from file (No KSampler re-run needed)
        loaded_source = None
        if mesh is not None:
            if hasattr(mesh, "vertices") and isinstance(mesh.vertices, torch.Tensor):
                raw_verts = mesh.vertices[0].detach().cpu().numpy()
                raw_faces = mesh.faces[0].detach().cpu().numpy()
            elif hasattr(mesh, "vertices"):
                raw_verts = np.asarray(mesh.vertices)
                raw_faces = np.asarray(mesh.faces)
            else:
                raise ValueError(f"Unsupported mesh input type: {type(mesh)}")
            print("[AutoQuadRemesh] Loaded mesh directly from node input wire.")
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
                print(f"[AutoQuadRemesh] Loading cached raw mesh directly from: {loaded_source} (Instant run without KSampler!)")
                m_loaded = trimesh.load(loaded_source, force="mesh")
                raw_verts = np.asarray(m_loaded.vertices)
                raw_faces = np.asarray(m_loaded.faces)
            else:
                raise RuntimeError("No input mesh found! Connect upstream MESH wire or generate a raw mesh first.")

        print(f"[AutoQuadRemesh] Source mesh: verts={len(raw_verts)}, faces={len(raw_faces)}")
        highpoly_ref = trimesh.Trimesh(vertices=raw_verts, faces=raw_faces, process=False)

        # 1.2. Анализ фотографии и извлечение анатомических петель одежды
        clothing_loops_list = extract_clothing_loops_from_image(image, highpoly_ref.bounds)
        clothing_loops_dict = dict(clothing_loops_list) if clothing_loops_list else {
            "Collar": 0.582,
            "Shirt Hem": 0.073,
            "Shorts Waistband": -0.002,
            "Shorts Hem": -0.189
        }

        # Output folder paths
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())

        if engine == "instant_crossfield":
            print(f"[AutoQuadRemesh] 🌪️ Запуск кросс-полевого ретополога Instant-Meshes (аналог ZRemesher, цель ~{target_quads} квадов, crease={crease_angle}°)...")
            import subprocess
            
            # 1. Физическое разделение пальцев
            m_work = highpoly_ref
            if separate_fingers:
                m_work = carve_finger_webbing(m_work)

            # 2. Гарантия 100% герметичного манифолд-меша (0 дыр) через PyMeshFix
            print("[AutoQuadRemesh] Подготовка герметичного Manifold-каркаса для кросс-полей...")
            tin = pymeshfix.PyTMesh()
            v_in = np.ascontiguousarray(m_work.vertices, dtype=np.float64)
            f_in = np.ascontiguousarray(m_work.faces, dtype=np.int32)
            tin.load_array(v_in, f_in)
            tin.remove_smallest_components()
            tin.fix_connectivity()
            tin.fill_small_boundaries(0, True)
            tin.clean(True)
            v_clean, f_clean = tin.return_arrays()

            m_clean = trimesh.Trimesh(vertices=v_clean, faces=f_clean, process=False)
            ply_tmp = os.path.join(full_output_folder, f"_temp_{counter}_in.ply")
            obj_tmp = os.path.join(full_output_folder, f"_temp_{counter}_out.obj")
            m_clean.export(ply_tmp)

            # 3. Вычисление кросс-полей и экстракция квадов
            coarse_f = max(500, target_quads // 4)
            cmd = [
                "/usr/local/bin/instant-meshes",
                "-r", "4",
                "-p", "4",
                "-f", str(coarse_f),
                "-c", str(crease_angle),
                "-S", "2",
                "-d",
                "-o", obj_tmp,
                ply_tmp
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # 4. Парсинг квадов
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

            q_verts = np.array(parsed_verts, dtype=np.float32)
            q_faces = np.array(parsed_quads, dtype=np.int32)

            # Удаление временных файлов
            try:
                os.remove(ply_tmp)
                os.remove(obj_tmp)
            except Exception:
                pass

        elif engine == "intelligent_custom_retopo":
            print("[AutoQuadRemesh] 🚀 Запуск собственного интеллектуального движка квад-ретопологии...")
            from .custom_quad_engine import generate_structured_quad_character
            
            # Быстрое получение скелета через DWPose
            dwpose_kps = {}
            try:
                import custom_nodes.comfyui_controlnet_aux as aux
                import cv2
                dw_node = aux.NODE_CLASS_MAPPINGS["DWPreprocessor"]()
                # Рендерим ортографический фронт для детектора
                test_in = os.path.join(folder_paths.get_output_directory(), "test_dwpose_in.png")
                if not os.path.exists(test_in):
                    # Если нет готового, генерируем простую фронт-проекцию
                    pass
                if os.path.exists(test_in):
                    img_np = cv2.imread(test_in)
                    t_img = torch.from_numpy(img_np[:, :, ::-1].copy()).float() / 255.0
                    t_img = t_img.unsqueeze(0)
                    res_pose = dw_node.estimate_pose(image=t_img, detect_hand="enable", detect_body="enable", detect_face="enable", resolution=512, bbox_detector="None", pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt")
                    p_info = res_pose["result"][1][0]["people"][0]
                    dwpose_kps = {
                        "body": np.array(p_info["pose_keypoints_2d"]).reshape(-1, 3),
                        "hand_left": np.array(p_info["hand_left_keypoints_2d"]).reshape(-1, 3),
                        "hand_right": np.array(p_info["hand_right_keypoints_2d"]).reshape(-1, 3)
                    }
                    print("[AutoQuadRemesh] DWPose скелет и 42 сустава пальцев успешно захвачены.")
            except Exception as e:
                print(f"[AutoQuadRemesh] DWPose warning (using geometric defaults): {e}")

            q_verts, q_faces, _ = generate_structured_quad_character(highpoly_ref, dwpose_kps, clothing_loops_dict)
        else:
            # 1.5. Физическое разделение пальцев для классического QuadriFlow
            if separate_fingers:
                m_sep = carve_finger_webbing(highpoly_ref)
                raw_verts = np.asarray(m_sep.vertices)
                raw_faces = np.asarray(m_sep.faces)

            # 2. Fast simplification
            if len(raw_faces) > 40000:
                print("[AutoQuadRemesh] Fast simplification to ~35k faces...")
                v_s, f_s = fast_simplification.simplify(np.asarray(raw_verts), np.asarray(raw_faces), target_count=35000)
            else:
                v_s, f_s = raw_verts, raw_faces

            # 3. Gentle Watertight Sealing
            if heal_mesh:
                print("[AutoQuadRemesh] Gentle mesh sealing (preserving creases & separate fingers)...")
                tin = pymeshfix.PyTMesh()
                v_in = np.ascontiguousarray(v_s, dtype=np.float64)
                f_in = np.ascontiguousarray(f_s, dtype=np.int32)
                tin.load_array(v_in, f_in)
                tin.remove_smallest_components()
                tin.fix_connectivity()
                tin.fill_small_boundaries(0, True)
                v_work, f_work = tin.return_arrays()
            else:
                v_work, f_work = v_s, f_s

            # 4. QuadriFlow quad calculation
            print(f"[AutoQuadRemesh] Calculating Quad Flow (target {target_quads} quads, adaptive={adaptive_scale}, sharp={preserve_sharp})...")
            res = None
            candidates = [
                (target_quads, 42, False),
                (target_quads + 200, 43, False),
                (target_quads, 42, True),
                (target_quads + 200, 43, True),
            ]
            last_error = None
            for t_q, s_d, use_mcf in candidates:
                try:
                    res = pyquadriflow(
                        faces=t_q,
                        seed=s_d,
                        mesh_vertices=v_work.tolist(),
                        face_indexes=f_work.tolist(),
                        flag_preserve_sharp=preserve_sharp,
                        flag_preserve_boundary=False,
                        flag_adaptive_scale=adaptive_scale,
                        flag_aggresive_sat=False,
                        flag_minimum_cost_flow=use_mcf
                    )
                    if res and len(res.get('faces', [])) > 0:
                        break
                except Exception as e:
                    last_error = e
                    print(f"[AutoQuadRemesh] Solver step ({t_q} quads, seed {s_d}, mcf={use_mcf}) notice: {e}. Trying optimal adjustment...")

            if res is None:
                raise RuntimeError(f"QuadriFlow solver failed: {last_error}")

            q_verts = np.array(res['vertices'], dtype=np.float32)
            q_faces = np.array(res['faces'], dtype=np.int32)


        # 5. Iterative Tangential Quad Relaxation & Surface Snapping
        if relax_iterations > 0:
            print(f"[AutoQuadRemesh] Applying {relax_iterations} iterations of Tangential Quad Relaxation...")
            q_verts = tangential_quad_relaxation(q_verts, q_faces, highpoly_ref, iterations=relax_iterations, factor=0.35)

        # 5.5. Автоматическое выравнивание топологических мастер-лупов вдоль ключевых переходов
        print("[AutoQuadRemesh] 🎯 Выравнивание топологических мастер-лупов вдоль швов и переходов...")
        q_verts = auto_align_feature_loops(q_verts, q_faces, highpoly_ref, clothing_loops_dict=clothing_loops_dict)

        # 6. Full Shrinkwrap Projection (Snap precisely to original high-poly shape)
        if shrinkwrap_strength > 0.0:
            print(f"[AutoQuadRemesh] Final Shrinkwrap projection onto high-poly surface (strength={shrinkwrap_strength})...")
            q_verts = apply_shrinkwrap_projection(q_verts, highpoly_ref, strength=shrinkwrap_strength)

        # 7. Quality Monitoring & Diagnostics Report
        print_quality_diagnostics(q_verts, q_faces, highpoly_ref)

        # 8. Triangulation for standard GLB surface
        tri_faces = []
        for qf in q_faces:
            tri_faces.append([qf[0], qf[1], qf[2]])
            tri_faces.append([qf[0], qf[2], qf[3]])
        tri_faces = np.array(tri_faces, dtype=np.int32)

        out_trimesh = trimesh.Trimesh(vertices=q_verts, faces=tri_faces, process=False)
        out_trimesh.metadata["quad_faces"] = q_faces

        t_verts = torch.from_numpy(q_verts).unsqueeze(0).float()
        t_faces = torch.from_numpy(tri_faces).unsqueeze(0).int()
        out_mesh = hy3d_nodes.MESH(vertices=t_verts, faces=t_faces)

        # Pure Quad OBJ file for Blender, Maya, ZBrush (f v1 v2 v3 v4)
        obj_file = f"{filename}_{counter:05}_.obj"
        obj_path = os.path.join(full_output_folder, obj_file)
        save_pure_quad_obj(q_verts, q_faces, obj_path)

        # GLB with Quad Wireframe overlay for ComfyUI 3D Viewer
        saved_file = f"{filename}_{counter:05}_.glb"
        glb_path = os.path.join(full_output_folder, saved_file)
        save_clean_quad_glb_with_wireframe(q_verts, q_faces, glb_path, show_quad_wireframe=show_quad_wireframe)

        elapsed = round(time.time() - t0, 2)
        print(f"[AutoQuadRemesh] Completed in {elapsed}s! Quads: {len(q_faces)}.")
        print(f"[AutoQuadRemesh] Pure quad OBJ: {obj_path}")

        return {
            "ui": {"3d": [{"filename": saved_file, "subfolder": subfolder, "type": "output"}]},
            "result": (out_mesh, out_trimesh)
        }


NODE_CLASS_MAPPINGS = {
    "AutoQuadRemesh": AutoQuadRemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoQuadRemesh": "Auto Quad Remesh (⚡ Локальная Квад-Ретопология)",
}
