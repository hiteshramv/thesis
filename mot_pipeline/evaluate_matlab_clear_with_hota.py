import sys
import os
import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ============================================================
# Configuration
# ============================================================

ROI_X_MIN, ROI_X_MAX = 0.0, 18.0
ROI_Y_MIN, ROI_Y_MAX = -15.0, 15.0

TARGET_EVAL_CLASSES = {"adult", "child", "cyclist"}

@dataclass(frozen=True)
class Box3D:
    scene_id: str
    timestamp_s: float
    frame_name: str
    obj_id: str
    class_name: str
    x: float
    y: float
    z: float
    l: float
    w: float
    h: float
    yaw: float


# ============================================================
# Timestamp and class helpers
# ============================================================

def parse_label_timestamp(label_stem: str) -> float:
    """
    Example:
      2026-02-27_12-34-43-500.json

    Converts to epoch seconds:
      2026-02-27 12:34:43.500
    """
    dt = datetime.strptime(label_stem, "%Y-%m-%d_%H-%M-%S-%f")
    return dt.timestamp()


def normalize_class(name: str) -> str:
    """
    Converts GT and prediction names to common evaluation names.

    GT:
      Pedestrian -> adult
      Child      -> child
      Bicycle    -> cyclist

    Prediction:
      Adult      -> adult
      Child      -> child
      Cyclist    -> cyclist
    """
    n = str(name).strip().lower()

    if n in {"pedestrian", "adult"}:
        return "adult"

    if n in {"child"}:
        return "child"

    if n in {"bicycle", "bycicle", "cyclist"}:
        return "cyclist"

    if n in {"car", "van", "truck", "bus", "motorcycle", "vehicle"}:
        return "vehicle"

    return n


def is_target_eval_class(class_name: str) -> bool:
    return normalize_class(class_name) in TARGET_EVAL_CLASSES


def compatible_class(gt_class: str, pred_class: str, strict_child: bool = False) -> bool:
    """
    Class-aware matching.

    strict_child = False:
      adult and child can match each other.
      cyclist only matches cyclist.

    strict_child = True:
      adult only matches adult.
      child only matches child.
      cyclist only matches cyclist.
    """
    g = normalize_class(gt_class)
    p = normalize_class(pred_class)

    if strict_child:
        return g == p

    if g in {"adult", "child"} and p in {"adult", "child"}:
        return True

    return g == p


def class_gate_m(gt_class: str, pred_class: str) -> float:
    """
    BEV center-distance gate in meters for custom similarity.
    Used only when --similarity_method custom_bev_gate.
    """
    g = normalize_class(gt_class)
    p = normalize_class(pred_class)

    if g in {"adult", "child"} or p in {"adult", "child"}:
        return 1.0

    if g == "cyclist" or p == "cyclist":
        return 1.0

    if g == "vehicle" or p == "vehicle":
        return 2.0

    return 1.0


# ============================================================
# Loading labels and predictions
# ============================================================

def load_labels_for_scene(scene_dir: Path) -> pd.DataFrame:
    scene_id = scene_dir.name
    label_dir = scene_dir / "label"

    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label folder: {label_dir}")

    rows: List[Box3D] = []

    for jf in sorted(label_dir.glob("*.json")):
        timestamp_s = parse_label_timestamp(jf.stem)

        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list in {jf}")

        for obj in data:
            if "obj_id" not in obj or "obj_type" not in obj or "psr" not in obj:
                continue

            psr = obj["psr"]
            pos = psr["position"]
            rot = psr.get("rotation", {})
            scale = psr["scale"]

            rows.append(
                Box3D(
                    scene_id=scene_id,
                    timestamp_s=timestamp_s,
                    frame_name=jf.stem,
                    obj_id=str(obj["obj_id"]),
                    class_name=str(obj["obj_type"]),
                    x=float(pos["x"]),
                    y=float(pos["y"]),
                    z=float(pos["z"]),
                    l=float(scale["x"]),
                    w=float(scale["y"]),
                    h=float(scale["z"]),
                    yaw=float(rot.get("z", 0.0)),
                )
            )

    return pd.DataFrame([r.__dict__ for r in rows])


def load_all_labels(root: Path) -> pd.DataFrame:
    all_rows = []

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        label_dir = scene_dir / "label"
        if not label_dir.exists():
            continue

        df = load_labels_for_scene(scene_dir)
        all_rows.append(df)

    if not all_rows:
        raise RuntimeError(f"No labels found under {root}")

    return pd.concat(all_rows, ignore_index=True)


def _find_prediction_files(root: Path, model_dir: str, csv_name: str = "predictions.csv") -> List[Path]:
    """
    Finds prediction CSV files under root/<scene>/<model_dir>/<csv_name>.
    csv_name is passed from --pred_csv_name (default: predictions.csv).
    """
    patterns = [f"*/{model_dir}/{csv_name}"]

    files: List[Path] = []
    seen = set()

    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            rp = p.resolve()
            if rp not in seen:
                files.append(p)
                seen.add(rp)

    return files


def load_all_predictions(root: Path, model_dir: str = "cv", csv_name: str = "predictions.csv") -> pd.DataFrame:
    """
    Expects files like:
      evaluation/new/cv/predictions.csv
      evaluation/new1/cv/predictions.csv
    csv_name is controlled by --pred_csv_name argument (default: predictions.csv).
    """
    dfs = []

    pred_files = _find_prediction_files(root, model_dir=model_dir, csv_name=csv_name)

    if not pred_files:
        raise RuntimeError(
            f"No prediction CSV files found under {root}. Expected "
            f"<scene>/{model_dir}/{csv_name}"
        )

    for pred_csv in pred_files:
        df = pd.read_csv(pred_csv)
        model_depth = len(Path(model_dir).parts)
        scene_id_from_folder = pred_csv.parents[model_depth].name
        df["scene_id"] = scene_id_from_folder
        dfs.append(df)

    pred = pd.concat(dfs, ignore_index=True)

    required = {
        "scene_id",
        "timestamp_s",
        "track_id",
        "class_name",
        "x", "y", "z",
        "l", "w", "h",
        "yaw",
    }

    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

    pred = pred.copy()
    pred["scene_id"] = pred["scene_id"].astype(str)
    pred["track_id"] = pred["track_id"].astype(str)
    pred["obj_id"] = pred["track_id"]
    pred["timestamp_s"] = pred["timestamp_s"].astype(float)

    for c in ["x", "y", "z", "l", "w", "h", "yaw"]:
        pred[c] = pred[c].astype(float)

    return pred


# ============================================================
# Filtering
# ============================================================

def filter_roi(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    return df[
        (df["x"] >= ROI_X_MIN) &
        (df["x"] <= ROI_X_MAX) &
        (df["y"] >= ROI_Y_MIN) &
        (df["y"] <= ROI_Y_MAX)
    ].copy()


# ============================================================
# Geometry and similarity
# ============================================================

def bev_distance(a, b) -> float:
    return float(math.hypot(
        float(a["x"]) - float(b["x"]),
        float(a["y"]) - float(b["y"]),
    ))


def distance_3d(a, b) -> float:
    return float(math.sqrt(
        (float(a["x"]) - float(b["x"])) ** 2 +
        (float(a["y"]) - float(b["y"])) ** 2 +
        (float(a["z"]) - float(b["z"])) ** 2
    ))



def bev_iou_aabb(a, b) -> float:
    """
    Axis-aligned BEV IoU using x/y center and l/w size. Yaw is ignored.
    This is a simple IoU similarity option for HOTA/CLEAR-style matching.
    """
    ax1 = float(a["x"]) - float(a["l"]) * 0.5
    ax2 = float(a["x"]) + float(a["l"]) * 0.5
    ay1 = float(a["y"]) - float(a["w"]) * 0.5
    ay2 = float(a["y"]) + float(a["w"]) * 0.5

    bx1 = float(b["x"]) - float(b["l"]) * 0.5
    bx2 = float(b["x"]) + float(b["l"]) * 0.5
    by1 = float(b["y"]) - float(b["w"]) * 0.5
    by2 = float(b["y"]) + float(b["w"]) * 0.5

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def iou_3d_aabb(a, b) -> float:
    """
    Axis-aligned 3D IoU using x/y/z center and l/w/h size. Yaw is ignored.
    Use this when you want an IoU-based HOTA similarity without adding Shapely.
    """
    ax1 = float(a["x"]) - float(a["l"]) * 0.5
    ax2 = float(a["x"]) + float(a["l"]) * 0.5
    ay1 = float(a["y"]) - float(a["w"]) * 0.5
    ay2 = float(a["y"]) + float(a["w"]) * 0.5
    az1 = float(a["z"]) - float(a["h"]) * 0.5
    az2 = float(a["z"]) + float(a["h"]) * 0.5

    bx1 = float(b["x"]) - float(b["l"]) * 0.5
    bx2 = float(b["x"]) + float(b["l"]) * 0.5
    by1 = float(b["y"]) - float(b["w"]) * 0.5
    by2 = float(b["y"]) + float(b["w"]) * 0.5
    bz1 = float(b["z"]) - float(b["h"]) * 0.5
    bz2 = float(b["z"]) + float(b["h"]) * 0.5

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    iz = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = ix * iy * iz

    vol_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) * max(0.0, az2 - az1)
    vol_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1) * max(0.0, bz2 - bz1)
    union = vol_a + vol_b - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def similarity_score(
    g,
    p,
    strict_child: bool,
    method: str = "euclidean",
    euclidean_scale: float = 2.0,
    use_3d: bool = False,
    class_aware: bool = True,
) -> float:
    """
    Returns similarity in [0, 1], matching MATLAB trackCLEARMetrics style.

    method="euclidean":
      sim = max(0, 1 - distance / euclidean_scale)

    method="custom_bev_gate":
      sim = max(0, 1 - BEV_distance / class_gate)

    method="bev_iou":
      sim = axis-aligned BEV IoU

    method="aabb_3d_iou":
      sim = axis-aligned 3D IoU

    class_aware=True keeps your VRU class matching logic.
    class_aware=False ignores class while matching, closer to vanilla MATLAB
    Euclidean mode if MATLAB receives only positions.
    """
    if class_aware and not compatible_class(g["class_name"], p["class_name"], strict_child):
        return 0.0

    if method == "euclidean":
        d = distance_3d(g, p) if use_3d else bev_distance(g, p)
        scale = max(1e-9, float(euclidean_scale))
        return float(max(0.0, 1.0 - d / scale))

    if method == "custom_bev_gate":
        gate = max(1e-9, class_gate_m(g["class_name"], p["class_name"]))
        d = bev_distance(g, p)
        return float(max(0.0, 1.0 - d / gate))

    if method == "bev_iou":
        return bev_iou_aabb(g, p)

    if method == "aabb_3d_iou":
        return iou_3d_aabb(g, p)

    raise ValueError(f"Unknown similarity method: {method!r}")


def nearest_prediction_frame(
    pred_scene: pd.DataFrame,
    label_ts: float,
    time_slop: float,
) -> pd.DataFrame:
    """
    Finds the nearest prediction timestamp to one label timestamp.

    If nearest timestamp is outside time_slop, returns an empty DataFrame.
    This adapts MATLAB's equal-time evaluation to your non-identical timestamps.
    """
    if len(pred_scene) == 0:
        return pred_scene.iloc[0:0].copy()

    unique_ts = np.array(sorted(pred_scene["timestamp_s"].unique()), dtype=float)
    nearest_idx = int(np.argmin(np.abs(unique_ts - label_ts)))
    nearest_ts = float(unique_ts[nearest_idx])

    if abs(nearest_ts - label_ts) > time_slop:
        return pred_scene.iloc[0:0].copy()

    return pred_scene[pred_scene["timestamp_s"] == nearest_ts].copy()


def build_similarity_matrix(
    gt_f: pd.DataFrame,
    pred_f: pd.DataFrame,
    strict_child: bool,
    method: str,
    euclidean_scale: float,
    use_3d: bool,
    class_aware: bool,
):
    gt_rows = list(gt_f.iterrows())
    pred_rows = list(pred_f.iterrows())

    S = np.zeros((len(gt_rows), len(pred_rows)), dtype=float)

    for gi, (_, g) in enumerate(gt_rows):
        for pi, (_, p) in enumerate(pred_rows):
            S[gi, pi] = similarity_score(
                g,
                p,
                strict_child=strict_child,
                method=method,
                euclidean_scale=euclidean_scale,
                use_3d=use_3d,
                class_aware=class_aware,
            )

    return S, gt_rows, pred_rows


DEFAULT_HOTA_ALPHAS = np.arange(0.05, 1.0, 0.05)


def _collect_hota_frame_data(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    strict_child: bool,
    method: str,
    euclidean_scale: float,
    use_3d: bool,
    class_aware: bool,
):
    """
    Collect per-truth-timestamp frame data for HOTA.

    Each frame stores the local GT/pred rows, their global identity keys, and the
    pairwise similarity matrix. Prediction frames are synchronized to label
    timestamps using nearest_prediction_frame(), same as the CLEAR evaluator.
    """
    frames = []
    gt_keys_all = set()
    pred_keys_all = set()

    for scene_id in sorted(gt["scene_id"].astype(str).unique()):
        gt_s = gt[gt["scene_id"].astype(str) == scene_id].copy()
        pred_s = pred[pred["scene_id"].astype(str) == scene_id].copy()

        for label_ts in sorted(gt_s["timestamp_s"].unique()):
            gt_f = gt_s[gt_s["timestamp_s"] == label_ts].copy()
            pred_f = nearest_prediction_frame(
                pred_scene=pred_s,
                label_ts=float(label_ts),
                time_slop=time_slop,
            )

            S, gt_rows, pred_rows = build_similarity_matrix(
                gt_f=gt_f,
                pred_f=pred_f,
                strict_child=strict_child,
                method=method,
                euclidean_scale=euclidean_scale,
                use_3d=use_3d,
                class_aware=class_aware,
            )

            gt_keys = [(str(g["scene_id"]), str(g["obj_id"])) for _, g in gt_rows]
            pred_keys = [(str(p["scene_id"]), str(p["obj_id"])) for _, p in pred_rows]

            gt_keys_all.update(gt_keys)
            pred_keys_all.update(pred_keys)

            frames.append({
                "scene_id": str(scene_id),
                "label_timestamp_s": float(label_ts),
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "gt_keys": gt_keys,
                "pred_keys": pred_keys,
                "similarity": S,
            })

    gt_key_to_idx = {k: i for i, k in enumerate(sorted(gt_keys_all))}
    pred_key_to_idx = {k: i for i, k in enumerate(sorted(pred_keys_all))}

    return frames, gt_key_to_idx, pred_key_to_idx


def evaluate_hota_official(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    strict_child: bool,
    method: str = "euclidean",
    euclidean_scale: float = 2.0,
    use_3d: bool = False,
    class_aware: bool = True,
    alphas: Optional[np.ndarray] = None,
) -> Tuple[Dict, pd.DataFrame]:

    if alphas is None:
        alphas = DEFAULT_HOTA_ALPHAS
    alphas = np.asarray(alphas, dtype=float)

    frames, gt_key_to_idx, pred_key_to_idx = _collect_hota_frame_data(
        gt=gt,
        pred=pred,
        time_slop=time_slop,
        strict_child=strict_child,
        method=method,
        euclidean_scale=euclidean_scale,
        use_3d=use_3d,
        class_aware=class_aware,
    )

    num_gt_ids = len(gt_key_to_idx)
    num_pred_ids = len(pred_key_to_idx)

    total_gt_dets = int(sum(len(fr["gt_keys"]) for fr in frames))
    total_pred_dets = int(sum(len(fr["pred_keys"]) for fr in frames))

    rows = []

    for alpha in alphas:
        gt_id_count = np.zeros(num_gt_ids, dtype=float)
        pred_id_count = np.zeros(num_pred_ids, dtype=float)
        potential_matches = np.zeros((num_gt_ids, num_pred_ids), dtype=float)

        for fr in frames:
            S = fr["similarity"]
            gt_indices = [gt_key_to_idx[k] for k in fr["gt_keys"]]
            pred_indices = [pred_key_to_idx[k] for k in fr["pred_keys"]]

            for gi in gt_indices:
                gt_id_count[gi] += 1.0
            for pi in pred_indices:
                pred_id_count[pi] += 1.0

            if S.size == 0:
                continue

            valid_rows, valid_cols = np.where(S >= alpha)
            for lr, lc in zip(valid_rows, valid_cols):
                gi = gt_indices[int(lr)]
                pi = pred_indices[int(lc)]
                potential_matches[gi, pi] += float(S[int(lr), int(lc)])

        denom = gt_id_count[:, None] + pred_id_count[None, :] - potential_matches
        global_alignment = np.divide(
            potential_matches,
            denom,
            out=np.zeros_like(potential_matches),
            where=denom > 0,
        )

        tp = 0
        loc_sum = 0.0
        ass_sum = 0.0
        ass_re_sum = 0.0
        ass_pr_sum = 0.0

        for fr in frames:
            S = fr["similarity"]
            if S.size == 0:
                continue

            gt_indices = [gt_key_to_idx[k] for k in fr["gt_keys"]]
            pred_indices = [pred_key_to_idx[k] for k in fr["pred_keys"]]

            ga = global_alignment[np.ix_(gt_indices, pred_indices)]
            score = S * ga
            score = score.copy()
            score[S < alpha] = 0.0

            if score.size == 0:
                continue

            row_ind, col_ind = linear_sum_assignment(-score)

            for r, c in zip(row_ind, col_ind):
                r = int(r)
                c = int(c)
                if score[r, c] <= 0.0:
                    continue

                gi = gt_indices[r]
                pi = pred_indices[c]
                sim = float(S[r, c])

                tp += 1
                loc_sum += sim

                tpa = potential_matches[gi, pi]
                fna = gt_id_count[gi] - tpa
                fpa = pred_id_count[pi] - tpa
                ass_denom = tpa + fna + fpa

                if ass_denom > 0:
                    ass_sum += float(tpa / ass_denom)
                if gt_id_count[gi] > 0:
                    ass_re_sum += float(tpa / gt_id_count[gi])
                if pred_id_count[pi] > 0:
                    ass_pr_sum += float(tpa / pred_id_count[pi])

        fn = total_gt_dets - tp
        fp = total_pred_dets - tp

        det_denom = tp + fn + fp
        deta = float(tp / det_denom) if det_denom > 0 else 0.0
        det_re = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        det_pr = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        assa = float(ass_sum / tp) if tp > 0 else 0.0
        ass_re = float(ass_re_sum / tp) if tp > 0 else 0.0
        ass_pr = float(ass_pr_sum / tp) if tp > 0 else 0.0
        hota = float(math.sqrt(deta * assa)) if deta > 0.0 and assa > 0.0 else 0.0
        rhota = float(math.sqrt(det_re * assa)) if det_re > 0.0 and assa > 0.0 else 0.0
        loca = float(loc_sum / tp) if tp > 0 else 0.0

        rows.append({
            "alpha": float(alpha),
            "HOTA": hota,
            "DetA": deta,
            "AssA": assa,
            "LocA": loca,
            "DetRe": det_re,
            "DetPr": det_pr,
            "AssRe": ass_re,
            "AssPr": ass_pr,
            "RHOTA": rhota,
            "HOTA_percent": 100.0 * hota,
            "DetA_percent": 100.0 * deta,
            "AssA_percent": 100.0 * assa,
            "LocA_percent": 100.0 * loca,
            "DetRe_percent": 100.0 * det_re,
            "DetPr_percent": 100.0 * det_pr,
            "AssRe_percent": 100.0 * ass_re,
            "AssPr_percent": 100.0 * ass_pr,
            "RHOTA_percent": 100.0 * rhota,
            "TP_alpha": int(tp),
            "FP_alpha": int(fp),
            "FN_alpha": int(fn),
        })

    hota_alpha_df = pd.DataFrame(rows)

    if len(hota_alpha_df) == 0:
        summary = {
            "HOTA_percent": 0.0,
            "DetA_percent": 0.0,
            "AssA_percent": 0.0,
            "LocA_percent": 0.0,
            "DetRe_percent": 0.0,
            "DetPr_percent": 0.0,
            "AssRe_percent": 0.0,
            "AssPr_percent": 0.0,
            "RHOTA_percent": 0.0,
        }
    else:
        summary = {
            "HOTA_percent": float(hota_alpha_df["HOTA_percent"].mean()),
            "DetA_percent": float(hota_alpha_df["DetA_percent"].mean()),
            "AssA_percent": float(hota_alpha_df["AssA_percent"].mean()),
            "LocA_percent": float(hota_alpha_df["LocA_percent"].mean()),
            "DetRe_percent": float(hota_alpha_df["DetRe_percent"].mean()),
            "DetPr_percent": float(hota_alpha_df["DetPr_percent"].mean()),
            "AssRe_percent": float(hota_alpha_df["AssRe_percent"].mean()),
            "AssPr_percent": float(hota_alpha_df["AssPr_percent"].mean()),
            "RHOTA_percent": float(hota_alpha_df["RHOTA_percent"].mean()),
        }

    summary.update({
        "HOTA_similarity_method": str(method),
        "HOTA_alpha_min": float(alphas[0]) if len(alphas) else float("nan"),
        "HOTA_alpha_max": float(alphas[-1]) if len(alphas) else float("nan"),
        "HOTA_alpha_step": float(alphas[1] - alphas[0]) if len(alphas) > 1 else float("nan"),
    })

    return summary, hota_alpha_df



def match_frame_matlab_clear(
    gt_f: pd.DataFrame,
    pred_f: pd.DataFrame,
    prev_pred_for_gt: Dict[Tuple[str, str], Optional[str]],
    strict_child: bool,
    similarity_threshold: float = 0.5,
    method: str = "euclidean",
    euclidean_scale: float = 2.0,
    use_3d: bool = False,
    class_aware: bool = True,
):
    if len(gt_f) == 0 and len(pred_f) == 0:
        return [], [], []

    if len(gt_f) == 0:
        return [], [], list(pred_f.index)

    if len(pred_f) == 0:
        return [], list(gt_f.index), []

    S, gt_rows, pred_rows = build_similarity_matrix(
        gt_f,
        pred_f,
        strict_child=strict_child,
        method=method,
        euclidean_scale=euclidean_scale,
        use_3d=use_3d,
        class_aware=class_aware,
    )

    used_g = set()
    used_p = set()
    matches = []

    pred_id_to_pi = {
        str(p["obj_id"]): pi
        for pi, (_, p) in enumerate(pred_rows)
    }

    for gi, (_, g) in enumerate(gt_rows):
        gt_key = (str(g["scene_id"]), str(g["obj_id"]))
        prev_pid = prev_pred_for_gt.get(gt_key)

        if prev_pid is None:
            continue

        pi = pred_id_to_pi.get(str(prev_pid))
        if pi is None:
            continue

        if pi in used_p:
            continue

        if S[gi, pi] >= similarity_threshold:
            gt_idx, g_row = gt_rows[gi]
            pred_idx, p_row = pred_rows[pi]

            matches.append({
                "scene_id": str(g_row["scene_id"]),
                "label_timestamp_s": float(g_row["timestamp_s"]),
                "pred_timestamp_s": float(p_row["timestamp_s"]),
                "frame_name": str(g_row["frame_name"]),
                "gt_id": str(g_row["obj_id"]),
                "pred_id": str(p_row["obj_id"]),
                "gt_class": str(g_row["class_name"]),
                "pred_class": str(p_row["class_name"]),
                "similarity": float(S[gi, pi]),
                "distance_bev_m": float(bev_distance(g_row, p_row)),
                "distance_3d_m": float(distance_3d(g_row, p_row)),
                "matched_by_history": True,
            })

            used_g.add(gi)
            used_p.add(pi)

    rem_g = [gi for gi in range(len(gt_rows)) if gi not in used_g]
    rem_p = [pi for pi in range(len(pred_rows)) if pi not in used_p]

    if rem_g and rem_p:
        subS = S[np.ix_(rem_g, rem_p)]

        BIG = 1e9
        cost = 1.0 - subS
        cost[subS < similarity_threshold] = BIG

        row_ind, col_ind = linear_sum_assignment(cost)

        for rr, cc in zip(row_ind, col_ind):
            if cost[rr, cc] >= BIG:
                continue

            gi = rem_g[int(rr)]
            pi = rem_p[int(cc)]

            gt_idx, g_row = gt_rows[gi]
            pred_idx, p_row = pred_rows[pi]

            matches.append({
                "scene_id": str(g_row["scene_id"]),
                "label_timestamp_s": float(g_row["timestamp_s"]),
                "pred_timestamp_s": float(p_row["timestamp_s"]),
                "frame_name": str(g_row["frame_name"]),
                "gt_id": str(g_row["obj_id"]),
                "pred_id": str(p_row["obj_id"]),
                "gt_class": str(g_row["class_name"]),
                "pred_class": str(p_row["class_name"]),
                "similarity": float(S[gi, pi]),
                "distance_bev_m": float(bev_distance(g_row, p_row)),
                "distance_3d_m": float(distance_3d(g_row, p_row)),
                "matched_by_history": False,
            })

            used_g.add(gi)
            used_p.add(pi)

    matched_gt_indices = {gt_rows[gi][0] for gi in used_g}
    matched_pred_indices = {pred_rows[pi][0] for pi in used_p}

    unmatched_gt = [idx for idx in gt_f.index if idx not in matched_gt_indices]
    unmatched_pred = [idx for idx in pred_f.index if idx not in matched_pred_indices]

    return matches, unmatched_gt, unmatched_pred


def evaluate_matlab_clear(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    strict_child: bool,
    similarity_threshold: float = 0.5,
    method: str = "euclidean",
    euclidean_scale: float = 2.0,
    use_3d: bool = False,
    class_aware: bool = True,
):
    """
    Computes MATLAB trackCLEARMetrics-style metrics.

    Returns:
      summary, matches_df, traj_df
    """
    all_matches = []

    total_truths = 0
    total_tracks = 0
    tp = 0
    fp = 0
    fn = 0
    idsw = 0
    frag = 0
    sim_sum = 0.0

    total_label_frames = 0
    synced_label_frames = 0
    no_sync_frames = 0

    gt_life = defaultdict(int)
    gt_matched = defaultdict(int)
    gt_class_for_key = {}

    for scene_id in sorted(gt["scene_id"].astype(str).unique()):
        gt_s = gt[gt["scene_id"].astype(str) == scene_id].copy()
        pred_s = pred[pred["scene_id"].astype(str) == scene_id].copy()

        prev_pred_for_gt: Dict[Tuple[str, str], Optional[str]] = {}
        prev_tracked: Dict[Tuple[str, str], bool] = {}
        last_truth_frame: Dict[Tuple[str, str], int] = {}
        ever_tracked = defaultdict(bool)

        frame_index = -1

        for label_ts in sorted(gt_s["timestamp_s"].unique()):
            frame_index += 1
            total_label_frames += 1

            gt_f = gt_s[gt_s["timestamp_s"] == label_ts].copy()
            pred_f = nearest_prediction_frame(
                pred_scene=pred_s,
                label_ts=float(label_ts),
                time_slop=time_slop,
            )

            if len(pred_f) > 0:
                synced_label_frames += 1
            else:
                no_sync_frames += 1

            total_truths += len(gt_f)
            total_tracks += len(pred_f)

            current_gt_keys = set()
            for _, g in gt_f.iterrows():
                key = (str(g["scene_id"]), str(g["obj_id"]))
                current_gt_keys.add(key)
                gt_life[key] += 1
                gt_class_for_key[key] = str(g["class_name"])

            matches, unmatched_gt, unmatched_pred = match_frame_matlab_clear(
                gt_f=gt_f,
                pred_f=pred_f,
                prev_pred_for_gt=prev_pred_for_gt,
                strict_child=strict_child,
                similarity_threshold=similarity_threshold,
                method=method,
                euclidean_scale=euclidean_scale,
                use_3d=use_3d,
                class_aware=class_aware,
            )

            matched_now: Dict[Tuple[str, str], str] = {}

            for m in matches:
                key = (str(m["scene_id"]), str(m["gt_id"]))
                pred_id = str(m["pred_id"])

                was_present_prev_truth_frame = (
                    last_truth_frame.get(key, None) == frame_index - 1
                )
                was_tracked_prev_truth_frame = (
                    was_present_prev_truth_frame and prev_tracked.get(key, False)
                )

                is_switch = False
                is_fragmentation = False

                if was_tracked_prev_truth_frame:
                    previous_pred = prev_pred_for_gt.get(key)
                    if previous_pred is not None and str(previous_pred) != pred_id:
                        idsw += 1
                        is_switch = True

                if (
                    was_present_prev_truth_frame
                    and not was_tracked_prev_truth_frame
                    and ever_tracked[key]
                ):
                    frag += 1
                    is_fragmentation = True

                m["is_id_switch"] = bool(is_switch)
                m["is_fragmentation"] = bool(is_fragmentation)

                matched_now[key] = pred_id
                ever_tracked[key] = True

                gt_matched[key] += 1
                sim_sum += float(m["similarity"])

            tp += len(matches)
            fp += len(unmatched_pred)
            fn += len(unmatched_gt)

            all_matches.extend(matches)

            for key in current_gt_keys:
                last_truth_frame[key] = frame_index

                if key in matched_now:
                    prev_tracked[key] = True
                    prev_pred_for_gt[key] = matched_now[key]
                else:
                    prev_tracked[key] = False
                    prev_pred_for_gt[key] = None


    mota = 100.0 * (1.0 - ((fn + fp + idsw) / total_truths)) if total_truths > 0 else 0.0
    motp = 100.0 * (sim_sum / tp) if tp > 0 else 0.0

    recall = 100.0 * (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = 100.0 * (tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    false_track_rate = fp / total_label_frames if total_label_frames > 0 else 0.0

    mean_bev_error_m = (
        float(np.mean([m["distance_bev_m"] for m in all_matches])) if all_matches else float("nan")
    )
    mean_3d_error_m = (
        float(np.mean([m["distance_3d_m"] for m in all_matches])) if all_matches else float("nan")
    )

    traj_rows = []
    mt = 0
    pt = 0
    ml = 0

    for key, life in sorted(gt_life.items()):
        matched = gt_matched.get(key, 0)
        ratio = matched / life if life > 0 else 0.0

        if ratio > 0.8:
            status = "MT"
            mt += 1
        elif ratio < 0.2:
            status = "ML"
            ml += 1
        else:
            status = "PT"
            pt += 1

        scene_id, obj_id = key
        traj_rows.append({
            "scene_id": scene_id,
            "obj_id": obj_id,
            "class_name": gt_class_for_key.get(key, "unknown"),
            "gt_frames": int(life),
            "matched_frames": int(matched),
            "tracked_ratio": float(ratio),
            "status": status,
        })

    n_traj = len(gt_life)

    mostly_tracked_pct = 100.0 * mt / n_traj if n_traj > 0 else 0.0
    partially_tracked_pct = 100.0 * pt / n_traj if n_traj > 0 else 0.0
    mostly_lost_pct = 100.0 * ml / n_traj if n_traj > 0 else 0.0

    matches_df = pd.DataFrame(all_matches)
    traj_df = pd.DataFrame(traj_rows)

    summary = {
        "Total_label_frames": int(total_label_frames),
        "Synced_label_frames": int(synced_label_frames),
        "Frames_no_prediction_sync": int(no_sync_frames),

        "GT_boxes": int(total_truths),
        "Pred_boxes_evaluated": int(total_tracks),

        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),

        "MOTA_percent": float(mota),
        "MOTP_percent": float(motp),

        "Mostly_Tracked_percent": float(mostly_tracked_pct),
        "Partially_Tracked_percent": float(partially_tracked_pct),
        "Mostly_Lost_percent": float(mostly_lost_pct),

        "Mostly_Tracked_count": int(mt),
        "Partially_Tracked_count": int(pt),
        "Mostly_Lost_count": int(ml),
        "GT_trajectories": int(n_traj),

        "False_Positive": int(fp),
        "False_Negative": int(fn),
        "Recall_percent": float(recall),
        "Precision_percent": float(precision),
        "False_Track_Rate": float(false_track_rate),
        "ID_Switches": int(idsw),
        "Fragmentations": int(frag),

        # Extra diagnostic fields, not MATLAB table columns.
        "Mean_BEV_error_m": float(mean_bev_error_m),
        "Mean_3D_error_m": float(mean_3d_error_m),
        "Similarity_method": str(method),
        "Similarity_threshold": float(similarity_threshold),
        "Euclidean_scale": float(euclidean_scale),
        "Use_3D_distance": bool(use_3d),
        "Class_aware_matching": bool(class_aware),
        "Strict_child": bool(strict_child),
    }

    return summary, matches_df, traj_df


# ============================================================
# Per-scene and per-class evaluation
# ============================================================

def evaluate_per_scene(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    strict_child: bool,
    similarity_threshold: float,
    method: str,
    euclidean_scale: float,
    use_3d: bool,
    class_aware: bool,
) -> pd.DataFrame:
    rows = []

    for scene_id in sorted(gt["scene_id"].astype(str).unique()):
        gt_s = gt[gt["scene_id"].astype(str) == scene_id].copy()
        pred_s = pred[pred["scene_id"].astype(str) == scene_id].copy()

        summary, _, _ = evaluate_matlab_clear(
            gt=gt_s,
            pred=pred_s,
            time_slop=time_slop,
            strict_child=strict_child,
            similarity_threshold=similarity_threshold,
            method=method,
            euclidean_scale=euclidean_scale,
            use_3d=use_3d,
            class_aware=class_aware,
        )
        summary["scene_id"] = scene_id
        rows.append(summary)

    return pd.DataFrame(rows)


def evaluate_per_class(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    similarity_threshold: float,
    method: str,
    euclidean_scale: float,
    use_3d: bool,
    class_aware: bool,
) -> pd.DataFrame:
    """
    Class-specific evaluation:
      - adult GT only against adult predictions
      - child GT only against child predictions
      - cyclist GT only against cyclist predictions

    This is a separate per-class evaluation pass, so MT/PT/ML and IDSW are
    diagnostic per-class values, not necessarily a direct decomposition of
    the overall relaxed adult-child summary.
    """
    rows = []

    for cls in sorted(TARGET_EVAL_CLASSES):
        gt_c = gt[gt["class_name"] == cls].copy()
        pred_c = pred[pred["class_name"] == cls].copy()

        if len(gt_c) == 0 and len(pred_c) == 0:
            continue

        if len(gt_c) == 0:
            rows.append({
                "class_name": cls,
                "Total_label_frames": 0,
                "Synced_label_frames": 0,
                "Frames_no_prediction_sync": 0,
                "GT_boxes": 0,
                "Pred_boxes_evaluated": int(len(pred_c)),
                "TP": 0,
                "FP": int(len(pred_c)),
                "FN": 0,
                "MOTA_percent": 0.0,
                "MOTP_percent": 0.0,
                "Mostly_Tracked_percent": 0.0,
                "Partially_Tracked_percent": 0.0,
                "Mostly_Lost_percent": 0.0,
                "Mostly_Tracked_count": 0,
                "Partially_Tracked_count": 0,
                "Mostly_Lost_count": 0,
                "GT_trajectories": 0,
                "False_Positive": int(len(pred_c)),
                "False_Negative": 0,
                "Recall_percent": 0.0,
                "Precision_percent": 0.0,
                "False_Track_Rate": 0.0,
                "ID_Switches": 0,
                "Fragmentations": 0,
                "Mean_BEV_error_m": float("nan"),
                "Mean_3D_error_m": float("nan"),
            })
            continue

        summary, _, _ = evaluate_matlab_clear(
            gt=gt_c,
            pred=pred_c,
            time_slop=time_slop,
            strict_child=True,
            similarity_threshold=similarity_threshold,
            method=method,
            euclidean_scale=euclidean_scale,
            use_3d=use_3d,
            class_aware=class_aware,
        )
        summary["class_name"] = cls
        rows.append(summary)

    return pd.DataFrame(rows)




def evaluate_pedestrian_combined(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    time_slop: float,
    similarity_threshold: float,
    method: str,
    euclidean_scale: float,
    use_3d: bool,
    class_aware: bool,
    hota_method: str,
    hota_alphas,
) -> pd.DataFrame:
    """
    Aggregate Adult + Child into a single 'pedestrian' class and run
    CLEAR + HOTA evaluation on the combined set.
    GT and pred class_name are temporarily remapped to 'pedestrian'.
    """
    ped_classes = {"adult", "child"}

    gt_ped = gt[gt["class_name"].isin(ped_classes)].copy()
    pred_ped = pred[pred["class_name"].isin(ped_classes)].copy()
    gt_ped["class_name"] = "pedestrian"
    pred_ped["class_name"] = "pedestrian"

    rows = []

    # Overall pedestrian
    if len(gt_ped) > 0:
        summary, _, _ = evaluate_matlab_clear(
            gt=gt_ped,
            pred=pred_ped,
            time_slop=time_slop,
            strict_child=False,          # already merged — no child distinction
            similarity_threshold=similarity_threshold,
            method=method,
            euclidean_scale=euclidean_scale,
            use_3d=use_3d,
            class_aware=class_aware,
        )
        summary["class_name"] = "pedestrian"

        # HOTA for pedestrian combined
        hs, _ = evaluate_hota_official(
            gt=gt_ped,
            pred=pred_ped,
            time_slop=time_slop,
            strict_child=False,
            method=hota_method,
            euclidean_scale=euclidean_scale,
            use_3d=use_3d,
            class_aware=class_aware,
            alphas=hota_alphas,
        )
        summary.update(hs)
        rows.append(summary)

    return pd.DataFrame(rows)

def classwise_from_overall_matches(matches: pd.DataFrame, traj: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optional diagnostic tables computed from the overall matching result.
    These decompose the overall run instead of re-running per class.
    """
    if len(matches) == 0:
        idsw_rows = []
    else:
        m = matches.sort_values(["scene_id", "gt_id", "label_timestamp_s"]).copy()
        m["prev_pred_id"] = m.groupby(["scene_id", "gt_id"])["pred_id"].shift()
        sw = m[m["prev_pred_id"].notna() & (m["prev_pred_id"].astype(str) != m["pred_id"].astype(str))]
        idsw_rows = []
        for cls, grp in sw.groupby("gt_class"):
            idsw_rows.append({"class_name": cls, "ID_Switches_from_overall_matches": int(len(grp))})

    idsw_df = pd.DataFrame(idsw_rows)

    if len(traj) == 0:
        traj_df = pd.DataFrame(columns=["class_name", "Mostly_Tracked_count", "Partially_Tracked_count", "Mostly_Lost_count"])
    else:
        rows = []
        for cls, grp in traj.groupby("class_name"):
            rows.append({
                "class_name": cls,
                "GT_trajectories": int(len(grp)),
                "Mostly_Tracked_count": int((grp["status"] == "MT").sum()),
                "Partially_Tracked_count": int((grp["status"] == "PT").sum()),
                "Mostly_Lost_count": int((grp["status"] == "ML").sum()),
            })
        traj_df = pd.DataFrame(rows)

    return idsw_df, traj_df


# ============================================================
# Main
# ============================================================

def build_summary_table(per_class_df: "pd.DataFrame", pedestrian_df: "pd.DataFrame") -> "pd.DataFrame":

    METRIC_MAP = [
        ("MOTA",              "MOTA_percent",              True),
        ("HOTA",              "HOTA_percent",              True),
        ("DetA",              "DetA_percent",              True),
        ("AssA",              "AssA_percent",              True),
        ("IDSW",              "ID_Switches",               False),
        ("FRAG",              "Fragmentations",            False),
        ("MOTP",              "MOTP_percent",              True),
        ("Recall",            "Recall_percent",            True),
        ("Precision",         "Precision_percent",         True),
        ("Mostly Tracked",    "Mostly_Tracked_percent",    True),
        ("Partially Tracked", "Partially_Tracked_percent", True),
        ("Mostly Lost",       "Mostly_Lost_percent",       True),
    ]

    CLASSES = ["adult", "child", "cyclist"]
    COL_NAMES = ["Adult", "Child", "Cyclist", "Pedestrian"]

    # Index per_class by class_name for quick lookup
    pc = {}
    if per_class_df is not None and len(per_class_df) > 0:
        for _, row in per_class_df.iterrows():
            pc[str(row["class_name"]).lower()] = row

    # Pedestrian row (Adult + Child combined)
    ped_row = None
    if pedestrian_df is not None and len(pedestrian_df) > 0:
        ped_row = pedestrian_df.iloc[0]

    rows = []
    for display_name, col, is_pct in METRIC_MAP:
        row = {"Metrics": display_name}

        for cls, col_name in zip(CLASSES, COL_NAMES[:3]):
            if cls in pc and col in pc[cls]:
                val = pc[cls][col]
                if is_pct:
                    row[col_name] = round(float(val), 2)
                else:
                    row[col_name] = int(val)
            else:
                row[col_name] = "" 

        # Pedestrian column
        if ped_row is not None and col in ped_row:
            val = ped_row[col]
            if is_pct:
                row["Pedestrian"] = round(float(val), 2)
            else:
                row["Pedestrian"] = int(val)
        else:
            row["Pedestrian"] = ""

        rows.append(row)

    return pd.DataFrame(rows, columns=["Metrics", "Adult", "Child", "Cyclist", "Pedestrian"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Common root containing both ground truth and predictions. "
            "Use this for the original single-root folder structure."
        ),
    )

    parser.add_argument(
        "--gt_root",
        type=Path,
        default=None,
        help="Ground-truth root containing <scene>/label/*.json.",
    )

    parser.add_argument(
        "--pred_root",
        type=Path,
        default=None,
        help=(
            "Prediction root containing "
            "<scene>/<model_dir>/<pred_csv_name>."
        ),
    )

    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Evaluate only one scene folder, e.g. new or new1. If omitted, all scenes are evaluated.",
    )

    parser.add_argument(
        "--scene_ids",
        nargs="+",
        default=None,
        help="Evaluate only the listed scene folders.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="cv",
        help="Prediction subfolder under each scene, e.g. cv, ca, ctrv, ctra.",
    )

    parser.add_argument(
        "--time_slop",
        type=float,
        default=0.15,
        help="Max allowed timestamp difference between label JSON time and prediction timestamp.",
    )

    parser.add_argument(
        "--strict_child",
        action="store_true",
        help="Require Child to match only Child. Without this, Child and Adult are compatible.",
    )

    parser.add_argument(
        "--similarity_method",
        choices=["euclidean", "custom_bev_gate", "bev_iou", "aabb_3d_iou"],
        default="euclidean",
        help="MATLAB-style similarity method. euclidean is closest to trackCLEARMetrics Euclidean mode.",
    )

    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.5,
        help="Minimum similarity for a valid GT-track match. MATLAB default is 0.5.",
    )

    parser.add_argument(
        "--euclidean_scale",
        type=float,
        default=2.0,
        help="Scale used for Euclidean similarity: sim=max(0,1-distance/scale).",
    )

    parser.add_argument(
        "--use_3d",
        action="store_true",
        help="Use 3D Euclidean distance. By default, uses BEV x-y distance.",
    )

    parser.add_argument(
        "--ignore_class_for_matching",
        action="store_true",
        help="Ignore class compatibility during matching. This is closest to vanilla MATLAB if only positions are supplied.",
    )


    parser.add_argument(
        "--hota_similarity_method",
        choices=["same", "euclidean", "custom_bev_gate", "bev_iou", "aabb_3d_iou"],
        default="same",
        help="Similarity method for official HOTA. Use 'same' to reuse --similarity_method. For IoU-HOTA use bev_iou or aabb_3d_iou.",
    )

    parser.add_argument(
        "--hota_alpha_min",
        type=float,
        default=0.05,
        help="Minimum HOTA alpha threshold.",
    )

    parser.add_argument(
        "--hota_alpha_max",
        type=float,
        default=0.95,
        help="Maximum HOTA alpha threshold.",
    )

    parser.add_argument(
        "--hota_alpha_step",
        type=float,
        default=0.05,
        help="HOTA alpha step.",
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("eval_results_matlab_clear"),
        help="Output folder for summary, matches, trajectory CSVs, and per-class CSV.",
    )

    parser.add_argument(
        "--pred_csv_name",
        type=str,
        default="predictions.csv",
        help="Name of the prediction CSV file to look for inside each scene/<model_dir>/ folder. "
             "Default: predictions.csv. Change this per test run instead of renaming files.",
    )

    args = parser.parse_args()

    gt_root = args.gt_root if args.gt_root is not None else args.root
    pred_root = args.pred_root if args.pred_root is not None else args.root

    if gt_root is None:
        parser.error(
            "Provide --gt_root, or use --root when GT and predictions "
            "are stored under the same root."
        )

    if pred_root is None:
        parser.error(
            "Provide --pred_root, or use --root when GT and predictions "
            "are stored under the same root."
        )

    if not gt_root.is_dir():
        parser.error(
            f"Ground-truth root does not exist: {gt_root}"
        )

    if not pred_root.is_dir():
        parser.error(
            f"Prediction root does not exist: {pred_root}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    class_aware = not bool(args.ignore_class_for_matching)
    hota_method = args.similarity_method if args.hota_similarity_method == "same" else args.hota_similarity_method
    hota_alphas = np.arange(
        float(args.hota_alpha_min),
        float(args.hota_alpha_max) + 0.5 * float(args.hota_alpha_step),
        float(args.hota_alpha_step),
    )

    gt = load_all_labels(gt_root)
    pred = load_all_predictions(
        pred_root,
        model_dir=args.model_dir,
        csv_name=args.pred_csv_name,
    )

    gt["class_name"] = gt["class_name"].apply(normalize_class)
    pred["class_name"] = pred["class_name"].apply(normalize_class)

    gt = gt[gt["class_name"].apply(is_target_eval_class)].copy()
    pred = pred[pred["class_name"].apply(is_target_eval_class)].copy()

    gt = filter_roi(gt)
    pred = filter_roi(pred)


    if args.scene_id is not None:
        selected_scenes = {args.scene_id}
    elif args.scene_ids is not None:
        selected_scenes = set(args.scene_ids)
    else:
        selected_scenes = None

    if selected_scenes is not None:
        gt = gt[gt["scene_id"].astype(str).isin(selected_scenes)].copy()
        pred = pred[pred["scene_id"].astype(str).isin(selected_scenes)].copy()

    print("\n===== EVALUATION INPUT =====")
    print(f"Root folder: {args.root}")
    print(f"Scene filter: {args.scene_id if args.scene_id is not None else 'ALL'}")
    print(f"Model dir: {args.model_dir}")
    print(f"Time slop: {args.time_slop:.3f} s")
    print(f"Strict child matching: {args.strict_child}")
    print(f"Similarity method: {args.similarity_method}")
    print(f"Similarity threshold: {args.similarity_threshold:.3f}")
    print(f"Euclidean scale: {args.euclidean_scale:.3f}")
    print(f"Use 3D distance: {args.use_3d}")
    print(f"Class-aware matching: {class_aware}")
    print(f"HOTA similarity method: {hota_method}")
    print(f"HOTA alphas: {hota_alphas[0]:.2f} to {hota_alphas[-1]:.2f}, step {args.hota_alpha_step:.2f}")
    print(f"ROI: x=[{ROI_X_MIN}, {ROI_X_MAX}], y=[{ROI_Y_MIN}, {ROI_Y_MAX}]")
    print(f"GT boxes after filtering: {len(gt)}")
    print(f"Prediction boxes after filtering: {len(pred)}")

    if len(gt) > 0:
        print("GT scenes:", sorted(gt["scene_id"].astype(str).unique()))
        print("GT classes:", sorted(gt["class_name"].astype(str).unique()))
    else:
        print("GT scenes: []")
        print("GT classes: []")

    if len(pred) > 0:
        print("Prediction scenes:", sorted(pred["scene_id"].astype(str).unique()))
        print("Prediction classes:", sorted(pred["class_name"].astype(str).unique()))
    else:
        print("Prediction scenes: []")
        print("Prediction classes: []")

    if len(gt) == 0:
        raise RuntimeError(
            "No ground-truth boxes found after filtering. "
            "Check --root, --scene_id, ROI, and class names in label JSON."
        )

    if len(pred) == 0:
        print("\nWARNING: No predictions found after filtering. Metrics will show all GT objects as missed.")

    summary, matches, traj = evaluate_matlab_clear(
        gt=gt,
        pred=pred,
        time_slop=args.time_slop,
        strict_child=args.strict_child,
        similarity_threshold=args.similarity_threshold,
        method=args.similarity_method,
        euclidean_scale=args.euclidean_scale,
        use_3d=args.use_3d,
        class_aware=class_aware,
    )

    hota_summary, hota_alpha = evaluate_hota_official(
        gt=gt,
        pred=pred,
        time_slop=args.time_slop,
        strict_child=args.strict_child,
        method=hota_method,
        euclidean_scale=args.euclidean_scale,
        use_3d=args.use_3d,
        class_aware=class_aware,
        alphas=hota_alphas,
    )
    summary.update(hota_summary)

    per_scene = evaluate_per_scene(
        gt=gt,
        pred=pred,
        time_slop=args.time_slop,
        strict_child=args.strict_child,
        similarity_threshold=args.similarity_threshold,
        method=args.similarity_method,
        euclidean_scale=args.euclidean_scale,
        use_3d=args.use_3d,
        class_aware=class_aware,
    )

    if len(per_scene) > 0:
        hota_scene_rows = []
        for scene_id in sorted(gt["scene_id"].astype(str).unique()):
            gt_s = gt[gt["scene_id"].astype(str) == scene_id].copy()
            pred_s = pred[pred["scene_id"].astype(str) == scene_id].copy()
            hs, _ = evaluate_hota_official(
                gt=gt_s,
                pred=pred_s,
                time_slop=args.time_slop,
                strict_child=args.strict_child,
                method=hota_method,
                euclidean_scale=args.euclidean_scale,
                use_3d=args.use_3d,
                class_aware=class_aware,
                alphas=hota_alphas,
            )
            hs["scene_id"] = scene_id
            hota_scene_rows.append(hs)
        per_scene_hota = pd.DataFrame(hota_scene_rows)
        per_scene = per_scene.merge(per_scene_hota, on="scene_id", how="left", suffixes=("", "_HOTA"))

    per_class = evaluate_per_class(
        gt=gt,
        pred=pred,
        time_slop=args.time_slop,
        similarity_threshold=args.similarity_threshold,
        method=args.similarity_method,
        euclidean_scale=args.euclidean_scale,
        use_3d=args.use_3d,
        class_aware=class_aware,
    )

    hota_class_rows = []
    for cls in sorted(TARGET_EVAL_CLASSES):
        gt_c = gt[gt["class_name"] == cls].copy()
        pred_c = pred[pred["class_name"] == cls].copy()
        if len(gt_c) == 0 and len(pred_c) == 0:
            continue
        if len(gt_c) == 0:
            hs = {
                "HOTA_percent": 0.0, "DetA_percent": 0.0, "AssA_percent": 0.0, "LocA_percent": 0.0,
                "DetRe_percent": 0.0, "DetPr_percent": 0.0, "AssRe_percent": 0.0, "AssPr_percent": 0.0,
                "RHOTA_percent": 0.0, "HOTA_similarity_method": hota_method,
                "HOTA_alpha_min": float(hota_alphas[0]), "HOTA_alpha_max": float(hota_alphas[-1]),
                "HOTA_alpha_step": float(args.hota_alpha_step),
            }
        else:
            hs, _ = evaluate_hota_official(
                gt=gt_c,
                pred=pred_c,
                time_slop=args.time_slop,
                strict_child=True,
                method=hota_method,
                euclidean_scale=args.euclidean_scale,
                use_3d=args.use_3d,
                class_aware=class_aware,
                alphas=hota_alphas,
            )
        hs["class_name"] = cls
        hota_class_rows.append(hs)
    if hota_class_rows:
        per_class_hota = pd.DataFrame(hota_class_rows)
        per_class = per_class.merge(per_class_hota, on="class_name", how="left", suffixes=("", "_HOTA"))

    class_idsw_from_overall, class_traj_from_overall = classwise_from_overall_matches(matches, traj)

    summary_path = args.out_dir / "summary.csv"
    per_scene_path = args.out_dir / "per_scene.csv"
    matches_path = args.out_dir / "matches.csv"
    traj_path = args.out_dir / "trajectory_coverage.csv"
    per_class_path = args.out_dir / "per_class.csv"
    class_idsw_overall_path = args.out_dir / "class_idsw_from_overall_matches.csv"
    class_traj_overall_path = args.out_dir / "class_trajectory_from_overall_matches.csv"
    hota_alpha_path = args.out_dir / "hota_alpha.csv"

    # Pedestrian combined (Adult + Child merged)
    pedestrian_combined = evaluate_pedestrian_combined(
        gt=gt,
        pred=pred,
        time_slop=args.time_slop,
        similarity_threshold=args.similarity_threshold,
        method=args.similarity_method,
        euclidean_scale=args.euclidean_scale,
        use_3d=args.use_3d,
        class_aware=class_aware,
        hota_method=hota_method,
        hota_alphas=hota_alphas,
    )
    pedestrian_path = args.out_dir / "pedestrian_combined.csv"
    pedestrian_combined.to_csv(pedestrian_path, index=False)

    # Wide-format summary table: metrics as rows, classes as columns
    summary_table = build_summary_table(per_class, pedestrian_combined)
    summary_table_path = args.out_dir / "summary_table.csv"
    summary_table.to_csv(summary_table_path, index=False)

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    per_scene.to_csv(per_scene_path, index=False)
    matches.to_csv(matches_path, index=False)
    traj.to_csv(traj_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    class_idsw_from_overall.to_csv(class_idsw_overall_path, index=False)
    class_traj_from_overall.to_csv(class_traj_overall_path, index=False)
    hota_alpha.to_csv(hota_alpha_path, index=False)

    print("\n===== MATLAB-STYLE CLEAR SUMMARY =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:36s}: {v:.4f}")
        else:
            print(f"{k:36s}: {v}")

    print("\n===== OUTPUT FILES =====")
    print(f"Summary:                         {summary_path}")
    print(f"Per-scene summary:               {per_scene_path}")
    print(f"Matches:                         {matches_path}")
    print(f"Trajectory coverage:             {traj_path}")
    print(f"Per-class summary:               {per_class_path}")
    print(f"Class IDSW from overall matches: {class_idsw_overall_path}")
    print(f"Class traj from overall matches: {class_traj_overall_path}")
    print(f"Official HOTA alpha table:       {hota_alpha_path}")
    print(f"Pedestrian combined (Adult+Child):{pedestrian_path}")
    print(f"Summary table (wide format):      {summary_table_path}")


if __name__ == "__main__":
    main()