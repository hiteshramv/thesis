from typing import Any, Dict, List
from collections import defaultdict
import numpy as np
from std_msgs.msg import ColorRGBA
import logging
import dataclasses


def class_color(cls_name: str) -> ColorRGBA:
    name = str(cls_name).strip().lower()

    #if name == "child":
    #    return ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)   # yellow
    if name in ("adult","child"):
        return ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)   # red
    elif name in ("cyclist"):
        return ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)   # green
    elif name in ("car", "van", "truck", "bus", "motorcycle"):
        return ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)   # blue

    return ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)       # default white



def aabb_iou3d(box_a: Dict, box_b: Dict) -> float:
    """
    Axis-aligned 3D IoU for boxes in dict format:
      box["center_xyz"] = (cx, cy, cz)
      box["dims_lwh"]   = (l, w, h)
    Ignores yaw.
    """
    ax, ay, az = box_a["center_xyz"]
    al, aw, ah = box_a["dims_lwh"]

    bx, by, bz = box_b["center_xyz"]
    bl, bw, bh = box_b["dims_lwh"]

    a_min = np.array([ax - al / 2.0, ay - aw / 2.0, az - ah / 2.0], dtype=float)
    a_max = np.array([ax + al / 2.0, ay + aw / 2.0, az + ah / 2.0], dtype=float)

    b_min = np.array([bx - bl / 2.0, by - bw / 2.0, bz - bh / 2.0], dtype=float)
    b_max = np.array([bx + bl / 2.0, by + bw / 2.0, bz + bh / 2.0], dtype=float)

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(0.0, inter_max - inter_min)

    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])
    if inter_vol <= 0.0:
        return 0.0

    vol_a = float(al * aw * ah)
    vol_b = float(bl * bw * bh)
    union = vol_a + vol_b - inter_vol
    if union <= 0.0:
        return 0.0

    return inter_vol / union


def _as_box_dict_from_track(t: Any) -> Dict:
    st = getattr(t, "state", None)
    return {
        "center_xyz": (float(st[0]), float(st[1]), float(st[2])),
        "dims_lwh":   (float(st[4]), float(st[5]), float(st[6])),
        "yaw":        float(st[3]),
    }


def _find_group_id(track_id: int, parent: Dict[int, int]) -> int:
    """DSU find with path compression."""
    if track_id not in parent:
        parent[track_id] = track_id
        return track_id
    p = parent[track_id]
    if p != track_id:
        parent[track_id] = _find_group_id(p, parent)
    return parent[track_id]


def _union_groups(a: int, b: int, parent: Dict[int, int]) -> int:
    """DSU union: keep smaller group id for stability."""
    ga = _find_group_id(a, parent)
    gb = _find_group_id(b, parent)
    if ga == gb:
        return ga
    g = ga if ga < gb else gb
    parent[ga] = g
    parent[gb] = g
    parent[a] = g
    parent[b] = g
    return g

def bev_iou(box_a: Dict, box_b: Dict,
            iou_thr:   float = 0.4,
            iomin_thr: float = 0.75) -> float:
    # """
    # Returns 1.0 if either:
    #   - IoU  >= iou_thr   (same-size duplicates)
    #   - IoMin >= iomin_thr (small box substantially inside large box)
    # Returns 0.0 otherwise.
    # Caller compares result >= iou_thr (0.45), so returning 1.0 always triggers,
    # returning 0.0 never triggers.
    # """
    ax, ay, _ = box_a["center_xyz"]
    al, aw, _ = box_a["dims_lwh"]
    bx, by, _ = box_b["center_xyz"]
    bl, bw, _ = box_b["dims_lwh"]

    ax1, ax2 = ax - al/2.0, ax + al/2.0
    ay1, ay2 = ay - aw/2.0, ay + aw/2.0
    bx1, bx2 = bx - bl/2.0, bx + bl/2.0
    by1, by2 = by - bw/2.0, by + bw/2.0

    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy

    if inter <= 0.0:
        return 0.0

    area_a = al * aw
    area_b = bl * bw
    union  = area_a + area_b - inter

    iou   = inter / union            if union              > 0.0 else 0.0
    iomin = inter / min(area_a, area_b) if min(area_a, area_b) > 0.0 else 0.0
    return float(max(iou, iomin))
    # if iou >= iou_thr or iomin >= iomin_thr:
    #     return 1.0   # caller's >= iou_thr will always trigger
    # return 0.0       # caller's >= iou_thr will never trigger
# """
# def choose_display_tracks_with_aliasing(
#     tracks_out:       List[Any],
#     group_parent:     Dict[int, int],
#     group_last_seen:  Dict[int, int],
#     frame_id:         int,
#     group_ttl:        int   = 15,
#     iou_thr:          float = 0.6,
#     iou_fn                  = bev_iou,
#     max_pairwise:     int   = 20000,
# ) -> List[Any]:

#     # ── Step 1: expire stale groups ──────────────────────────────────────
#     stale_roots = {
#         root for root, last_frame in group_last_seen.items()
#         if (frame_id - last_frame) > group_ttl
#     }
#     for k in list(group_parent.keys()):
#         if _find_group_id(k, group_parent) in stale_roots:
#             group_parent[k] = k          # reset: track is its own root again
#     for r in stale_roots:
#         group_last_seen.pop(r, None)

#     # ── Step 2: build valid track + box lists ────────────────────────────
#     valid, boxes = [], []
#     for t in tracks_out:
#         st = getattr(t, "state", None)
#         if st is None or len(st) < 7:
#             continue
#         tid = int(getattr(t, "track_id", 10**9))
#         _find_group_id(tid, group_parent)
#         valid.append(t)
#         boxes.append(_as_box_dict_from_track(t))

#     n = len(valid)
#     if n == 0:
#         return []

#     # ── Step 3: pairwise grouping with class guard ───────────────────────
#     # pairwise grouping (O(n^2)) with cap
#     pairs = n * (n - 1) // 2
#     if pairs <= max_pairwise:
#         for i in range(n):
#             ti = int(getattr(valid[i], "track_id", 10**9))
#             for j in range(i + 1, n):
#                 tj = int(getattr(valid[j], "track_id", 10**9))
#                 if iou_fn(boxes[i], boxes[j]) >= float(iou_thr):
#                     gid = _union_groups(ti, tj, group_parent)
#                     group_last_seen[gid] = frame_id
#     else:
#         # log a warning rather than silently doing nothing
#         logging.getLogger("aliasing").warning(
#             f"choose_display_tracks: {n} tracks → {pairs} pairs exceeds hard cap, skipping grouping"
#         )

#     # ── Step 4: group members and choose leader ──────────────────────────
#     groups: Dict[int, list] = defaultdict(list)
#     for t in valid:
#         tid = int(getattr(t, "track_id", 10**9))
#         gid = _find_group_id(tid, group_parent)
#         groups[gid].append(t)

#     def rank_key(t: Any):
#         return (
#             -int(getattr(t, "hits", 0)),
#             int(getattr(t, "time_since_update", 999)),
#             int(getattr(t, "track_id", 10**9)),
#         )

#     display = []
#     for gid, members in groups.items():
#         leader = sorted(members, key=rank_key)[0]
#         leader_copy = dataclasses.replace(leader, track_id=int(gid))
#         display.append(leader_copy)

#     return display

# """
def choose_display_tracks_with_aliasing(
    tracks_out: List[Any],
    group_parent: Dict[int, int],
    iou_thr: float = 0.5,
    iou_fn = bev_iou,
    max_pairwise: int = 20000,
) -> List[Any]:


    # filter valid tracks
    valid = []
    boxes = []
    for t in tracks_out:
        st = getattr(t, "state", None)
        if st is None or len(st) < 7:
            continue
        tid = int(getattr(t, "track_id", 10**9))
        _find_group_id(tid, group_parent)  # ensure parent exists
        valid.append(t)
        boxes.append(_as_box_dict_from_track(t))

    n = len(valid)
    if n == 0:
        return []

    # pairwise grouping (O(n^2)) with cap
    pairs = n * (n - 1) // 2
    if pairs <= max_pairwise:
        for i in range(n):
            ti = int(getattr(valid[i], "track_id", 10**9))
            for j in range(i + 1, n):
                tj = int(getattr(valid[j], "track_id", 10**9))
                if iou_fn(boxes[i], boxes[j]) >= float(iou_thr):
                    _union_groups(ti, tj, group_parent)
    else:
        # log a warning rather than silently doing nothing
        logging.getLogger("aliasing").warning(
            f"choose_display_tracks: {n} tracks → {pairs} pairs exceeds hard cap, skipping grouping"
        )

    # group members
    groups = defaultdict(list)
    for t in valid:
        tid = int(getattr(t, "track_id", 10**9))
        gid = _find_group_id(tid, group_parent)
        groups[gid].append(t)

    # choose leader per group
    def rank_key(t: Any):
        return (
            -int(getattr(t, "hits", 0)),
            int(getattr(t, "time_since_update", 999)),
            int(getattr(t, "track_id", 10**9)),
        )

    display = []
    for gid, members in groups.items():
        leader = sorted(members, key=rank_key)[0]
        # IMPORTANT: overwrite leader id to stable group id for visualization
        leader_copy = dataclasses.replace(leader, track_id=int(gid))
        #leader.track_id = int(gid)
        display.append(leader_copy)

    return display

def cleanup_group_parent(group_parent: Dict[int, int], alive_track_ids: List[int]) -> None:
    """
    Prune group_parent to only keep entries reachable by alive tracks.
    Safe: never deletes a root that an alive track depends on.
    """
    alive = set(int(x) for x in alive_track_ids)

    # Step 1: find all roots that alive tracks depend on — these must be kept
    alive_roots = set()
    for tid in alive:
        if tid in group_parent:
            alive_roots.add(_find_group_id(tid, group_parent))

    # Step 2: keep only keys that are either alive tracks or alive roots
    keep = alive | alive_roots
    for k in list(group_parent.keys()):
        if k not in keep:
            group_parent.pop(k, None)

    # Step 3: path-compress surviving entries
    for k in list(group_parent.keys()):
        group_parent[k] = _find_group_id(k, group_parent)
        

# def cleanup_group_parent(group_parent: Dict[int, int], alive_track_ids: List[int]) -> None:
    
#     #Optional: prevent group_parent from growing forever.
#     #Keeps only mappings for IDs that are currently alive.
    
#     alive = set(int(x) for x in alive_track_ids)
#     for k in list(group_parent.keys()):
#         if k not in alive:
#             group_parent.pop(k, None)
#     # compress to alive roots
#     for k in list(group_parent.keys()):
#         group_parent[k] = _find_group_id(group_parent[k], group_parent)

# def bev_iou(box_a: Dict, box_b: Dict) -> float:
    
#     #BEV IoU combined with IoMin (intersection-over-minimum).
#     #Returns max(IoU, IoMin) so both same-size duplicates
#     #and containment cases are caught.
    
#     ax, ay, _ = box_a["center_xyz"]
#     al, aw, _ = box_a["dims_lwh"]

#     bx, by, _ = box_b["center_xyz"]
#     bl, bw, _ = box_b["dims_lwh"]

#     ax1, ax2 = ax - al / 2.0, ax + al / 2.0
#     ay1, ay2 = ay - aw / 2.0, ay + aw / 2.0
#     bx1, bx2 = bx - bl / 2.0, bx + bl / 2.0
#     by1, by2 = by - bw / 2.0, by + bw / 2.0

#     ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
#     iy = max(0.0, min(ay2, by2) - max(ay1, by1))
#     inter = ix * iy

#     if inter <= 0.0:
#         return 0.0

#     area_a = al * aw
#     area_b = bl * bw
#     union  = area_a + area_b - inter

#     iou  = inter / union if union > 0.0 else 0.0
#     iomin = inter / min(area_a, area_b) if min(area_a, area_b) > 0.0 else 0.0

#     return float(max(iou, iomin))
    