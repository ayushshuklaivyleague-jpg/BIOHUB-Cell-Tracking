"""
Mitotic Division Recovery Engine for 4D Cell Tracking.
Recovers cytokinesis bifurcations directly within global assignment by un-strangling
mitotic spindle geometry (cos theta <= 0.50 vs restrictive 0.20) and widening sister
separation boundaries (16.0 um vs restrictive 14.0 um).
Author: Ayush Shukla (ayushshuklaivyleague@gmail.com)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Set, Any
import numpy as np

from src.config import CFG, PHYSICAL_VOXEL_SCALE
from src.tracker import point_to_physical_um, physical_distance_um


class MitoticDivisionRecovery:
    """
    Recovers mitotic cleavage bifurcations (1 mother cell -> 2 daughter cells).
    Enforces biophysical plausibility:
      1. Mother-to-daughter distance <= div_max_parent_um (10.5 um)
      2. Sister-to-sister separation <= div_max_sister_um (16.0 um)
      3. Mitotic spindle alignment cos(theta) <= div_max_cos_angle (0.50)
      4. Joint cytokinesis pairing bonus
    """

    def __init__(
        self,
        div_max_parent_um: float = CFG["DIV_MAX_PARENT_UM"],
        div_max_sister_um: float = CFG["DIV_MAX_SISTER_UM"],
        div_max_cos_angle: float = CFG["DIV_MAX_COS_ANGLE"],
        div_joint_bonus: float = CFG["DIV_JOINT_BONUS"],
        div_global_frac_cap: float = CFG["DIV_GLOBAL_FRAC_CAP"],
        div_frame_frac_cap: float = CFG["DIV_FRAME_FRAC_CAP"],
    ):
        self.div_max_parent_um = div_max_parent_um
        self.div_max_sister_um = div_max_sister_um
        self.div_max_cos_angle = div_max_cos_angle
        self.div_joint_bonus = div_joint_bonus
        self.div_global_frac_cap = div_global_frac_cap
        self.div_frame_frac_cap = div_frame_frac_cap

    def compute_spindle_cosine(
        self,
        p_mother: np.ndarray,
        p_daughter1: np.ndarray,
        p_daughter2: np.ndarray,
    ) -> float:
        """
        Compute directional cosine between mother-to-daughter vectors:
          cos(theta) = (v1 . v2) / (|v1| * |v2|)
        For true cleavage, daughter cells separate in opposite directions (cos theta ~ -1).
        A threshold cos theta <= 0.50 permits up to 120-degree angular deviation in 3D.
        """
        v1 = p_daughter1 - p_mother
        v2 = p_daughter2 - p_mother
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 1.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def recover_mitotic_divisions(
        self,
        nodes_by_id: Dict[int, Dict[str, Any]],
        existing_edges: List[Dict[str, Any]],
        learned_edge_probs: Dict[Tuple[int, int], float] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Scan unlinked candidate daughter cells in frame t+1 and verify whether
        an existing tracked parent cell at frame t can bifurcate into two daughters.
        """
        learned_edge_probs = learned_edge_probs or {}

        # Map current edges
        out_by_source: Dict[int, List[Dict[str, Any]]] = {}
        target_in_edge: Set[int] = set()
        for edge in existing_edges:
            s_id = int(edge["source_id"])
            t_id = int(edge["target_id"])
            out_by_source.setdefault(s_id, []).append(edge)
            target_in_edge.add(t_id)

        # Group nodes by time frame
        ids_by_t: Dict[int, List[int]] = {}
        for nid, node in nodes_by_id.items():
            ids_by_t.setdefault(int(node["t"]), []).append(nid)

        # Precompute positions
        pos_um = {
            nid: point_to_physical_um([node["z"], node["y"], node["x"]])
            for nid, node in nodes_by_id.items()
        }

        # Global division cap
        max_total_divisions = max(1, int(round(len(existing_edges) * self.div_global_frac_cap)))
        added_edges: List[Dict[str, Any]] = []

        used_secondary_daughters: Set[int] = set()

        for t in sorted(ids_by_t.keys()):
            if len(added_edges) >= max_total_divisions:
                break

            current_frame_ids = ids_by_t.get(t, [])
            next_frame_ids = ids_by_t.get(t + 1, [])
            if not current_frame_ids or not next_frame_ids:
                continue

            # Identify unlinked target candidates in t+1
            unlinked_daughters = [
                t_id for t_id in next_frame_ids
                if t_id not in target_in_edge and t_id not in used_secondary_daughters
            ]
            if not unlinked_daughters:
                continue

            frame_div_cap = max(1, int(round(len(next_frame_ids) * self.div_frame_frac_cap)))
            frame_divisions = 0

            # Test parents that already have exactly one outgoing edge
            candidate_divisions: List[Tuple[float, int, int, float]] = []

            for s_id in current_frame_ids:
                curr_edges = out_by_source.get(s_id, [])
                if len(curr_edges) != 1:
                    continue

                d1_id = int(curr_edges[0]["target_id"])
                p_mother = pos_um[s_id]
                p_d1 = pos_um[d1_id]

                for d2_id in unlinked_daughters:
                    if d2_id in used_secondary_daughters:
                        continue

                    p_d2 = pos_um[d2_id]

                    # 1. Check mother-to-daughter distance
                    dist_mother_d2 = physical_distance_um(p_mother, p_d2)
                    if dist_mother_d2 > self.div_max_parent_um:
                        continue

                    # 2. Check sister-to-sister separation
                    dist_sister = physical_distance_um(p_d1, p_d2)
                    if dist_sister > self.div_max_sister_um:
                        continue

                    # 3. Check mitotic spindle cosine angle
                    cos_angle = self.compute_spindle_cosine(p_mother, p_d1, p_d2)
                    if cos_angle > self.div_max_cos_angle:
                        continue

                    # Joint cytokinesis confidence score
                    prob_d2 = float(learned_edge_probs.get((s_id, d2_id), 0.5))
                    score = prob_d2 * self.div_joint_bonus - 0.1 * (dist_sister / self.div_max_sister_um)
                    candidate_divisions.append((score, s_id, d2_id, prob_d2))

            # Sort candidate divisions by confidence score descending
            candidate_divisions.sort(key=lambda x: x[0], reverse=True)

            for score, s_id, d2_id, prob_d2 in candidate_divisions:
                if frame_divisions >= frame_div_cap or len(added_edges) >= max_total_divisions:
                    break
                if d2_id in used_secondary_daughters:
                    continue
                if len(out_by_source.get(s_id, [])) >= 2:
                    continue

                new_edge = {
                    "source_id": s_id,
                    "target_id": d2_id,
                    "prob": prob_d2,
                    "is_division": True,
                    "sister_id": int(out_by_source[s_id][0]["target_id"]),
                }
                added_edges.append(new_edge)
                out_by_source[s_id].append(new_edge)
                used_secondary_daughters.add(d2_id)
                frame_divisions += 1

        all_edges = list(existing_edges) + added_edges
        return all_edges
