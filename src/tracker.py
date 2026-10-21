"""
Neural-Aware Global Linker for 4D Spatiotemporal Cell Tracking.
Formulates a global bipartite assignment objective combining log-odds calibrated
neural edge affinities with physical search radii and EMA velocity vector persistence.

Neural Architecture & Provenance:
  In the full competition pipeline (see notebooks/WINNING_NOTEBOOK.ipynb), edge probabilities
  P_neural are predicted by a 50-epoch 3D U-Net Transformer backbone with 8-view geometric TTA
  (artifact: 'biohub-tracking-support-pack-50ep-v1', script: 'predict_unet_transformer.py').
  Candidate pairs without neural scores default to neutral baseline probability 0.5 (logit = 0),
  where assignment is governed by physical distance and motion persistence.
Author: Ayush Shukla (ayushshuklaivyleague@gmail.com)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import CFG, PHYSICAL_VOXEL_SCALE


def point_to_physical_um(coords: np.ndarray | List[float]) -> np.ndarray:
    """Convert (z, y, x) voxel coordinates to physical metric micrometers."""
    return np.asarray(coords, dtype=np.float64) * PHYSICAL_VOXEL_SCALE


def physical_distance_um(p1_um: np.ndarray, p2_um: np.ndarray) -> float:
    """Compute Euclidean distance in physical metric micrometers."""
    diff = p1_um - p2_um
    return float(np.sqrt(np.dot(diff, diff)))


class NeuralAwareGlobalLinker:
    """
    Spatiotemporal Frame-to-Frame Linker formulating a global bipartite matching objective:
      C(i, j) = -w_learned * logit(P_neural) + w_dist * (d_phys / d_0) + w_vel * (d_motion / d_0)
    solved via the Hungarian algorithm (linear_sum_assignment).
    """

    def __init__(
        self,
        candidate_radius_um: float = CFG["CANDIDATE_RADIUS_UM"],
        learned_edge_weight: float = CFG["LEARNED_EDGE_WEIGHT"],
        distance_weight: float = CFG["DISTANCE_WEIGHT"],
        velocity_weight: float = CFG["VELOCITY_WEIGHT"],
        velocity_ema_alpha: float = CFG["VELOCITY_EMA_ALPHA"],
        min_link_prob: float = CFG["MIN_LINK_PROB"],
    ):
        self.candidate_radius_um = candidate_radius_um
        self.learned_edge_weight = learned_edge_weight
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.velocity_ema_alpha = velocity_ema_alpha
        self.min_link_prob = min_link_prob

    def compute_logit(self, p: float, eps: float = 1e-6) -> float:
        """Compute numerically stable log-odds logit(p)."""
        p_clipped = np.clip(p, eps, 1.0 - eps)
        return float(np.log(p_clipped / (1.0 - p_clipped)))

    def link_consecutive_frames(
        self,
        source_nodes: Dict[int, Dict[str, Any]],
        target_nodes: Dict[int, Dict[str, Any]],
        learned_edge_probs: Dict[Tuple[int, int], float] | None = None,
        node_velocities: Dict[int, np.ndarray] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, np.ndarray]]:
        """
        Match nodes between frame t (source) and frame t+1 (target).
        Returns list of matched edges and updated velocity vectors.
        """
        learned_edge_probs = learned_edge_probs or {}
        node_velocities = node_velocities or {}

        source_ids = sorted(source_nodes.keys())
        target_ids = sorted(target_nodes.keys())

        if not source_ids or not target_ids:
            return [], node_velocities

        # Precompute physical positions (z, y, x) -> (z_um, y_um, x_um)
        source_pos_um = {
            nid: point_to_physical_um([source_nodes[nid]["z"], source_nodes[nid]["y"], source_nodes[nid]["x"]])
            for nid in source_ids
        }
        target_pos_um = {
            nid: point_to_physical_um([target_nodes[nid]["z"], target_nodes[nid]["y"], target_nodes[nid]["x"]])
            for nid in target_ids
        }

        # Build candidate pairs within candidate_radius_um
        n_src = len(source_ids)
        n_tgt = len(target_ids)
        cost_matrix = np.full((n_src, n_tgt), fill_value=1e6, dtype=np.float64)

        src_id_to_idx = {nid: i for i, nid in enumerate(source_ids)}
        tgt_id_to_idx = {nid: j for j, nid in enumerate(target_ids)}

        valid_candidate_count = 0
        for i, s_id in enumerate(source_ids):
            s_pos = source_pos_um[s_id]
            v_ema = node_velocities.get(s_id, np.zeros(3, dtype=np.float64))

            # Predicted position based on inertia
            pred_pos = s_pos + v_ema

            for j, t_id in enumerate(target_ids):
                t_pos = target_pos_um[t_id]
                dist_um = physical_distance_um(s_pos, t_pos)

                if dist_um > self.candidate_radius_um:
                    continue

                prob = float(learned_edge_probs.get((s_id, t_id), 0.5))
                if prob < self.min_link_prob:
                    continue

                motion_disp_um = physical_distance_um(pred_pos, t_pos)

                # Neural log-odds cost + physical penalty
                logit_val = self.compute_logit(prob)
                cost = (
                    -self.learned_edge_weight * logit_val
                    + self.distance_weight * (dist_um / self.candidate_radius_um)
                    + self.velocity_weight * (motion_disp_um / self.candidate_radius_um)
                )

                cost_matrix[i, j] = cost
                valid_candidate_count += 1

        if valid_candidate_count == 0:
            return [], node_velocities

        # Solve optimal bipartite assignment via Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        selected_edges: List[Dict[str, Any]] = []
        updated_velocities = dict(node_velocities)

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] >= 1e5:
                continue

            s_id = source_ids[r]
            t_id = target_ids[c]
            prob = float(learned_edge_probs.get((s_id, t_id), 0.5))

            edge = {
                "source_id": s_id,
                "target_id": t_id,
                "prob": prob,
                "cost": float(cost_matrix[r, c]),
            }
            selected_edges.append(edge)

            # Update EMA velocity
            s_pos = source_pos_um[s_id]
            t_pos = target_pos_um[t_id]
            curr_v = t_pos - s_pos

            prev_v = node_velocities.get(s_id, curr_v)
            new_v = self.velocity_ema_alpha * curr_v + (1.0 - self.velocity_ema_alpha) * prev_v
            updated_velocities[t_id] = new_v

        return selected_edges, updated_velocities
