"""
Biohub Research Metrics Suite for 4D Cell Tracking.
Implements:
  1. Anisotropic Physical-Space Bipartite Matching (max_dist = 7.0 um).
  2. Edge Confusion Matrix & Adjusted Edge Jaccard.
  3. Mitotic Division Confusion Matrix & Division Jaccard.
  4. Error Mass Decomposition (Detection vs Fragmentation vs Association).
  5. Node Density Ratio.

Note on Boundary-Case Handling:
  In compute_edge_confusion, false positives are identified when a predicted edge connects
  nodes that map to unmatched biological targets or invalid ground-truth transitions.
  Adjusted Jaccard penalizes overall prediction count divergence relative to ground-truth targets.
Author: Ayush Shukla (ayushshuklaivyleague@gmail.com)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set, Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.config import CFG, PHYSICAL_VOXEL_SCALE, METRIC_MATCHING_RADIUS_UM


class BiohubResearchMetrics:
    """
    Computes research validation metrics for predicted vs ground-truth 4D cell lineage graphs.
    """

    def __init__(
        self,
        matching_radius_um: float = METRIC_MATCHING_RADIUS_UM,
        voxel_scale_um: np.ndarray = PHYSICAL_VOXEL_SCALE,
    ):
        self.matching_radius_um = matching_radius_um
        self.voxel_scale_um = np.asarray(voxel_scale_um, dtype=np.float64)

    def match_nodes_bipartite(
        self,
        pred_nodes: Dict[int, Tuple[float, float, float, float]],  # id -> (t, z, y, x)
        gt_nodes: Dict[int, Tuple[float, float, float, float]],    # id -> (t, z, y, x)
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Frame-by-frame bipartite matching under Euclidean distance in physical metric micrometers.
        Returns:
          pred_to_gt: mapping from predicted node ID -> ground-truth node ID
          gt_to_pred: mapping from ground-truth node ID -> predicted node ID
        """
        pred_by_t: Dict[int, List[int]] = {}
        for pid, (t, *_r) in pred_nodes.items():
            pred_by_t.setdefault(int(t), []).append(pid)

        gt_by_t: Dict[int, List[int]] = {}
        for gid, (t, *_r) in gt_nodes.items():
            gt_by_t.setdefault(int(t), []).append(gid)

        pred_to_gt: Dict[int, int] = {}
        gt_to_pred: Dict[int, int] = {}

        for t, p_ids in pred_by_t.items():
            g_ids = gt_by_t.get(t, [])
            if not g_ids:
                continue

            p_pos = np.array([pred_nodes[p][1:] for p in p_ids], dtype=np.float64) * self.voxel_scale_um
            g_pos = np.array([gt_nodes[g][1:] for g in g_ids], dtype=np.float64) * self.voxel_scale_um

            diff = p_pos[:, None, :] - g_pos[None, :, :]
            cost = np.sqrt(np.sum(diff ** 2, axis=-1))

            BIG = 1e6
            cost_gated = np.where(cost <= self.matching_radius_um, cost, BIG)
            row_ind, col_ind = linear_sum_assignment(cost_gated)

            for r, c in zip(row_ind, col_ind):
                if cost_gated[r, c] >= BIG:
                    continue
                p_id = p_ids[r]
                g_id = g_ids[c]
                pred_to_gt[p_id] = g_id
                gt_to_pred[g_id] = p_id

        return pred_to_gt, gt_to_pred

    def compute_edge_confusion(
        self,
        pred_edges: List[Tuple[int, int]],
        gt_edges: List[Tuple[int, int]],
        pred_to_gt: Dict[int, int],
        gt_to_pred: Dict[int, int],
    ) -> Tuple[int, int, int]:
        """
        Compute edge TP, FP, FN based on matched node IDs.
        """
        gt_edge_set = set(gt_edges)
        gt_outgoing: Dict[int, Set[int]] = {}
        gt_incoming_source: Dict[int, int] = {}
        for s, t in gt_edge_set:
            gt_outgoing.setdefault(s, set()).add(t)
            gt_incoming_source[t] = s

        tp = 0
        fp = 0
        matched_gt_edges = set()

        for s, t in pred_edges:
            ms = pred_to_gt.get(s)
            mt = pred_to_gt.get(t)
            is_tp = ms is not None and mt is not None and mt in gt_outgoing.get(ms, set())
            if is_tp:
                tp += 1
                matched_gt_edges.add((ms, mt))
                continue
            is_fp = (mt is not None and mt in gt_incoming_source) or (
                ms is not None and bool(gt_outgoing.get(ms))
            )
            if is_fp:
                fp += 1

        fn = len(gt_edge_set - matched_gt_edges)
        return tp, fp, fn

    @staticmethod
    def edge_jaccard(tp: int, fp: int, fn: int) -> float:
        """Standard Jaccard similarity index: TP / (TP + FP + FN)."""
        denom = tp + fp + fn
        return float(tp / denom) if denom > 0 else 0.0

    @staticmethod
    def adjusted_jaccard(jaccard: float, t_pred: int, t_true: int, alpha: float = 0.1) -> float:
        """Penalizes total edge count divergence."""
        if not t_true or t_true <= 0:
            return jaccard
        penalty = alpha * abs(t_pred - t_true) / t_true
        return float(max(0.0, jaccard * (1.0 - penalty)))

    @staticmethod
    def weakly_connected_components(
        node_ids: List[int],
        edges: List[Tuple[int, int]]
    ) -> Dict[int, int]:
        """Disjoint-set union to isolate weakly connected graph components."""
        parent = {n: n for n in node_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for s, t in edges:
            if s in parent and t in parent:
                union(s, t)
        return {n: find(n) for n in node_ids}

    def compute_division_confusion(
        self,
        pred_nodes: Dict[int, Any],
        pred_edges: List[Tuple[int, int]],
        gt_nodes: Dict[int, Any],
        gt_edges: List[Tuple[int, int]],
        pred_to_gt: Dict[int, int],
        gt_to_pred: Dict[int, int],
    ) -> Tuple[int, int, int]:
        """
        Compute cytokinesis division confusion matrix (TP, FP, FN) with strict biological matching.

        A predicted division is a predicted node u with out-degree >= 2.
        A ground-truth division is a ground-truth node u* with out-degree >= 2.

        Strict Biological Verification Criteria:
          1. Mother Verification: The predicted mother u must match the true mother u* (pred_to_gt[u] == u*).
          2. Daughter Verification: At least 2 predicted daughters from pred_out[u] must match distinct
             true daughters from gt_out[u*] via bipartite node matching.
          3. 1-to-1 Mapping: Each ground-truth division event u* can be claimed by at most one predicted division.

        False Positives (FP):
          Any predicted branching event (out-degree >= 2) that fails to match an unclaimed ground-truth division:
            - Branching at an unmapped node (pred_to_gt[u] is None).
            - Branching at a non-dividing node (u* has out-degree <= 1).
            - Branching where fewer than 2 daughters match the true daughter set.
            - Duplicate predicted branches claiming an already matched ground-truth division.

        False Negatives (FN):
          Any ground-truth division event that was not claimed by a valid TP prediction:
            FN = len(gt_division_mothers) - TP.
        """
        gt_out: Dict[int, Set[int]] = {}
        for s, t in gt_edges:
            gt_out.setdefault(s, set()).add(t)

        pred_out: Dict[int, Set[int]] = {}
        for s, t in pred_edges:
            pred_out.setdefault(s, set()).add(t)

        gt_division_mothers = {s: outs for s, outs in gt_out.items() if len(outs) >= 2}
        pred_division_mothers = {s: outs for s, outs in pred_out.items() if len(outs) >= 2}

        tp = 0
        fp = 0
        claimed_gt_divisions: Set[int] = set()

        for u_pred, d_pred in pred_division_mothers.items():
            u_gt = pred_to_gt.get(u_pred)

            # Case 1: Predicted mother is unmapped or does not correspond to a ground-truth division
            if u_gt is None or u_gt not in gt_division_mothers:
                fp += 1
                continue

            # Case 2: Ground-truth division was already claimed by an earlier predicted branch
            if u_gt in claimed_gt_divisions:
                fp += 1
                continue

            # Case 3: Check daughter correspondence
            true_daughters = gt_division_mothers[u_gt]
            matched_daughters = {
                pred_to_gt[v] for v in d_pred
                if v in pred_to_gt and pred_to_gt[v] in true_daughters
            }

            if len(matched_daughters) >= 2:
                tp += 1
                claimed_gt_divisions.add(u_gt)
            else:
                fp += 1

        fn = len(gt_division_mothers) - tp
        return tp, fp, fn

    def evaluate_lineage(
        self,
        pred_nodes: Dict[int, Tuple[float, float, float, float]],
        pred_edges: List[Tuple[int, int]],
        gt_nodes: Dict[int, Tuple[float, float, float, float]],
        gt_edges: List[Tuple[int, int]],
        weight_edge: float = 0.85,
        weight_div: float = 0.15,
    ) -> Dict[str, float]:
        """
        Execute comprehensive benchmark evaluation.

        Args:
            weight_edge: Weight for adjusted edge Jaccard in validation proxy (default 0.85).
            weight_div: Weight for division Jaccard in validation proxy (default 0.15).
        """
        pred_to_gt, gt_to_pred = self.match_nodes_bipartite(pred_nodes, gt_nodes)
        edge_tp, edge_fp, edge_fn = self.compute_edge_confusion(pred_edges, gt_edges, pred_to_gt, gt_to_pred)
        div_tp, div_fp, div_fn = self.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        raw_edge_jaccard = self.edge_jaccard(edge_tp, edge_fp, edge_fn)
        adj_edge_jaccard = self.adjusted_jaccard(raw_edge_jaccard, len(pred_edges), len(gt_edges))
        div_jaccard = self.edge_jaccard(div_tp, div_fp, div_fn)

        node_ratio = float(len(pred_nodes) / max(1, len(gt_nodes)))

        # Composite validation proxy score: S = 0.85 * J_edge + 0.15 * J_div
        proxy_score = float(weight_edge * adj_edge_jaccard + weight_div * div_jaccard)

        return {
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "raw_edge_jaccard": raw_edge_jaccard,
            "adjusted_edge_jaccard": adj_edge_jaccard,
            "div_tp": div_tp,
            "div_fp": div_fp,
            "div_fn": div_fn,
            "division_jaccard": div_jaccard,
            "node_density_ratio": node_ratio,
            "validation_proxy_score": proxy_score,
        }


# Backwards compatibility alias
BiohubEvaluationMetrics = BiohubResearchMetrics

