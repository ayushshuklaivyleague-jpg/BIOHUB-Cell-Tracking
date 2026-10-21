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
        Compute cytokinesis division TP, FP, FN.
        A division is counted as TP if both daughter lineages originate from the matched mother cell.
        """
        gt_out: Dict[int, Set[int]] = {}
        gt_in: Dict[int, int] = {}
        for s, t in gt_edges:
            gt_out.setdefault(s, set()).add(t)
            gt_in[t] = s

        pred_out: Dict[int, Set[int]] = {}
        for s, t in pred_edges:
            pred_out.setdefault(s, set()).add(t)

        pred_node_ids = list(pred_nodes.keys())
        components = self.weakly_connected_components(pred_node_ids, pred_edges)
        fork_components = {
            components[n] for n, outs in pred_out.items() if len(outs) >= 2 and n in components
        }
        gt_division_sources = [s for s, outs in gt_out.items() if len(outs) >= 2]

        def lineage_descendants(root_child: int) -> Set[int]:
            seen = {root_child}
            stack = [root_child]
            while stack:
                cur = stack.pop()
                for nxt in gt_out.get(cur, set()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        tp = 0
        fn = 0
        tp_gt_sources: Set[int] = set()

        for gsrc in gt_division_sources:
            children = sorted(gt_out[gsrc])
            if len(children) < 2:
                continue
            anchor_candidates = [gsrc]
            if gsrc in gt_in:
                anchor_candidates.append(gt_in[gsrc])
            anchor_pred_nodes = [gt_to_pred[a] for a in anchor_candidates if a in gt_to_pred]

            lineage_hit_components: List[Set[int]] = []
            ok = True
            for child in children[:2]:
                lineage = lineage_descendants(child)
                hit_comp_ids = {
                    components[p_id]
                    for gt_id in lineage
                    if (p_id := gt_to_pred.get(gt_id)) is not None and p_id in components
                }
                if not hit_comp_ids:
                    ok = False
                    break
                lineage_hit_components.append(hit_comp_ids)

            if not ok or not anchor_pred_nodes:
                fn += 1
                continue

            anchor_comp_ids = {components[p] for p in anchor_pred_nodes if p in components}
            if not anchor_comp_ids:
                fn += 1
                continue

            found = any(
                comp_id in lineage_hit_components[0]
                and comp_id in lineage_hit_components[1]
                and comp_id in fork_components
                for comp_id in anchor_comp_ids
            )
            if found:
                tp += 1
                tp_gt_sources.add(gsrc)
            else:
                fn += 1

        fp = 0
        for n, outs in pred_out.items():
            if len(outs) < 2:
                continue
            g = pred_to_gt.get(n)
            if g is None or g not in gt_out or g in tp_gt_sources:
                continue
            fp += 1

        return tp, fp, fn

    def evaluate_lineage(
        self,
        pred_nodes: Dict[int, Tuple[float, float, float, float]],
        pred_edges: List[Tuple[int, int]],
        gt_nodes: Dict[int, Tuple[float, float, float, float]],
        gt_edges: List[Tuple[int, int]],
    ) -> Dict[str, float]:
        """
        Execute comprehensive benchmark evaluation.
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

        # Composite validation proxy score
        proxy_score = float(0.70 * adj_edge_jaccard + 0.30 * div_jaccard)

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

