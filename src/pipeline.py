"""
End-to-End 4D Volumetric Cell Tracking Pipeline.
Orchestrates:
  1. Detection & Subvoxel Localization with sequence-metadata node calibration.
  2. Frame-to-frame candidate extraction with expanded physical search radius (8.8 um).
  3. Neural-Aware Global Linker (calibrated log-odds assignment).
  4. Mitotic cytokinesis division recovery (relaxed 3D spindle alignment cos theta <= 0.50).
  5. Metric-preserving graph pruning (safeguarding against artificial shortcuts).
Author: Ayush Shukla (ayushshuklaivyleague@gmail.com)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

from src.config import CFG
from src.tracker import NeuralAwareGlobalLinker
from src.division import MitoticDivisionRecovery
from src.detection import NodeCountCalibrator


class VolumetricTrackingPipeline:
    """
    Modular execution engine for 4D spatiotemporal cell tracking in anisotropic light-sheet volumes.
    """

    def __init__(self, cfg_overrides: Dict[str, Any] | None = None):
        cfg_dict = dict(CFG)
        if cfg_overrides:
            cfg_dict.update(cfg_overrides)

        self.cfg = cfg_dict
        self.node_calibrator = NodeCountCalibrator(
            target_node_ratio=self.cfg["TARGET_NODE_RATIO"],
            max_threshold_shift=self.cfg["MAX_NODE_THRESHOLD_SHIFT"],
            base_threshold=self.cfg["DET_THRESHOLD"],
        )
        self.linker = NeuralAwareGlobalLinker(
            candidate_radius_um=self.cfg["CANDIDATE_RADIUS_UM"],
            learned_edge_weight=self.cfg["LEARNED_EDGE_WEIGHT"],
            distance_weight=self.cfg["DISTANCE_WEIGHT"],
            velocity_weight=self.cfg["VELOCITY_WEIGHT"],
            velocity_ema_alpha=self.cfg["VELOCITY_EMA_ALPHA"],
            min_link_prob=self.cfg["MIN_LINK_PROB"],
        )
        self.division_recovery = MitoticDivisionRecovery(
            div_max_parent_um=self.cfg["DIV_MAX_PARENT_UM"],
            div_max_sister_um=self.cfg["DIV_MAX_SISTER_UM"],
            div_max_cos_angle=self.cfg["DIV_MAX_COS_ANGLE"],
            div_joint_bonus=self.cfg["DIV_JOINT_BONUS"],
            div_global_frac_cap=self.cfg["DIV_GLOBAL_FRAC_CAP"],
            div_frame_frac_cap=self.cfg["DIV_FRAME_FRAC_CAP"],
        )

    def track_sequence(
        self,
        raw_detections: List[Dict[str, Any]],
        estimated_number_of_nodes: int | float | None = None,
        learned_edge_probs: Dict[Tuple[int, int], float] | None = None,
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Execute full lineage reconstruction for a 4D video sequence.
        Returns:
          nodes_by_id: dict of verified nodes
          edges: list of directed lineage edges
        """
        learned_edge_probs = learned_edge_probs or {}

        # 1. Dynamic Node Calibration
        if self.cfg["CALIBRATE_NODE_COUNT"]:
            calibrated_thresh = self.node_calibrator.calibrate_threshold(
                raw_detections, estimated_number_of_nodes
            )
            filtered_detections = self.node_calibrator.filter_nodes(
                raw_detections, calibrated_thresh
            )
        else:
            filtered_detections = [d for d in raw_detections if d.get("score", 1.0) >= self.cfg["DET_THRESHOLD"]]

        # Build nodes_by_id
        nodes_by_id: Dict[int, Dict[str, Any]] = {
            int(d["id"]): d for d in filtered_detections
        }

        # Group by time frame
        frames_dict: Dict[int, Dict[int, Dict[str, Any]]] = {}
        for nid, node in nodes_by_id.items():
            frames_dict.setdefault(int(node["t"]), {})[nid] = node

        # 2. Sequential Neural-Aware Global Linking
        sorted_frames = sorted(frames_dict.keys())
        all_edges: List[Dict[str, Any]] = []
        node_velocities: Dict[int, np.ndarray] = {}

        for i in range(len(sorted_frames) - 1):
            t_curr = sorted_frames[i]
            t_next = sorted_frames[i + 1]

            # Only link consecutive frames (t -> t+1) to safeguard bipartite metric integrity
            if t_next != t_curr + 1:
                continue

            src_nodes = frames_dict[t_curr]
            tgt_nodes = frames_dict[t_next]

            step_edges, node_velocities = self.linker.link_consecutive_frames(
                source_nodes=src_nodes,
                target_nodes=tgt_nodes,
                learned_edge_probs=learned_edge_probs,
                node_velocities=node_velocities,
            )
            all_edges.extend(step_edges)

        # 3. Mitotic Cytokinesis Recovery
        if self.cfg["ENABLE_DIVISIONS"]:
            all_edges = self.division_recovery.recover_mitotic_divisions(
                nodes_by_id=nodes_by_id,
                existing_edges=all_edges,
                learned_edge_probs=learned_edge_probs,
            )

        # 4. Metric-Preserving Graph Pruning
        if self.cfg["PRUNE_ISOLATED"]:
            connected_node_ids = set()
            for e in all_edges:
                connected_node_ids.add(int(e["source_id"]))
                connected_node_ids.add(int(e["target_id"]))

            # Note: We keep nodes from single-frame terminal lineages or clean true orphans
            # To preserve nascent daughters, we only prune points completely isolated across time
            nodes_by_id = {nid: n for nid, n in nodes_by_id.items() if nid in connected_node_ids}

        return nodes_by_id, all_edges
