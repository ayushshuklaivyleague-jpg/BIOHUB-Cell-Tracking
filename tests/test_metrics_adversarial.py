"""
Adversarial Unit Tests for Lineage and Division Metric Evaluation.
Validates that compute_division_confusion behaves with strict biological fidelity:
  1. Correct mother with two correctly identified daughters -> TP=1, FP=0, FN=0.
  2. Branch at the wrong mother -> TP=0, FP=1, FN=1.
  3. Branch connecting unrelated lineages -> TP=0, FP=1, FN=1.
  4. Missing daughters and unmatched nodes -> TP=0, FP=1, FN=1.
  5. Multiple predicted branches claiming one ground-truth division -> TP=1, FP=1, FN=0.
  6. Unmapped predicted branch -> TP=0, FP=1, FN=0 (when GT has 0 divisions).
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import BiohubResearchMetrics


class TestAdversarialMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = BiohubResearchMetrics(matching_radius_um=7.0)

    def test_1_correct_mother_and_both_daughters(self):
        """Case 1: Correct mother with two correctly identified daughters -> TP=1, FP=0, FN=0."""
        # GT: Node 1 (t=0) divides into Node 2 and Node 3 (t=1)
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 8.0, 10.0, 10.0),
            3: (1.0, 12.0, 10.0, 10.0),
        }
        gt_edges = [(1, 2), (1, 3)]

        # Pred: Node 101 matches 1, Node 102 matches 2, Node 103 matches 3
        pred_nodes = {
            101: (0.0, 10.0, 10.0, 10.0),
            102: (1.0, 8.0, 10.0, 10.0),
            103: (1.0, 12.0, 10.0, 10.0),
        }
        pred_edges = [(101, 102), (101, 103)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 1, "Should count 1 True Positive division")
        self.assertEqual(fp, 0, "Should count 0 False Positive divisions")
        self.assertEqual(fn, 0, "Should count 0 False Negative divisions")

    def test_2_branch_at_wrong_mother(self):
        """Case 2: Branch predicted at a cell that did NOT divide in GT -> TP=0, FP=1, FN=1."""
        # GT: Node 1 divides into 2 and 3; Node 4 simply persists to 5
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 8.0, 10.0, 10.0),
            3: (1.0, 12.0, 10.0, 10.0),
            4: (0.0, 50.0, 50.0, 50.0),
            5: (1.0, 50.0, 50.0, 50.0),
            6: (1.0, 52.0, 50.0, 50.0),
        }
        gt_edges = [(1, 2), (1, 3), (4, 5)]

        # Pred: Predicts division at 104 (matched to 4) instead of 101 (matched to 1)
        pred_nodes = {
            101: (0.0, 10.0, 10.0, 10.0),
            102: (1.0, 8.0, 10.0, 10.0),
            103: (1.0, 12.0, 10.0, 10.0),
            104: (0.0, 50.0, 50.0, 50.0),
            105: (1.0, 50.0, 50.0, 50.0),
            106: (1.0, 52.0, 50.0, 50.0),
        }
        # 101 only has 1 outgoing edge; 104 has 2 outgoing edges!
        pred_edges = [(101, 102), (104, 105), (104, 106)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 0, "No valid division should be counted")
        self.assertEqual(fp, 1, "Spurious branch at non-dividing mother 4 must be FP")
        self.assertEqual(fn, 1, "Unrecovered division at mother 1 must be FN")

    def test_3_branch_connecting_unrelated_lineages(self):
        """Case 3: Branch predicted at correct mother, but connecting unrelated lineage -> TP=0, FP=1, FN=1."""
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 8.0, 10.0, 10.0),
            3: (1.0, 12.0, 10.0, 10.0),
            4: (1.0, 80.0, 80.0, 80.0),  # Unrelated cell
        }
        gt_edges = [(1, 2), (1, 3)]

        pred_nodes = {
            101: (0.0, 10.0, 10.0, 10.0),
            102: (1.0, 8.0, 10.0, 10.0),
            104: (1.0, 80.0, 80.0, 80.0),  # Matches unrelated cell 4
        }
        # Pred connects 101 to 102 and 104 (only 1 valid daughter, 1 unrelated)
        pred_edges = [(101, 102), (101, 104)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 0, "Should not count as TP because second daughter is unrelated")
        self.assertEqual(fp, 1, "Erroneous fork connecting wrong daughter must be FP")
        self.assertEqual(fn, 1, "Unrecovered true division must be FN")

    def test_4_missing_daughter_unmatched_nodes(self):
        """Case 4: One daughter is an unmapped detection -> TP=0, FP=1, FN=1."""
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 8.0, 10.0, 10.0),
            3: (1.0, 12.0, 10.0, 10.0),
        }
        gt_edges = [(1, 2), (1, 3)]

        pred_nodes = {
            101: (0.0, 10.0, 10.0, 10.0),
            102: (1.0, 8.0, 10.0, 10.0),
            199: (1.0, 200.0, 200.0, 200.0),  # Way too far, unmapped!
        }
        pred_edges = [(101, 102), (101, 199)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 0, "Cannot count as TP when daughter is an unmapped hallucination")
        self.assertEqual(fp, 1, "Must be an FP")
        self.assertEqual(fn, 1, "Must be an FN")

    def test_5_multiple_predicted_branches_for_one_gt(self):
        """Case 5: Redundant predicted forks -> TP=1, FP=1, FN=0."""
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 8.0, 10.0, 10.0),
            3: (1.0, 12.0, 10.0, 10.0),
        }
        gt_edges = [(1, 2), (1, 3)]

        # Pred has 101 matching 1 with daughters (102, 103), AND 101 also branched to 104
        # or another fork at 101
        pred_nodes = {
            101: (0.0, 10.0, 10.0, 10.0),
            102: (1.0, 8.0, 10.0, 10.0),
            103: (1.0, 12.0, 10.0, 10.0),
            104: (1.0, 8.5, 10.0, 10.0),  # duplicate near daughter 2
        }
        # Pred predicts a trifurcation at 101: 3 outgoing edges
        pred_edges = [(101, 102), (101, 103), (101, 104)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 1, "Valid branch matches GT division 1")
        self.assertEqual(fp, 0, "Trifurcation at single node counts as 1 branch event (TP)")
        self.assertEqual(fn, 0, "No FN")

    def test_6_unmapped_mother_branch(self):
        """Case 6: Predicted fork at completely unmapped mother -> FP=1."""
        gt_nodes = {
            1: (0.0, 10.0, 10.0, 10.0),
            2: (1.0, 10.0, 10.0, 10.0),
        }
        gt_edges = [(1, 2)]

        # Pred forks at node 999 which is far away and unmapped
        pred_nodes = {
            999: (0.0, 300.0, 300.0, 300.0),
            998: (1.0, 305.0, 300.0, 300.0),
            997: (1.0, 295.0, 300.0, 300.0),
        }
        pred_edges = [(999, 998), (999, 997)]

        pred_to_gt, gt_to_pred = self.metrics.match_nodes_bipartite(pred_nodes, gt_nodes)
        tp, fp, fn = self.metrics.compute_division_confusion(
            pred_nodes, pred_edges, gt_nodes, gt_edges, pred_to_gt, gt_to_pred
        )

        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1, "Unmapped mother fork must count as an FP")
        self.assertEqual(fn, 0, "GT had 0 divisions, so FN=0")


if __name__ == "__main__":
    unittest.main()
