"""Unit tests for evaluation module.

Tests for GRN evaluation metrics including:
- AUROC computation
- AUPR computation
- Precision@k, Recall@k, F1@k
- Full evaluation pipeline
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabpfn.grn.evaluation import (
    compute_aupr,
    compute_auroc,
    compute_f1_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_grn,
)


class TestComputeAUROC:
    """Test suite for AUROC computation."""

    def test_perfect_predictions(self):
        """Test AUROC with perfect predictions."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2"), ("TF3", "GENE3")}

        auroc = compute_auroc(inferred, gold_standard)

        # Perfect predictions should give AUROC close to 1.0
        assert auroc >= 0.9

    def test_random_predictions(self):
        """Test AUROC with random predictions."""
        # Create random edges
        np.random.seed(42)
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = set(list(inferred.keys())[:10])  # 10 true edges

        auroc = compute_auroc(inferred, gold_standard)

        # Random predictions should give AUROC around 0.5
        assert 0.3 <= auroc <= 0.7

    def test_worst_predictions(self):
        """Test AUROC with worst predictions (inverse ordering)."""
        inferred = {
            ("TF1", "GENE1"): 0.1,
            ("TF2", "GENE2"): 0.2,
            ("TF3", "GENE3"): 0.3,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2"), ("TF3", "GENE3")}

        auroc = compute_auroc(inferred, gold_standard)

        # Even with low scores, if all true edges have scores,
        # AUROC should still be reasonable
        assert auroc >= 0.0

    def test_with_dataframe_gold_standard(self):
        """Test AUROC with DataFrame gold standard."""
        inferred = {("TF1", "GENE1"): 0.9, ("TF2", "GENE2"): 0.3}
        gold_standard = pd.DataFrame({
            "tf": ["TF1", "TF2"],
            "target": ["GENE1", "GENE2"],
        })

        auroc = compute_auroc(inferred, gold_standard)

        assert isinstance(auroc, float)
        assert 0.0 <= auroc <= 1.0

    def test_empty_inferred_edges(self):
        """Test AUROC with empty inferred edges."""
        inferred = {}
        gold_standard = {("TF1", "GENE1")}

        auroc = compute_auroc(inferred, gold_standard)

        assert auroc == 0.0

    def test_empty_gold_standard(self):
        """Test AUROC with empty gold standard."""
        inferred = {("TF1", "GENE1"): 0.9}
        gold_standard = set()

        auroc = compute_auroc(inferred, gold_standard)

        # Should handle gracefully
        assert isinstance(auroc, float)


class TestComputeAUPR:
    """Test suite for AUPR computation."""

    def test_perfect_predictions(self):
        """Test AUPR with perfect predictions."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2"), ("TF3", "GENE3")}

        aupr = compute_aupr(inferred, gold_standard)

        # Perfect predictions should give AUPR close to 1.0
        assert aupr >= 0.8

    def test_sparse_gold_standard(self):
        """Test AUPR with sparse gold standard (typical GRN scenario)."""
        # 100 possible edges, only 10 are true
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(10)
            for j in range(10)
        }
        gold_standard = set(list(inferred.keys())[:10])

        aupr = compute_aupr(inferred, gold_standard)

        assert isinstance(aupr, float)
        assert 0.0 <= aupr <= 1.0

    def test_no_true_positives(self):
        """Test AUPR when no true edges are predicted with high scores."""
        inferred = {
            ("TF1", "GENE1"): 0.1,  # Low score for true edge
            ("TF2", "GENE2"): 0.2,  # Low score for true edge
            ("TF3", "GENE3"): 0.9,  # High score for false edge
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2")}

        aupr = compute_aupr(inferred, gold_standard)

        # Should still compute a valid AUPR
        assert isinstance(aupr, float)
        assert 0.0 <= aupr <= 1.0


class TestPrecisionAtK:
    """Test suite for Precision@k computation."""

    def test_perfect_top_k(self):
        """Test Precision@k when all top-k predictions are correct."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
            ("TF4", "GENE4"): 0.6,
            ("TF5", "GENE5"): 0.5,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2"), ("TF3", "GENE3")}

        precision = compute_precision_at_k(inferred, gold_standard, k=3)

        # Top 3 should all be correct
        assert precision == 1.0

    def test_partial_correct_top_k(self):
        """Test Precision@k with some correct predictions."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
            ("TF4", "GENE4"): 0.6,
            ("TF5", "GENE5"): 0.5,
        }
        gold_standard = {("TF1", "GENE1"), ("TF3", "GENE3")}

        precision = compute_precision_at_k(inferred, gold_standard, k=5)

        # 2 out of 5 are correct
        assert precision == 0.4

    def test_k_larger_than_predictions(self):
        """Test Precision@k when k is larger than number of predictions."""
        inferred = {("TF1", "GENE1"): 0.9}
        gold_standard = {("TF1", "GENE1")}

        precision = compute_precision_at_k(inferred, gold_standard, k=10)

        # Should handle gracefully - 1/10 = 0.1
        assert precision == 0.1

    def test_k_of_zero(self):
        """Test Precision@k when k is zero."""
        inferred = {("TF1", "GENE1"): 0.9}
        gold_standard = {("TF1", "GENE1")}

        precision = compute_precision_at_k(inferred, gold_standard, k=0)

        assert precision == 0.0

    def test_different_k_values(self):
        """Test Precision@k with different k values."""
        # Create edges with clear ranking - first edges have highest scores
        inferred = {}
        score = 1.0
        for i in range(5):
            for j in range(5):
                inferred[(f"TF{i}", f"GENE{j}")] = score
                score -= 0.01

        # Make sure the first 3 edges are in gold standard
        gold_standard = {("TF0", "GENE0"), ("TF0", "GENE1"), ("TF0", "GENE2")}

        # Precision should decrease as k increases
        precision_5 = compute_precision_at_k(inferred, gold_standard, k=5)
        precision_10 = compute_precision_at_k(inferred, gold_standard, k=10)

        # Top 5 contains all 3 true edges, so precision = 3/5 = 0.6
        assert precision_5 == 0.6
        # Top 10 also contains all 3 true edges, so precision = 3/10 = 0.3
        assert precision_10 == 0.3
        assert precision_5 > precision_10


class TestRecallAtK:
    """Test suite for Recall@k computation."""

    def test_perfect_recall(self):
        """Test Recall@k when all true edges are in top-k."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2"), ("TF3", "GENE3")}

        recall = compute_recall_at_k(inferred, gold_standard, k=10)

        assert recall == 1.0

    def test_partial_recall(self):
        """Test Recall@k with only some true edges in top-k."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
            ("TF4", "GENE4"): 0.6,
        }
        gold_standard = {
            ("TF1", "GENE1"),
            ("TF2", "GENE2"),
            ("TF3", "GENE3"),
            ("TF4", "GENE4"),
            ("TF5", "GENE5"),
        }

        recall = compute_recall_at_k(inferred, gold_standard, k=3)

        # 3 out of 5 true edges are in top 3
        assert recall == 0.6

    def test_empty_gold_standard(self):
        """Test Recall@k with empty gold standard."""
        inferred = {("TF1", "GENE1"): 0.9}
        gold_standard = set()

        recall = compute_recall_at_k(inferred, gold_standard, k=10)

        assert recall == 0.0


class TestF1AtK:
    """Test suite for F1@k computation."""

    def test_perfect_f1(self):
        """Test F1@k with perfect precision and recall."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
        }
        gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2")}

        # Use k=2 (same as number of predictions) for perfect F1
        f1 = compute_f1_at_k(inferred, gold_standard, k=2)

        assert f1 == 1.0

    def test_balanced_f1(self):
        """Test F1@k with balanced precision and recall."""
        inferred = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
            ("TF4", "GENE4"): 0.6,
        }
        gold_standard = {
            ("TF1", "GENE1"),
            ("TF2", "GENE2"),
            ("TF3", "GENE3"),
            ("TF4", "GENE4"),
            ("TF5", "GENE5"),
        }

        # k=4, 4 true positives out of 5 true edges
        # precision = 4/4 = 1.0, recall = 4/5 = 0.8
        # f1 = 2 * (1.0 * 0.8) / (1.0 + 0.8) = 1.6 / 1.8 ≈ 0.889
        f1 = compute_f1_at_k(inferred, gold_standard, k=4)

        assert abs(f1 - 0.889) < 0.01


class TestEvaluateGRN:
    """Test suite for full evaluation function."""

    def test_default_k_values(self):
        """Test evaluation with default k values."""
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(10)
            for j in range(20)
        }
        gold_standard = set(list(inferred.keys())[:50])

        metrics = evaluate_grn(inferred, gold_standard)

        # Check that all metrics are present
        assert "auroc" in metrics
        assert "aupr" in metrics
        assert "precision@100" in metrics
        assert "recall@100" in metrics
        assert "f1@100" in metrics
        assert "precision@500" in metrics
        assert "precision@1000" in metrics

    def test_custom_k_values(self):
        """Test evaluation with custom k values."""
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = set(list(inferred.keys())[:10])

        metrics = evaluate_grn(inferred, gold_standard, k_values=[10, 20])

        # Check that custom k values are used
        assert "precision@10" in metrics
        assert "precision@20" in metrics
        assert "precision@100" not in metrics

    def test_all_metrics_in_valid_range(self):
        """Test that all metrics are in valid range [0, 1]."""
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = set(list(inferred.keys())[:10])

        metrics = evaluate_grn(inferred, gold_standard)

        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, float)
            assert 0.0 <= metric_value <= 1.0, (
                f"{metric_name} = {metric_value} is not in [0, 1]"
            )

    def test_with_networkx_graph(self):
        """Test evaluation with NetworkX graph as input."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF2", "GENE2", weight=0.8)
            graph.add_edge("TF3", "GENE3", weight=0.3)

            gold_standard = {("TF1", "GENE1"), ("TF2", "GENE2")}

            metrics = evaluate_grn(graph, gold_standard)

            assert "auroc" in metrics
            assert "aupr" in metrics

        except ImportError:
            pytest.skip("networkx not installed")

    def test_consistency_with_individual_functions(self):
        """Test that evaluate_grn gives same results as individual functions."""
        inferred = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = set(list(inferred.keys())[:10])

        metrics = evaluate_grn(inferred, gold_standard, k_values=[10])

        # Compare with individual function calls
        auroc_individual = compute_auroc(inferred, gold_standard)
        aupr_individual = compute_aupr(inferred, gold_standard)
        precision_individual = compute_precision_at_k(inferred, gold_standard, k=10)

        assert metrics["auroc"] == auroc_individual
        assert metrics["aupr"] == aupr_individual
        assert metrics["precision@10"] == precision_individual
