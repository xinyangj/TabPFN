"""Unit tests for visualization module.

Tests for GRN visualization utilities including:
- GRNNetworkVisualizer
- AttentionHeatmapVisualizer
- EdgeScoreVisualizer
- create_evaluation_summary_plot
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tabpfn.grn.visualization import (
    AttentionHeatmapVisualizer,
    EdgeScoreVisualizer,
    GRNNetworkVisualizer,
    create_evaluation_summary_plot,
)


class TestGRNNetworkVisualizer:
    """Test suite for GRNNetworkVisualizer class."""

    def test_init(self):
        """Test visualizer initialization."""
        visualizer = GRNNetworkVisualizer(figsize=(10, 8), dpi=150)

        assert visualizer.figsize == (10, 8)
        assert visualizer.dpi == 150

    def test_init_with_defaults(self):
        """Test visualizer initialization with default parameters."""
        visualizer = GRNNetworkVisualizer()

        assert visualizer.figsize == (12, 10)
        assert visualizer.dpi == 100

    def test_plot_network_basic(self, tmp_path):
        """Test basic network plotting without saving."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF2", "GENE2", weight=0.8)
            graph.add_edge("TF1", "GENE2", weight=0.3)

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_network(graph)

            assert fig is not None
            assert hasattr(fig, "axes")

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_save_to_file(self, tmp_path):
        """Test network plotting with file output."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF2", "GENE2", weight=0.8)

            visualizer = GRNNetworkVisualizer()
            output_path = tmp_path / "network.pdf"

            result = visualizer.plot_network(graph, output_path)

            assert result is None
            assert output_path.exists()

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_different_layouts(self, tmp_path):
        """Test network plotting with different layout algorithms."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF2", "GENE2", weight=0.8)

            visualizer = GRNNetworkVisualizer()

            for layout in ["spring", "circular", "kamada_kawai", "spectral", "shell"]:
                fig = visualizer.plot_network(graph, layout=layout)
                assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_invalid_layout(self, tmp_path):
        """Test network plotting with invalid layout raises error."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)

            visualizer = GRNNetworkVisualizer()

            with pytest.raises(ValueError, match="Unknown layout"):
                visualizer.plot_network(graph, layout="invalid_layout")

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_max_nodes(self, tmp_path):
        """Test network plotting with node limit."""
        try:
            import networkx as nx

            # Create larger network
            graph = nx.DiGraph()
            for i in range(50):
                graph.add_edge(f"TF{i}", f"GENE{i}", weight=np.random.random())

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_network(graph, max_nodes=10)

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_custom_colors(self, tmp_path):
        """Test network plotting with custom colors."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_network(
                graph,
                tf_color="#FF0000",
                target_color="#0000FF",
            )

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_subnetwork(self, tmp_path):
        """Test subnetwork extraction and plotting."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF1", "GENE2", weight=0.8)
            graph.add_edge("GENE1", "GENE3", weight=0.7)
            graph.add_edge("GENE2", "GENE4", weight=0.6)

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_subnetwork(graph, focus_genes=["TF1"])

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_subnetwork_with_distance(self, tmp_path):
        """Test subnetwork with custom max distance."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("GENE1", "GENE2", weight=0.8)
            graph.add_edge("GENE2", "GENE3", weight=0.7)

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_subnetwork(
                graph,
                focus_genes=["TF1"],
                max_distance=1,
            )

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_community_structure(self, tmp_path):
        """Test community detection visualization."""
        try:
            import networkx as nx

            graph = nx.DiGraph()
            # Create two communities
            graph.add_edge("TF1", "GENE1", weight=0.9)
            graph.add_edge("TF1", "GENE2", weight=0.8)
            graph.add_edge("TF2", "GENE3", weight=0.9)
            graph.add_edge("TF2", "GENE4", weight=0.8)

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_community_structure(graph)

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_plot_network_empty_graph(self, tmp_path):
        """Test plotting empty network."""
        try:
            import networkx as nx

            graph = nx.DiGraph()

            visualizer = GRNNetworkVisualizer()
            fig = visualizer.plot_network(graph)

            assert fig is not None

        except ImportError:
            pytest.skip("networkx not installed")


class TestAttentionHeatmapVisualizer:
    """Test suite for AttentionHeatmapVisualizer class."""

    def test_init(self):
        """Test attention visualizer initialization."""
        visualizer = AttentionHeatmapVisualizer(figsize=(8, 6), dpi=120)

        assert visualizer.figsize == (8, 6)
        assert visualizer.dpi == 120

    def test_init_with_defaults(self):
        """Test attention visualizer initialization with defaults."""
        visualizer = AttentionHeatmapVisualizer()

        assert visualizer.figsize == (10, 8)
        assert visualizer.dpi == 100

    def test_plot_attention_heatmap_basic(self, tmp_path):
        """Test basic attention heatmap plotting."""
        np.random.seed(42)
        attention = np.random.randn(5, 5)

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_attention_heatmap(attention)

        assert fig is not None

    def test_plot_attention_heatmap_with_labels(self, tmp_path):
        """Test attention heatmap with TF names."""
        np.random.seed(42)
        attention = np.random.randn(3, 3)
        tf_names = ["TF1", "TF2", "TF3"]

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_attention_heatmap(attention, tf_names=tf_names)

        assert fig is not None

    def test_plot_attention_heatmap_save_to_file(self, tmp_path):
        """Test attention heatmap with file output."""
        np.random.seed(42)
        attention = np.random.randn(5, 5)

        visualizer = AttentionHeatmapVisualizer()
        output_path = tmp_path / "attention.pdf"

        result = visualizer.plot_attention_heatmap(attention, output_path=output_path)

        assert result is None
        assert output_path.exists()

    def test_plot_attention_heatmap_custom_colormap(self, tmp_path):
        """Test attention heatmap with custom colormap."""
        np.random.seed(42)
        attention = np.random.randn(5, 5)

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_attention_heatmap(attention, cmap="coolwarm")

        assert fig is not None

    def test_plot_attention_heatmap_custom_scale(self, tmp_path):
        """Test attention heatmap with custom value scale."""
        np.random.seed(42)
        attention = np.random.randn(5, 5)

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_attention_heatmap(
            attention, vmin=-1.0, vmax=1.0
        )

        assert fig is not None

    def test_plot_multi_layer_attention(self, tmp_path):
        """Test multi-layer attention visualization."""
        np.random.seed(42)

        attention_dict = {
            "layer_0": {"between_features": np.random.randn(3, 3)},
            "layer_1": {"between_features": np.random.randn(3, 3)},
            "layer_2": {"between_features": np.random.randn(3, 3)},
        }

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_multi_layer_attention(attention_dict)

        assert fig is not None

    def test_plot_multi_layer_attention_with_labels(self, tmp_path):
        """Test multi-layer attention with TF names."""
        np.random.seed(42)

        attention_dict = {
            "layer_0": {"between_features": np.random.randn(3, 3)},
            "layer_1": {"between_features": np.random.randn(3, 3)},
        }
        tf_names = ["TF1", "TF2", "TF3"]

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_multi_layer_attention(attention_dict, tf_names=tf_names)

        assert fig is not None

    def test_plot_multi_layer_attention_save_to_file(self, tmp_path):
        """Test multi-layer attention with file output."""
        np.random.seed(42)

        attention_dict = {
            "layer_0": {"between_features": np.random.randn(3, 3)},
        }

        visualizer = AttentionHeatmapVisualizer()
        output_path = tmp_path / "multi_layer.pdf"

        result = visualizer.plot_multi_layer_attention(
            attention_dict, output_path=output_path
        )

        assert result is None
        assert output_path.exists()

    def test_plot_multi_layer_attention_many_layers(self, tmp_path):
        """Test multi-layer attention with many layers."""
        np.random.seed(42)

        # Create 6 layers (should wrap to multiple rows)
        attention_dict = {
            f"layer_{i}": {"between_features": np.random.randn(3, 3)}
            for i in range(6)
        }

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_multi_layer_attention(attention_dict)

        assert fig is not None

    def test_plot_attention_heatmap_rectangular_matrix(self, tmp_path):
        """Test attention heatmap with non-square matrix."""
        np.random.seed(42)
        attention = np.random.randn(5, 10)

        visualizer = AttentionHeatmapVisualizer()
        fig = visualizer.plot_attention_heatmap(attention)

        assert fig is not None


class TestEdgeScoreVisualizer:
    """Test suite for EdgeScoreVisualizer class."""

    def test_init(self):
        """Test edge score visualizer initialization."""
        visualizer = EdgeScoreVisualizer(figsize=(10, 6), dpi=120)

        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 120

    def test_init_with_defaults(self):
        """Test edge score visualizer initialization with defaults."""
        visualizer = EdgeScoreVisualizer()

        assert visualizer.figsize == (12, 5)
        assert visualizer.dpi == 100

    def test_plot_score_distribution_from_list(self, tmp_path):
        """Test score distribution from list."""
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100).tolist()

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_score_distribution(scores)

        assert fig is not None

    def test_plot_score_distribution_from_dict(self, tmp_path):
        """Test score distribution from dict."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_score_distribution(edge_scores)

        assert fig is not None

    def test_plot_score_distribution_with_gold_standard(self, tmp_path):
        """Test score distribution with gold standard highlighting."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = {
            (f"TF{i}", f"GENE{j}")
            for i in range(5)
            for j in range(3)
        }

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_score_distribution(edge_scores, gold_standard=gold_standard)

        assert fig is not None

    def test_plot_score_distribution_save_to_file(self, tmp_path):
        """Test score distribution with file output."""
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 100).tolist()

        visualizer = EdgeScoreVisualizer()
        output_path = tmp_path / "distribution.pdf"

        result = visualizer.plot_score_distribution(scores, output_path=output_path)

        assert result is None
        assert output_path.exists()

    def test_plot_precision_recall_curve(self, tmp_path):
        """Test precision-recall curve plotting."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = {
            (f"TF{i}", f"GENE{j}")
            for i in range(5)
            for j in range(3)
        }

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_precision_recall_curve(edge_scores, gold_standard)

        assert fig is not None

    def test_plot_precision_recall_curve_save_to_file(self, tmp_path):
        """Test precision-recall curve with file output."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = {
            (f"TF{i}", f"GENE{j}")
            for i in range(5)
            for j in range(3)
        }

        visualizer = EdgeScoreVisualizer()
        output_path = tmp_path / "pr_curve.pdf"

        result = visualizer.plot_precision_recall_curve(
            edge_scores, gold_standard, output_path=output_path
        )

        assert result is None
        assert output_path.exists()

    def test_plot_roc_curve(self, tmp_path):
        """Test ROC curve plotting."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = {
            (f"TF{i}", f"GENE{j}")
            for i in range(5)
            for j in range(3)
        }

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_roc_curve(edge_scores, gold_standard)

        assert fig is not None

    def test_plot_roc_curve_save_to_file(self, tmp_path):
        """Test ROC curve with file output."""
        np.random.seed(42)
        edge_scores = {
            (f"TF{i}", f"GENE{j}"): np.random.random()
            for i in range(5)
            for j in range(10)
        }
        gold_standard = {
            (f"TF{i}", f"GENE{j}")
            for i in range(5)
            for j in range(3)
        }

        visualizer = EdgeScoreVisualizer()
        output_path = tmp_path / "roc_curve.pdf"

        result = visualizer.plot_roc_curve(
            edge_scores, gold_standard, output_path=output_path
        )

        assert result is None
        assert output_path.exists()

    def test_plot_roc_curve_perfect_predictions(self, tmp_path):
        """Test ROC curve with perfect predictions."""
        edge_scores = {
            ("TF1", "GENE1"): 0.9,
            ("TF2", "GENE2"): 0.8,
            ("TF3", "GENE3"): 0.7,
            ("TF1", "GENE4"): 0.1,
        }
        gold_standard = {
            ("TF1", "GENE1"),
            ("TF2", "GENE2"),
            ("TF3", "GENE3"),
        }

        visualizer = EdgeScoreVisualizer()
        fig = visualizer.plot_roc_curve(edge_scores, gold_standard)

        assert fig is not None

    def test_plot_score_distribution_empty_scores(self, tmp_path):
        """Test score distribution with empty scores."""
        edge_scores = {}

        visualizer = EdgeScoreVisualizer()
        # Should handle gracefully
        fig = visualizer.plot_score_distribution(edge_scores)

        assert fig is not None


class TestCreateEvaluationSummaryPlot:
    """Test suite for create_evaluation_summary_plot function."""

    def test_basic_summary_plot(self, tmp_path):
        """Test basic evaluation summary plot."""
        metrics = {
            "auroc": 0.85,
            "aupr": 0.65,
            "precision@100": 0.75,
            "recall@100": 0.55,
        }

        fig = create_evaluation_summary_plot(metrics)

        assert fig is not None

    def test_summary_plot_with_baseline(self, tmp_path):
        """Test summary plot with baseline comparison."""
        metrics = {
            "auroc": 0.85,
            "aupr": 0.65,
            "precision@100": 0.75,
        }
        baseline_metrics = {
            "auroc": 0.70,
            "aupr": 0.50,
            "precision@100": 0.60,
        }

        fig = create_evaluation_summary_plot(metrics, baseline_metrics=baseline_metrics)

        assert fig is not None

    def test_summary_plot_save_to_file(self, tmp_path):
        """Test summary plot with file output."""
        metrics = {"auroc": 0.85, "aupr": 0.65}
        output_path = tmp_path / "summary.pdf"

        result = create_evaluation_summary_plot(metrics, output_path=output_path)

        assert result is None
        assert output_path.exists()

    def test_summary_plot_with_precision_at_k(self, tmp_path):
        """Test summary plot with precision@k metrics."""
        metrics = {
            "auroc": 0.85,
            "aupr": 0.65,
            "precision@100": 0.75,
            "precision@500": 0.60,
            "recall@100": 0.55,
        }

        fig = create_evaluation_summary_plot(metrics)

        assert fig is not None

    def test_summary_plot_empty_metrics(self, tmp_path):
        """Test summary plot with empty metrics."""
        metrics = {}

        fig = create_evaluation_summary_plot(metrics)

        assert fig is not None

    def test_summary_plot_all_metrics_in_range(self, tmp_path):
        """Test summary plot ensures all metrics are in valid range."""
        metrics = {
            "auroc": 0.5,
            "aupr": 1.0,
            "precision@100": 0.0,
            "recall@100": 0.75,
        }

        fig = create_evaluation_summary_plot(metrics)

        assert fig is not None


class TestIntegrationVisualization:
    """Integration tests for visualization workflow."""

    def test_full_visualization_workflow(self, tmp_path):
        """Test complete visualization workflow from edge scores to plots."""
        try:
            import networkx as nx

            np.random.seed(42)

            # Create synthetic edge scores
            edge_scores = {
                (f"TF{i}", f"GENE{j}"): np.random.random()
                for i in range(5)
                for j in range(10)
            }

            # Create gold standard
            gold_standard = {
                (f"TF{i}", f"GENE{j}")
                for i in range(5)
                for j in range(3)
            }

            # Create network graph
            graph = nx.DiGraph()
            for (tf, target), score in edge_scores.items():
                if score > 0.7:
                    graph.add_edge(tf, target, weight=score)

            # Visualize network
            network_viz = GRNNetworkVisualizer()
            fig1 = network_viz.plot_network(graph)
            assert fig1 is not None

            # Visualize edge scores
            score_viz = EdgeScoreVisualizer()
            fig2 = score_viz.plot_score_distribution(edge_scores, gold_standard)
            assert fig2 is not None

            # Visualize PR curve
            fig3 = score_viz.plot_precision_recall_curve(edge_scores, gold_standard)
            assert fig3 is not None

            # Visualize ROC curve
            fig4 = score_viz.plot_roc_curve(edge_scores, gold_standard)
            assert fig4 is not None

            # Create evaluation summary
            from tabpfn.grn.evaluation import evaluate_grn
            metrics = evaluate_grn(edge_scores, gold_standard, k_values=[10, 20])
            fig5 = create_evaluation_summary_plot(metrics)
            assert fig5 is not None

        except ImportError:
            pytest.skip("networkx not installed")

    def test_attention_and_network_visualization(self, tmp_path):
        """Test visualization of attention patterns and resulting network."""
        try:
            import networkx as nx

            np.random.seed(42)

            # Create synthetic attention data
            attention_dict = {
                "layer_0": {"between_features": np.random.randn(5, 5)},
                "layer_1": {"between_features": np.random.randn(5, 5)},
            }
            tf_names = [f"TF{i}" for i in range(5)]

            # Visualize attention
            attn_viz = AttentionHeatmapVisualizer()
            fig1 = attn_viz.plot_multi_layer_attention(attention_dict, tf_names)
            assert fig1 is not None

            # Create corresponding network
            graph = nx.DiGraph()
            for i, tf in enumerate(tf_names):
                for j in range(3):
                    graph.add_edge(tf, f"GENE{j}", weight=np.random.random())

            # Visualize network
            network_viz = GRNNetworkVisualizer()
            fig2 = network_viz.plot_network(graph)
            assert fig2 is not None

        except ImportError:
            pytest.skip("networkx not installed")
