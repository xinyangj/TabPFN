"""Visualization utilities for Gene Regulatory Network (GRN) inference.

This module provides visualization functionality for GRN analysis including:
- Network visualization using NetworkX and Matplotlib
- Attention heatmaps
- Edge score distributions
- Precision-recall curves
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    import pandas as pd

# Set style for better-looking plots
sns.set_style("whitegrid")


class GRNNetworkVisualizer:
    """Visualize gene regulatory networks inferred from TabPFN.

    Provides methods for:
    - Network visualization with node/edge styling
    - Subnetwork extraction for focused views
    - Community detection for functional modules

    Examples
    --------
    >>> from tabpfn.grn import GRNNetworkVisualizer
    >>> import networkx as nx
    >>> graph = nx.DiGraph()
    >>> graph.add_edge("TF1", "GENE1", weight=0.9)
    >>> visualizer = GRNNetworkVisualizer()
    >>> visualizer.plot_network(graph, "output/network.pdf")
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 10),
        dpi: int = 100,
    ) -> None:
        """Initialize the visualizer.

        Parameters
        ----------
        figsize : tuple, default=(12, 10)
            Figure size (width, height) in inches
        dpi : int, default=100
            Dots per inch for output resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_network(
        self,
        graph: "nx.DiGraph",
        output_path: str | Path | None = None,
        *,
        layout: str = "spring",
        max_nodes: int | None = None,
        edge_width_scale: float = 1.0,
        node_size_scale: float = 1.0,
        tf_color: str = "#FF6B6B",
        target_color: str = "#4ECDC4",
        title: str = "Inferred Gene Regulatory Network",
    ) -> plt.Figure | None:
        """Plot the GRN as a network graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph of regulatory edges with 'weight' attribute
        output_path : str or Path, optional
            Path to save the figure. If None, displays the figure.
        layout : str, default='spring'
            Network layout algorithm. Options: 'spring', 'circular',
            'kamada_kawai', 'spectral', 'shell'
        max_nodes : int, optional
            Maximum number of nodes to display (for large networks)
        edge_width_scale : float, default=1.0
            Scale factor for edge widths
        node_size_scale : float, default=1.0
            Scale factor for node sizes
        tf_color : str, default='#FF6B6B'
            Color for transcription factor nodes
        target_color : str, default='#4ECDC4'
            Color for target gene nodes
        title : str, default='Inferred Gene Regulatory Network'
            Title for the plot

        Returns
        -------
        fig : plt.Figure or None
            The figure object if not saved to file, None otherwise
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for network visualization. "
                "Install with: pip install networkx"
            )

        # Limit nodes if specified
        if max_nodes and graph.number_of_nodes() > max_nodes:
            # Get top edges by weight
            edges_with_weights = [
                (u, v, data.get("weight", 0))
                for u, v, data in graph.edges(data=True)
            ]
            edges_with_weights.sort(key=lambda x: x[2], reverse=True)

            # Keep top nodes
            top_edges = edges_with_weights[:max_nodes * 2]
            graph = nx.DiGraph()
            for u, v, w in top_edges:
                graph.add_edge(u, v, weight=w)

        # Get node types
        tf_set = set()
        target_set = set()
        for u, v, data in graph.edges(data=True):
            tf_set.add(u)
            target_set.add(v)

        # All nodes are either TFs or targets
        all_nodes = list(tf_set | target_set)

        # Create layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        elif layout == "shell":
            pos = nx.shell_layout(graph)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Draw nodes with different colors for TFs and targets
        tf_nodes = list(tf_set)
        target_nodes = list(target_set)

        # Node sizes based on degree
        in_degrees = dict(graph.in_degree())
        node_sizes = [in_degrees[n] * 200 * node_size_scale + 100 for n in all_nodes]

        # Create node to index mapping
        node_to_idx = {n: i for i, n in enumerate(all_nodes)}

        # Draw TF nodes
        if tf_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=tf_nodes,
                node_color=tf_color,
                node_size=[node_sizes[node_to_idx[n]] for n in tf_nodes],
                alpha=0.9,
                ax=ax,
                label="Transcription Factors",
            )

        # Draw target nodes
        if target_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=target_nodes,
                node_color=target_color,
                node_size=[node_sizes[node_to_idx[n]] for n in target_nodes],
                alpha=0.9,
                ax=ax,
                label="Target Genes",
            )

        # Draw edges
        edges = list(graph.edges(data=True))
        if edges:
            edge_weights = [d["weight"] for u, v, d in edges]
            # Normalize edge widths
            max_weight = max(edge_weights) if edge_weights else 1.0
            edge_widths = [w / max_weight * 3 * edge_width_scale for w in edge_weights]

            nx.draw_networkx_edges(
                graph,
                pos,
                width=edge_widths,
                alpha=0.6,
                edge_color="gray",
                ax=ax,
            )

        # Draw labels for important nodes only (high degree)
        degree_threshold = sorted(in_degrees.values(), reverse=True)[
            min(len(all_nodes) // 5, 10)
        ] if len(all_nodes) > 20 else 0
        high_degree_nodes = [n for n, d in in_degrees.items() if d >= degree_threshold]

        if high_degree_nodes:
            nx.draw_networkx_labels(
                graph,
                pos,
                labels={n: n for n in high_degree_nodes},
                font_size=8,
                ax=ax,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=tf_color, label="Transcription Factors"),
            Patch(facecolor=target_color, label="Target Genes"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig

    def plot_subnetwork(
        self,
        graph: "nx.DiGraph",
        focus_genes: list[str],
        output_path: str | Path | None = None,
        *,
        max_distance: int = 2,
    ) -> plt.Figure | None:
        """Plot a focused subnetwork around specified genes.

        Parameters
        ----------
        graph : nx.DiGraph
            Full GRN
        focus_genes : list of str
            Genes to focus on (extract subnetwork around these)
        output_path : str or Path, optional
            Path to save the figure
        max_distance : int, default=2
            Maximum distance from focus genes to include

        Returns
        -------
        fig : plt.Figure or None
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required")

        # Extract subnetwork
        subgraph_nodes = set()
        for gene in focus_genes:
            if gene in graph:
                # Get nodes within max_distance
                subgraph_nodes.add(gene)
                for dist in range(1, max_distance + 1):
                    current_level = set(subgraph_nodes)
                    for node in current_level:
                        if node in graph:
                            for neighbor in graph.neighbors(node):
                                subgraph_nodes.add(neighbor)

        subgraph = graph.subgraph(subgraph_nodes)

        return self.plot_network(
            subgraph,
            output_path,
            title=f"GRN Subnetwork around {', '.join(focus_genes[:3])}",
        )

    def plot_community_structure(
        self,
        graph: "nx.DiGraph",
        output_path: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot network with community detection highlighting functional modules.

        Parameters
        ----------
        graph : nx.DiGraph
            GRN to visualize
        output_path : str or Path, optional
            Path to save the figure

        Returns
        -------
        fig : plt.Figure or None
        """
        try:
            import networkx as nx
            from networkx.algorithms import community
        except ImportError:
            raise ImportError("networkx is required for community detection")

        # Detect communities
        communities = community.greedy_modularity_communities(
            graph.to_undirected()
        )

        # Create color map
        n_communities = len(communities)
        colors = plt.cm.Set3(np.linspace(0, 1, n_communities))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

        # Draw nodes colored by community
        for i, community in enumerate(communities):
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(community),
                node_color=[colors[i]],
                label=f"Community {i+1}",
                alpha=0.9,
                node_size=300,
                ax=ax,
            )

        # Draw edges
        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.3,
            edge_color="gray",
            ax=ax,
        )

        ax.set_title(
            f"GRN Community Structure ({n_communities} modules detected)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig


class AttentionHeatmapVisualizer:
    """Visualize attention patterns from TabPFN models.

    Creates heatmaps showing:
    - TF-TF attention (between-features)
    - Sample-sample attention (between-items)
    - Attention aggregation across layers/heads
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 100,
    ) -> None:
        """Initialize the attention visualizer.

        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            Figure size (width, height) in inches
        dpi : int, default=100
            Dots per inch for output resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        tf_names: list[str] | None = None,
        output_path: str | Path | None = None,
        *,
        title: str = "Attention Weights",
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> plt.Figure | None:
        """Plot attention weights as a heatmap.

        Parameters
        ----------
        attention_matrix : np.ndarray
            Attention weight matrix of shape (n_entities, n_entities)
        tf_names : list of str, optional
            Names of TFs for axis labels
        output_path : str or Path, optional
            Path to save the figure
        title : str, default='Attention Weights'
            Title for the heatmap
        cmap : str, default='viridis'
            Colormap name
        vmin : float, optional
            Minimum value for color scale
        vmax : float, optional
            Maximum value for color scale

        Returns
        -------
        fig : plt.Figure or None
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        im = ax.imshow(
            attention_matrix,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Target", fontsize=10)
        ax.set_ylabel("Source", fontsize=10)

        if tf_names is not None:
            # Show subset of labels if too many
            n_labels = min(len(tf_names), 20)
            step = max(1, len(tf_names) // n_labels)
            ax.set_xticks(range(0, len(tf_names), step))
            ax.set_yticks(range(0, len(tf_names), step))
            ax.set_xticklabels([tf_names[i] for i in range(0, len(tf_names), step)], rotation=90)
            ax.set_yticklabels([tf_names[i] for i in range(0, len(tf_names), step)])

        fig.colorbar(im, ax=ax, label="Attention Weight")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig

    def plot_multi_layer_attention(
        self,
        attention_dict: dict[str, dict[str, np.ndarray]],
        tf_names: list[str] | None = None,
        output_path: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot attention heatmaps for multiple layers side by side.

        Parameters
        ----------
        attention_dict : dict
            Nested dict {layer_name: {attn_type: attention_matrix}}
        tf_names : list of str, optional
            Names for axis labels
        output_path : str or Path, optional
            Path to save the figure

        Returns
        -------
        fig : plt.Figure or None
        """
        # Get between-features attention for each layer
        layers = sorted([k for k in attention_dict.keys() if k.startswith("layer_")])
        if not layers:
            layers = sorted(attention_dict.keys())

        n_layers = len(layers)
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 4, n_rows * 3.5),
            dpi=self.dpi,
        )
        axes = axes.flatten() if n_layers > 1 else [axes]

        for ax, layer_key in zip(axes, layers):
            layer_data = attention_dict[layer_key]
            attn = layer_data.get("between_features", None)

            if attn is not None and attn.ndim == 2:
                # Take mean across heads if needed
                if attn.shape[0] < attn.shape[1]:
                    attn = attn.T
                im = ax.imshow(attn, cmap="viridis", aspect="auto")
                ax.set_title(f"{layer_key.replace('_', ' ').title()}")
                fig.colorbar(im, ax=ax, label="Attention")

                if tf_names and len(tf_names) <= 20:
                    ax.set_xticks(range(len(tf_names)))
                    ax.set_yticks(range(len(tf_names)))
                    ax.set_xticklabels(tf_names, rotation=90)
                    ax.set_yticklabels(tf_names)

        # Hide unused subplots
        for ax in axes[n_layers:]:
            ax.axis("off")

        plt.suptitle("Multi-Layer Attention Patterns", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig


class EdgeScoreVisualizer:
    """Visualize edge score distributions and evaluation metrics."""

    def __init__(
        self,
        figsize: tuple[int, int] = (12, 5),
        dpi: int = 100,
    ) -> None:
        """Initialize the edge score visualizer.

        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size (width, height) in inches
        dpi : int, default=100
            Dots per inch for output resolution
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_score_distribution(
        self,
        edge_scores: dict[tuple[str, str], float] | list[float],
        gold_standard: set[tuple[str, str]] | None = None,
        output_path: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot distribution of edge scores.

        Parameters
        ----------
        edge_scores : dict or list
            Edge scores as dict of (tf, target) -> score or list of scores
        gold_standard : set, optional
            Set of true edges for highlighting
        output_path : str or Path, optional
            Path to save the figure

        Returns
        -------
        fig : plt.Figure or None
        """
        # Convert to list if dict
        if isinstance(edge_scores, dict):
            scores = list(edge_scores.values())
        else:
            scores = edge_scores

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot histogram
        ax.hist(scores, bins=50, alpha=0.7, color="skyblue", edgecolor="black")

        # Mark true edges if provided
        if gold_standard and isinstance(edge_scores, dict):
            true_scores = [
                s for (tf, tgt), s in edge_scores.items() if (tf, tgt) in gold_standard
            ]
            false_scores = [
                s for (tf, tgt), s in edge_scores.items() if (tf, tgt) not in gold_standard
            ]

            if true_scores:
                ax.hist(
                    true_scores,
                    bins=50,
                    alpha=0.5,
                    color="green",
                    label="True Edges",
                )
            if false_scores:
                ax.hist(
                    false_scores,
                    bins=50,
                    alpha=0.3,
                    color="red",
                    label="False Edges",
                )
            ax.legend()

        ax.set_xlabel("Edge Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title("Distribution of Predicted Edge Scores", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig

    def plot_precision_recall_curve(
        self,
        edge_scores: dict[tuple[str, str], float],
        gold_standard: set[tuple[str, str]],
        output_path: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot precision-recall curve for inferred GRN.

        Parameters
        ----------
        edge_scores : dict
            Dictionary of (tf, target) -> score
        gold_standard : set
            Set of true edges
        output_path : str or Path, optional
            Path to save the figure

        Returns
        -------
        fig : plt.Figure or None
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score

        # Create binary labels and scores
        all_pairs = list(edge_scores.keys())
        y_true = [1 if pair in gold_standard else 0 for pair in all_pairs]
        y_score = [edge_scores[pair] for pair in all_pairs]

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(recall, precision, linewidth=2, color="darkorange", label=f"AUPR = {aupr:.3f}")
        ax.fill_between(recall, precision, alpha=0.2, color="darkorange")

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.05])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig

    def plot_roc_curve(
        self,
        edge_scores: dict[tuple[str, str], float],
        gold_standard: set[tuple[str, str]],
        output_path: str | Path | None = None,
    ) -> plt.Figure | None:
        """Plot ROC curve for inferred GRN.

        Parameters
        ----------
        edge_scores : dict
            Dictionary of (tf, target) -> score
        gold_standard : set
            Set of true edges
        output_path : str or Path, optional
            Path to save the figure

        Returns
        -------
        fig : plt.Figure or None
        """
        from sklearn.metrics import roc_curve, auc

        # Create binary labels and scores
        all_pairs = list(edge_scores.keys())
        y_true = [1 if pair in gold_standard else 0 for pair in all_pairs]
        y_score = [edge_scores[pair] for pair in all_pairs]

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        ax.plot(fpr, tpr, linewidth=2, color="darkblue", label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Random (AUC = 0.5)")

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1.0])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)
            return None
        else:
            return fig


def create_evaluation_summary_plot(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure | None:
    """Create a summary bar chart comparing evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric names to values (e.g., {'aupr': 0.5, 'auroc': 0.7})
    baseline_metrics : dict, optional
        Baseline metrics to compare against
    output_path : str or Path, optional
        Path to save the figure

    Returns
    -------
    fig : plt.Figure or None
    """
    # Filter to only the main metrics for comparison
    main_metrics = {
        k: v for k, v in metrics.items()
        if "@" not in k or k in ["aupr", "auroc"]
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    x = list(main_metrics.keys())
    y = list(main_metrics.values())

    bars = ax.bar(x, y, color="steelblue", alpha=0.8)

    if baseline_metrics:
        baseline_y = [baseline_metrics.get(k, 0) for k in x]
        ax.plot(x, baseline_y, "o--", color="red", markersize=8, label="Baseline")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("GRN Inference Performance Metrics", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.legend()

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig
