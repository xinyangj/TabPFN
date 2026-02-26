"""Wrapper classes for multi-target regression with automatic target exclusion.

This module provides wrapper classes for GRN baseline methods:
- SklearnForestWrapper - for GENIE3 (RandomForest) and GRNBoost2 (GradientBoosting)
- LinearRegressionWrapper - for Correlation and MutualInfo
- TabPFNWrapper - for TabPFN-based GRN inference

Each wrapper trains one model per target gene, excluding that target from
input features if it's also a TF (allow_overlap=True), preventing information leakage.

Architecture: Option 1 - Simplified Design
- NO base class inheritance (removed MultiTargetRegressorBase)
- Each wrapper class manages its own models directly
- Wrappers call sklearn's fit() and predict() directly
- Preprocessing is handled separately by prepare_target_features()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch  # For GPU memory management
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    from sklearn.linear_model import LinearRegression

from tabpfn.grn.utils import compute_target_feature_indices


def prepare_target_features(
    X: np.ndarray,
    y: np.ndarray,
    tf_names: list[str],
    target_genes: list[str],
) -> dict[int, tuple[np.ndarray, list[str]]]:
    """Prepare features for each target, excluding self if needed.

    This function SEPARATES preprocessing from model training.
    It computes which TFs to use for each target gene and prepares
    the corresponding feature matrices.

    Parameters
    ----------
    X : np.ndarray
        Full TF expression matrix (n_samples, n_TFs)
    y : np.ndarray
        Target expression matrix (n_samples, n_targets)
    tf_names : list[str]
        Transcription factor names (must match columns of X)
    target_genes : list[str]
        Target gene names (must match columns of y)

    Returns
    -------
    prepared_features : dict
        Dictionary mapping target_idx to (X_for_target, tf_names_for_target)
        - X_for_target: Feature matrix with target excluded (if it's a TF)
        - tf_names_for_target: List of TF names used for this target

    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    >>> y = np.array([[7], [8], [9]])
    >>> tf_names = ['TF1', 'TF2', 'TF3']
    >>> target_genes = ['TF1', 'TF2', 'GENE1']
    >>> features = prepare_target_features(X, y, tf_names, target_genes)
    >>> # For target TF1 (index 0), it will be excluded from its own features
    >>> X_0, tf_names_0 = features[0]
    >>> print(X_0.shape)  # (n_samples, 2) - TF1 excluded
    """
    prepared_features = {}

    for target_idx, target_name in enumerate(target_genes):
        # Get TF indices to use (exclude this target if it's also a TF)
        tf_indices, tf_names_for_target, _ = compute_target_feature_indices(
            tf_names=tf_names,
            target_name=target_name,
            warn_on_single=True,
        )

        # Subset X for this target
        X_for_target = X[:, tf_indices]
        prepared_features[target_idx] = (X_for_target, tf_names_for_target)

    return prepared_features


class SklearnForestWrapper:
    """Wrapper for sklearn forest-based methods (RandomForest, GradientBoosting).

    Used by GENIE3 (RandomForest) and GRNBoost2 (GradientBoosting).

    Each target gene gets its own sklearn model, with target exclusion
    handled during training (no information leakage).

    Parameters
    ----------
    estimator_class : type
        The sklearn forest class (e.g., RandomForestRegressor)
    estimator_kwargs : dict
        Additional keyword arguments to pass to estimator
    n_estimators : int, default=100
        Number of trees in the forest
    random_state : int, default=42
        Random state for reproducibility

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> wrapper = SklearnForestWrapper(
    ...     estimator_class=RandomForestRegressor,
    ...     estimator_kwargs={"max_features": "sqrt"},
    ...     n_estimators=100,
    ...     random_state=42
    ... )
    """

    def __init__(
        self,
        estimator_class: type,
        estimator_kwargs: dict,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.estimator_class = estimator_class
        self.estimator_kwargs = estimator_kwargs
        self.n_estimators = n_estimators
        self.random_state = random_state
        # Dictionary of fitted sklearn models, one per target
        self.models_: dict[int, Any] = {}
        # For each target, which TF indices were used (for predict)
        self.tf_indices_per_target_: dict[int, list[int]] = {}

    def _create_estimator(self) -> Any:
        """Create a forest estimator with specified kwargs and n_estimators."""
        return self.estimator_class(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            **self.estimator_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, tf_names: list[str] | None = None, target_genes: list[str] | None = None):
        """Train one sklearn forest model per target, excluding target from features.

        PREPROCESSING: Uses prepare_target_features() to select features for each target.
        TRAINING: Calls sklearn's fit() directly on each model.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_tfs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)
        tf_names : list[str], optional
            Names of TFs (columns of X)
        target_genes : list[str], optional
            Names of target genes (columns of y)

        Returns
        -------
        self
        """
        # Generate default names if not provided
        if tf_names is None:
            tf_names = [f"TF{i}" for i in range(X.shape[1])]
        if target_genes is None:
            target_genes = [f"Target{i}" for i in range(y.shape[1])]

        # Prepare features for each target (preprocessing step)
        prepared = prepare_target_features(
            X=X, y=y, tf_names=tf_names, target_genes=target_genes
        )

        # Train one sklearn model per target
        for target_idx, (X_for_target, _) in prepared.items():
            y_target = y[:, target_idx]

            # Create and fit sklearn model DIRECTLY
            model = self._create_estimator()
            model.fit(X_for_target, y_target)

            # Store the fitted model and feature indices
            self.models_[target_idx] = model

            # Store which TF indices were used (for predict)
            # Map feature names back to original TF indices
            tf_to_original_idx = {name: i for i, name in enumerate(tf_names)}
            tf_names_for_target = prepared[target_idx][1]
            tf_indices_original = [tf_to_original_idx[name] for name in tf_names_for_target]
            self.tf_indices_per_target_[target_idx] = tf_indices_original

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression for all targets using fitted sklearn models.

        Each target's model makes predictions using the features it was trained on.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_tfs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)

        Raises
        ------
        ValueError
            If models haven't been fitted yet
        """
        if not self.models_:
            raise ValueError("Models must be fitted before prediction")

        n_samples = X.shape[0]
        n_targets = len(self.models_)
        predictions = np.zeros((n_samples, n_targets))

        # Predict using each fitted model directly
        for target_idx, model in self.models_.items():
            # Get the TF indices that were used for this target during training
            tf_indices = self.tf_indices_per_target_[target_idx]
            # Subset X to the features that were used during training
            X_for_target = X[:, tf_indices]
            predictions[:, target_idx] = model.predict(X_for_target)

        return predictions

    def get_edge_scores(
        self,
        tf_names: list[str],
        target_genes: list[str],
    ) -> dict[tuple[str, str], float]:
        """Extract edge scores from fitted Random Forest/Gradient Boosting models.

        Uses sklearn's feature_importances_ as edge scores.

        Parameters
        ----------
        tf_names : list[str]
            Transcription factor names (columns of X during fit)
        target_genes : list[str]
            Target gene names (columns of y during fit)

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> importance score

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        from tabpfn.grn.utils import create_edge_score_dict

        if not self.models_:
            raise ValueError("Model must be fitted before extracting edge scores")

        # Build importance matrix (n_tfs, n_targets)
        n_tfs = len(tf_names)
        n_targets = len(target_genes)
        importance_matrix = np.zeros((n_tfs, n_targets))

        for target_idx, model in self.models_.items():
            # feature_importances_ has shape (n_features_for_target,)
            # Need to map back to original TF indices
            tf_indices = self.tf_indices_per_target_[target_idx]
            for i, tf_idx in enumerate(tf_indices):
                importance_matrix[tf_idx, target_idx] = model.feature_importances_[i]

        # Create edge score dictionary
        return create_edge_score_dict(
            tf_names=tf_names,
            target_genes=target_genes,
            scores=importance_matrix,
            skip_self_edges=True,
        )


class LinearRegressionWrapper:
    """Wrapper for linear regression methods (Correlation, MutualInfo).

    Each target gene gets its own sklearn LinearRegression model, with target
    exclusion handled during training (no information leakage).

    Stores coefficients in a full matrix for edge score extraction.

    Parameters
    ----------
    random_state : int, default=42
        Random state (passed for consistency, not used by LinearRegression)

    Attributes
    ----------
    coef_ : np.ndarray
        Coefficient matrix of shape (n_TFs, n_targets)
        where coef_[tf_idx, target_idx] is the coefficient for TF->target
        and coef_[i, j] = 0 if TF i was excluded when training target j
    intercept_ : np.ndarray
        Intercept vector of length n_targets

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> wrapper = LinearRegressionWrapper(random_state=42)
    >>> wrapper.fit(X, y, tf_names=tf_names, target_genes=target_genes)
    >>> edge_scores = wrapper.get_edge_scores(tf_names, target_genes)
    """

    def __init__(self, random_state: int = 42):
        # LinearRegression doesn't use n_estimators
        self.random_state = random_state
        from sklearn.linear_model import LinearRegression
        self._lr_class = LinearRegression
        # Coefficient matrices
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def _create_estimator(self) -> "LinearRegression":
        """Create a LinearRegression estimator."""
        return self._lr_class()

    def fit(self, X: np.ndarray, y: np.ndarray, tf_names: list[str] | None = None, target_genes: list[str] | None = None):
        """Train one LinearRegression model per target, excluding target from features.

        PREPROCESSING: Uses prepare_target_features() to select features for each target.
        TRAINING: Calls sklearn's fit() directly on each model.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)
        tf_names : list[str], optional
            Names of TFs (columns of X)
        target_genes : list[str], optional
            Names of target genes (columns of y)

        Returns
        -------
        self
        """
        # Generate default names if not provided
        if tf_names is None:
            tf_names = [f"TF{i}" for i in range(X.shape[1])]
        if target_genes is None:
            target_genes = [f"Target{i}" for i in range(y.shape[1])]

        # Prepare features for each target (preprocessing step)
        prepared = prepare_target_features(
            X=X, y=y, tf_names=tf_names, target_genes=target_genes
        )

        # Initialize coefficient matrices (store full size for edge extraction)
        n_tfs = X.shape[1]
        n_targets = y.shape[1]
        self.coef_ = np.zeros((n_tfs, n_targets))
        self.intercept_ = np.zeros(n_targets)

        # Train one LR model per target on pre-selected features (JUST TRAINING)
        for target_idx in range(n_targets):
            X_for_target, tf_names_for_target = prepared[target_idx]
            y_target = y[:, target_idx]

            # Map TF names back to original indices for coef_ matrix
            tf_to_original_idx = {name: i for i, name in enumerate(tf_names)}
            tf_indices_original = [tf_to_original_idx[name] for name in tf_names_for_target]

            # Fit LinearRegression DIRECTLY
            lr = self._lr_class()
            lr.fit(X_for_target, y_target)

            # Store intercept
            self.intercept_[target_idx] = lr.intercept_

            # Store coefficients in full matrix (zeros for excluded features)
            for i, tf_idx in enumerate(tf_indices_original):
                self.coef_[tf_idx, target_idx] = lr.coef_[i]

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression for all targets using fitted LinearRegression models.

        Each target's model makes predictions using the features it was trained on.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)

        Raises
        ------
        ValueError
            If models haven't been fitted yet
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Models must be fitted before prediction")

        n_samples = X.shape[0]
        n_targets = self.coef_.shape[1]
        predictions = np.zeros((n_samples, n_targets))

        # Predict using each fitted model directly
        for target_idx in range(n_targets):
            # Get the TF indices that were used for this target during training
            # For linear regression, all TFs are always used (no exclusion in predict)
            # So we can just use all TF indices
            tf_indices = list(range(X.shape[1]))
            # Subset X to all TF features
            X_for_target = X
            coef_for_target = self.coef_[:, target_idx]
            predictions[:, target_idx] = X_for_target @ coef_for_target + self.intercept_[target_idx]

        return predictions

    def get_edge_scores(
        self,
        tf_names: list[str],
        target_genes: list[str],
    ) -> dict[tuple[str, str], float]:
        """Extract edge scores from fitted Linear Regression models.

        Uses absolute coefficient values as edge scores.

        Parameters
        ----------
        tf_names : list[str]
            Transcription factor names (columns of X during fit)
        target_genes : list[str]
            Target gene names (columns of y during fit)

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> |coefficient|

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        from tabpfn.grn.utils import create_edge_score_dict

        if self.coef_ is None:
            raise ValueError("Model must be fitted before extracting edge scores")

        # Use absolute coefficients as edge scores
        # coef_ already has shape (n_TFs, n_targets)
        return create_edge_score_dict(
            tf_names=tf_names,
            target_genes=target_genes,
            scores=np.abs(self.coef_),
            skip_self_edges=True,
        )


class TabPFNWrapper:
    """Wrapper for TabPFN-based GRN inference.

    This class wraps TabPFNGRNRegressor to provide a consistent interface
    with other baseline methods (SklearnForestWrapper, LinearRegressionWrapper).

    Unlike TabPFNGRNRegressor which takes tf_names and target_genes in __init__,
    this wrapper follows the sklearn-style pattern where they are passed to fit().
    This allows it to be used interchangeably with other wrappers in
    GRNBaselineRunner.

    Parameters
    ----------
    n_estimators : int, default=4
        Number of TabPFN estimators per target
    attention_aggregation : str, default='mean'
        Method to aggregate attention across layers and heads
    edge_score_strategy : str, default='self_attention'
        Method to extract edge scores from attention patterns
    device : str, default='auto'
        Device to use for computation
    random_state : int, default=42
        Random state for reproducibility
    keep_model : bool, default=False
        If False (default), delete the TabPFN model after fitting to free GPU memory,
        keeping only attention weights for edge score extraction. If True, keep the
        model for expression prediction. Set to True only if you need predict().

    Examples
    --------
    >>> from tabpfn.grn.baseline_models import TabPFNWrapper
    >>> wrapper = TabPFNWrapper(
    ...     n_estimators=2,
    ...     edge_score_strategy='self_attention'
    ... )
    >>> wrapper.fit(X, y, tf_names, target_genes)
    >>> edge_scores = wrapper.get_edge_scores(tf_names, target_genes)
    """

    def __init__(
        self,
        n_estimators: int = 4,
        attention_aggregation: str = "mean",
        edge_score_strategy: str = "self_attention",
        device: str = "auto",
        random_state: int = 42,
        keep_model: bool = False,
        ig_n_folds: int = 1,
        ig_baseline: str = "zero",
        rise_n_masks: int = 500,
        rise_mask_prob: float = 0.5,
        rise_baseline: str = "zero",
        rise_n_folds: int = 1,
        shapley_n_permutations: int = 200,
        shapley_n_folds: int = 1,
        shapley_exact_threshold: int = 15,
        shapley_method: str = "auto",
    ):
        self.n_estimators = n_estimators
        self.attention_aggregation = attention_aggregation
        self.edge_score_strategy = edge_score_strategy
        self.device = device
        self.random_state = random_state
        self.keep_model = keep_model
        self.ig_n_folds = ig_n_folds
        self.ig_baseline = ig_baseline
        self.rise_n_masks = rise_n_masks
        self.rise_mask_prob = rise_mask_prob
        self.rise_baseline = rise_baseline
        self.rise_n_folds = rise_n_folds
        self.shapley_n_permutations = shapley_n_permutations
        self.shapley_n_folds = shapley_n_folds
        self.shapley_exact_threshold = shapley_exact_threshold
        self.shapley_method = shapley_method
        # Store individual regressors (one per target gene)
        self._regressors: dict[str, Any] = {}
        # Store fit parameters for prediction
        self._tf_names: list[str] | None = None
        self._target_genes: list[str] | None = None
        # Store X for prediction
        self._X: np.ndarray | None = None
        # Store prepared features for predict
        self._prepared_features: dict[int, tuple[np.ndarray, list[str]]] = {}
        # Shared model architecture for IG/RISE (loaded once, reused across targets)
        self._shared_model_arch = None
        self._shared_device = None
        self._shared_criterion = None

    def _ensure_shared_model_arch(self):
        """Load TabPFN model architecture once for reuse across targets.

        For IG strategy, we only need the pre-trained architecture (identical
        for all targets). Loading it once avoids redundant ``torch.load()``
        calls per target.
        """
        if self._shared_model_arch is not None:
            return

        import torch
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
        )
        # Fit with small random data to trigger weight loading and device placement
        rng = np.random.RandomState(0)
        dummy_X = rng.randn(20, 3).astype(np.float32)
        dummy_y = rng.randn(20).astype(np.float32)
        model.fit(dummy_X, dummy_y)
        self._shared_model_arch = model.models_[0]
        self._shared_criterion = model.znorm_space_bardist_
        if hasattr(model, "devices_") and model.devices_:
            self._shared_device = model.devices_[0]
        else:
            self._shared_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

    def fit_one_target_gene(
        self,
        target_idx: int,
        target_name: str,
        X_for_target: np.ndarray,
        y_target: np.ndarray,
        tf_names_for_target: list[str],
    ) -> "TabPFNWrapper":
        """Fit a TabPFN model for a single target gene.

        This method creates, fits, and optionally cleans up a model for one target.
        Use this in a loop for memory-efficient expression prediction.

        Parameters
        ----------
        target_idx : int
            Index of the target gene
        target_name : str
            Name of the target gene
        X_for_target : np.ndarray
            TF expression matrix for this target (n_samples, n_tfs_for_target)
        y_target : np.ndarray
            Target expression values (n_samples,)
        tf_names_for_target : list[str]
            TF names for this target (with target excluded if it's a TF)

        Returns
        -------
        self
            Returns self for method chaining
        """
        from tabpfn.grn.grn_regressor import TabPFNGRNRegressor

        # For IG, RISE, or Shapley, load model arch once and reuse across all targets
        shared_model_arch = None
        shared_device = None
        if self.edge_score_strategy in ("integrated_gradients", "rise", "shapley"):
            self._ensure_shared_model_arch()
            shared_model_arch = self._shared_model_arch
            shared_device = self._shared_device

        # Create a separate regressor for this single target
        single_target_regressor = TabPFNGRNRegressor(
            tf_names=tf_names_for_target,  # Target excluded from TFs
            target_genes=[target_name],
            n_estimators=self.n_estimators,
            attention_aggregation=self.attention_aggregation,
            edge_score_strategy=self.edge_score_strategy,
            device=self.device,
            random_state=self.random_state,
            ig_n_folds=self.ig_n_folds,
            ig_baseline=self.ig_baseline,
            rise_n_masks=self.rise_n_masks,
            rise_mask_prob=self.rise_mask_prob,
            rise_baseline=self.rise_baseline,
            rise_n_folds=self.rise_n_folds,
            shapley_n_permutations=self.shapley_n_permutations,
            shapley_n_folds=self.shapley_n_folds,
            shapley_exact_threshold=self.shapley_exact_threshold,
            shapley_method=self.shapley_method,
        )

        # Fit on the target-specific features
        single_target_regressor.fit(
            X_for_target, y_target.reshape(-1, 1),
            shared_model_arch=shared_model_arch,
            shared_device=shared_device,
            shared_criterion=self._shared_criterion,
        )

        # Clean up the heavy TabPFN model to free GPU memory
        # We only need the attention weights for edge score computation
        # unless keep_model=True for expression prediction
        if not self.keep_model:
            single_target_regressor.cleanup_model()

        # Store the regressor for this target
        self._regressors[target_name] = single_target_regressor

        return self

    def predict_one_target_gene(
        self,
        target_name: str,
        X: np.ndarray,
        tf_names_for_target: list[str] | None = None,
        cleanup_after: bool = False,
    ) -> np.ndarray:
        """Predict expression for a single target gene and optionally cleanup.

        This method predicts for one target and optionally deletes the model
        after prediction to free GPU memory. Use this in a loop for
        memory-efficient expression prediction.

        Parameters
        ----------
        target_name : str
            Name of the target gene to predict
        X : np.ndarray
            Full TF expression matrix (n_samples, n_tfs)
        tf_names_for_target : list[str], optional
            TF names that were used for this target during training.
            If None, uses stored prepared features.
        cleanup_after : bool, default=False
            If True, delete the model after prediction to free GPU memory.

        Returns
        -------
        predictions : np.ndarray
            Predicted expression values (n_samples,)

        Raises
        ------
        ValueError
            If target model has been cleaned up and cannot predict
        """
        import gc

        if target_name not in self._regressors:
            raise ValueError(f"No fitted model found for target: {target_name}")

        regressor = self._regressors[target_name]

        # Check if model was cleaned up
        if not regressor.target_models_:
            raise ValueError(
                f"Cannot predict for {target_name}: model was cleaned up. "
                "Use keep_model=True or fit the target again."
            )

        # Get the TF indices that were used for this target during training
        if tf_names_for_target is None:
            # Use stored prepared features
            target_idx = self._target_genes.index(target_name)
            tf_names_for_target = self._prepared_features[target_idx][1]

        # Map TF names back to column indices in X
        tf_to_idx = {name: i for i, name in enumerate(self._tf_names)}
        tf_indices = [tf_to_idx[name] for name in tf_names_for_target]

        # Subset X to the features that were used during training
        X_for_target = X[:, tf_indices]
        predictions = regressor.predict(X_for_target)

        # Clean up after prediction if requested
        if cleanup_after:
            regressor.cleanup_model()
            # CRITICAL: Delete the regressor from _regressors dict to free memory
            # The regressor contains attention_weights_, X_, y_ and other tensors
            # that accumulate in GPU memory even after model cleanup
            del self._regressors[target_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray, tf_names: list[str] | None = None,
            target_genes: list[str] | None = None) -> "TabPFNWrapper":
        """Train one TabPFN model per target, excluding target from features.

        This method uses prepare_target_features() to properly exclude each target
        gene from its own input features when fitting models, preventing information
        leakage (self-correlation = 1.0).

        Internally calls fit_one_target_gene() for each target.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_tfs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)
        tf_names : list[str], optional
            Names of TFs (columns of X)
        target_genes : list[str], optional
            Names of target genes (columns of y)

        Returns
        -------
        self
        """
        # Generate default names if not provided
        if tf_names is None:
            tf_names = [f"TF{i}" for i in range(X.shape[1])]
        if target_genes is None:
            target_genes = [f"Target{i}" for i in range(y.shape[1])]

        # Store for later use
        self._tf_names = tf_names
        self._target_genes = target_genes
        # Store X for prediction
        self._X = X

        # Prepare features for each target (preprocessing step)
        # This properly excludes each target from its own input features
        prepared = prepare_target_features(
            X=X, y=y, tf_names=tf_names, target_genes=target_genes
        )
        self._prepared_features = prepared

        # For each target, create and fit a separate regressor
        # Each regressor is trained with that target excluded from input features
        for target_idx, target_name in enumerate(target_genes):
            X_for_target, _ = prepared[target_idx]
            y_target = y[:, target_idx]
            tf_names_for_target = prepared[target_idx][1]

            # Call fit_one_target_gene for each target
            self.fit_one_target_gene(
                target_idx=target_idx,
                target_name=target_name,
                X_for_target=X_for_target,
                y_target=y_target,
                tf_names_for_target=tf_names_for_target,
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression for all targets using fitted TabPFN models.

        Each target's model makes predictions using the features it was trained on.
        Internally calls predict_one_target_gene() for each target.

        Parameters
        ----------
        X : np.ndarray
            Full TF expression matrix (n_samples, n_tfs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)

        Raises
        ------
        ValueError
            If models haven't been fitted yet
        """
        if not self._regressors:
            raise ValueError("Models must be fitted before prediction")

        # Check if models were cleaned up (keep_model=False)
        if not self.keep_model:
            # Check if any regressor has an empty target_models_
            first_regressor = next(iter(self._regressors.values()))
            if not first_regressor.target_models_:
                raise ValueError(
                    "Cannot predict: models were cleaned up to save GPU memory. "
                    "Initialize TabPFNWrapper with keep_model=True if you need predict()."
                )

        n_samples = X.shape[0]
        n_targets = len(self._regressors)
        predictions = np.zeros((n_samples, n_targets))

        # Predict using each fitted regressor
        # Uses predict_one_target_gene() for each target
        for target_idx, (target_name, _) in enumerate(self._regressors.items()):
            # Get TF names that were used for this target during training
            tf_names_for_target = self._prepared_features[target_idx][1]
            # Call predict_one_target_gene for this target
            predictions[:, target_idx] = self.predict_one_target_gene(
                target_name=target_name,
                X=X,
                tf_names_for_target=tf_names_for_target,
                cleanup_after=False,  # Don't cleanup here, models already kept
            )

        return predictions

    def get_edge_score_one_target_gene(
        self,
        target_name: str,
        edge_score_strategy: str | None = None,
    ) -> dict[tuple[str, str], float]:
        """Get edge scores for a single target gene before cleanup.

        This method extracts edge scores for a single target gene,
        which can be called before cleanup_model() to avoid OOM errors.
        It uses the regressor's existing get_edge_scores() method
        which handles all edge score strategies properly.

        Parameters
        ----------
        target_name : str
            Name of the target gene to get edge scores for
        edge_score_strategy : str, optional
            Edge score strategy to use. If None, uses the strategy from __init__.
            Options: 'self_attention', 'tf_to_target', 'target_to_tf',
            'sequential_rollout', 'integrated_gradients'

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target_name) -> importance score for this target only

        Raises
        ------
        ValueError
            If model for target_name hasn't been fitted yet

        Examples
        --------
        >>> from tabpfn.grn.baseline_models import TabPFNWrapper
        >>> wrapper = TabPFNWrapper(n_estimators=1)
        >>> wrapper.fit_one_target_gene(0, "G1", X_train, y_train[:, 0], tf_names)
        >>> # Get edge scores before cleanup
        >>> edge_scores_g1 = wrapper.get_edge_score_one_target_gene("G1")
        >>> # Now safe to cleanup
        >>> wrapper._regressors["G1"].cleanup_model()
        """
        if target_name not in self._regressors:
            raise ValueError(f"Model for {target_name} hasn't been fitted yet")

        # Get the regressor for this target
        regressor = self._regressors[target_name]

        # Use provided strategy or fall back to the default from __init__
        strategy = edge_score_strategy if edge_score_strategy is not None else self.edge_score_strategy

        # Filter edge scores for this target only from the regressor's scores
        all_edge_scores = regressor.get_edge_scores(edge_score_strategy=strategy)

        # Filter to only include edges for this target
        target_edge_scores = {
            (tf, tgt): score
            for (tf, tgt), score in all_edge_scores.items()
            if tgt == target_name
        }

        return target_edge_scores

    def get_edge_scores(
        self,
        tf_names: list[str],
        target_genes: list[str],
        edge_score_strategy: str | None = None,
    ) -> dict[tuple[str, str], float]:
        """Extract edge scores from fitted TabPFN models.

        Delegates to each individual regressor's get_edge_scores() method,
        which handles all edge score strategies properly.

        Parameters
        ----------
        tf_names : list[str]
            Transcription factor names (columns of X during fit)
            (Note: this parameter is kept for API compatibility but not used,
             as each regressor has its own TF names with target excluded)
        target_genes : list[str]
            Target gene names (columns of y during fit)
        edge_score_strategy : str, optional
            Edge score strategy to use. If None, uses the strategy from __init__.
            Options: 'self_attention', 'tf_to_target', 'target_to_tf',
            'sequential_rollout', 'integrated_gradients'
            This allows computing different edge score strategies without re-fitting.

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> importance score

        Raises
        ------
        ValueError
            If model has not been fitted yet
        """
        if not self._regressors:
            raise ValueError("Model must be fitted before extracting edge scores")

        # Use provided strategy or fall back to the default from __init__
        strategy = edge_score_strategy if edge_score_strategy is not None else self.edge_score_strategy

        # Collect edge scores from each individual regressor
        # Each regressor already has the correct TF names (with target excluded)
        # and pre-computed edge scores from fit()
        edge_scores = {}
        for target_name, regressor in self._regressors.items():
            # Call the regressor's get_edge_scores method with the strategy
            # If strategy differs from fit time, it will recompute from attention weights
            target_edge_scores = regressor.get_edge_scores(edge_score_strategy=strategy)
            edge_scores.update(target_edge_scores)

        return edge_scores
