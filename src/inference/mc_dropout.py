"""
MC Dropout Inference for Uncertainty Quantification
====================================================

Implements Monte Carlo Dropout for predicting with uncertainty estimates.

The key insight: Run predictions multiple times with dropout ENABLED
to sample from the approximate posterior distribution.

Author: Shek Lun Leung  
Date: January 2026
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def predict_with_uncertainty(
    model,
    params,
    x_input: jnp.ndarray,
    num_samples: int = 100,
    seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Monte Carlo Dropout prediction with uncertainty quantification.
    
    Standard approach: dropout OFF during test → deterministic prediction
    MC Dropout: dropout ON during test → stochastic predictions → uncertainty!
    
    Args:
        model: Flax BayesianMLP model
        params: Trained model parameters
        x_input: Input parameters [batch, input_dim] or [input_dim]
        num_samples: Number of MC forward passes (more = better uncertainty estimate)
        seed: Random seed for reproducibility
    
    Returns:
        mean: Mean prediction [batch, output_dim] or [output_dim]
        std: Standard deviation (uncertainty) [batch, output_dim] or [output_dim]
        samples: All MC samples [num_samples, batch, output_dim] or [num_samples, output_dim]
    """
    # Ensure batch dimension
    single_input = x_input.ndim == 1
    if single_input:
        x_input = x_input[None, :]  # [1, input_dim]
    
    # Generate multiple stochastic predictions
    predictions = []
    
    key = jax.random.PRNGKey(seed)
    for i in range(num_samples):
        key, subkey = jax.random.split(key)
        
        # CRITICAL: training=True keeps dropout active during inference!
        pred = model.apply(
            params,
            x_input,
            training=True,  # <-- This is the magic!
            rngs={'dropout': subkey}
        )
        predictions.append(pred)
    
    # Stack: [num_samples, batch, output_dim]
    predictions = jnp.stack(predictions, axis=0)
    
    # Compute uncertainty statistics
    mean = jnp.mean(predictions, axis=0)  # [batch, output_dim]
    std = jnp.std(predictions, axis=0)    # [batch, output_dim]
    
    # Remove batch dimension if input was single sample
    if single_input:
        mean = mean[0]
        std = std[0]
        predictions = predictions[:, 0, :]
    
    return mean, std, predictions


def calibration_plot(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    confidence_levels: np.ndarray = np.array([0.68, 0.95, 0.99]),
    save_path: Optional[str] = None
):
    """
    Plot calibration curve: expected vs observed coverage.
    
    A well-calibrated model should have:
    - 68% of true values within ±1σ
    - 95% of true values within ±1.96σ
    - 99% of true values within ±2.58σ
    
    Args:
        y_true: Ground truth values [n_samples, output_dim]
        y_pred_mean: Predicted means [n_samples, output_dim]
        y_pred_std: Predicted std devs [n_samples, output_dim]
        confidence_levels: Confidence levels to evaluate
        save_path: Where to save plot (optional)
    """
    # Compute z-scores: how many std devs away is truth from prediction?
    z_scores = np.abs((y_true - y_pred_mean) / (y_pred_std + 1e-8))
    z_scores_flat = z_scores.flatten()
    
    # For each confidence level, compute observed coverage
    expected_coverage = []
    observed_coverage = []
    
    for conf in confidence_levels:
        # Expected: e.g., 95% should be within ±1.96σ
        expected_coverage.append(conf)
        
        # Corresponding z-score threshold
        from scipy.stats import norm
        z_threshold = norm.ppf((1 + conf) / 2)
        
        # Observed: fraction of samples within threshold
        within_bounds = (z_scores_flat <= z_threshold).mean()
        observed_coverage.append(within_bounds)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    plt.plot(expected_coverage, observed_coverage, 'bo-', 
             label='Model', markersize=10, linewidth=2)
    
    plt.xlabel('Expected Coverage', fontsize=14)
    plt.ylabel('Observed Coverage', fontsize=14)
    plt.title('Calibration Curve', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Annotate points
    for exp, obs in zip(expected_coverage, observed_coverage):
        plt.annotate(f'{exp:.0%}', 
                    xy=(exp, obs), 
                    xytext=(10, -10),
                    textcoords='offset points',
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved calibration plot to {save_path}")
    
    plt.show()
    
    # Print calibration metrics
    print("\n📊 Calibration Metrics:")
    for exp, obs in zip(expected_coverage, observed_coverage):
        error = abs(obs - exp)
        print(f"   {exp:.0%} CI: Expected={exp:.3f}, Observed={obs:.3f}, Error={error:.3f}")


def plot_uncertainty_1d(
    x_coords: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    title: str = "Glacier Flow with Uncertainty",
    xlabel: str = "Position (m)",
    ylabel: str = "Velocity (m/s)",
    save_path: Optional[str] = None
):
    """
    Plot 1D predictions with uncertainty bands.
    
    Args:
        x_coords: Spatial coordinates [N]
        mean_values: Predicted mean [N]
        std_values: Predicted std [N]
        y_true: Ground truth (optional) [N]
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Where to save plot
    """
    plt.figure(figsize=(14, 6))
    
    # Ensure 1D arrays (flatten if needed)
    mean_values = np.asarray(mean_values).flatten()
    std_values = np.asarray(std_values).flatten()
    if y_true is not None:
        y_true = np.asarray(y_true).flatten()
    
    # Mean prediction
    plt.plot(x_coords, mean_values, 'b-', linewidth=2.5, label='Mean Prediction')
    
    # Confidence intervals
    ci_95_lower = (mean_values - 1.96 * std_values).flatten()
    ci_95_upper = (mean_values + 1.96 * std_values).flatten()
    ci_68_lower = (mean_values - 1.0 * std_values).flatten()
    ci_68_upper = (mean_values + 1.0 * std_values).flatten()
    
    plt.fill_between(x_coords, ci_95_lower, ci_95_upper,
                     alpha=0.2, color='blue', label='95% CI (±1.96σ)')
    plt.fill_between(x_coords, ci_68_lower, ci_68_upper,
                     alpha=0.3, color='blue', label='68% CI (±1σ)')
    
    # Ground truth
    if y_true is not None:
        plt.plot(x_coords, y_true, 'r--', linewidth=2, label='FEM Ground Truth', alpha=0.7)
    
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved uncertainty plot to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Testing MC Dropout Inference\n" + "="*60)
    
    # Create dummy model and data
    from src.models.bayesian_mlp import create_model
    
    input_dim = 2
    output_dim = 100
    model, params = create_model(input_dim, output_dim)
    
    # Test input
    x_test = jnp.array([[1.0e-16, 5e-10]])  # Single input
    
    # MC Dropout prediction
    print("\n🎲 Running MC Dropout (100 samples)...")
    mean, std, samples = predict_with_uncertainty(
        model, params, x_test, num_samples=100
    )
    
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std shape: {std.shape}")
    print(f"   Samples shape: {samples.shape}")
    print(f"\n   Mean velocity range: [{mean.min():.2e}, {mean.max():.2e}]")
    print(f"   Uncertainty range: [{std.min():.2e}, {std.max():.2e}]")
    
    # Visualize
    x_coords = np.linspace(0, 5000, output_dim)
    plot_uncertainty_1d(
        x_coords, mean, std,
        title="MC Dropout Uncertainty Quantification (Demo)"
    )
    
    print("\n✅ MC Dropout inference module ready!")
