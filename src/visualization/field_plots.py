"""
Advanced Field Visualization for Glacier Flow
==============================================

Professional 2D velocity field plotting using tricontourf,
adapted from glacier-dynamics-fem project.

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple
import os


def plot_velocity_field_2d(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    velocity: np.ndarray,
    title: str = "Glacier Velocity Field",
    xlabel: str = "Distance along glacier (m)",
    ylabel: str = "Elevation (m)",
    save_path: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (14, 6)
):
    """
    Create professional tricontourf plot of 2D velocity field.
    
    Args:
        x_coords: X coordinates of mesh points
        z_coords: Z coordinates of mesh points  
        velocity: Velocity values at each point
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (optional)
        cmap: Colormap name
        figsize: Figure size (width, height)
    """
    # Ensure 1D arrays
    x_coords = np.asarray(x_coords).flatten()
    z_coords = np.asarray(z_coords).flatten()
    velocity = np.asarray(velocity).flatten()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Tricontourf plot
    levels = 50
    tcf = ax.tricontourf(x_coords, z_coords, velocity, 
                         levels=levels, cmap=cmap)
    
    # Colorbar
    cbar = plt.colorbar(tcf, ax=ax, label='Horizontal Velocity (m/s)')
    cbar.ax.tick_params(labelsize=11)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=13, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='medium')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add statistics annotation
    v_max = velocity.max()
    v_mean = velocity.mean()
    v_min = velocity.min()
    stats_text = f"Max: {v_max:.2e} m/s\\nMean: {v_mean:.2e} m/s"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved velocity field to {save_path}")
    
    return fig, ax


def plot_fem_bnn_comparison(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    fem_velocity: np.ndarray,
    bnn_velocity: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Side-by-side comparison of FEM and BNN velocity fields.
    
    Args:
        x_coords: X coordinates
        z_coords: Z coordinates
        fem_velocity: FEM ground truth velocities
        bnn_velocity: BNN predicted velocities
        uncertainty: Optional uncertainty estimates
        save_path: Path to save figure
    """
    # Flatten arrays
    x = np.asarray(x_coords).flatten()
    z = np.asarray(z_coords).flatten()
    v_fem = np.asarray(fem_velocity).flatten()
    v_bnn = np.asarray(bnn_velocity).flatten()
    
    # Compute error
    error = np.abs(v_fem - v_bnn)
    relative_error = error / (np.abs(v_fem) + 1e-10) * 100  # Percentage
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Common colormap limits
    v_max = max(v_fem.max(), v_bnn.max())
    v_min = min(v_fem.min(), v_bnn.min())
    
    # Plot 1: FEM Ground Truth
    ax1 = axes[0, 0]
    tcf1 = ax1.tricontourf(x, z, v_fem, levels=50, cmap='viridis',
                           vmin=v_min, vmax=v_max)
    plt.colorbar(tcf1, ax=ax1, label='Velocity (m/s)')
    ax1.set_title('FEM Ground Truth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_aspect('equal')
    
    # Plot 2: BNN Prediction
    ax2 = axes[0, 1]
    tcf2 = ax2.tricontourf(x, z, v_bnn, levels=50, cmap='viridis',
                           vmin=v_min, vmax=v_max)
    plt.colorbar(tcf2, ax=ax2, label='Velocity (m/s)')
    ax2.set_title('Bayesian NN Prediction', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_aspect('equal')
    
    # Plot 3: Absolute Error
    ax3 = axes[1, 0]
    tcf3 = ax3.tricontourf(x, z, error, levels=50, cmap='Reds')
    plt.colorbar(tcf3, ax=ax3, label='Absolute Error (m/s)')
    ax3.set_title(f'Absolute Error (RMSE: {np.sqrt(np.mean(error**2)):.2e})', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Elevation (m)')
    ax3.set_aspect('equal')
    
    # Plot 4: Relative Error % or Uncertainty
    ax4 = axes[1, 1]
    if uncertainty is not None:
        unc = np.asarray(uncertainty).flatten()
        tcf4 = ax4.tricontourf(x, z, unc, levels=50, cmap='plasma')
        plt.colorbar(tcf4, ax=ax4, label='Uncertainty σ (m/s)')
        ax4.set_title('Predictive Uncertainty (MC Dropout)', 
                      fontsize=14, fontweight='bold')
    else:
        tcf4 = ax4.tricontourf(x, z, relative_error, levels=50, cmap='RdYlGn_r')
        plt.colorbar(tcf4, ax=ax4, label='Relative Error (%)')
        ax4.set_title(f'Relative Error (Mean: {relative_error.mean():.2f}%)', 
                      fontsize=14, fontweight='bold')
    ax4.set_xlabel('Distance (m)')
    ax4.set_ylabel('Elevation (m)')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot to {save_path}")
    
    return fig, axes


def plot_vertical_profile(
    z_coords: np.ndarray,
    fem_velocity: np.ndarray,
    bnn_velocity: np.ndarray,
    uncertainty: Optional[np.ndarray] = None,
    x_location: float = 2500.0,
    save_path: Optional[str] = None
):
    """
    Plot vertical velocity profile at a given x-location.
    
    Args:
        z_coords: Vertical coordinates
        fem_velocity: FEM velocities
        bnn_velocity: BNN velocities
        uncertainty: Optional uncertainty
        x_location: X coordinate of profile
        save_path: Path to save figure
    """
    # Sort by elevation
    idx = np.argsort(z_coords)
    z = z_coords[idx]
    v_fem = fem_velocity[idx]
    v_bnn = bnn_velocity[idx]
    
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # Plot FEM
    ax.plot(v_fem, z, 'r-', linewidth=2.5, label='FEM Ground Truth', marker='o', 
            markersize=4, alpha=0.7)
    
    # Plot BNN with uncertainty
    if uncertainty is not None:
        unc = uncertainty[idx]
        ax.plot(v_bnn, z, 'b-', linewidth=2.5, label='BNN Prediction', marker='s',
                markersize=4, alpha=0.7)
        ax.fill_betweenx(z, v_bnn - 2*unc, v_bnn + 2*unc, 
                         alpha=0.3, color='blue', label='95% CI')
    else:
        ax.plot(v_bnn, z, 'b--', linewidth=2.5, label='BNN Prediction', marker='s',
                markersize=4, alpha=0.7)
    
    # Styling
    ax.set_xlabel('Horizontal Velocity (m/s)', fontsize=13, fontweight='medium')
    ax.set_ylabel('Elevation (m)', fontsize=13, fontweight='medium')
    ax.set_title(f'Vertical Profile at x = {x_location:.0f} m', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add RMSE annotation
    rmse = np.sqrt(np.mean((v_fem - v_bnn)**2))
    mae = np.mean(np.abs(v_fem - v_bnn))
    stats_text = f"RMSE: {rmse:.2e} m/s\\nMAE: {mae:.2e} m/s"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved vertical profile to {save_path}")
    
    return fig, ax


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing field_plots module...")
    
    # Create synthetic glacier-like mesh
    x = np.linspace(0, 5000, 200)
    z = np.linspace(0, 500, 40)
    X, Z = np.meshgrid(x, z)
    
    # Synthetic velocity (parabolic profile)
    V = 0.001 * (1 - (Z / 500)**2) * (X / 5000)
    
    # Flatten
    x_flat = X.flatten()
    z_flat = Z.flatten()
    v_flat = V.flatten()
    
    # Test plot
    plot_velocity_field_2d(
        x_flat, z_flat, v_flat,
        title="Demo Velocity Field",
        save_path="plots/demo_field.png"
    )
    
    print("✅ Demo complete!")
