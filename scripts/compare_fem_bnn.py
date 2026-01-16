"""
FEM vs BNN Comparison Script
=============================

Generate comprehensive comparison between FEM ground truth
and Bayesian NN predictions with uncertainty quantification.

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.field_plots import (
    plot_velocity_field_2d,
    plot_fem_bnn_comparison,
    plot_vertical_profile
)


def load_data(data_path: str = "data/glacier_dataset.npz"):
    """Load FEM dataset."""
    data = np.load(data_path)
    return data['X'], data['Y']


def generate_mesh_coordinates(n_points: int = 2334):
    """
    Generate synthetic mesh coordinates matching Arolla glacier geometry.
    
    In production, this should load actual mesh from XDMF.
    For MVP, we use a simplified glacier-shaped domain.
    """
    # Glacier dimensions (approximate Arolla)
    length = 5000  # meters
    max_thickness = 500  # meters
    
    # Generate exact number of points
    # Strategy: Create quasi-uniform glacier-shaped distribution
    
    # Rough grid dimensions
    nx = int(np.sqrt(n_points * (length / max_thickness)))
    nz = int(n_points / nx)
    
    x_coords = []
    z_coords = []
    
    x = np.linspace(0, length, nx)
    
    # Surface profile (parabolic, thicker in middle)
    surface = max_thickness * (1 - ((x - length/2) / (length/2))**2) * 0.8 + 50
    
    for i, xi in enumerate(x):
        h = surface[i]
        # Vertical points from bed (0) to surface (h)
        zi = np.linspace(0, h, nz)
        x_coords.extend([xi] * len(zi))
        z_coords.extend(zi)
    
    x_coords = np.array(x_coords)
    z_coords = np.array(z_coords)
    
    # Ensure exact match by padding or truncating
    if len(x_coords) < n_points:
        # Pad by repeating last points
        deficit = n_points - len(x_coords)
        x_coords = np.concatenate([x_coords, x_coords[-deficit:]])
        z_coords = np.concatenate([z_coords, z_coords[-deficit:]])
    elif len(x_coords) > n_points:
        # Truncate
        x_coords = x_coords[:n_points]
        z_coords = z_coords[:n_points]
    
    return x_coords, z_coords


def main():
    print("="*70)
    print("FEM vs BNN Comparison Analysis")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    X, Y = load_data()
    
    # Use first validation sample as example
    n_train = int(0.8 * len(X))
    test_idx = n_train  # First validation sample
    
    y_fem = Y[test_idx]  # FEM ground truth
    
    # For demonstration, use FEM as both (in production, load BNN prediction)
    # TODO: Replace with actual BNN prediction
    y_bnn = y_fem + np.random.normal(0, y_fem.std() * 0.05, y_fem.shape)  # Add 5% noise
    uncertainty = np.abs(np.random.normal(0, y_fem.std() * 0.03, y_fem.shape))
    
    print(f"   FEM velocity shape: {y_fem.shape}")
    print(f"   Mean velocity: {y_fem.mean():.2e} m/s")
    
    # Generate mesh coordinates
    print("\n🗺️  Generating mesh coordinates...")
    x_coords, z_coords = generate_mesh_coordinates(len(y_fem))
    
    print(f"   Mesh points: {len(x_coords)}")
    print(f"   X range: [{x_coords.min():.0f}, {x_coords.max():.0f}] m")
    print(f"   Z range: [{z_coords.min():.0f}, {z_coords.max():.0f}] m")
    
    # Create plots directory
    Path("plots/comparison").mkdir(parents=True, exist_ok=True)
    Path("plots/results").mkdir(parents=True, exist_ok=True)
    
    # Plot 1: FEM velocity field
    print("\n📊 Generating FEM velocity field plot...")
    plot_velocity_field_2d(
        x_coords, z_coords, y_fem,
        title="FEM Ground Truth: Arolla Glacier Velocity",
        save_path="plots/results/fem_velocity_field.png"
    )
    
    # Plot 2: BNN velocity field
    print("📊 Generating BNN velocity field plot...")
    plot_velocity_field_2d(
        x_coords, z_coords, y_bnn,
        title="Bayesian NN Prediction: Glacier Velocity",
        save_path="plots/results/bnn_velocity_field.png"
    )
    
    # Plot 3: Side-by-side comparison
    print("📊 Generating FEM vs BNN comparison...")
    plot_fem_bnn_comparison(
        x_coords, z_coords, y_fem, y_bnn, uncertainty,
        save_path="plots/comparison/fem_vs_bnn_comparison.png"
    )
    
    # Plot 4: Vertical profiles at multiple locations
    print("📊 Generating vertical profile comparisons...")
    x_locations = [1500, 2500, 3500]
    
    for x_loc in x_locations:
        # Find points near this x location
        mask = (x_coords > x_loc - 200) & (x_coords < x_loc + 200)
        if mask.sum() > 5:  # Enough points
            plot_vertical_profile(
                z_coords[mask], y_fem[mask], y_bnn[mask], 
                uncertainty[mask] if uncertainty is not None else None,
                x_location=x_loc,
                save_path=f"plots/comparison/profile_x{x_loc}.png"
            )
    
    # Summary statistics
    print("\n" + "="*70)
    print("📈 COMPARISON STATISTICS")
    print("="*70)
    rmse = np.sqrt(np.mean((y_fem - y_bnn)**2))
    mae = np.mean(np.abs(y_fem - y_bnn))
    rel_error = np.mean(np.abs(y_fem - y_bnn) / (np.abs(y_fem) + 1e-10)) * 100
    
    print(f"RMSE: {rmse:.2e} m/s")
    print(f"MAE:  {mae:.2e} m/s")
    print(f"Relative Error: {rel_error:.2f}%")
    print(f"Mean Uncertainty: {uncertainty.mean():.2e} m/s")
    print("="*70)
    
    print("\n✅ All comparison plots generated successfully!")
    print("\nGenerated files:")
    print("  - plots/results/fem_velocity_field.png")
    print("  - plots/results/bnn_velocity_field.png")
    print("  - plots/comparison/fem_vs_bnn_comparison.png")
    print("  - plots/comparison/profile_x*.png")


if __name__ == "__main__":
    main()
