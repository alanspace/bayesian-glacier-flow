"""
Data Generation Script for Bayesian Glacier Surrogate
======================================================

Generate training dataset by running FEM solver with varying parameters.
Optimized for rapid prototyping (100 samples in ~1 hour).

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import time

# Add existing FEM solver to path
fem_path = Path(__file__).parents[1] / "picard-fem-glacier-flow" / "src"
sys.path.insert(0, str(fem_path))

try:
    from nonlinear_solver import solve_nonlinear_glacier
    import helpfunctions
    FEM_AVAILABLE = True
except ImportError:
    print("Warning: Could not import FEM solver. Using dummy data.")
    FEM_AVAILABLE = False


from src.fem.solver import GlacierFEMSolver

def generate_single_sample(A: float, epsilon: float, mesh_path: str, solver: GlacierFEMSolver) -> tuple:
    """
    Run single FEM simulation with given parameters.
    
    Args:
        A: Flow parameter (Pa^-3 yr^-1)
        epsilon: Regularization parameter
        mesh_path: Path to glacier mesh
        solver: Initialized GlacierFEMSolver instance
        
    Returns:
        velocity: Velocity field array
        solve_time: Time taken for simulation
    """
    if not FEM_AVAILABLE:
        # Dummy data for testing
        return np.random.randn(1200) * 1e-3, 0.5
    
    # Run the actual nonlinear solve
    velocity, iterations, solve_time = solver.solve_nonlinear(
        A=A, 
        epsilon=epsilon,
        max_iter=50,
        tol=1e-6
    )
    
    return velocity, solve_time


def generate_dataset(
    n_samples: int = 100,
    save_path: str = "data/glacier_dataset.npz",
    seed: int = 42
):
    """
    Generate training dataset with varying parameters.
    
    Strategy:
    - Vary A (flow parameter): 5e-17 to 2e-16 Pa^-3 yr^-1
    - Vary epsilon (regularization): 1e-11 to 1e-9
    - Fixed geometry (Arolla glacier)
    
    Args:
        n_samples: Number of simulations to run
        save_path: Where to save dataset
        seed: Random seed for reproducibility
        
    Returns:
        X: Input parameters [n_samples, 2]
        Y: Velocity fields [n_samples, n_dofs]
    """
    np.random.seed(seed)
    
    # Parameter ranges
    A_min, A_max = 5e-17, 2e-16
    eps_min, eps_max = 1e-11, 1e-9
    
    # Sample parameters uniformly (could use LHS for better coverage)
    A_samples = np.random.uniform(A_min, A_max, n_samples)
    eps_samples = np.random.uniform(eps_min, eps_max, n_samples)
    
    X_samples = []
    Y_samples = []
    times = []
    
    print(f"🔬 Generating {n_samples} FEM simulations...")
    print(f"   A range: [{A_min:.2e}, {A_max:.2e}]")
    print(f"   ε range: [{eps_min:.2e}, {eps_max:.2e}]")
    print()
    
    # Mesh path
    mesh_path = str(
        Path(__file__).parents[1] / 
        "picard-fem-glacier-flow" / "data" / "arolla.xdmf"
    )

    # Initialize Solver once
    print(f"🏗️ Initializing FEM Solver with mesh: {mesh_path}")
    solver = GlacierFEMSolver(mesh_path)
    
    start_time = time.time()
    
    for i in tqdm(range(n_samples), desc="Simulations"):
        A = A_samples[i]
        epsilon = eps_samples[i]
        
        # Run FEM solver
        velocity, solve_time = generate_single_sample(A, epsilon, mesh_path, solver)
        
        # Store
        X_samples.append([A, epsilon])
        Y_samples.append(velocity)
        times.append(solve_time)
        
        # Progress update every 10 samples
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = np.mean(times)
            remaining = avg_time * (n_samples - i - 1)
            print(f"   Progress: {i+1}/{n_samples} | "
                  f"Avg time: {avg_time:.1f}s | "
                  f"Est. remaining: {remaining/60:.1f} min")
    
    # Convert to arrays
    X = np.array(X_samples, dtype=np.float32)
    Y = np.array(Y_samples, dtype=np.float32)
    
    total_time = time.time() - start_time
    
    # Save
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        save_path,
        X=X,
        Y=Y,
        metadata={
            'n_samples': n_samples,
            'A_range': (A_min, A_max),
            'eps_range': (eps_min, eps_max),
            'total_time_seconds': total_time,
            'avg_solve_time': np.mean(times)
        }
    )
    
    # Summary
    print("\n" + "="*60)
    print("✅ Dataset Generation Complete!")
    print("="*60)
    print(f"Samples generated: {n_samples}")
    print(f"Input shape (X):   {X.shape}")
    print(f"Output shape (Y):  {Y.shape}")
    print(f"Total time:        {total_time/60:.1f} minutes")
    print(f"Avg per sample:    {np.mean(times):.1f} seconds")
    print(f"Saved to:          {save_path}")
    print("="*60)
    
    return X, Y


def load_dataset(load_path: str = "data/glacier_dataset.npz"):
    """Load previously generated dataset."""
    data = np.load(load_path, allow_pickle=True)
    X = data['X']
    Y = data['Y']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    print(f"📊 Loaded dataset from {load_path}")
    print(f"   X shape: {X.shape}")
    print(f"   Y shape: {Y.shape}")
    if metadata:
        print(f"   Samples: {metadata.get('n_samples', 'unknown')}")
    
    return X, Y, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate glacier FEM dataset")
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of simulations (default: 100)')
    parser.add_argument('--output', type=str, default='data/glacier_dataset.npz',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Generate dataset
    X, Y = generate_dataset(
        n_samples=args.n_samples,
        save_path=args.output,
        seed=args.seed
    )
    
    # Quick validation
    print("\n📈 Dataset Statistics:")
    print(f"X (inputs):")
    print(f"  A - min: {X[:,0].min():.2e}, max: {X[:,0].max():.2e}, mean: {X[:,0].mean():.2e}")
    print(f"  ε - min: {X[:,1].min():.2e}, max: {X[:,1].max():.2e}, mean: {X[:,1].mean():.2e}")
    print(f"\nY (velocities):")
    print(f"  min: {Y.min():.2e}, max: {Y.max():.2e}, mean: {Y.mean():.2e}")
