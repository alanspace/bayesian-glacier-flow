"""
Parameter Sampling for FEM Data Generation

This module implements smart parameter sampling strategies for generating
diverse training datasets from the FEM glacier solver.

Strategies:
- Latin Hypercube Sampling (LHS)
- Sobol sequences
- Random sampling with constraints

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional
import json


class ParameterSampler:
    """
    Sample glacier parameters for FEM simulations using various strategies.
    
    Parameters define glacier geometry and physics:
    - bed_elevation: Bedrock topography profile
    - surface_elevation: Ice surface profile
    - A: Flow parameter (Pa^-3 yr^-1)
    - epsilon: Regularization parameter
    - temperature: Ice temperature (affects A)
    """
    
    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        seed: int = 42
    ):
        """
        Args:
            param_ranges: Dict mapping parameter names to (min, max) tuples
            seed: Random seed for reproducibility
        """
        self.param_ranges = param_ranges
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def latin_hypercube(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Latin Hypercube Sampling for space-filling parameter coverage.
        
        Args:
            n_samples: Number of parameter sets to generate
            
        Returns:
            Dictionary mapping parameter names to arrays of sampled values
        """
        n_params = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=self.seed)
        
        # Sample unit hypercube
        unit_samples = sampler.random(n=n_samples)
        
        # Scale to parameter ranges
        samples = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val = self.param_ranges[param_name]
            samples[param_name] = qmc.scale(
                unit_samples[:, i:i+1],
                [min_val],
                [max_val]
            ).flatten()
            
        return samples
    
    def sobol_sequence(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Sobol quasi-random sequence for low-discrepancy sampling.
        
        Better coverage than pure random, good for integration/sensitivity analysis.
        
        Args:
            n_samples: Number of parameter sets
            
        Returns:
            Dictionary of sampled parameters
        """
        n_params = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=self.seed)
        
        # Generate samples (must be power of 2 for Sobol)
        n_sobol = int(2 ** np.ceil(np.log2(n_samples)))
        unit_samples = sampler.random(n=n_sobol)[:n_samples]
        
        # Scale to ranges
        samples = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val = self.param_ranges[param_name]
            samples[param_name] = min_val + (max_val - min_val) * unit_samples[:, i]
            
        return samples
    
    def random_uniform(self, n_samples: int) -> Dict[str, np.ndarray]:
        """
        Simple uniform random sampling within parameter ranges.
        
        Args:
            n_samples: Number of parameter sets
            
        Returns:
            Dictionary of sampled parameters
        """
        samples = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            samples[param_name] = self.rng.uniform(
                low=min_val,
                high=max_val,
                size=n_samples
            )
        return samples
    
    def glacier_geometry_samples(
        self,
        n_samples: int,
        x_coords: np.ndarray,
        method: str = "lhs"
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate diverse glacier geometries by varying bed elevation profiles.
        
        Uses parameterized functions for realistic topography:
        - Sinusoidal bed with varying amplitude/frequency
        - Linear slopes with varying gradients
        - Polynomial profiles
        
        Args:
            n_samples: Number of geometries to generate
            x_coords: Spatial coordinates for profile discretization
            method: Sampling method ('lhs', 'sobol', 'random')
            
        Returns:
            List of dicts, each containing 'bed', 'surface', and physics params
        """
        # Parameter ranges for geometry generation
        geometry_params = {
            'bed_amplitude': (50.0, 200.0),      # m
            'bed_frequency': (0.0001, 0.001),    # 1/m
            'bed_mean_slope': (-0.2, -0.05),     # dimensionless
            'ice_thickness_mean': (100.0, 300.0), # m
            'surface_slope': (0.02, 0.1),        # dimensionless
            'A': (5e-17, 2e-16),                 # Pa^-3 yr^-1
            'epsilon': (1e-11, 1e-9)             # dimensionless
        }
        
        # Sample parameters
        if method == "lhs":
            param_samples = ParameterSampler(
                geometry_params, seed=self.seed
            ).latin_hypercube(n_samples)
        elif method == "sobol":
            param_samples = ParameterSampler(
                geometry_params, seed=self.seed
            ).sobol_sequence(n_samples)
        else:
            param_samples = ParameterSampler(
                geometry_params, seed=self.seed
            ).random_uniform(n_samples)
        
        # Generate geometries
        geometries = []
        for i in range(n_samples):
            # Bed elevation: linear trend + sinusoidal variation
            bed = (
                param_samples['bed_mean_slope'][i] * x_coords +
                param_samples['bed_amplitude'][i] * 
                np.sin(param_samples['bed_frequency'][i] * x_coords)
            )
            
            # Surface: bed + ice thickness
            thickness = param_samples['ice_thickness_mean'][i]
            surface = bed + thickness + param_samples['surface_slope'][i] * x_coords
            
            geometries.append({
                'x': x_coords,
                'bed': bed,
                'surface': surface,
                'A': param_samples['A'][i],
                'epsilon': param_samples['epsilon'][i],
                'geometry_id': i
            })
            
        return geometries
    
    def save_samples(self, samples: Dict, filename: str):
        """Save sampled parameters to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_samples = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in samples.items()
        }
        
        with open(filename, 'w') as f:
            json.dump(json_samples, f, indent=2)
        
        print(f"Saved {len(next(iter(samples.values())))} samples to {filename}")
    
    @staticmethod
    def load_samples(filename: str) -> Dict[str, np.ndarray]:
        """Load sampled parameters from JSON file."""
        with open(filename, 'r') as f:
            json_samples = json.load(f)
        
        # Convert lists back to numpy arrays
        samples = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in json_samples.items()
        }
        
        return samples


if __name__ == "__main__":
    # Example usage
    print("Parameter Sampler Demo\n" + "="*50)
    
    # Define parameter ranges
    param_ranges = {
        'A': (5e-17, 2e-16),
        'epsilon': (1e-11, 1e-9),
        'surface_slope': (0.02, 0.1)
    }
    
    # Create sampler
    sampler = ParameterSampler(param_ranges, seed=42)
    
    # Generate LHS samples
    n_samples = 10
    lhs_samples = sampler.latin_hypercube(n_samples)
    
    print(f"\nLatin Hypercube Sampling ({n_samples} samples):")
    for param, values in lhs_samples.items():
        print(f"  {param}: {values[:3]} ... (showing first 3)")
    
    # Generate glacier geometries
    print(f"\nGenerating glacier geometries...")
    x_coords = np.linspace(0, 5000, 200)  # 5 km glacier
    geometries = sampler.glacier_geometry_samples(
        n_samples=5,
        x_coords=x_coords,
        method='lhs'
    )
    
    print(f"Generated {len(geometries)} glacier geometries")
    print(f"  Geometry 0: A={geometries[0]['A']:.2e}, "
          f"epsilon={geometries[0]['epsilon']:.2e}")
    
    # Save samples
    sampler.save_samples(lhs_samples, 'test_samples.json')
    
    print("\n✅ Parameter sampling module ready!")
