"""
Complete MVP Workflow: Bayesian Glacier Surrogate
==================================================

End-to-end pipeline from data generation to uncertainty visualization.

Usage:
    python run_mvp.py --generate_data  # Generate 100 FEM samples
    python run_mvp.py --train          # Train model
    python run_mvp.py --predict        # Predict with uncertainty

Author: Shek Lun Leung
Date: January 2026
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Add src to path so we can import modules from src/
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    parser = argparse.ArgumentParser(description="Bayesian Glacier Surrogate MVP")
    parser.add_argument('--generate_data', action='store_true',
                       help='Generate training data (100 FEM sims)')
    parser.add_argument('--train', action='store_true',
                       help='Train Bayesian MLP')
    parser.add_argument('--predict', action='store_true',
                       help='Predict with uncertainty')
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of FEM samples to generate')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--mc_samples', type=int, default=100,
                       help='MC Dropout samples for uncertainty')
    
    args = parser.parse_args()
    
    # If no args, run all
    if not any([args.generate_data, args.train, args.predict]):
        args.all = True
    
    print("="*70)
    print("🚀 Bayesian Glacier Surrogate - MVP Workflow")
    print("="*70)
    
    data_path = Path("data/glacier_dataset.npz")
    model_path = Path("models/trained_bayesian_mlp.pkl")
    
    # ========== Step 1: Data Generation ==========
    if args.generate_data or args.all:
        print("\n📊 STEP 1: Generating Training Data")
        print("-" * 70)
        
        # Lazy import to avoid needing FEM dependencies in JAX env
        try:
            from generate_training_data import generate_dataset
        except ImportError as e:
            print(f"❌ Could not import generate_dataset: {e}")
            print("   (This step requires the 'glacier-fem' environment)")
            return
        
        X, Y = generate_dataset(
            n_samples=args.n_samples,
            save_path=str(data_path)
        )
        
        print(f"✅ Generated {len(X)} samples")
    
    # Load data
    if not data_path.exists():
        print(f"\n❌ Error: No data found at {data_path}")
        print("   Run with --generate_data first")
        return
    
    print(f"\n📂 Loading data from {data_path}...")
    data = np.load(data_path)
    X, Y = data['X'], data['Y']
    print(f"   X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Train/val split
    n_total = len(X)
    n_train = int(0.8 * n_total)
    
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    
    print(f"   Train: {n_train}, Val: {n_total - n_train}")
    
    # ========== Step 2: Training ==========
    if args.train or args.all:
        print("\n🧠 STEP 2: Training Bayesian MLP")
        print("-" * 70)
        
        # Lazy imports for JAX/Flax
        try:
            import jax.numpy as jnp
            from models.bayesian_mlp import create_model, train_model
        except ImportError as e:
            print(f"❌ Could not import JAX/Flax models: {e}")
            print("   (This step requires the 'bayesian-glacier' environment)")
            return
        # Create model
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
        
        model, _ = create_model(input_dim, output_dim)
        
        # Train
        params, train_losses, val_losses = train_model(
            model,
            jnp.array(X_train),
            jnp.array(Y_train),
            jnp.array(X_val),
            jnp.array(Y_val),
            epochs=args.epochs,
            batch_size=32,
            learning_rate=1e-3
        )
        
        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump({'params': params, 'model_config': {
                'input_dim': input_dim,
                'output_dim': output_dim
            }}, f)
        
        print(f"\n✅ Model saved to {model_path}")
        
        # Plot training curves
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to training_curves.png")
    
    # ========== Step 3: Prediction with Uncertainty ==========
    if args.predict or args.all:
        print("\n🎲 STEP 3: MC Dropout Uncertainty Quantification")
        print("-" * 70)
        
        # Lazy imports
        try:
            import jax.numpy as jnp
            from models.bayesian_mlp import create_model
            from inference.mc_dropout import predict_with_uncertainty, plot_uncertainty_1d
        except ImportError as e:
            print(f"❌ Could not import JAX modules: {e}")
            print("   (This step requires the 'bayesian-glacier' environment)")
            return
        
        # Load model
        if not model_path.exists():
            print(f"❌ Error: No trained model at {model_path}")
            print("   Run with --train first")
            return
        
        import pickle
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
            params = saved['params']
            config = saved['model_config']
        
        model, _ = create_model(config['input_dim'], config['output_dim'])
        
        # Test on validation set
        test_idx = 0
        x_test = X_val[test_idx:test_idx+1]
        y_true = Y_val[test_idx]
        
        print(f"\nTest input: A={x_test[0,0]:.2e}, ε={x_test[0,1]:.2e}")
        
        # MC Dropout prediction
        mean_pred, std_pred, samples = predict_with_uncertainty(
            model, params, jnp.array(x_test),
            num_samples=args.mc_samples
        )
        
        # Statistics
        print(f"\nPrediction Statistics:")
        print(f"   Mean velocity: [{mean_pred.min():.2e}, {mean_pred.max():.2e}] m/s")
        print(f"   Uncertainty (std): [{std_pred.min():.2e}, {std_pred.max():.2e}] m/s")
        print(f"   Relative uncertainty: {(std_pred/mean_pred).mean():.1%}")
        
        # Visualize
        mean_pred_flat = np.asarray(mean_pred).flatten()
        std_pred_flat = np.asarray(std_pred).flatten()
        y_true_flat = np.asarray(y_true).flatten()
        
        x_coords = np.linspace(0, 5000, len(mean_pred_flat))
        plot_uncertainty_1d(
            x_coords,
            mean_pred_flat,
            std_pred_flat,
            y_true=y_true_flat,
            title="Bayesian Glacier Flow Prediction with Uncertainty",
            save_path="uncertainty_visualization.png"
        )
        
        print(f"\n✅ Uncertainty visualization saved!")
        
        # Performance metrics
        mse = np.mean((mean_pred - y_true)**2)
        mae = np.mean(np.abs(mean_pred - y_true))
        
        print(f"\nPrediction Accuracy:")
        print(f"   MSE: {mse:.2e}")
        print(f"   MAE: {mae:.2e}")
        print(f"   RMSE: {np.sqrt(mse):.2e}")
    
    # ========== Final Summary ==========
    print("\n" + "="*70)
    print("✅ MVP WORKFLOW COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Email Prof. Azizpour with results")
    print("  2. Scale to 500-1000 samples")
    print("  3. Build Streamlit webapp")
    print("  4. Write paper")
    print("\nGenerated files:")
    print(f"  - {data_path}")
    if model_path.exists():
        print(f"  - {model_path}")
    print(f"  - uncertainty_visualization.png")
    print(f"  - training_curves.png")
    print("="*70)


if __name__ == "__main__":
    main()
