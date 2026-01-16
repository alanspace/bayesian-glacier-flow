"""
Bayesian MLP with MC Dropout for Uncertainty Quantification
=============================================================

Simple Multi-Layer Perceptron with dropout layers for Bayesian inference
via Monte Carlo Dropout.

Author: Shek Lun Leung
Date: January 2026
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence
import optax
from tqdm import tqdm


class BayesianMLP(nn.Module):
    """
    Bayesian Neural Network using MC Dropout.
    
    Architecture:
        Input → Dense(256) → ReLU → Dropout(0.2)
             → Dense(256) → ReLU → Dropout(0.2)
             → Dense(256) → ReLU → Dropout(0.2)
             → Dense(output_dim)
    
    Key Feature: Keep dropout ACTIVE during inference for uncertainty estimation.
    """
    features: Sequence[int] = (256, 256, 256)
    output_dim: int = 1200  # Velocity field dimension
    dropout_rate: float = 0.2
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass through network.
        
        Args:
            x: Input parameters [batch, input_dim]
            training: If True, apply dropout (used for both training AND MC sampling)
        
        Returns:
            predictions: Velocity field [batch, output_dim]
        """
        # Hidden layers with dropout
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'dense_{i}')(x)
            x = nn.relu(x)
            # CRITICAL: deterministic=not training means dropout is ON when training=True
            x = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not training
            )(x)
        
        # Output layer (no activation, no dropout)
        x = nn.Dense(self.output_dim, name='output')(x)
        
        return x


def create_model(input_dim: int, output_dim: int, seed: int = 0):
    """
    Initialize Bayesian MLP with random weights.
    
    Args:
        input_dim: Number of input features
        output_dim: Number of output values (velocity DOFs)
        seed: Random seed for initialization
    
    Returns:
        model: Flax model
        params: Initialized parameters
    """
    model = BayesianMLP(output_dim=output_dim)
    key = jax.random.PRNGKey(seed)
    
    # Initialize with dummy input
    dummy_input = jnp.ones([1, input_dim])
    params = model.init(key, dummy_input, training=False)
    
    return model, params


def train_model(
    model,
    X_train: jnp.ndarray,
    Y_train: jnp.ndarray,
    X_val: jnp.ndarray = None,
    Y_val: jnp.ndarray = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42
):
    """
    Train Bayesian MLP using Adam optimizer.
    
    Args:
        model: Flax BayesianMLP model
        X_train: Training inputs [n_train, input_dim]
        Y_train: Training outputs [n_train, output_dim]
        X_val: Validation inputs (optional)
        Y_val: Validation outputs (optional)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam
        seed: Random seed
    
    Returns:
        params: Trained model parameters
        train_losses: Training loss history
        val_losses: Validation loss history (if validation data provided)
    """
    # Initialize model
    key = jax.random.PRNGKey(seed)
    params = model.init(key, X_train[:1], training=False)
    
    # Optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(params)
    
    # Loss function
    @jax.jit
    def loss_fn(params, X, Y, key):
        """MSE loss with dropout (training mode)."""
        predictions = model.apply(
            params, X, 
            training=True,
            rngs={'dropout': key}
        )
        return jnp.mean((predictions - Y) ** 2)
    
    @jax.jit
    def eval_loss(params, X, Y):
        """Evaluation loss without dropout."""
        predictions = model.apply(params, X, training=False)
        return jnp.mean((predictions - Y) ** 2)
    
    # Update function
    @jax.jit
    def update(params, opt_state, X_batch, Y_batch, key):
        """Single optimization step."""
        loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, Y_batch, key)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training loop
    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size
    
    train_losses = []
    val_losses = []
    
    print(f"🧠 Training Bayesian MLP...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Training samples: {n_train}")
    print()
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Shuffle data
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)
        X_shuffled = X_train[perm]
        Y_shuffled = Y_train[perm]
        
        # Mini-batch training
        epoch_losses = []
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_train)
            
            X_batch = X_shuffled[start_idx:end_idx]
            Y_batch = Y_shuffled[start_idx:end_idx]
            
            key, subkey = jax.random.split(key)
            params, opt_state, batch_loss = update(
                params, opt_state, X_batch, Y_batch, subkey
            )
            epoch_losses.append(batch_loss)
        
        # Record training loss
        train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(float(train_loss))
        
        # Validation loss
        if X_val is not None and Y_val is not None:
            val_loss = eval_loss(params, X_val, Y_val)
            val_losses.append(float(val_loss))
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1:3d}: Train Loss = {train_loss:.6f}"
            if val_losses:
                msg += f", Val Loss = {val_losses[-1]:.6f}"
            tqdm.write(msg)
    
    print("\n✅ Training complete!")
    print(f"   Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"   Final val loss: {val_losses[-1]:.6f}")
    
    return params, train_losses, val_losses


if __name__ == "__main__":
    print("Testing Bayesian MLP\n" + "="*60)
    
    # Create dummy data
    n_samples = 1000
    input_dim = 2
    output_dim = 100
    
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (n_samples, input_dim))
    Y = jax.random.normal(key, (n_samples, output_dim))
    
    # Split train/val
    n_train = 800
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y[:n_train], Y[n_train:]
    
    print(f"Data: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"      X_val {X_val.shape}, Y_val {Y_val.shape}\n")
    
    # Create and train model
    model, _ = create_model(input_dim, output_dim)
    
    params, train_losses, val_losses = train_model(
        model, X_train, Y_train, X_val, Y_val,
        epochs=20,
        batch_size=64,
        learning_rate=1e-3
    )
    
    print("\n✅ Bayesian MLP module ready!")
