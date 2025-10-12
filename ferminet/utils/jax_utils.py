import jax
import jax.numpy as jnp

def build_array(X: jnp.ndarray, Y: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """
    Constructs an array of shape (N, N, d), where:
      - arr[i, j, 0] = X[i, j]
      - arr[i, j, 1] = Y[i, j]
      - arr[i, j, 2..d-1] = positions[2..d-1]
    
    Arguments:
    - X: shape (N, N)
    - Y: shape (N, N)
    - positions: shape (d,) with d >= 2
    
    Returns:
    - arr: shape (N, N, d)
    """
    # Basic checks
    N1, N2 = X.shape
    assert (N1, N2) == Y.shape, "X and Y must have the same shape."
    d = positions.shape[0]
    assert d >= 2, "`positions` must be at least 2D."

    # Expand X and Y along a new last axis, so they become (N, N, 1).
    # Then broadcast the remaining positions[2:] to shape (N, N, d-2).
    return jnp.concatenate(
        [
            X[..., None],  # shape (N, N, 1)
            Y[..., None],  # shape (N, N, 1)
            jnp.broadcast_to(positions[2:], (N1, N2, d - 2))  # shape (N, N, d-2)
        ],
        axis=-1
    )