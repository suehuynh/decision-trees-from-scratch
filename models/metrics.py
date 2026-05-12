import numpy as np

def entropy(p: float) -> float:
    """Calculates Cross Entropy for a probability p."""
    if p <= 0 or p >= 1:
        return 0.0
    
    metrics = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return metrics

def gini(p: float) -> float:
    """Calculates Gini Impurity: 1 - sum(p_i^2)."""
    metrics = 1 - p ** 2 - (1 - p) ** 2
    return metrics

def error(p: float) -> float:
    """Calculates Misclassification Error: min(p, 1 - p)."""
    metrics = float(min(p, 1 - p))
    return metrics

def variance(y: np.ndarray) -> float:
    """Calculates variance of labels for regression."""
    if len(y) == 0:
        return 0.0
    return np.var(y)

if __name__ == "__main__":
    p = np.random.random()
    
    print(f"--- Metrics Test with p={p:.4f} ---")

    assert isinstance(entropy(p), float)
    assert isinstance(gini(p), float)
    assert isinstance(error(p), float)
    
    print(f"Entropy: {entropy(p):.4f}")
    print(f"Gini:    {gini(p):.4f}")
    print(f"Error:   {error(p):.4f}")