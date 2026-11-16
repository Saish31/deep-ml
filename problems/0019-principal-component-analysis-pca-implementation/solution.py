import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std

    cov_matrix = np.cov(standardized_data.T)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    top_k_vectors = eigenvectors[:, idx[:k]]

    for j in range(k):
        for val in top_k_vectors[:, j]:
            if abs(val) > 1e-10:
                if val < 0:
                    top_k_vectors[:, j] *= -1
                break
    return np.round(top_k_vectors, 4)