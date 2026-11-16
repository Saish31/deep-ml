import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    
    points_arr = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    for _ in range(max_iterations):
        # 1. Assign each point to the closest centroid
        labels = np.array([
            np.argmin(np.linalg.norm(centroids - p, axis=1)) 
            for p in points_arr
        ])

        # 2. Compute new centroids as the mean of assigned points
        new_centroids = np.copy(centroids)
        for j in range(k):
            cluster_points = points_arr[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)

        # 3. Stop early if centroids stop moving
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    # 4. Round to 4 decimals and return as a list of tuples
    return [tuple(row) for row in np.round(centroids, 4)]