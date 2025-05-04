import numpy as np

def solve_tight_radius(covar: np.ndarray, prefactor: float, y: float = 1.0/255.0) -> np.ndarray:
    """Solve the tight axis-aligned bounding box radius for a Gaussian defined as
    y = prefactor * exp(-1/2 * xᵀ * covar⁻¹ * x)
    at the given y value.

    Args:
        covar: 2x2 covariance matrix
        prefactor: Prefactor of the Gaussian
        y: Threshold value (default: 1/255)

    Returns:
        np.ndarray: Radius vector [x, y]
    """
    if prefactor < y:
        return np.zeros(2)

    # Q = xᵀ * covar⁻¹ * x
    Q = -2.0 * np.log(y / prefactor)
    return np.sqrt(Q * covar[[0, 1], [0, 1]])


def generate_points_on_aabb_edges(radius: np.ndarray, N: int) -> np.ndarray:
    """Generate points on the edges of an AABB defined by radius.
    
    Args:
        radius: [x, y] radius of the AABB
        N: Number of points to generate per edge
        
    Returns:
        np.ndarray: Array of points on the AABB edges
    """
    # Points on the top and bottom edges
    x_points = np.linspace(-radius[0], radius[0], N)
    top_points = np.column_stack((x_points, np.full(N, radius[1])))
    bottom_points = np.column_stack((x_points, np.full(N, -radius[1])))
    
    # Points on the left and right edges
    y_points = np.linspace(-radius[1], radius[1], N)
    left_points = np.column_stack((np.full(N, -radius[0]), y_points))
    right_points = np.column_stack((np.full(N, radius[0]), y_points))
    
    # Combine all points
    return np.vstack((top_points, bottom_points, left_points, right_points))

def test_radius_with_edge_points():
    """Test the radius calculation by evaluating points on the AABB edges.
    
    The test generates points on the edges of the AABB defined by the radius and verifies
    that the maximum value of the Gaussian is close to the threshold y.
    """
    # Test parameters
    N = 1000  # Number of points per edge
    prefactor = 0.8
    y_threshold = 1.0 / 255.0
    
    # Test different covariance matrices
    test_covars = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),  # Identity
        np.array([[2.0, 0.0], [0.0, 1.0]]),  # Stretched in x
        np.array([[1.0, 0.5], [0.5, 1.0]]),  # Correlated
        np.array([[1.0, 1.0], [1.0, 2.0]])   # Strongly correlated
    ]
    
    for i, covar in enumerate(test_covars):
        print(f"\nTest case {i+1}:")
        print(f"Covariance matrix:\n{covar}")
        
        # Calculate radius
        radius = solve_tight_radius(covar, prefactor, y_threshold)
        print(f"Calculated radius: {radius}")
        
        # Generate points on the AABB edges
        points = generate_points_on_aabb_edges(radius, N)
        
        # Evaluate Gaussian for each point
        covar_inv = np.linalg.inv(covar)
        values = np.zeros(len(points))
        for j in range(len(points)):
            x = points[j]
            exponent = -0.5 * x @ covar_inv @ x
            values[j] = prefactor * np.exp(exponent)
        
        # Find maximum value
        max_value = np.max(values)
        print(f"Maximum value: {max_value}")
        print(f"Threshold: {y_threshold}")
        print(f"Ratio (max/threshold): {max_value/y_threshold}")
        
        # Verify that maximum value is close to threshold
        assert abs(max_value - y_threshold) < 1e-6, f"Test {i+1} failed: maximum value {max_value} not close to threshold {y_threshold}"

if __name__ == "__main__":
    test_radius_with_edge_points()


