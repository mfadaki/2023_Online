import numpy as np


def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


def cem_optimize(objective, mean, cov, num_samples=100, num_elites=10, num_iterations=100):
    for _ in range(num_iterations):
        # Sample candidate solutions from the current distribution
        samples = np.random.multivariate_normal(mean, cov, size=num_samples)

        # Evaluate the objective function for each candidate solution
        scores = np.array([objective(x, y) for x, y in samples])

        # Select the elite samples
        elite_indices = np.argsort(scores)[:num_elites]
        elite_samples = samples[elite_indices]

        # Update the probability distribution
        mean = np.mean(elite_samples, axis=0)
        cov = np.cov(elite_samples, rowvar=False)

    return mean


if __name__ == "__main__":
    initial_mean = np.array([0, 0])
    initial_cov = np.eye(2)

    optimal_solution = cem_optimize(rosenbrock, initial_mean, initial_cov)
    print("Optimal solution found:", optimal_solution)
