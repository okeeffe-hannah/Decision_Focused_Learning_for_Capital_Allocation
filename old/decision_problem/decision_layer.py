import torch
from scipy.stats import norm

def expected_shortfall_mixture(alpha, p_hat, mu_g, sigma_g, 
                                mu_b, sigma_b, K, K_bar):
    """Equation (10) from the document."""
    ES_g = gaussian_ES(alpha, mu_g, sigma_g, K, K_bar)
    ES_b = gaussian_ES(alpha, mu_b, sigma_b, K, K_bar)
    return (1 - p_hat) * ES_g + p_hat * ES_b

def gaussian_ES(alpha, mu, sigma, K, K_bar):
    """Equation (8) from the document."""
    r_thresh = (K_bar/K - 1) / (alpha + 1e-8)
    z = (r_thresh - mu) / sigma
    ES = (K_bar - K*(1 + alpha*mu)) * norm.cdf(z) + K*alpha*sigma * norm.pdf(z)
    return max(ES, 0)

def soft_optimal_alpha(p_hat, params, n_grid=100, temperature=0.01):
    """Option C — soft argmax over grid."""
    alphas = torch.linspace(1e-6, 1-1e-6, n_grid)
    objectives = torch.tensor([
        objective(a.item(), p_hat, params) for a in alphas
    ])
    weights = torch.softmax(objectives / temperature, dim=0)
    return (weights * alphas).sum()