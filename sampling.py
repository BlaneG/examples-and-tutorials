import numpy as np


def bootstrap_sample(x, n_samples, smoothed=True, replace=True, p=None):
    """
    Parameters
    --------------
    x : 1-D array-like.
        Sample to compute bootstrap statistics from.
    n_samples : int
        Number of bootstrap samples to return.
    smoothed : bool
        Whether smoothed sampling is used.
    replace : bool
        Whether sampling with replacement is used.
    p : 1-D array-like
        Probability for each value of x for weighted sampling.
        
    Notes
    --------------
    References:  https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Smoothed_bootstrap
    """
    size = (n_samples, len(x))
    bootstrap_samples = np.random.choice(x, size, replace, p)
    bootstrap_result = np.mean(bootstrap_samples, axis=1)
    
    if smoothed:
        noise = gaussian_noise(x, n_samples)
        return bootstrap_result + noise

    return bootstrap_result

def gaussian_noise(x, n_samples)->np.ndarray:
    """Create guassian noise of len(n_samples).
    
    Parameters
    ------------
    x : ndarray
    n_samples : int
    
    References
    ------------
    researchgate.net/publication/292390585_Unbiasing_the_bootstrap-bootknife_sampling_vs_smoothing
    """
    sigma = np.std(x)/np.sqrt(len(x))
    noise = np.random.normal(0, sigma, n_samples)
    return noise