def z_score(x):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    import numpy as np
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm, mu, sigma

def skl(x):
    """
    computes  X, zcore normalized by column using scikit-learn
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    return x_norm

