import numpy as np

def colorfulness(image):
    """
    Compute the "colorfulness" metric of an RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image, shape (3, H, W), values in [0,255] or [0,1].
        If values are [0,1], function will scale internally to [0,255].
    
    Returns
    -------
    float
        Colorfulness score (higher means more colorful)
    """
    if image.max() <= 1.0:
        image = image * 255.0

    R = image[0]
    G = image[1]
    B = image[2]

    # rg = R - G
    rg = R - G
    # yb = 0.5*(R + G) - B
    yb = 0.5 * (R + G) - B

    # Compute the standard deviation and mean of rg and yb
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)

    # Combine the mean and standard deviation
    colorfulness_score = np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)
    return colorfulness_score


def calculate_colorfulness(images):
    colorfulness_scores = [colorfulness(image) for image in images]
    return np.mean(colorfulness_scores)