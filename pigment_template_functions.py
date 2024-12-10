import numpy as np

def govardovskii_template(
        wavelengths, 
        lambda_max, 
        A1_proportion=100, 
        omit_beta_band=False
        ):
    """
    Implements Govardovskii's (2000) visual pigment template with A1/A2 chromophore mixing.
    
    Parameters:
    wavelengths : array-like
        Wavelengths at which to evaluate the template (nm)
    lambda_max : float
        Peak wavelength of the visual pigment (nm)
    A1_proportion : float
        Percentage of A1 chromophore (0-100), default 100 (pure A1)
        
    Returns:
    tuple: (wavelengths, sensitivities)
        Arrays of wavelengths and corresponding normalized sensitivity values
    """
    wavelengths = np.asarray(wavelengths)
    x = lambda_max / wavelengths
    
    # A1 template parameters
    a1 = 0.8795 + 0.0459 * np.exp(-((lambda_max - 300)**2) / 11940)
    lambda_beta_A1 = 189 + 0.315 * lambda_max
    beta_bandwidth_A1 = -40.5 + 0.195 * lambda_max
    
    # A1 alpha and beta bands
    alpha_A1 = 1 / (np.exp(69.7 * (a1 - x)) + 
                    np.exp(28 * (0.922 - x)) + 
                    np.exp(-14.9 * (1.104 - x)) + 
                    0.674)
    beta_A1 = 0.26 * np.exp(-((wavelengths - lambda_beta_A1) / 
                              beta_bandwidth_A1)**2)
    sensitivity_A1 = alpha_A1 + (0 if omit_beta_band else beta_A1)
    
    # A2 template parameters
    a2 = 0.875 + 0.0268 * np.exp((lambda_max - 665) / 40.7)
    A2_peak = 62.7 + 1.834 * np.exp((lambda_max - 625) / 54.2)
    lambda_beta_A2 = 216.7 + 0.287 * lambda_max
    beta_bandwidth_A2 = 317 - 1.149 * lambda_max + 0.00124 * (lambda_max**2)
    
    # A2 alpha and beta bands
    alpha_A2 = 1 / (np.exp(A2_peak * (a2 - x)) + 
                    np.exp(20.85 * (0.9101 - x)) + 
                    np.exp(-10.37 * (1.1123 - x)) + 
                    0.5343)
    beta_A2 = 0.26 * np.exp(-((wavelengths - lambda_beta_A2) / 
                              beta_bandwidth_A2)**2)
    sensitivity_A2 = alpha_A2 + (0 if omit_beta_band else beta_A2)
    
    # Combine A1 and A2 templates according to proportion
    A1_weight = A1_proportion / 100
    A2_weight = 1 - A1_weight
    total_sensitivity = (A1_weight * sensitivity_A1 + 
                        A2_weight * sensitivity_A2)
    
    # Normalize
    normalized_sensitivity = total_sensitivity / np.max(total_sensitivity)
    
    return normalized_sensitivity

def lamb_template(
        wavelengths, 
        lambda_max
        ):
    """
    Computes the Lamb visual pigment template.

    Parameters:
    wavelengths : array-like
        Wavelengths at which to evaluate the template (nm)
    lambda_max : float
        Peak wavelength of the visual pigment (nm)

    Returns:
    normalized_sensitivity : array
        Array of normalized sensitivity values
    """
    # Constants
    A = 0.880
    B = 0.924
    C = 1.104
    D = 0.655
    a = 70
    b = 28.5
    c = -14.1

    # Compute spectral sensitivity for each wavelength in the specified range
    wavelengths = np.asarray(wavelengths)
    r = lambda_max / wavelengths
    a_ = a * (A - r)
    b_ = b * (B - r)
    c_ = c * (C - r)

    S_lambda_1 = np.exp(a_) + np.exp(b_) + np.exp(c_) + D
    S_lambda = 1 / S_lambda_1

    # Normalize
    normalized_sensitivity = S_lambda / np.max(S_lambda)

    return normalized_sensitivity
