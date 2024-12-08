import numpy as np 

def find_orthogonal_vectors(normal):
    """
    Obtain the two vectors spanning the plane defined by a normal vector.
    """
    n = normal / np.linalg.norm(normal)
    # Create a vector that is not collinear with the normal
    if n[0] == 0 and n[1] == 0:
        arbitrary_vector = np.array([1, 0, 0])
    else:
        arbitrary_vector = np.array([0, 0, 1])
    vec_a = np.cross(n, arbitrary_vector)
    vec_a /= np.linalg.norm(vec_a)
    vec_b = np.cross(n, vec_a)
    vec_b /= np.linalg.norm(vec_b)
    return vec_a, vec_b

# Taken from https://mk.bcgsc.ca/colorblind/math.mhtml.
lms_to_rgb_transformation = np.array([
    [ 5.47221206, -4.6419601 ,  0.16963708],
    [-1.1252419 ,  2.29317094, -0.1678952 ],
    [ 0.02980165, -0.19318073,  1.16364789]])

# Gamma correction for sRGB space.
def gamma_correct(rgb):
    rgb_corrected = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1 / 2.4) - 0.055
    )
    return rgb_corrected

def lms_to_rgb(lms, to_lms_transformation, max_basis=True):
    # TODO: Note that this uses the human LMS to RGB, and consider implications for non-human OCS
    if max_basis:
        return gamma_correct(np.clip(lms_to_rgb_transformation @ np.linalg.inv(to_lms_transformation) @ lms, 0, 1))
    else:
        return gamma_correct(np.clip(lms_to_rgb_transformation @ lms, 0, 1))

# Need an OCSR class that holds:
    # it'd be nice to have a non-blocking portion to the initialization
    # it'll be responsible for its own slicing and generation stuff
# illuminant, pass in with the wavelengths
# store the transformation from original (i.e. inverse max basis if max basis, else just identity)
    # depends because max-basis is computed dynamically now
# edges
# vertices

