from chromalab.observer import Observer
from chromalab.spectra import Spectra, Illuminant
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#%matplotlib widget

import bisect
import pandas as pd
import os
from tqdm import tqdm

def read_cone_response(csv_file_path, min_wavelength, max_wavelength, max_points=None):
    # First, try to read the file without a header

    df = None
    try:
        df = pd.read_csv(csv_file_path, header=None)
    except FileNotFoundError:
        print("File not found!")
        return None, None, None, None
    except:
        print("File not found!")
        return None, None, None, None
    
    # Check if the first value in the file is numeric to determine if there's a header
    try:
        float(df.iloc[0, 0])  # Try converting the first value to a float
        has_header = False     # If successful, there's no header
    except ValueError:
        has_header = True      # If it raises ValueError, there is a header
    
    # If there is a header, reload the CSV with the header
    if has_header:
        df = pd.read_csv(csv_file_path)
    
    # Check if we have 3 or 4 cones based on the number of columns
    if len(df.columns) == 4:
        df.columns = ['Wavelength', 'S-Response', 'M-Response', 'L-Response']
    elif len(df.columns) == 5:
        df.columns = ['Wavelength', 'S-Response', 'Q-Response', 'M-Response', 'L-Response']
    
    df = df[(df['Wavelength'] >= min_wavelength) & (df['Wavelength'] <= max_wavelength)]

    # Enforce a maximum number of datapoints if specified
    if max_num_points:
        wavelength_idxs = get_idxs(collection=df, max_num_points=max_num_points)  # decimal values are possible, so make it int
        
        print(wavelength_idxs, df.shape[0])
        df = df.iloc[wavelength_idxs, :]

    # Check the step size
    # wavelength_step = df['Wavelength'].iloc[1] - df['Wavelength'].iloc[0]
    
    # if wavelength_step < 10:
    #     # Filter rows where the wavelength is a multiple of 10
    #     df = df[df['Wavelength'] % 10 == 0]

    # Extract the relevant columns as arrays
    wavelengths = df['Wavelength'].to_numpy()
    s_response = df['S-Response'].to_numpy()
    m_response = df['M-Response'].to_numpy()
    l_response = df['L-Response'].to_numpy()
    
    return wavelengths, s_response, m_response, l_response

def quads_to_triangles(quads: np.ndarray) -> np.ndarray:
    """
    Convert an array of quads (n, 4, 3) to an array of triangles (2n, 3, 3).
    
    Parameters:
    quads (np.ndarray): An array of shape (n, 4, 3) where n is the number of quads, 
                        each quad has 4 vertices in 3D space.
                        
    Returns:
    np.ndarray: An array of shape (2n, 3, 3) containing triangles formed from the quads.
    """
    # Ensure input is the correct shape
    assert quads.shape[1:] == (4, 3), "Input array must have shape (n, 4, 3)"
    
    # Number of quads
    n = quads.shape[0]
    
    # First triangle for each quad: [v0, v1, v2]
    triangles_1 = quads[:, [0, 1, 2]]
    
    # Second triangle for each quad: [v0, v2, v3]
    triangles_2 = quads[:, [0, 2, 3]]
    
    # Stack the two sets of triangles together along the first axis
    triangles = np.vstack((triangles_1, triangles_2))
    
    return triangles

def triangles_to_vertices_indices(triangles: np.ndarray):
    """
    Convert a list of triangles into a list of unique vertices and their indices.
    
    Parameters:
    triangles (np.ndarray): An array of shape (n, 3, 3) where n is the number of triangles,
                            each triangle has 3 vertices in 3D space.
                            
    Returns:
    tuple: A tuple containing:
        - vertices (np.ndarray): An array of unique vertices of shape (m, 3), where m is the number of unique vertices.
        - indices (np.ndarray): An array of shape (n, 3) representing the indices of the triangles in the vertex list.
    """
    # Dictionary to store unique vertices and their corresponding indices
    vertex_to_index = {}
    vertices = []
    indices = []

    # Iterate through all triangles
    for triangle in triangles:
        triangle_indices = []
        for vertex in triangle:
            # Convert the vertex (numpy array) to a tuple so it can be used as a dictionary key
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in vertex_to_index:
                # Assign a new index if the vertex is not already in the dictionary
                vertex_to_index[vertex_tuple] = len(vertices)
                vertices.append(vertex_tuple)
            # Append the index of the vertex for this triangle
            triangle_indices.append(vertex_to_index[vertex_tuple])
        # Append the triangle indices
        indices.append(triangle_indices)
    
    # Convert the list of vertices and indices to numpy arrays
    vertices = np.array(vertices)
    indices = np.array(indices)
    
    return vertices, indices

import os
def generate_OCS(min_wavelength: int, max_wavelength: int, response_file_name: str, max_num_points: int = None):
def generate_OCS(min_wavelength: int, max_wavelength: int, response_file_name: str):
    
    csv_file_path = os.path.join(os.getcwd(), "res/uploads/", response_file_name)
    wavelengths, s_response, m_response, l_response = read_cone_response(csv_file_path, min_wavelength, max_wavelength, max_num_points)
    print("BRUH", wavelengths)

    if wavelengths is None:
        # Cone responses of a typical trichromat.
        freq = 15
        wavelengths = np.arange(min_wavelength, max_wavelength + 1, freq)
        standard_trichromat = Observer.trichromat(np.arange(min_wavelength, max_wavelength + 1, freq))
        s_response, m_response, l_response = standard_trichromat.sensors[0].data, standard_trichromat.sensors[1].data, standard_trichromat.sensors[2].data 
        
        # Only grab max_num_points amount of data points
        if max_num_points:
            wavelength_idxs = get_idxs(s_response, max_num_points)
            s_response, m_response, l_response = s_response[wavelength_idxs], m_response[wavelength_idxs], l_response[wavelength_idxs]
            
    else:
        # Update the indices to the wavelengths we care about
        # start_idx = bisect.bisect_left(wavelengths, min_wavelength)
        # end_idx = bisect.bisect_left(wavelengths, max_wavelength)
        # s_response, m_response, l_response = s_response[start_idx:end_idx], m_response[start_idx:end_idx], l_response[start_idx:end_idx]
        print(f"Loaded response file: {response_file_name}")
        print(f"Wavelengths: {wavelengths}")
        print(f"Wavelengths: {len(wavelengths)}")
        print(f"S-Response: {len(s_response)}")
        print(f"M-Response: {len(m_response)}")
        print(f"L-Response: {len(l_response)}")

    n = len(wavelengths)
    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)

    # Each point has an indicator reflectance function where R = 1 at a single wavelength and 0 elsewhere.
    # These points can be thought of as vectors which form a (linearly dependent) basis.
    # The Minkowski sum of these vectors span the object color solid.
    # Each point in the solid can be represented as some (non-unique) linear combination of these vectors.
    # This represents equations (9), (10), (11), (12), (13).
    lms_responses = np.vstack(( s_response, 
                                m_response, 
                                l_response)) * illuminant.data

    points = np.copy(lms_responses).T

    # As shown in Centore's paper, these vertices form the shape of the solid.
    # This represents the matrix in (7).
    vertices = np.zeros((n + 1, n, 3))
    for i in range(1, n + 1):
        for j in range(n):
            vertices[i, j] = vertices[i - 1, j] + points[(i + j - 1) % n]

    # This represents the diagram in (8)
    faces = np.zeros((n * (n - 1), 4, 3))
    face_colors = np.zeros((n * (n - 1), 3))
    for i in tqdm(range(1, n)):
        for j in range(n):
            faces[((i - 1) * n) + j, 0] = vertices[i, j]
            faces[((i - 1) * n) + j, 1] = vertices[i - 1, (j + 1) % n]
            faces[((i - 1) * n) + j, 2] = vertices[i, (j + 1) % n]
            faces[((i - 1) * n) + j, 3] = vertices[i + 1, j]
            
            # Calculate the reflectance on each face by using the reflectance of one of its vertices.
            # Since each vertex can be thought of as a linear combination of the basis vectors, 
            # the vertex's reflectance is the sum of reflectances of those vectors that made up the vertex.
            reflectance_data = np.zeros(n)
            for k in range(i):
                reflectance_data[(j + k) % n] = 1
            reflectance = Spectra(wavelengths=wavelengths, data=reflectance_data)
            face_colors[(i - 1) * n + j] = reflectance.to_rgb(illuminant)    # Bottleneck. Takes about 3ms. 

    # Uses ideas from Jessica's paper, on chapter 3.2 The Max Basis.
    # We use the cutpoints that Jessica shows to be optimal for the trichromatic case.
    cutpoint_1 = 487
    cutpoint_2 = 573
    index_1 = None
    index_2 = None
    for i, wavelength in enumerate(wavelengths):
        if index_1 is None and wavelength > cutpoint_1:
            index_1 = i
        if index_2 is None and wavelength >= cutpoint_2:
            index_2 = i
            break

    # We calculate the vectors p1, p2 and p3 as shown in the paper.
    # We "project the partition into the cone response basis" by summing up all the lms_responses within each partition.
    # Note that our earlier calculations for lms_responses includes the illuminant already.
    p1 = np.sum(lms_responses[:, :index_1], axis=1).reshape((3, 1))
    p2 = np.sum(lms_responses[:, index_1:index_2], axis=1).reshape((3, 1))
    p3 = np.sum(lms_responses[:, index_2:], axis=1).reshape((3, 1))

    # We then create a transformation matrix that maps p1 to (1, 0, 0), p2 to (0, 1, 0) and p3 to (1, 0, 0).
    # p1, p2 and p3 correspond to the ideal R, G, B points on our object color solid, 
    # and we are mapping them onto the R, G, B points on the RGB cube.
    # We are essentially "stretching" our object color solid so that it approximates the RGB cube.
    transformation_matrix = np.linalg.inv(np.hstack((p1, p2, p3)))
    faces_transformed = np.matmul(faces, transformation_matrix.T)
    faces = faces_transformed

    tris = quads_to_triangles(faces)
    vertices, indices = triangles_to_vertices_indices(tris)

    # Normalize vertices to [0, 1] range
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    # TODO: The bird color solid changes in shape when normalizing vs. not
    normalized_vertices =  (vertices - min_coords) / range_coords

    # Ensure the colors array has enough values to match the number of vertices
    if len(face_colors) < len(vertices):
        num_missing_colors = len(vertices) - len(face_colors)
        missing_colors = np.array([[1.0, 1.0, 1.0]] * num_missing_colors)  # Create a NumPy array of white [R, G, B] for missing colors
        
        # Use np.append to concatenate the arrays
        face_colors = np.append(face_colors, missing_colors, axis=0)  # Append along the correct axis

    if wavelengths is None:
        wavelengths = np.arange(min_wavelength, max_wavelength + 1, 3)[wavelength_idxs].tolist()
    else:
        wavelengths = wavelengths.tolist()

    return normalized_vertices.tolist(), indices.tolist(), face_colors.tolist(), wavelengths, s_response.tolist(), m_response.tolist(), l_response.tolist()