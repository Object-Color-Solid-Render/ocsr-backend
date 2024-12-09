import numpy as np
import pywavefront

def load_obj(file_path):
    """Loads an obj model and extract vertices and indices"""
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = scene.vertices
    indices = [index for mesh in scene.mesh_list for index in mesh.faces] # library calls indices, "faces"
    return vertices, indices

def calculate_normals(vertices, indices):
    """Calculate normals for each vertex based on the indices"""
    
    center = np.mean(vertices, axis=0)

    normals = np.zeros_like(vertices)

    for index in indices:
        v0, v1, v2 = [np.array(vertices[i]) for i in index]
        normal = np.cross(v1 - v0, v2 - v0)
        
        # flip normal if it points towards center
        if np.dot(normal, v0 - center) < 0:
            normal *= -1

        for i in index:
            normals[i] += normal            
    
    norms = np.linalg.norm(normals, axis=1)
    zeros = norms == 0
    norms[zeros] = 1 # since it's is already a zero vector, it's fine to just leave it like this
    norms = norms[:, np.newaxis]
    return (normals/norms).tolist()

def convex_2d_hull_to_vbo_ibo(
        coords: list[tuple[float, float]],
        vertex_colors: list[tuple[float, float, float]]
        ):
    """
    Generate vertex position list, vertex color list, and triangle indices for a convex shape in R^3.

    Parameters:
        coords (list of tuple): Unordered list of 2D coordinates forming a convex shape.
        vertex_colors (list of tuple): List of RGB colors (or RGBA) corresponding to each vertex.

    Returns:
        tuple: A tuple (vertices, colors, indices), where:
            - vertices (np.ndarray): Unique list of (x, y, z) vertex positions in R^3.
            - colors (np.ndarray): Corresponding list of (r, g, b[, a]) vertex colors.
            - indices (np.ndarray): Array of shape (n, 3) representing triangle indices.
    """
    assert len(coords) == len(vertex_colors), "Each coordinate must have a corresponding color."
    
    # Calculate the bounding box of the coordinates
    coords_array = np.array(coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    range_coords = max_coords - min_coords

    # Normalize coordinates to be in the range [0, 1]
    normalized_coords = (coords_array - min_coords) / range_coords

    # Compute centroid
    centroid = tuple(np.mean(normalized_coords, axis=0))  # Centroid coordinates (2D)
    centroid_color = tuple(np.mean(vertex_colors, axis=0))  # Average color for the centroid

    # Sort the coordinates counterclockwise around the centroid
    sorted_indices = sorted(
        range(len(normalized_coords)), 
        key=lambda i: np.arctan2(normalized_coords[i][1] - centroid[1], normalized_coords[i][0] - centroid[0])
    )
    sorted_coords = [normalized_coords[i] for i in sorted_indices]
    sorted_colors = [vertex_colors[i] for i in sorted_indices]

    # Create vertex positions and colors in R^3
    vertex_positions = [(x, y, 0.0) for (x, y) in sorted_coords]
    vertex_colors = sorted_colors
    
    # Add the centroid as a vertex
    vertex_positions.append((centroid[0], centroid[1], 0.0))  # Add z=0 for 3D compatibility
    vertex_colors.append(centroid_color)

    # Convert to numpy arrays for easy indexing
    vertices = np.array(vertex_positions, dtype=np.float32)
    colors = np.array(vertex_colors, dtype=np.float32)

    # Generate triangle indices
    centroid_index = len(vertices) - 1
    indices = []
    for i in range(len(sorted_coords)):
        next_i = (i + 1) % len(sorted_coords)
        indices.append([i, next_i, centroid_index])  # Each triangle as a triplet

    return vertices, colors, np.array(indices, dtype=np.int32)

def quads_to_triangles(quads: np.ndarray, invert_winding: bool = False) -> np.ndarray:
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
    triangles_1 = quads[:, [0, 1, 2]] if invert_winding else quads[:, [0, 2, 1]]
    
    # Second triangle for each quad: [v0, v2, v3]
    triangles_2 = quads[:, [0, 2, 3]] if invert_winding else quads[:, [0, 3, 2]]
    
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
