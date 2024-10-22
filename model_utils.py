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

    return (normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]).tolist()