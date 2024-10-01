from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
import pywavefront

app = Flask(__name__)
CORS(app)

def read_file(file_path: str) -> str:
    file_content: str = ""
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

@app.route('/get_teapot_data', methods=['GET'])
def get_teapot_data():
    VERT_SHADER_PATH: str = "shaders/teapot.vert"
    FRAG_SHADER_PATH: str = "shaders/teapot.frag"

    """Fetch teapot 3D model data, calculate normals, and return shaders"""
    scene = pywavefront.Wavefront("meshes/teapot.obj", collect_faces=True)

    # Extract vertices and faces
    vertices = scene.vertices
    faces = [face for mesh in scene.mesh_list for face in mesh.faces]

    # Calculate vertex normals
    normals = calculate_normals(vertices, faces)

    # Shaders
    vertex_shader = read_file(VERT_SHADER_PATH)
    fragment_shader = read_file(FRAG_SHADER_PATH)

    return jsonify({
        'vertices': vertices,
        'faces': faces,
        'normals': normals,
        'vertexShader': vertex_shader,
        'fragmentShader': fragment_shader
    })

def calculate_normals(vertices, faces):
    """Calculate normals for each vertex based on the faces"""
    normals = np.zeros_like(vertices)
    
    for face in faces:
        v0, v1, v2 = [np.array(vertices[i]) for i in face]
        normal = np.cross(v1 - v0, v2 - v0)
        
        for i in face:
            normals[i] += normal
    
    # Normalize the result
    return (normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]).tolist()

if __name__ == '__main__':
    get_teapot_data()
    #app.run(debug=True)
