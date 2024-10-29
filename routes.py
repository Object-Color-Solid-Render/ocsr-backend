import numpy as np
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from model_utils import load_obj, calculate_normals
from ocs_generator import generate_OCS
from shaders import get_vertex_shader, get_fragment_shader
from ocs_slice import get_y_slice

teapot_routes = Blueprint('teapot_routes', __name__)
ocs_routes = Blueprint('ocs_routes', __name__)
file_routes = Blueprint('file_routes', __name__)

UPLOAD_FOLDER = 'res/uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@file_routes.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        return jsonify({'message': 'File successfully uploaded', 'filename': filename}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@ocs_routes.route('/get_ocs_data', methods=['GET'])
def get_ocs_data():
    """Generate object color solid geometry, colors, normals, and return shaders"""
    min_wavelength, max_wavelength = int(request.args.get('minWavelength', 390)), int(request.args.get('maxWavelength', 700))
    max_num_points = int(request.args.get('maxNumPoints', 20))
    response_file_name = request.args.get('responseFileName', '')
    is_max_basis = True # TODO # request.args.get('isMaxBasis', False)

    print(f"min: {min_wavelength}")
    print(f"response file name: {response_file_name}")
    print("generating ocs")
    vertices, indices, colors, wavelengths, s_response, m_response, l_response = generate_OCS(min_wavelength, max_wavelength, response_file_name, is_max_basis)
    
    normals = calculate_normals(vertices, indices)

    if (len(vertices) != len(colors)):
        print("ERROR: vertices and colors have different lengths")

    return jsonify({
        'vertices': vertices,
        'indices': indices,
        'normals': normals,
        'colors': colors,
        'vertexShader': get_vertex_shader(),
        'fragmentShader': get_fragment_shader(),
        'wavelengths': wavelengths,
        's_response': s_response,
        'm_response': m_response, 
        'l_response': l_response
    })

# Use a POST request when it later gets adapted to using vertices
# @ocs_routes.route('/compute_ocs_slice', methods=['POST']) 
@ocs_routes.route('/compute_ocs_slice', methods=['GET']) 
def compute_ocs_slice():
    """Return the vertices of the OCS intersected by the plane specified"""
    # data = request.get_json()
    # vertices, colors, num_wavelengths, y = (
    #     data.get('vertices', np.array([])), 
    #     data.get('colors', np.array([])), 
    #     int(data.get('num_wavelengths', 0)),
    #     float(data.get('y', 0)),
    # )
    vertices, colors, y = request.args.get('vertices', []), request.args.get('colors', []), float(request.args.get('y', 0))
    num_wavelengths = 0 # TODO: FIX THIS
    intersection_vertices, intersection_colors = get_y_slice(vertices, colors, num_wavelengths, y) 
    return jsonify(
        {
            'vertices': intersection_vertices,
            'colors': intersection_colors,
            'vertexShader': get_vertex_shader(),
            'fragmentShader': get_fragment_shader()
        }
    )



@teapot_routes.route('/get_teapot_data', methods=['GET'])
def get_teapot_data():
    """Fetch teapot 3D model data, calculate normals, and return shaders"""
    vertices, indices = load_obj('res/models/teapot.obj')
    normals = calculate_normals(vertices, indices)
    colors = [0.5, 0.5, 0.5] * len(vertices)

    return jsonify({
        'vertices': vertices,
        'indices': indices,
        'normals': normals,
        'colors': colors,
        'vertexShader': get_vertex_shader(),
        'fragmentShader': get_fragment_shader()
    })