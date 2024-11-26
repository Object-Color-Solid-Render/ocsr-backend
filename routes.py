import numpy as np
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from model_utils import load_obj, calculate_normals
from ocs_generator import generate_OCS, generate_OCS2
from shaders import get_vertex_shader, get_fragment_shader
from ocs_slice import get_y_slice
from govardovskii import govardovskii_template
from spectralDBLoader import read_csv

teapot_routes = Blueprint('teapot_routes', __name__)
ocs_routes = Blueprint('ocs_routes', __name__)
file_routes = Blueprint('file_routes', __name__)

UPLOAD_FOLDER = 'res/uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# turns np arrays to lists and lists to lists
def to_list(l):
    return l.tolist() if isinstance(l, np.ndarray) else l

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
    
    min_wavelength = int(request.args.get('minWavelength', 390))
    max_wavelength = int(request.args.get('maxWavelength', 700))

    # default values should yield garbage; we always want request to work
    wavelength_sample_resolution = int(request.args.get('wavelengthSampleResolution', 5))
    is_max_basis = request.args.get('isMaxBasis', False)
    omit_beta_band = request.args.get('omitBetaBand', True)
    peakWavelength1 = int(request.args.get('peakWavelength1', 500))
    peakWavelength2 = int(request.args.get('peakWavelength2', 510))
    peakWavelength3 = int(request.args.get('peakWavelength3', 520))
    peakWavelength4 = int(request.args.get('peakWavelength4', 530))   # not used currently
    isCone1Active = request.args.get('isCone1Active', False)
    isCone2Active = request.args.get('isCone2Active', False)
    isCone3Active = request.args.get('isCone3Active', False)
    isCone4Active = request.args.get('isCone4Active', False)

    print("===== Parameters =====")
    print("Wavelength Bouunds: ", min_wavelength, max_wavelength)
    print("Peak Wavelengths: ", peakWavelength1, peakWavelength2, peakWavelength3, peakWavelength4)
    print("Active Cones: ", isCone1Active, isCone2Active, isCone3Active, isCone4Active)
    print("Omit Beta Band: ", omit_beta_band)
    print("Is Max Basis: ", is_max_basis)
    print("Wavelength Sample Resolution: ", wavelength_sample_resolution)
    print("======================")

    peaks = [peakWavelength1, peakWavelength2, peakWavelength3, peakWavelength4]

    wavelengths = np.linspace(min_wavelength, max_wavelength, num=wavelength_sample_resolution)

    # curves is [S, M, L, Q]
    curves = [govardovskii_template(wavelengths=wavelengths, 
                            lambda_max=peak,
                            A1_proportion=100,
                            omit_beta_band=omit_beta_band) for peak in peaks]

    vertices, indices, colors = generate_OCS2(curves, wavelengths, is_max_basis)
    normals = calculate_normals(vertices, indices)

    if len(vertices) != len(colors):
        print("ERROR: vertices and colors have different lengths")

    # Convert all numpy arrays to lists
    response_data = {
        'vertices': to_list(vertices),
        'indices': to_list(indices),
        'normals': to_list(normals),
        'colors': to_list(colors),
        'vertexShader': get_vertex_shader(),
        'fragmentShader': get_fragment_shader(),
        'wavelengths': to_list(wavelengths),
        's_response': to_list(curves[0]),
        'm_response': to_list(curves[1]),
        'l_response': to_list(curves[2])
    }

    return jsonify(response_data)


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
    print("GENERATING SLICE!")
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