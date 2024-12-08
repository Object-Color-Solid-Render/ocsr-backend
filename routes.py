import numpy as np
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from model_utils import load_obj, calculate_normals
from ocs_generator import generate_OCS, OCSContext4D, OCSGeometry4D, get_4d_ocs_geometry
from shaders import get_vertex_shader, get_fragment_shader
from ocs_slice import get_ostwald_slice
from govardovskii import govardovskii_template
from spectralDBLoader import read_csv
import ocs_generator


teapot_routes = Blueprint('teapot_routes', __name__)
ocs_routes = Blueprint('ocs_routes', __name__)
file_routes = Blueprint('file_routes', __name__)
db_routes = Blueprint('db_routes', __name__)

UPLOAD_FOLDER = 'res/uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}

# turns np arrays to lists and lists to lists
def to_list(l):
    return l.tolist() if isinstance(l, np.ndarray) else l

# TODO, remove this stuff
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
    """
    Generate object color solid geometry, colors, normals, and return shaders

    """
    
    # deprecated
    min_wavelength = int(request.args.get('minWavelength', 390))
    max_wavelength = int(request.args.get('maxWavelength', 700))

    # FIXME: algo to calculate bounds for min and maxes
    min_wavelength = 390
    max_wavelength = 700

    #FIXME: frontend should input a list of ocs datas
    # default values should yield garbage; we always want request to work
    is_max_basis = request.args.get('isMaxBasis', False) == "true"
    peakWavelength1 = int(request.args.get('peakWavelength1', 500))
    peakWavelength2 = int(request.args.get('peakWavelength2', 510))
    peakWavelength3 = int(request.args.get('peakWavelength3', 520))
    peakWavelength4 = int(request.args.get('peakWavelength4', 530))   # not used currently
    isCone1Active = request.args.get('isCone1Active', False) == "true"
    isCone2Active = request.args.get('isCone2Active', False) == "true"
    isCone3Active = request.args.get('isCone3Active', False) == "true"
    isCone4Active = request.args.get('isCone4Active', False) == "true"

    # how many samples per nm wavelength to take?
    # FIXME: add frontend customization
    SAMPLES_PER_NM:float = 0.1

    peaks = [peakWavelength1, peakWavelength2, peakWavelength3, peakWavelength4]
    activeCones = [isCone1Active, isCone2Active, isCone3Active, isCone4Active]

    # FIXME: Once the frontend renders multiple OCS, pass in the correct index based on the OCS being referenced
    generate_context: OCSContext4D = OCSContext4D(min_wavelength, max_wavelength, SAMPLES_PER_NM, peaks, activeCones, is_max_basis, idx=0)

    geometry: OCSGeometry4D = get_4d_ocs_geometry(generate_context)

    
    # Convert all numpy arrays to lists
    response_data = {
        'vertices': to_list(geometry.vertices),
        'indices': to_list(geometry.indices),
        'normals': to_list(geometry.normals),
        'colors': to_list(geometry.colors),
        'vertexShader': get_vertex_shader(),
        'fragmentShader': get_fragment_shader(),
        'wavelengths': to_list(geometry.wavelengths),
        's_response': to_list(geometry.curves[0]),
        'm_response': to_list(geometry.curves[1]),
        'l_response': to_list(geometry.curves[2]),
        'q_response': to_list(geometry.curves[3])
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
    a, b, c, d = float(request.args.get('a', 0)), float(request.args.get('b', 0)), float(request.args.get('c', 0)), float(request.args.get('d', 0))
    # TODO: When there are multiple OCS, add indexing
    intersection_vertices, intersection_colors, indices = get_ostwald_slice(ocs_generator.active_ocs[0], a, b, c, d) 
    return jsonify(
        {
            'vertices': intersection_vertices,
            'colors': intersection_colors,
            'indices': indices,
            'vertexShader': get_vertex_shader(),
            'fragmentShader': get_fragment_shader()
        }
    )


@db_routes.route('/get_spectral_db', methods=['GET'])
def get_spectral_db():

    data = read_csv("res/Spectral Sensitivity Database.csv")

    return jsonify({
        'data': data
    })

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
    
