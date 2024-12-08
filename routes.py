import numpy as np
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from model_utils import load_obj, calculate_normals
from ocs_generator import OCSContext4D, OCSGeometry4D, get_4d_ocs_geometry, center_4d_ocs_geometry
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
    Generate object color solid geometries for multiple entries.
    """

    entries = []
    args = request.args

    # Parse entries from request arguments
    index = 0
    while f'entries[{index}][minWavelength]' in args:
        entry = {
            'min_wavelength': int(args.get(f'entries[{index}][minWavelength]', 390)),
            'max_wavelength': int(args.get(f'entries[{index}][maxWavelength]', 700)),
            'omitBetaBand': args.get(f'entries[{index}][omitBetaBand]', 'False') == 'true',
            'isMaxBasis': args.get(f'entries[{index}][isMaxBasis]', 'False') == 'true',
            'wavelengthSampleResolution': float(args.get(f'entries[{index}][wavelengthSampleResolution]', 1.0)),
            'peaks': [
                int(args.get(f'entries[{index}][peakWavelength1]', 500)),
                int(args.get(f'entries[{index}][peakWavelength2]', 510)),
                int(args.get(f'entries[{index}][peakWavelength3]', 520)),
                int(args.get(f'entries[{index}][peakWavelength4]', 530)),
            ],
            'activeCones': [
                args.get(f'entries[{index}][isCone1Active]', 'False') == 'true',
                args.get(f'entries[{index}][isCone2Active]', 'False') == 'true',
                args.get(f'entries[{index}][isCone3Active]', 'False') == 'true',
                args.get(f'entries[{index}][isCone4Active]', 'False') == 'true',
            ],
            'idx': index
        }
        entries.append(entry)
        index += 1

    response_data = []
    for entry in entries:
        generate_context = OCSContext4D(
            entry['min_wavelength'],
            entry['max_wavelength'],
            0.1,
            # entry['wavelengthSampleResolution'],
            entry['peaks'],
            entry['activeCones'],
            entry['isMaxBasis'],
            entry['idx']
        )

        geometry: OCSGeometry4D = get_4d_ocs_geometry(generate_context)
        geometry: OCSGeometry4D = center_4d_ocs_geometry(geometry)

        data = {
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
            'q_response': to_list(geometry.curves[3]),
        }
        response_data.append(data)

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
    num_ocs = int(request.args.get('num_ocs', 0))

    response_data = []
    for i in range(num_ocs):
        intersection_vertices, intersection_colors, indices = get_ostwald_slice(ocs_generator.active_ocs[i], a, b, c, d) 
        data = {
            'vertices': intersection_vertices,
            'colors': intersection_colors,
            'indices': indices,
            'vertexShader': get_vertex_shader(),
            'fragmentShader': get_fragment_shader()
        }
        response_data.append(data)
    
    return jsonify(response_data)


@db_routes.route('/get_spectral_db', methods=['GET'])
def get_spectral_db():

    data = read_csv("res/Spectral Sensitivity Database.csv")

    return jsonify({
        'data': data
    })

@teapot_routes.route('/get_teapot_data', methods=['GET'])  # Corrected syntax
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

