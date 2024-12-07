import numpy as np
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from model_utils import load_obj, calculate_normals
from ocs_generator import generate_OCS
from shaders import get_vertex_shader, get_fragment_shader
from ocs_slice import get_y_slice
from govardovskii import govardovskii_template
from spectralDBLoader import read_csv

from typing import List
from dataclasses import dataclass

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


@dataclass
class OCSContext4D:
    """
    Contexts for generating a single OCS geometry
    """
    min_sample_wavelength: int
    max_sample_wavelength: int
    sample_per_wavelength: int
    peak_wavelengths: List[int] # 4 peak wavelengths
    active_cones: List[bool] # 4 active cones
    is_max_basis: bool
    

@dataclass
class OCSGeometry4D:
    """
    Geometry of a single OCS
    """
    vertices: List[float]
    indices: List[int]
    colors: List[float]
    normals: List[float]
    wavelengths: List[int]
    curves: List[List[float]] # l, m, s, q responses

# generate OCS data for a single color solid
def get_single_ocs_geometry(ocs_ctx: OCSContext4D) -> OCSGeometry4D:
    # derive min, max wavelengths and the sample resolution
    assert len(ocs_ctx.peak_wavelengths) == 4
    assert len(ocs_ctx.active_cones) == 4

    print("===== Parameters =====")
    print("Wavelength Bouunds: ", ocs_ctx.min_sample_wavelength, ocs_ctx.max_sample_wavelength)
    print("Peak Wavelengths: ", ocs_ctx.peak_wavelengths)
    print("Active Cones: ", ocs_ctx.active_cones)

    wavelength_sample_resolution: int = ocs_ctx.sample_per_wavelength * (ocs_ctx.max_sample_wavelength - ocs_ctx.min_sample_wavelength + 1)

    wavelengths: List[int] = np.linspace(ocs_ctx.min_sample_wavelength, ocs_ctx.max_sample_wavelength, num=wavelength_sample_resolution)

    curves = []
    for peak, is_active in zip(ocs_ctx.peak_wavelengths, ocs_ctx.active_cones):
        if is_active:
            curve = govardovskii_template(wavelengths=wavelengths,
                                        lambda_max=peak,
                                        A1_proportion=100,
                                        omit_beta_band=True)
        else:
            curve = np.zeros(wavelength_sample_resolution) + 1e-6
        curves.append(curve)

    assert len(curves) == 4

    vertices, indices, colors = generate_OCS2(curves, wavelengths, ocs_ctx.is_max_basis)
    normals = calculate_normals(vertices, indices)
    
    assert len(vertices) == len(colors)


    # generate OCS data for a single color solid
    ret: OCSGeometry4D = OCSGeometry4D(
            vertices=to_list(vertices),
            indices=to_list(indices),
            normals=to_list(normals),
            colors=to_list(colors),
            wavelengths=to_list(wavelengths),
            curves=[to_list(curve) for curve in curves]
        )

    return ret

@ocs_routes.route('/get_ocs_data', methods=['GET'])
def get_ocs_data():
    """Generate object color solid geometry, colors, normals, and return shaders"""
    
    min_wavelength = int(request.args.get('minWavelength', 390))
    max_wavelength = int(request.args.get('maxWavelength', 700))

    # default values should yield garbage; we always want request to work
    #wavelength_sample_resolution = int(request.args.get('wavelengthSampleResolution', 5))
    is_max_basis = request.args.get('isMaxBasis', False) == "true"
    #omit_beta_band = request.args.get('omitBetaBand', True) == "true"
    peakWavelength1 = int(request.args.get('peakWavelength1', 500))
    peakWavelength2 = int(request.args.get('peakWavelength2', 510))
    peakWavelength3 = int(request.args.get('peakWavelength3', 520))
    peakWavelength4 = int(request.args.get('peakWavelength4', 530))   # not used currently
    isCone1Active = request.args.get('isCone1Active', False) == "true"
    isCone2Active = request.args.get('isCone2Active', False) == "true"
    isCone3Active = request.args.get('isCone3Active', False) == "true"
    isCone4Active = request.args.get('isCone4Active', False) == "true"

    # we dont need to take input for these anymore
    omit_beta_band = True   # always omit beta band
    wavelength_sample_resolution = 40   # code is fast enough where it can do 50 in an instant and 30 is more than enough

    print("===== Parameters =====")
    print("Wavelength Bouunds: ", min_wavelength, max_wavelength)
    print("Peak Wavelengths: ", peakWavelength1, peakWavelength2, peakWavelength3, peakWavelength4)
    print("Active Cones: ", isCone1Active, isCone2Active, isCone3Active, isCone4Active)
    print("Omit Beta Band: ", omit_beta_band)
    print("Is Max Basis: ", is_max_basis)
    print("Wavelength Sample Resolution: ", wavelength_sample_resolution)
    print("======================")

    peaks = [peakWavelength1, peakWavelength2, peakWavelength3, peakWavelength4]
    activeCones = [isCone1Active, isCone2Active, isCone3Active, isCone4Active]

    # get list of only active peaks and sort
    activePeaks = [peak for peak in peaks if activeCones[peaks.index(peak)]]
    activePeaks.sort()

    wavelengths = np.linspace(min_wavelength, max_wavelength, num=wavelength_sample_resolution)

    # curves is [S, M, L, Q]
    # cheap hack to allow us to use 3D OCS code for lower dimensions
    # add curves for all active peaks, 
    # then for inactive peaks (code breaks when zero response function is passed in)
    # add curves for the last active peak + epsilon which acts as a linearly dependant function
    # and keep dimentions virutally the same
    curves = [
    govardovskii_template(
        wavelengths=wavelengths,
        lambda_max=activePeaks[i] if i < len(activePeaks) else activePeaks[-1] + 1e-6,
        A1_proportion=100,
        omit_beta_band=omit_beta_band
    )
    for i in range(len(peaks))
    ]

    vertices, indices, colors = generate_OCS(curves, wavelengths, is_max_basis)
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
        'l_response': to_list(curves[2]),
        'q_response': to_list(curves[3])
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
    
