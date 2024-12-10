from collections import defaultdict
from typing import List
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import pandas as pd

from chromalab.observer import Observer
from chromalab.spectra import Spectra, Illuminant

from pigment_template_functions import govardovskii_template, lamb_template
from model_utils import load_obj, calculate_normals, convex_2d_hull_to_vbo_ibo, quads_to_triangles, triangles_to_vertices_indices

# Each OCS may have multiple 3D OCS associated with it. (i.e. 4D has 4 of them)
# Map from index : [OCSs]
active_ocs = defaultdict(list) 

def peaks_to_curves(
        pigment_template_function: str,
        cone_peaks: List[float], 
        active_cones: List[bool], 
        sampling_wavelengths: List[float]
        ):
    """Convert a list of peaks into a list of curves"""
    
    assert len(cone_peaks) == 4
    assert len(active_cones) == 4
    assert len(sampling_wavelengths) > 0
    assert all(isinstance(x, (int, float)) for x in cone_peaks)           # every element is a number
    assert all(isinstance(x, bool) for x in active_cones)                 # every element is a boolean
    assert all(isinstance(x, (int, float)) for x in sampling_wavelengths) # every element is a number

    # get list of only active peaks and sort
    activePeaks = [
        peak for peak in cone_peaks if active_cones[cone_peaks.index(peak)]]
    activePeaks.sort()

    curves = []
    for peak in activePeaks:
        if pigment_template_function == "Govardovskii":
            spectral_sensitivity = govardovskii_template(
                sampling_wavelengths, 
                peak, 
                A1_proportion=100,
                omit_beta_band=True
                )
        elif pigment_template_function == "Lamb":
            spectral_sensitivity = lamb_template(
                sampling_wavelengths, 
                peak, 
                )
        else:   # default to Govardovskii
            spectral_sensitivity = govardovskii_template(
                sampling_wavelengths, 
                peak, 
                A1_proportion=100,
                omit_beta_band=True
                )
        curves.append(spectral_sensitivity)

    return curves

# code shamelessly from https://github.com/chromalab/chromalab/blob/main/chromalab/max_basis.py

class MaxBasis:
    dim4SampleConst = 10
    dim3SampleConst = 2
    
    def __init__(self, observer, verbose=False) -> None:
        self.verbose = verbose
        self.observer = observer
        self.wavelengths = observer.wavelengths
        self.matrix = observer.get_normalized_sensor_matrix()
        self.dimension = observer.dimension
        self.step_size = self.observer.wavelengths[1] - self.observer.wavelengths[0]
        self.dim_sample_const = self.dim4SampleConst if self.dimension == 4 else self.dim3SampleConst

        self.__findMaxCutpoints()
        
    def __computeVolume(self, wavelengths):
        # wavelengths = [matrix.wavelengths[idx] for idx in indices]
        transitions = self.getCutpointTransitions(wavelengths)
        cone_vals = np.array([np.dot(self.matrix, Spectra.from_transitions(x, 1 if i == 0 else 0, self.wavelengths).data) for i, x in enumerate(transitions)])
        vol = np.abs(np.linalg.det(cone_vals))
        return vol

    def __findMaxCutpoints(self, rng=None):
        if self.dimension == 2:
            X = np.arange(self.observer.wavelengths[0] + self.step_size,
                        self.observer.wavelengths[-1] - self.step_size,
                        self.step_size)
            
            # Compute volumes in a vectorized manner
            Zidx = np.array([self.__computeVolume([wavelength]) for wavelength in tqdm(X, disable=not self.verbose)])
            
            maxvol = np.max(Zidx)
            idx = np.argmax(Zidx)
            self.cutpoints = [X[idx], Zidx[idx]]
            self.listvol = [X, Zidx]
            return self.cutpoints

        elif self.dimension == 3:
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size,
                            self.observer.wavelengths[-1] - self.step_size,
                            self.step_size)
                Y = X.copy()  # Same range as X
            else:
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)

            # Precompute meshgrid indices
            Xidx, Yidx = np.meshgrid(X, Y, indexing='ij')

            # Parallel computation of Zidx
            def compute_volume(i, j):
                if i <= j:
                    wavelengths = sorted([X[i], Y[j]])
                    return self.__computeVolume(wavelengths)
                return -np.inf

            Zidx = np.array([[compute_volume(i, j) for j in range(len(Y))] for i in tqdm(range(len(X)), disable=not self.verbose)])
            
            maxvol = np.max(Zidx)
            idx = np.unravel_index(np.argmax(Zidx), Zidx.shape)
            self.cutpoints = [Xidx[idx], Yidx[idx], Zidx[idx]]
            self.listvol = [Xidx, Yidx, Zidx]
            return self.cutpoints

        elif self.dimension == 4:
            if not rng:
                X = np.arange(self.observer.wavelengths[0] + self.step_size,
                            self.observer.wavelengths[-1] - self.step_size,
                            self.step_size)
                Y = X.copy()
                W = X.copy()
            else:
                X = np.arange(rng[0][0], rng[0][1], self.step_size)
                Y = np.arange(rng[1][0], rng[1][1], self.step_size)
                W = np.arange(rng[2][0], rng[2][1], self.step_size)

            # Precompute meshgrid indices
            Xidx, Yidx, Widx = np.meshgrid(X, Y, W, indexing='ij')

            # Parallel computation of Zidx
            def compute_volume(i, j, k):
                if i <= j <= k:
                    wavelengths = sorted([X[i], Y[j], W[k]])
                    return self.__computeVolume(wavelengths)
                return -np.inf

            Zidx = np.array([[[compute_volume(i, j, k) for k in range(len(W))]
                            for j in range(len(Y))]
                            for i in tqdm(range(len(X)), disable=not self.verbose)])

            maxvol = np.max(Zidx)
            idx = np.unravel_index(np.argmax(Zidx), Zidx.shape)
            self.cutpoints = [Xidx[idx], Yidx[idx], Widx[idx], Zidx[idx]]
            self.listvol = [Xidx, Yidx, Widx, Zidx]
            return self.cutpoints

        else:
            raise NotImplementedError
    
    def get_cutpoints(self):
        return self.cutpoints
    
    def get_cmf(self):
        return self.maximal_sensors
    
    def getCutpointTransitions(self, wavelengths):
        transitions = [[wavelengths[0]], [wavelengths[len(wavelengths)-1]]]
        transitions += [[wavelengths[i], wavelengths[i+1]] for i in range(len(wavelengths)-1)]
        transitions.sort()
        return transitions

class ObjectColorSolidTrichromat:
    
    IS_HUMAN_TRICHROMAT = True  # to be accurate, this should only be set true for humans
                                # but even though we dont have arbitrary CMF, still looks good for non human case

    def __init__(self, 
                 observer: Observer, 
                 illuminant: Illuminant, 
                 wavelengths: list, 
                 is_max_basis: bool = False, 
                 indices: list=[2,1,0]
                 ):
        
        self.rgbcmf = self.loadciergb(wavelengths) # shape (n, 3)
        self.observer = observer # instance of Observer (from chromalab)
        self.illuminant = illuminant # instance of Illuminant (from chromalab)
        self.wavelengths = wavelengths # shape n
        self.coneresponses = np.vstack((self.observer.sensors[indices[0]].data, 
                                        self.observer.sensors[indices[1]].data, 
                                        self.observer.sensors[indices[2]].data
                                        )) # shape (3, n)
        self.numreceptors = self.coneresponses.shape[0]
        self.vertices = self.computeVertexBuffer() # shape (n + 1, n, 3) as described in paul centore paper
        
        self.max_basis_vertices = self.vertices
        self.is_max_basis = is_max_basis
        self.transformation = np.eye(3)
        self.edges = np.array([])  # create edges for the first time on the first slice
        if is_max_basis:
            self.transformation = self.computeMaxBasisTransformation()
            self.max_basis_vertices = np.matmul(self.vertices, self.transformation.T)
        
        self.faces, self.facecolors = self.computeFacesAndColorsBuffer() # faces: shape (n * (n - 1), 4, 3), face_colors: shape((n * (n - 1), 3))

    def computeVertexBuffer(self):
        n = len(self.wavelengths)
        points = np.copy(self.coneresponses).T
        vertices = np.zeros((n + 1, n, 3))
        for i in range(1, n + 1):
            for j in range(n):
                vertices[i, j] = vertices[i - 1, j] + points[(i + j - 1) % n]
        # normalize all vertices so that whitepoint is (1,1,1); this is based on jessicas paper
        vertices[:,:,0] = vertices[:,:,0] / np.max(vertices[:,:,0])
        vertices[:,:,1] = vertices[:,:,1] / np.max(vertices[:,:,1])
        vertices[:,:,2] = vertices[:,:,2] / np.max(vertices[:,:,2])
        return vertices
    
    def compute_edges(self):
        # Builds up the set of edges from the diagram in (8).
        # There are 2 * (n ** 2) edges in total where n is the number of generating vectors.
        n = len(self.wavelengths)
        self.edges = np.zeros((2 * (n ** 2), 2, 3))
        index = 0
        for j in range(n):
            # Vertical edges.
            for i in range(n):
                self.edges[index][0] = self.max_basis_vertices[i][j]
                self.edges[index][1] = self.max_basis_vertices[i + 1][j]
                index += 1
                
            # Diagonal edges.
            for i in range(1, n + 1):
                self.edges[index][0] = self.max_basis_vertices[i][j]
                self.edges[index][1] = self.max_basis_vertices[i - 1][(j + 1) % n]
                index += 1

    def computeMaxBasisTransformation(self):
        max_basis = MaxBasis(self.observer)
        cutpoints = max_basis.get_cutpoints()
        cutpoint_1 = cutpoints[0]
        cutpoint_2 = cutpoints[1]

        index_1 = None
        index_2 = None
        for i, wavelength in enumerate(self.wavelengths):
            if index_1 is None and wavelength > cutpoint_1:
                index_1 = i
            if index_2 is None and wavelength >= cutpoint_2:
                index_2 = i
                break

        # We calculate the vectors p1, p2 and p3 as shown in the paper.
        # We "project the partition into the cone response basis" by summing up all the lms_responses within each partition.
        # Note that our earlier calculations for lms_responses includes the illuminant already.
        lms_responses = self.coneresponses * self.illuminant.data
        p1 = np.sum(lms_responses[:, :index_1], axis=1).reshape((3, 1))
        p2 = np.sum(lms_responses[:, index_1:index_2], axis=1).reshape((3, 1))
        p3 = np.sum(lms_responses[:, index_2:], axis=1).reshape((3, 1))

        # We then create a transformation matrix that maps p1 to (1, 0, 0), p2 to (0, 1, 0) and p3 to (1, 0, 0).
        # p1, p2 and p3 correspond to the ideal R, G, B points on our object color solid, 
        # and we are mapping them onto the R, G, B points on the RGB cube.
        # We are essentially "stretching" our object color solid so that it approximates the RGB cube.
        transformation_matrix = np.linalg.inv(np.hstack((p1, p2, p3)))
        return transformation_matrix

    def getCutpointIndices(self, cutpoint_1, cutpoint_2):
        index_1 = next(i for i, wl in enumerate(self.wavelengths) if wl > cutpoint_1)
        index_2 = next(i for i, wl in enumerate(self.wavelengths) if wl >= cutpoint_2)
        return index_1, index_2

    def computeFacesAndColorsBuffer(self):
        n = len(self.wavelengths)
        faces = np.zeros((n * (n - 1), 4, 3))
        face_colors = np.zeros((n * (n - 1), 3))
        for i in tqdm(range(1, n)):
            for j in range(n):
                faces[((i - 1) * n) + j, 0] = self.max_basis_vertices[i, j]
                faces[((i - 1) * n) + j, 1] = self.max_basis_vertices[i - 1, (j + 1) % n]
                faces[((i - 1) * n) + j, 2] = self.max_basis_vertices[i, (j + 1) % n]
                faces[((i - 1) * n) + j, 3] = self.max_basis_vertices[i + 1, j]
                
                reflectance = np.zeros(n)
                for k in range(i):
                    reflectance[(j + k) % n] = 1
                face_colors[(i - 1) * n + j] = self.coneresponse2rgb(self.vertices[i, j], reflectance)

        # min max normalize
        face_colors[:,0] = (face_colors[:,0] - np.min(face_colors[:,0]))  / (np.max(face_colors[:,0]) - np.min(face_colors[:,0])) 
        face_colors[:,1] = (face_colors[:,1] - np.min(face_colors[:,1]))  / (np.max(face_colors[:,1]) - np.min(face_colors[:,1])) 
        face_colors[:,2] = (face_colors[:,2] - np.min(face_colors[:,2]))  / (np.max(face_colors[:,2]) - np.min(face_colors[:,2])) 
        
        return faces, face_colors

    def coneresponse2rgb(self, lmscoord, reflectance):

        if not ObjectColorSolidTrichromat.IS_HUMAN_TRICHROMAT or reflectance is None:
            objectcolorlms = self.coneresponses * self.illuminant.data * np.vstack((reflectance,reflectance,reflectance))
            A = objectcolorlms.T @ np.linalg.inv(self.coneresponses @ self.coneresponses.T) # moore penrose
            return A @ lmscoord

        objectcolorlms = self.coneresponses * self.illuminant.data * np.vstack((reflectance,reflectance,reflectance))
        A = self.rgbcmf @ objectcolorlms.T @ np.linalg.inv(self.coneresponses @ self.coneresponses.T) # moore penrose
        return A @ lmscoord


    def loadciergb(self, wavelengths):
        if not ObjectColorSolidTrichromat.IS_HUMAN_TRICHROMAT:
            return None

        # Load CIE RGB data from file
        df = pd.read_csv('res/sbrgb2.csv')  # Assuming the first column is wavelengths

        # Extract wavelength and RGB data from dataframe
        ciewavelengths = df.iloc[:, 0].values  # First column
        ciergb = df.iloc[:, 1:].values.T      # Remaining columns (transpose for easier indexing)

        # Create a new dataframe with the same length as wavelengths
        extended_rgb = np.zeros((3, len(wavelengths)))  # Initialize new RGB array
        for i in range(3):  # Interpolate for each channel (R, G, B)
            extended_rgb[i] = np.interp(wavelengths, ciewavelengths, ciergb[i], left=0, right=0)

        # Replace ciewavelengths and ciergb with the interpolated data
        ciewavelengths = np.array(wavelengths)  # Update ciewavelengths to match wavelengths
        ciergb = extended_rgb                   # Update ciergb to the interpolated values

        # Continue with original computation using the now aligned `ciewavelengths` and `ciergb`
        ciergbs = np.zeros((3, len(wavelengths)))
        curridx = 0
        for i, wavelength in enumerate(wavelengths[:-2]):
            if ciewavelengths[curridx] + 5 < wavelength:
                curridx += 1
            if wavelength < ciewavelengths[0] or wavelength > ciewavelengths[-1]:
                ciergbs[:, i] = np.array([0, 0, 0])
            else:
                wavelength1 = ciewavelengths[curridx]
                rgb1 = ciergb[:, curridx]
                rgb2 = ciergb[:, curridx + 1]
                percent = (wavelength - wavelength1) / 5
                ciergbs[:, i] = rgb2 * percent + rgb1 * (1 - percent)

        return ciergbs

# turns np arrays to lists and lists to lists
# TODO: need some type safety please instead of using this
def to_list(l):
    return l.tolist() if isinstance(l, np.ndarray) else l

@dataclass
class OCSContext4D:
    """
    Contexts for generating a single OCS geometry
    """
    min_sample_wavelength: int
    max_sample_wavelength: int
    sample_per_wavelength: float
    peak_wavelengths: List[int]  # 4 peak wavelengths
    active_cones: List[bool]  # 4 active cones
    is_max_basis: bool
    pigment_template_function: str
    idx: int  # The index associated with this OCS

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
    curves: List[List[float]]  # l, m, s, q responses

def center_4d_ocs_geometry(ocs_geometry: OCSGeometry4D) -> OCSGeometry4D:
    """
    Given a geometry, center all vertices around the origin.
    We derive the origin by taking the average of all vertices.
    """
    
    vertices = np.array(ocs_geometry.vertices)
    avg = np.mean(vertices, axis=0)
    centered_vertices = vertices - avg

    return OCSGeometry4D(
        vertices=centered_vertices.tolist(),
        indices=ocs_geometry.indices,
        colors=ocs_geometry.colors,
        normals=ocs_geometry.normals,
        wavelengths=ocs_geometry.wavelengths,
        curves=ocs_geometry.curves
    )

def get_4d_ocs_geometry(ocs_ctx: OCSContext4D) -> OCSGeometry4D:
    """
    Generate a single OCS geometry based on the given context
    """
    print("==== Generating 4D OCS Geometry ====")
    # derive min, max wavelengths and the sample resolution
    assert len(ocs_ctx.peak_wavelengths) == 4
    assert len(ocs_ctx.active_cones) == 4

    print("===== Parameters =====")
    print("Wavelength Bounds: ", ocs_ctx.min_sample_wavelength,
          ocs_ctx.max_sample_wavelength)
    print("Peak Wavelengths: ", ocs_ctx.peak_wavelengths)
    print("Active Cones: ", ocs_ctx.active_cones)
    print("Pigment Template Function: ", ocs_ctx.pigment_template_function)

    wavelength_sample_resolution: int = int(ocs_ctx.sample_per_wavelength *
                                            (ocs_ctx.max_sample_wavelength -
                                             ocs_ctx.min_sample_wavelength + 1))

    print("sample per wavelength: ", ocs_ctx.sample_per_wavelength)

    wavelengths: list[int] = np.linspace(
        ocs_ctx.min_sample_wavelength, 
        ocs_ctx.max_sample_wavelength, 
        num=wavelength_sample_resolution)  # type: ignore

    curves = peaks_to_curves(ocs_ctx.pigment_template_function, 
                             ocs_ctx.peak_wavelengths, 
                             ocs_ctx.active_cones, 
                             wavelengths
                            )

    if len(curves) == 4:
        print("NOT IMPLEMENTED")
        vertices, indices, colors = generate_4D_OCS(curves, wavelengths, ocs_ctx.is_max_basis)
    elif len(curves) == 3:
        vertices, indices, colors = generate_3D_OCS(curves, wavelengths, ocs_ctx.is_max_basis, ocs_ctx.idx)
    elif len(curves) == 2:
        vertices, indices, colors = generate_2D_OCS(curves, wavelengths, ocs_ctx.is_max_basis)

    normals = calculate_normals(vertices, indices)

    assert len(vertices) == len(colors)

    # now that we've generated the OCS, pad out the number of curves to 4
    for i in range(len(curves), 4):
        curves.append(np.zeros_like(curves[0]))

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

def generate_2D_OCS(curves: List[List[float]], wavelengths: List[float], is_max_basis: bool, ocs_idx: int = 0):
    assert len(curves) == 2
    assert len(wavelengths) == len(curves[0])

    S, M = curves

    n = len(wavelengths)
    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)
    dichrom_responses = np.vstack((S, M))  # Combine responses into a 2D array (2 x n)
    points = np.copy(dichrom_responses).T # generating vectors

    # generate a list of optimal color vertices from the locus 
    vertices = np.zeros((n * 2, 2))
    vertex_colors = np.zeros((n * 2, 3))

    for i in range(1,n + 1):
        # fills up vertices from 1 to n
        vertices[i] = vertices[i - 1] + points[i - 1]
        
        # fill in the reflectance
        reflectance_data = np.zeros(n)
        for j in range(i):
            # from black to white point
            reflectance_data[j] = 1 # give 1 to all generating vectors associating with the particular position
        reflectance = Spectra(wavelengths=wavelengths, data=reflectance_data)
        vertex_colors[i] = reflectance.to_rgb(illuminant)

    for i in range(1, n):
        # fills up vertices from n+1 to 2n
        vertices[i + n] = vertices[i + n - 1] - points[i - 1]
        reflectance_data = np.zeros(n)
        for j in reversed(range(i, n)):
            
            # from black to white point
            reflectance_data[j] = 1 # give 1 to all generating vectors associating with the particular position
        reflectance = Spectra(wavelengths=wavelengths, data=reflectance_data)
        vertex_colors[i + n] = reflectance.to_rgb(illuminant)

    vertices, vertex_colors, indices = convex_2d_hull_to_vbo_ibo(vertices, vertex_colors)

    return vertices, indices, vertex_colors

def generate_3D_OCS(curves: List[List[float]], wavelengths: List[float], is_max_basis: bool, ocs_idx: int = 0):
    
    assert len(curves) == 3
    assert len(wavelengths) == len(curves[0])

    S, M, L = curves

    s1 = Spectra(data=S, wavelengths=wavelengths)
    s2 = Spectra(data=M, wavelengths=wavelengths)
    s3 = Spectra(data=L, wavelengths=wavelengths)
    observer = Observer([s1, s2, s3])

    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)

    ocs = ObjectColorSolidTrichromat(observer, illuminant, wavelengths, is_max_basis)
    active_ocs[ocs_idx].append(ocs)  # Update the global dictionary of active OCSs

    tris = quads_to_triangles(ocs.faces, invert_winding=is_max_basis)
    vertices, indices = triangles_to_vertices_indices(tris)

    # Normalize vertices to [0, 1] range
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    normalized_vertices =  (vertices - min_coords) / range_coords

    vertex_colors = ocs.facecolors

    # Ensure the colors array has enough values to match the number of vertices
    if len(vertex_colors) < len(vertices):
        num_missing_colors = len(vertices) - len(vertex_colors)
        missing_colors = np.array([[1.0, 1.0, 1.0]] * num_missing_colors)  # Create a NumPy array of white [R, G, B] for missing colors
        vertex_colors = np.append(vertex_colors, missing_colors, axis=0)  # Append along the correct axis
    
    return normalized_vertices, indices, vertex_colors


def generate_4D_OCS(curves: List[List[float]], wavelengths: List[float], is_max_basis: bool, ocs_idx: int):

    assert len(curves) == 4
    assert len(wavelengths) == len(curves[0])

    S, M, L, Q = curves

    QMS = [Q, M, S] # no L
    QLS = [Q, L, S] # no M
    QLM = [Q, L, M] # no S

    print("NOT IMPLEMENTED")

    # lol
    #return generate_3D_OCS(QMS, wavelengths, is_max_basis), 
    #        generate_3D_OCS(QLS, wavelengths, is_max_basis), 
    #        generate_3D_OCS(QLM, wavelengths, is_max_basis)

