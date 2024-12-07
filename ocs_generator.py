from tqdm import tqdm
import numpy as np
import pandas as pd

from chromalab.observer import Observer
from chromalab.spectra import Spectra, Illuminant

from govardovskii import govardovskii_template

def peaks_to_curves(
        peaks: list, 
        sample_frequency: int, 
        min_wavelength: int, 
        max_wavelength: int, 
        ommit_beta_band: bool
        ):
    """Convert a list of peaks into a list of curves, each with a start and end wavelength"""
    
    assert len(peaks) == 4
    assert sample_frequency > 0
    assert min_wavelength < max_wavelength
    assert min_wavelength > 0
    assert type(peaks[0]) == int
    assert type(peaks[1]) == int
    assert type(peaks[2]) == int
    assert type(peaks[3]) == int
    
    sampling_wavelengths = np.linspace(min_wavelength, max_wavelength, sample_frequency)
    
    curves = []
    for peak in peaks:
        
        spectral_sensitivity = govardovskii_template(
            sampling_wavelengths, 
            peak, 
            ommit_beta_band=ommit_beta_band
            )
        
        curves.append(spectral_sensitivity)

    return curves

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
        self.is_max_basis = is_max_basis
        if is_max_basis:
            self.applyMaxBasisTransformation()
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

    def applyMaxBasisTransformation(self):
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
        vertices_transformed = np.matmul(self.vertices, transformation_matrix.T)
        self.vertices = vertices_transformed

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
                faces[((i - 1) * n) + j, 0] = self.vertices[i, j]
                faces[((i - 1) * n) + j, 1] = self.vertices[i - 1, (j + 1) % n]
                faces[((i - 1) * n) + j, 2] = self.vertices[i, (j + 1) % n]
                faces[((i - 1) * n) + j, 3] = self.vertices[i + 1, j]
                
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



def generate_OCS(curves: list, wavelengths: list, is_max_basis: bool):
    
    assert len(curves) == 4
    assert len(wavelengths) == len(curves[0])

    s_response, m_response, l_response, _ = curves

    s1 = Spectra(data=s_response, wavelengths=wavelengths)
    s2 = Spectra(data=m_response, wavelengths=wavelengths)
    s3 = Spectra(data=l_response, wavelengths=wavelengths)
    observer = Observer([s1, s2, s3])

    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)

    ocs = ObjectColorSolidTrichromat(observer, illuminant, wavelengths, is_max_basis)

    tris = quads_to_triangles(ocs.faces, invert_winding=is_max_basis)
    vertices, indices = triangles_to_vertices_indices(tris)

    # Normalize vertices to [0, 1] range
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    range_coords = max_coords - min_coords
    normalized_vertices =  (vertices - min_coords) / range_coords

    face_colors = ocs.facecolors

    # Ensure the colors array has enough values to match the number of vertices
    if len(face_colors) < len(vertices):
        num_missing_colors = len(vertices) - len(face_colors)
        missing_colors = np.array([[1.0, 1.0, 1.0]] * num_missing_colors)  # Create a NumPy array of white [R, G, B] for missing colors
        
        # Use np.append to concatenate the arrays
        face_colors = np.append(face_colors, missing_colors, axis=0)  # Append along the correct axis

    return normalized_vertices.tolist(), indices.tolist(), face_colors.tolist()



def generate_OCS_old(curves: list, wavelengths: list, max_basis: bool):
    
    assert len(curves) == 4
    assert len(wavelengths) == len(curves[0])

    n = len(wavelengths)
    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)

    s_response, m_response, l_response, _ = curves

    # Each point has an indicator reflectance function where R = 1 at a single wavelength and 0 elsewhere.
    # These points can be thought of as vectors which form a (linearly dependent) basis.
    # The Minkowski sum of these vectors span the object color solid.
    # Each point in the solid can be represented as some (non-unique) linear combination of these vectors.
    # This represents equations (9), (10), (11), (12), (13).
    valid_responses = [curve for curve in [s_response, m_response, l_response] if not np.all(curve == 0)]
    lms_responses = np.vstack(valid_responses) * illuminant.data

    points = np.copy(lms_responses).T

    # As shown in Centore's paper, these vertices form the shape of the solid.
    # This represents the matrix in (7).
    dim = len(valid_responses)
    vertices = np.zeros((n + 1, n, dim))
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

    if (max_basis):
        # Uses ideas from Jessica's paper, on chapter 3.2 The Max Basis.
        # Compute optimal cutpoints dynamically using Jessica's code.
        s1 = Spectra(data=s_response, wavelengths=wavelengths)
        s2 = Spectra(data=m_response, wavelengths=wavelengths)
        s3 = Spectra(data=l_response, wavelengths=wavelengths)
        observer = Observer([s1, s2, s3])
        max_basis = MaxBasis(observer)
        cutpoints = max_basis.get_cutpoints()
        cutpoint_1 = cutpoints[0]
        cutpoint_2 = cutpoints[1]

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

    return normalized_vertices.tolist(), indices.tolist(), face_colors.tolist()
