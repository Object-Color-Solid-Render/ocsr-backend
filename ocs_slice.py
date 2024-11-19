from chromalab.observer import Observer
from chromalab.spectra import Spectra, Illuminant
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from colour import sd_to_XYZ, XYZ_to_xy, XYZ_to_sRGB


# TODO: Generalize this to non-human responses. THus using the vertices and colors provided. Currently the args aren't doing anything yet
def get_y_slice(vertices, colors, wavelengths, y):
    wavelengths = np.arange(390, 701, 5)
    n = len(wavelengths)
    standard_trichromat = Observer.trichromat(wavelengths)
    illuminant = Illuminant.get("D65").interpolate_values(wavelengths)
    illuminant_colour = illuminant.to_colour()
    chromaticity_coord = XYZ_to_xy(sd_to_XYZ(illuminant_colour) / 100)

    # ------------- May not need later start

    # Each point has an indicator reflectance function where R = 1 at a single wavelength and 0 elsewhere.
    # These points can be thought of as vectors which form a (linearly dependent) basis.
    # The Minkowski sum of these vectors span the object color solid.
    # Each point in the solid can be represented as some (non-unique) linear combination of these vectors.
    # This represents equations (9), (10), (11), (12), (13).
    lms_responses = np.vstack((standard_trichromat.sensors[0].data, 
                            standard_trichromat.sensors[1].data, 
                            standard_trichromat.sensors[2].data)) * illuminant.data

    # Uses ideas from Jessica's paper, on chapter 3.2 The Max Basis.
    # We use the cutpoints that Jessica shows to be optimal for the trichromatic case.
    cutpoint_1 = 487
    cutpoint_2 = 573
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
    points = np.matmul(transformation_matrix, lms_responses).T


    # -------------

    # As shown in Centore's paper, these vertices form the shape of the solid.
    # This represents the matrix in (7).
    vertices = np.zeros((n + 1, n, 3))
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

    # -------

    vertices = np.zeros((n + 1, n, 3))
    vertices_xyz = np.zeros((n + 1, n, 3))
    for i in tqdm(range(1, n + 1)):
        for j in range(n):
            reflectance_data = np.zeros(n)
            vertices[i, j] = vertices[i - 1, j] + points[(i + j - 1) % n]
            for k in range(i):
                reflectance_data[(j + k) % n] = 1
            reflectance = Spectra(wavelengths=wavelengths, data=reflectance_data)
            vertices_xyz[i, j] = sd_to_XYZ(reflectance.to_colour(), illuminant=illuminant_colour) / 100


    # ------------- May not need later

    edges = np.zeros((2 * (n ** 2), 2, 3))
    edges_xyz = np.zeros((2 * (n ** 2), 2, 3))
    index = 0
    for j in range(n):
        # Vertical edges.
        for i in range(n):
            edges[index][0] = vertices[i][j]
            edges[index][1] = vertices[i + 1][j]
            edges_xyz[index][0] = vertices_xyz[i][j]
            edges_xyz[index][1] = vertices_xyz[i + 1][j]
            index += 1
            
        # Diagonal edges.
        for i in range(1, n + 1):
            edges[index][0] = vertices[i][j]
            edges[index][1] = vertices[i - 1][(j + 1) % n]
            edges_xyz[index][0] = vertices_xyz[i][j]
            edges_xyz[index][1] = vertices_xyz[i - 1][(j + 1) % n]
            index += 1


    # Up until here and above can be generated right when we render the solid and stored into the session object

    # ---------
    # Here, d will get updated with the value passed into the backend
    intersection_points = []
    intersection_xyz = []
    a, b, c, d = 0, 0, 1, y    # ax + by + cz = d
    for i in range(edges.shape[0]):
        p1 = edges[i][0]
        p2 = edges[i][1]
        v = p2 - p1
        denom = a * v[0] + b * v[1] + c * v[2]
        if abs(denom) < 1e-6:    # Edge is parallel to plane.
            continue
        t = (d - a * p1[0] - b * p1[1] - c * p1[2]) / denom
        if 0 <= t <= 1:
            i_point = ((1 - t) * p1) + (t * p2)
            i_xyz = ((1 - t) * edges_xyz[i][0]) + (t * edges_xyz[i][1])
            intersection_points.append(i_point)
            intersection_xyz.append(i_xyz)
    assert len(intersection_points) > 0, "No intersection points"
    intersection_rgb = []
    for i_xyz in intersection_xyz:
        intersection_rgb.append(np.clip(XYZ_to_sRGB(i_xyz, illuminant=chromaticity_coord), 0, 1))
    intersection_points = np.array(intersection_points)
    intersection_rgb = np.array(intersection_rgb)

    # project the intersection points onto the cut plane
    

    # tris = quads_to_triangles(faces)
    # vertices, indices = triangles_to_vertices_indices(tris)

    return intersection_points.tolist(), intersection_rgb.tolist()