import numpy as np
from math_utils import lms_to_rgb, find_orthogonal_vectors
from scipy.spatial import ConvexHull, Delaunay
from ocs_generator import to_list

# Note that all calculations are done based on the all OCS being centered at the origin (as done in the generation code)
def get_ostwald_slice(ocs: "ObjectColorSolid", a: float, b: float, c: float, d: float):
    # Iterates through each edge and checks whether it intersects the plane.
    # If it does, store the coordinates of the intersection and the XYZ color coordinates of the intersection.
    # ax + by + cz + d = 0
    intersection_points = []
    
    if ocs.edges.shape[0] == 0:
        ocs.compute_edges()

    for i in range(ocs.edges.shape[0]):
        p1 = ocs.edges[i][0]
        p2 = ocs.edges[i][1]
        v = p2 - p1
        denom = a * v[0] + b * v[1] + c * v[2]
        if abs(denom) < 1e-6:    # Edge is parallel to plane.
            continue
        t = -(d + a * p1[0] + b * p1[1] + c * p1[2]) / denom
        if 0 <= t <= 1:
            i_point = ((1 - t) * p1) + (t * p2)
            intersection_points.append(i_point)
    assert len(intersection_points) > 0, "No intersection points"
    intersection_rgb = [lms_to_rgb(i_point, np.linalg.inv(ocs.transformation)) for i_point in intersection_points]

    # In the end, we obtain an array of points along the boundary of the intersection and their RGB values.
    intersection_points = np.array(intersection_points)
    intersection_rgb = np.array(intersection_rgb)
    
    # We then connect these vertices together to form the outline of the intersection.
    # We do this by first projecting the points onto the 2D plane and then finding the convex hull of those points.
    # ConvexHull will order the points CCW
    vec_a, vec_b = find_orthogonal_vectors(np.array([a, b, c]))
    projection_matrix = np.vstack((vec_a, vec_b))
    projected_intersection_points = intersection_points @ projection_matrix.T
    hull = ConvexHull(projected_intersection_points) 
    hull_vertices = projected_intersection_points[hull.vertices]
    boundary_points = np.hstack(
        (hull_vertices,
        np.zeros(hull_vertices.shape[0]).reshape(-1, 1)) # have this lie flat on the XY plane
    )
    boundary_colors = intersection_rgb[hull.vertices]
    
    delaunay = Delaunay(hull_vertices)
    indices = delaunay.simplices

    print(boundary_points.shape)
    print(indices.shape)
    return to_list(boundary_points), to_list(boundary_colors), to_list(indices)

# def plane_vertices_to_indices(vertices: 'np.NDArray'): 
#     indices = []