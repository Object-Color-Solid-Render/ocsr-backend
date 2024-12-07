import numpy as np
from math_utils import lms_to_rgb, find_orthogonal_vectors
from scipy.spatial import ConvexHull, Delaunay


# TODO: Generalize this to non-human responses. 
# TODO: Pass in plane equation (a, b, c, d)!
def get_y_slice(ocs: "ObjectColorSolid", y: float):
    # Iterates through each edge and checks whether it intersects the plane.
    # If it does, store the coordinates of the intersection and the XYZ color coordinates of the intersection.
    intersection_points = []
    # ax + by + cz + d = 0
    a = 0
    b = 0
    c = 1
    d = y
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

    boundary_points = np.hstack(
        projected_intersection_points[hull.vertices], 
        np.zeros(projected_intersection_points.shape[0]) # have this lie flat on the XY plane
    )
    boundary_colors = intersection_rgb[hull.vertices]
    
    delaunay = Delaunay(projected_intersection_points)
    indices = delaunay.simplices()

    return boundary_points.tolist(), boundary_colors.to_list(), indices.tolist()

# def plane_vertices_to_indices(vertices: 'np.NDArray'): 
#     indices = []