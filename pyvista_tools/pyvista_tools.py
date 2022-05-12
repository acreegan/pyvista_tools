from __future__ import annotations
import itertools
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray, ArrayLike
import pyvista as pv
from tqdm import tqdm
import pymeshfix
from pyvista_tools.geometry_tools import find_sequence, winding_order_agrees_with_normal, compute_normal,\
     dihedral_angle

"""
pyvista_tools is a module that provides tools for working with pyvista objects. This module should contain functions 
that are not necessarily useful on their own, but can be used to support development of pyvista features.
"""


def pyvista_faces_to_2d(faces: ArrayLike) -> NDArray:
    """
    Convert pyvista faces from the native 1d array to a 2d array with one face per row. Padding is trimmed.

    Only works on a list of faces with all the same number of points per face.

    Parameters
    ----------
    faces
        Faces to be reshaped

    Returns
    -------
    2d array of faces
    """
    points_per_face = faces[0]
    faces = faces.reshape(-1, points_per_face+1)
    if not np.all(faces[:, 0] == faces[:, 0]):
        raise ValueError("pyvista_faces_to_2d requires all to be the same shape")

    return faces[:, 1:]


def pyvista_faces_to_1d(faces: NDArray) -> NDArray:
    """
    Convert 2d array of faces to the pyvista native 1d format, inserting the padding.

    Only works on a list of faces with all the same number of points per face.

    Parameters
    ----------
    faces
        Faces to be reshaped

    Returns
    -------
    1d array of faces

    """
    padding = len(faces[0])
    return np.insert(faces, 0, values=padding, axis=1).ravel()


def pyvista_faces_by_dimension(faces: NDArray) -> Dict[int, NDArray]:
    """
    You can also do this by casting to UnstructuredGrid, where the face types are available in a dict

    Parameters
    ----------
    faces

    Returns
    -------

    """
    output = {}
    i = 0
    while i < len(faces):
        # Preceding each face is the "padding" indicating the number of elements in the face
        num_elems = faces[i]
        # Append padding plus each element to the output dict
        if num_elems in output:
            output[num_elems] = np.append(output[num_elems], np.array([faces[i + j] for j in range(num_elems + 1)]))
        else:
            output[num_elems] = np.array([faces[i + j] for j in range(num_elems + 1)])
        # Increment index to the next padding number
        i += num_elems + 1
    return output


def select_shared_faces(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = None) -> Tuple[list, list]:
    """
    Select that faces that two meshes share. Shared faces are determined by selecting the faces that use shared points.

    Parameters
    ----------
    mesh_a
        First mesh
    mesh_b
        Second mesh
    tolerance
        Tolerance for selecting shared points

    Returns
    -------
    Tuple of lists containing the indices of the shared faces in mesh_a and mesh_b

    """
    shared_points_kwargs = {"mesh_a": mesh_a, "mesh_b": mesh_b, "tolerance": tolerance}
    shared_points = select_shared_points(**{k: v for k, v in shared_points_kwargs.items() if v is not None})

    mesh_a_faces = select_faces_using_points(mesh_a, shared_points[0])
    mesh_b_faces = select_faces_using_points(mesh_b, shared_points[1])

    return mesh_a_faces, mesh_b_faces


def select_shared_points(mesh_a: pv.PolyData, mesh_b: pv.PolyData, tolerance: float = 1e-05, progress_bar: bool = False) \
        -> Tuple[List[int], List[int]]:
    """
    Select the points that two meshes share. Points are considered shared if they are within a specified euclidian
    distance from one another.

    Parameters
    ----------
    mesh_a
        First mesh
    mesh_b
        Second mesh
    tolerance
        Maximum euclidian distance between points to consider them shared
    progress_bar

    Returns
    -------
    shared_points
        Tuple containing indices of shared points in mesh_a, and shared points in mesh_b

    """
    shared_points_a = []
    shared_points_b = []
    for i_a, point_a in tqdm(list(enumerate(mesh_a.points)), disable=not progress_bar, desc="Selecting Shared Points"):
        for i_b, point_b in enumerate(mesh_b.points):
            # linalg.norm calculates euclidean distance
            if np.linalg.norm(point_a - point_b) <= tolerance:
                # Need to remember the index of the shared point in both meshes so we can find faces that use it in both
                shared_points_a.append(i_a)
                shared_points_b.append(i_b)

    return shared_points_a, shared_points_b


def select_faces_using_points(mesh: pv.PolyData, points: list) -> List[int]:
    """
    Select all faces in a mesh that contain only the specified points.

    Parameters
    ----------
    mesh:
        Mesh to select from
    points
        The only points in the given mesh that the selected faces may contain

    Returns
    -------
    mesh_faces
        List of indices of the selected faces in the mesh

    """
    mesh_faces = []
    faces = pyvista_faces_to_2d(mesh.faces)
    points_set = set(points)
    for j, face in enumerate(faces):
        # Check if all of the points in each face are in the target points set
        if set(face).issubset(points_set):
            mesh_faces.append(j)

    return mesh_faces


def select_points_used_by_faces(mesh: pv.PolyData, points: List[int] = None, faces: List[int] = None,
                                exclusive: bool = False) -> List[int]:
    """
    Select points used in the faces of a mesh. Optionally specify a subset of points and/or faces to check. When
    exclusive is set to True, selects only points that are not used in the remaining faces.

    Only works on meshes with all the same number of points per face.

    Todo: not crash if you don't specify points (till then you can use list(range(len(mesh.points)))

    Parameters
    ----------
    mesh
        Mesh to select from
    points
        Optional subset of points in mesh to select from
    faces
        Optional subset of faces in mesh to test
    exclusive
        If true, selects only points exclusively used in the specified faces

    Returns
    -------
    List of points used in the specified faces

    """
    mesh_faces_2d = pyvista_faces_to_2d(mesh.faces)
    faces_2d = mesh_faces_2d[faces]
    remaining_faces = np.delete(mesh_faces_2d, faces, 0)

    # Check if any given point exists in the faces to search
    used_in_faces = []
    for point in points:
        if any(point in face for face in faces_2d):
            used_in_faces.append(point)

    # If exclusive is set, remove points that also exist in the remaining faces
    excluded_points = []
    if exclusive:
        for point in used_in_faces:
            if any(point in face for face in remaining_faces):
                excluded_points.append(point)

    used_in_faces = list(set(used_in_faces) - set(excluded_points))

    return used_in_faces


def select_faces_using_edges(mesh: pv.PolyData, edges: pv.PolyData) -> List[int]:
    """
    Extract all the faces of a Pyvista Polydata object that use a given set of edges

    Parameters
    ----------
    mesh
        Mesh to extract edges from
    edges
        Edges to use

    Returns
    -------
    faces_using_edges
        List of faces in mesh that use edges

    """
    mesh = mesh.copy()
    mesh = mesh.merge(edges)

    faces_using_edges = []
    for i, face in enumerate(pyvista_faces_to_2d(mesh.faces)):
        for line in pyvista_faces_to_2d(mesh.lines):
            if find_sequence(face, line, check_reverse=True) >= 0:
                faces_using_edges.append(i)

    return faces_using_edges


def compute_face_agreement_with_normals(mesh: pv.PolyData) -> List[bool]:
    """
    Calculate whether the normal vectors implied by the winding order of a mesh's faces agree with the mesh's normal
    vector attributes.

    Parameters
    ----------
    mesh

    Returns
    -------
    agreements:
        List of bool representing whether the normal vector implied by the winding order of each face is in the same
        direction as the face's normal vector attribute


    """
    faces = pyvista_faces_to_2d(mesh.faces)
    face_coords = mesh.points[faces]

    agreements = []
    for coords, normal in zip(face_coords, mesh.face_normals):
        agreement = winding_order_agrees_with_normal(coords, normal)
        agreements.append(agreement)

    return agreements


def rewind_face(mesh, face_num):
    """
    Reverse the winding direction of a single face of a pyvista polydata

    Note: this operates inplace

    Parameters
    ----------
    mesh
    face_num
    inplace

    Returns
    -------

    """

    faces = pyvista_faces_to_2d(mesh.faces)
    face = faces[face_num]
    face = [face[0], *face[-1:0:-1]]  # Start at same point, then reverse the rest of the face nodes
    faces[face_num] = face
    mesh.faces = pyvista_faces_to_1d(faces)


def rewind_neighbor(mesh: pv.PolyData, face: int, neighbor: int, shared_line: Tuple[int, int], wind_opposite=True):
    """
    Rewind a face's neighbor such that the neighbor has opposite handedness.

    Parameters
    ----------
    mesh
    face
    neighbor
    shared_line
    wind_opposite
    """
    if find_sequence(pyvista_faces_to_2d(mesh.faces)[face], shared_line, check_reverse=False) == -1:
        oriented_line = shared_line[::-1]
    else:
        oriented_line = shared_line

    if find_sequence(pyvista_faces_to_2d(mesh.faces)[neighbor], oriented_line, check_reverse=False) == -1:
        if wind_opposite:
            rewind_face(mesh, neighbor)
    else:
        if not wind_opposite:
            rewind_face(mesh, neighbor)


def triangulate_loop_with_stitch(loop: ArrayLike[Tuple[int, int]], points: ArrayLike = None) -> List[List[int]]:
    """
    Triangulate a loop by stitching back and forth across it.
    *Note* This algorithm can create self intersecting geometry in loops with concave sections

    Parameters
    ----------
    loop
        List of lines making up the loop to be triangulated. Lines are represented by list of two ints referring to
        indices in a points array
    points
        Array of points representing the 3D coordinates referred to by the elements of the loop.
        Unused for this algorithm

    Returns
    -------
    faces
        List of faces, each represented as a list of three indices to the points array

    """
    loop = list(zip(*loop))[0]  # Just need to look at the line starts
    faces = [[loop[-1], loop[0], loop[1]]]
    next_up_node = 2  # Already used 2 nodes from start of loop, 1 from end
    next_down_node = -2
    for i in range(len(loop) - 3):
        # Next face always starts with the new node
        # If i is even, count up from 0, if i is odd, count down from -1
        if i % 2 == 0:
            new_node = loop[next_up_node]
            next_up_node += 1
            faces.append([new_node, faces[-1][0], faces[-1][2]])  # on even iterations, go to opposite node first
        else:
            new_node = loop[next_down_node]
            next_down_node -= 1
            faces.append([new_node, faces[-1][1], faces[-1][0]])  # on odd iterations, go to adjacent node first

    return faces


def triangulate_loop_with_nearest_neighbors(loop: ArrayLike[Tuple[int, int]], points: ArrayLike) -> List[List[int]]:
    """
    Triangulate loop by building triangles using the nearest neighbor point to existing triangle edges.

    Parameters
    ----------
    loop
        List of lines making up the loop to be triangulated. Lines are represented by list of two ints referring to
        indices in a points array
    points
        Array of points representing the 3D coordinates referred to by the elements of the loop.
        Unused for this algorithm

    Returns
    -------
    faces
        List of faces, each represented as a list of three indices to the points array

    """
    loop = list(zip(*loop))[0]  # Just need to look at where each line starts
    faces = []

    # Start with a single face consisting of point 0 and its nearest neighbors
    a = loop[0]
    neighbors = sorted(loop, key=lambda neighbor: np.linalg.norm(points[a] - points[neighbor]))
    b = neighbors[1]
    c = neighbors[2]
    faces.append([a, b, c])

    # Recursively build faces off the first face
    continue_triangulating_with_nearest_neighbors(faces, loop, points)

    return faces


def continue_triangulating_with_nearest_neighbors(faces, loop, points):
    a0, b0, c0 = faces[-1]
    # Check all lines in latest triangle to recursively build off
    for a, b in [(c0, b0), (b0, a0), (a0, c0)]:
        # If the points are adjacent in the loop, they are on the edge and don't need to be built off
        points_adjacent = find_sequence(loop, [a, b], check_reverse=True) >= 0

        # If the line a,b is already found in two triangles, don't build any more
        line_in_two_faces = \
            np.count_nonzero([find_sequence(face, [a, b], check_reverse=True) >= 0 for face in faces]) >= 2

        if not points_adjacent and not line_in_two_faces:

            # Look for neighbors that are not a or b and don't already have a triangle with a or b
            # But at least one line in the triangle must be in the loop
            valid_neighbors = []
            for point in loop:
                if point not in [a, b]:
                    line_a_to_point_used = np.any(
                        [find_sequence(face, [a, point], check_reverse=True) >= 0 for face in faces])
                    line_b_to_point_used = np.any(
                        [find_sequence(face, [b, point], check_reverse=True) >= 0 for face in faces])

                    line_in_loop_a = find_sequence(loop, [a, point], check_reverse=True) >= 0
                    line_in_loop_b = find_sequence(loop, [b, point], check_reverse=True) >= 0

                    if not line_a_to_point_used and not line_b_to_point_used:
                        if line_in_loop_a or line_in_loop_b:
                            valid_neighbors.append(point)

            if not valid_neighbors:
                continue

            neighbors = sorted(valid_neighbors, key=lambda neighbor: np.linalg.norm(points[a] - points[neighbor]))
            c = neighbors[0]
            faces.append([a, b, c])
            continue_triangulating_with_nearest_neighbors(faces, loop, points)


loop_triangulation_algorithms = {
    "stitch": triangulate_loop_with_stitch,
    "nearest_neighbor": triangulate_loop_with_nearest_neighbors
}


def select_intersecting_triangles(mesh: pv.PolyData, quiet=False, *args, **kwargs) -> NDArray[int]:
    """
    Wrapper around the pymeshfix function for selecting self intersecting triangles

    Parameters
    ----------
    mesh
        Mesh to select from
    quiet
        Enable or disable verbose output from pymehsfix
        *NOTE* pymeshfix seems to have this backward. Quiet=True makes it loud. Quiet=False makes it quiet
    args
        args for PyTMesh.select_intersecting_triangles
    kwargs
        kwargs for PyTMesh.select_intersecting_Triangles

    Returns
    -------
    intersecting
        NDArray of intersecting triangles

    """
    tin = pymeshfix.PyTMesh(quiet)
    tin.load_array(mesh.points, pyvista_faces_to_2d(mesh.faces))
    intersecting = tin.select_intersecting_triangles(*args, **kwargs)
    return intersecting


def identify_neighbors(surface: pv.PolyData) -> Tuple[Dict, Dict]:
    """
    Identify neighbors of each face in a surface. Returns a dict of faces with each face's neighbors, grouped by the
    lines that the faces share. Also returns a dict of lines in the surface with each face that uses each line.


    Parameters
    ----------
    surface

    Returns
    -------
    neighbors_dict, lines_dict

    """
    # Create a dict of unique lines in the mesh, recording which faces use which lines
    lines_dict: Dict[Tuple, List] = {}
    for face_index, face in enumerate(pyvista_faces_to_2d(surface.faces)):
        for (a, b) in pairwise([*face, face[0]]):
            key = (a, b) if (a, b) in lines_dict else (b, a) if (b, a) in lines_dict else None
            if key:
                lines_dict[key].append(face_index)
            else:
                lines_dict[(a, b)] = [face_index]

    # Create a dict of faces, recording the neighbors of each face on each shared line
    neighbors_dict: Dict[int, Dict[Tuple, List]] = {}
    for line, face_list in lines_dict.items():
        for face_index in face_list:
            if face_index in neighbors_dict:
                neighbors_dict[face_index][line] = [f for f in face_list if f is not face_index]
            else:
                neighbors_dict[face_index] = {line: [f for f in face_list if f is not face_index]}

    return neighbors_dict, lines_dict


def pairwise(iterable):
    """
    Temporary implementation of itertools.pairwise for compatibility with older versions of python
    """
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def compute_neighbor_angles(surface: pv.PolyData, known_face: int, neighbors: List[int], shared_line: Tuple[int, int],
                            use_winding_order_normal=False) \
        -> List[float]:
    """
    Compute the dihedral angles between one face of a surface mesh and its neighbors along a given shared line. The
    angles are given as 0 < angle < 2*pi, where the positive direction is equivalent to rotation of the known_face
    in the direction of its normal vector about the shared line.

    Only works on surfaces with all faces having the same number of points. (Since pyvista_faces_to_2d is used)

    Parameters
    ----------
    surface
        Pyvista PolyData representing a surface mesh
    known_face
        Index of the face in surface that is known to be on the "true" surface
    neighbors
        List of indices to faces that are neighbors to known_face along a single shared line
    shared_line
        Tuple of indices to points in surface that represents the shared line between known_face and its neighbors
    use_winding_order_normal
        use the winding order of the known face to calculate the known face normal instead of the built in normal
        attribute

    Returns
    -------
    neighbors_angles
        List of dihedral angles between known_face and neighbors

    """
    surface_c = surface.copy()

    face_points = pyvista_faces_to_2d(surface_c.faces)[known_face]
    neighbors_points = pyvista_faces_to_2d(surface_c.faces)[neighbors]

    # Get the shared line in the order it is in the known face
    if find_sequence(face_points, shared_line, check_reverse=False) == -1:
        shared_line = shared_line[::-1]

    # Get the shared line vector, also known as the normal to the plane on which the face normals lie
    shared_line_points = [surface_c.points[shared_line[0]], surface_c.points[shared_line[1]]]
    plane_normal = shared_line_points[1] - shared_line_points[0]

    # Wind the neighbor faces with the opposite handedness to known face (shared line in the same direction) which will
    # make their calculated normals point away
    for i, neighbor_face_points in enumerate(neighbors_points):
        if find_sequence(neighbor_face_points, shared_line, check_reverse=False) == -1:
            neighbors_points[i] = neighbor_face_points[::-1]

    neighbors_normals = [compute_normal(points_coords) for points_coords in surface_c.points[neighbors_points]]
    if use_winding_order_normal:
        known_face_normal = compute_normal(surface_c.points[face_points])
    else:
        known_face_normal = surface_c.face_normals[known_face]
    neighbors_angles = [dihedral_angle(known_face_normal, neighbor_normal, plane_normal)
                        for neighbor_normal in neighbors_normals]
    return neighbors_angles


def choose_outer_surface_face(surface: pv.PolyData, known_face: int, neighbors: List[int], shared_line: Tuple[int, int]) \
        -> int:
    """
    Choose which neighbor of a given face on a given shared line must lie on the true surface of the given surface mesh.
    The neighbor on the true surface is that which has the lowest dihedral angle with the known surface face.

    Only works on surfaces with all faces having the same number of points. (Since pyvista_faces_to_2d is used)

    Parameters
    ----------
    surface
        Pyvista PolyData representing a surface mesh
    known_face
        Index of the face in surface that is known to be on the "true" surface
    neighbors
        List of indices to faces that are neighbors to known_face along a single shared line
    shared_line
        Tuple of indices to points in surface that represents the shared line between known_face and its neighbors

    Returns
    -------
    surface face
        The neighbor which lies on the true surface

    """

    neighbors_angles = compute_neighbor_angles(surface, known_face, neighbors, shared_line)

    min = np.argmin(neighbors_angles)
    surface_face = neighbors[min]
    return surface_face


def choose_inner_surface_face(surface: pv.PolyData, known_face: int, neighbors: List[int], shared_line: Tuple[int, int]) \
        -> int:

    agreements = compute_face_agreement_with_normals(surface)
    if agreements[known_face]:
        rewind_face(surface, known_face)
    neighbors_angles = compute_neighbor_angles(surface, known_face, neighbors, shared_line, use_winding_order_normal=True)

    min = np.argmin(neighbors_angles)
    surface_face = neighbors[min]
    return surface_face


def find_face_on_outer_surface(surface: pv.PolyData) -> int:
    """
    Find a face on the outer surface by casting a long ray from the first surface and choosing the last face it hits

    Parameters
    ----------
    surface

    Returns
    -------
    face
        A face guaranteed to be on the outer surface of the input surface mesh

    """
    #
    stop = surface.cell_centers().points[0] - surface.face_normals[0]
    b = surface.bounds
    distance = np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
    start = stop + (surface.face_normals[0] * distance)
    _, intersection_cells = surface.ray_trace(start, stop, first_point=True)
    face = intersection_cells[0]

    return face
