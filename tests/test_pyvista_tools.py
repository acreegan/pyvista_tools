from pyvista_tools import remove_shared_faces, select_shared_faces, select_points_used_by_faces, \
    pyvista_faces_by_dimension, pyvista_faces_to_2d, pyvista_faces_to_1d, select_shared_points, \
    select_faces_using_points, remove_shared_faces_with_ray_trace, find_sequence, select_faces_using_edges, \
    find_loops_and_chains, triangulate_loop_with_stitch, triangulate_loop_with_nearest_neighbors, \
    select_intersecting_triangles, dihedral_angle, compute_normal, extract_outer_surface, identify_neighbors, \
    remove_boundary_faces_recursively, extract_enclosed_regions, compute_neighbor_angles, rewind_face, \
    rewind_neighbor, remove_shared_faces_with_merge

import numpy as np
import numpy.testing
import pyvista as pv
import collections
from matplotlib import cm
import sys

mesh_a_verts = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1],
                         [0, 0.5, 0.5],
                         [1, 0.5, 0.5]], dtype=float)

mesh_a_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 5, 9], [3, 5, 7, 9], [3, 7, 6, 9], [3, 6, 4, 9],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_a = pv.PolyData(mesh_a_verts, mesh_a_faces)

mesh_b_verts = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [-1, 0, 0],
                         [-1, 0, 1],
                         [-1, 1, 0],
                         [-1, 1, 1],
                         [0, 0.5, 0.5]], dtype=float)

mesh_b_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 6, 7], [3, 4, 7, 5],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_b = pv.PolyData(mesh_b_verts, mesh_b_faces)

mesh_c_verts = np.array([[1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1],
                         [2, 0, 0],
                         [2, 0, 1],
                         [2, 1, 0],
                         [2, 1, 1],
                         [1, 0.5, 0.5]], dtype=float)

mesh_c_faces = np.hstack([[3, 0, 1, 5], [3, 0, 5, 4],
                          [3, 1, 3, 7], [3, 1, 7, 5],
                          [3, 3, 2, 6], [3, 3, 6, 7],
                          [3, 2, 0, 4], [3, 2, 4, 6],
                          [3, 4, 6, 7], [3, 4, 7, 5],
                          [3, 0, 1, 8], [3, 1, 3, 8], [3, 3, 2, 8], [3, 2, 0, 8]])

mesh_c = pv.PolyData(mesh_c_verts, mesh_c_faces)

shared_faces_ab = ([12, 13, 14, 15], [10, 11, 12, 13])
shared_points_ab = ([0, 1, 2, 3, 8], [0, 1, 2, 3, 8])


def test_select_faces_using_points():
    faces = select_faces_using_points(mesh_a, shared_points_ab[0])
    assert faces == shared_faces_ab[0]


def test_select_shared_points():
    shared_points = select_shared_points(mesh_a, mesh_b)
    assert shared_points == shared_points_ab


def test_select_shared_faces():
    shared_faces = select_shared_faces(mesh_a, mesh_b)
    assert shared_faces == shared_faces_ab


def test_pyvista_faces_to_2d():
    mesh_b_faces_2d = np.array([[0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5], [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6],
                                [4, 6, 7], [4, 7, 5], [0, 1, 8], [1, 3, 8], [3, 2, 8], [2, 0, 8]])

    faces_2d = pyvista_faces_to_2d(mesh_b.faces)

    assert np.array_equal(mesh_b_faces_2d, faces_2d)


def test_pyvista_faces_to_1d():
    mesh_b_faces_2d = np.array([[0, 1, 5], [0, 5, 4], [1, 3, 7], [1, 7, 5], [3, 2, 6], [3, 6, 7], [2, 0, 4], [2, 4, 6],
                                [4, 6, 7], [4, 7, 5], [0, 1, 8], [1, 3, 8], [3, 2, 8], [2, 0, 8]])

    faces_1d = pyvista_faces_to_1d(mesh_b_faces_2d)
    assert np.array_equal(faces_1d, mesh_b.faces)


def test_select_points_in_faces():
    exclusive_points = [8]
    test_points = [0, 1, 2, 3, 8]
    test_faces = [12, 13, 14, 15]

    points_in_test_faces = select_points_used_by_faces(mesh_a, test_points, test_faces, exclusive=True)

    assert np.array_equal(points_in_test_faces, exclusive_points)


def test_remove_shared_faces():
    a_merged_points = mesh_a_verts[:-2]
    a_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_a_faces)[:-8])
    a_merged = pv.PolyData(a_merged_points, a_merged_faces)
    b_merged_points = mesh_b_verts[:-1]
    b_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_b_faces)[:-4])
    b_merged = pv.PolyData(b_merged_points, b_merged_faces)
    c_merged_points = mesh_c_verts[:-1]
    c_merged_faces = pyvista_faces_to_1d(pyvista_faces_to_2d(mesh_c_faces)[:-4])
    c_merged = pv.PolyData(c_merged_points, c_merged_faces)
    a_b_c_merged = a_merged.merge(b_merged).merge(c_merged)

    trimmed = remove_shared_faces([mesh_a, mesh_b, mesh_c])
    merged = pv.PolyData()
    for mesh in trimmed:
        merged = merged.merge(mesh)

    assert np.array_equal(a_b_c_merged.points, merged.points)
    assert np.array_equal(a_b_c_merged.faces, merged.faces)

    p = pv.PolyData(merged.points, merged.faces)
    assert p.is_manifold


def test_remove_shared_faces_again():
    shared_a = pv.Box(quads=False).rotate_z(90, inplace=False).translate([-2, 0, 0], inplace=False)
    shared_b = pv.Box(quads=False)
    not_shared = pv.Box(quads=False).translate([4, 0, 0], inplace=False)

    removed = remove_shared_faces([shared_a, shared_b, not_shared])

    # p = pv.Plotter()
    # for mesh in [shared_a, shared_b, not_shared]:
    #     p.add_mesh(mesh, style="wireframe")
    # p.add_title("Original Meshes")
    # p.show()

    # p = pv.Plotter()
    # for mesh in removed:
    #     p.add_mesh(mesh, style="wireframe")
    # p.show()

    correct_removed_0_points = np.array([
        [-1., -1., -1.], [-1., -1., 1.], [-3., -1., 1.],
        [-3., -1., -1.], [-1., 1., 1.], [-1., 1., -1.],
        [-3., 1., -1.], [-3., 1., 1.], [1., -1., 1.],
        [1., -1., -1.], [1., 1., -1.], [1., 1., 1.]])

    correct_removed_0_faces = np.array(
        [3, 0, 1, 2, 3, 0, 2, 3, 3, 4, 5, 6, 3, 4, 6, 7, 3,
         2, 7, 6, 3, 2, 6, 3, 3, 5, 0, 3, 3, 5, 3, 6, 3, 1,
         4, 7, 3, 1, 7, 2, 3, 8, 9, 10, 3, 8, 10, 11, 3, 0, 9,
         8, 3, 0, 8, 1, 3, 4, 11, 10, 3, 4, 10, 5, 3, 9, 0, 5,
         3, 9, 5, 10, 3, 1, 8, 11, 3, 1, 11, 4])
    correct_removed_1_points = np.array([
        [3., -1., -1.], [5., -1., -1.], [3., 1., -1.],
        [5., 1., -1.], [3., -1., 1.], [5., -1., 1.],
        [3., 1., 1.], [5., 1., 1.]])
    correct_removed_1_faces = np.array(
        [3, 0, 4, 6, 3, 0, 6, 2, 3, 5, 1, 3, 3, 5, 3, 7, 3, 0, 1, 5, 3, 0,
         5, 4, 3, 6, 7, 3, 3, 6, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 3, 4, 5, 7,
         3, 4, 7, 6])

    assert np.array_equal(removed[0].points, correct_removed_0_points)
    assert np.array_equal(removed[0].faces, correct_removed_0_faces)
    assert np.array_equal(removed[1].points, correct_removed_1_points)
    assert np.array_equal(removed[1].faces, correct_removed_1_faces)


def test_remove_shared_faces_with_merge():
    shared_a = pv.Box(quads=False).rotate_z(90, inplace=False).translate([-2, 0, 0], inplace=False)
    shared_b = pv.Box(quads=False)

    merged = remove_shared_faces_with_merge([shared_a, shared_b])
    # merged.plot(style="wireframe")

    assert merged.n_faces == 20


def test_remove_shared_faces_with_merge_3_surfaces():
    shared_a = pv.Box(quads=False).rotate_z(90, inplace=False).translate([-2, 0, 0], inplace=False)
    shared_b = pv.Box(quads=False)
    shared_c = pv.Box(quads=False).rotate_z(90, inplace=False).translate([2, 0, 0], inplace=False)

    merged = remove_shared_faces_with_merge([shared_a, shared_b, shared_c])
    # merged.plot(style="wireframe")

    assert merged.n_faces == 28


def test_remove_shared_faces_with_ray_trace():
    shared_a = pv.Box(quads=False).rotate_z(90, inplace=False).translate([-2, 0, 0], inplace=False)
    shared_b = pv.Box(quads=False)
    not_shared = pv.Box(quads=False).translate([4, 0, 0], inplace=False)

    removed = remove_shared_faces_with_ray_trace([shared_a, shared_b, not_shared])

    # p = pv.Plotter()
    # for mesh in [shared_a, shared_b, not_shared]:
    #     p.add_mesh(mesh, style="wireframe")
    # p.add_title("Original Meshes")
    # p.show()
    #
    # p = pv.Plotter()
    # for mesh in removed:
    #     p.add_mesh(mesh, style="wireframe")
    # p.show()

    correct_removed_0_points = np.array([
        [-1., -1., -1.], [-1., -1., 1.], [-3., -1., 1.],
        [-3., -1., -1.], [-1., 1., 1.], [-1., 1., -1.],
        [-3., 1., -1.], [-3., 1., 1.], [1., -1., 1.],
        [1., -1., -1.], [1., 1., -1.], [1., 1., 1.]])

    correct_removed_0_faces = np.array(
        [3, 0, 1, 2, 3, 0, 2, 3, 3, 4, 5, 6, 3, 4, 6, 7, 3,
         2, 7, 6, 3, 2, 6, 3, 3, 5, 0, 3, 3, 5, 3, 6, 3, 1,
         4, 7, 3, 1, 7, 2, 3, 8, 9, 10, 3, 8, 10, 11, 3, 0, 9,
         8, 3, 0, 8, 1, 3, 4, 11, 10, 3, 4, 10, 5, 3, 9, 0, 5,
         3, 9, 5, 10, 3, 1, 8, 11, 3, 1, 11, 4])
    correct_removed_1_points = np.array([
        [3., -1., -1.], [5., -1., -1.], [3., 1., -1.],
        [5., 1., -1.], [3., -1., 1.], [5., -1., 1.],
        [3., 1., 1.], [5., 1., 1.]])
    correct_removed_1_faces = np.array(
        [3, 0, 4, 6, 3, 0, 6, 2, 3, 5, 1, 3, 3, 5, 3, 7, 3, 0, 1, 5, 3, 0,
         5, 4, 3, 6, 7, 3, 3, 6, 3, 2, 3, 1, 0, 2, 3, 1, 2, 3, 3, 4, 5, 7,
         3, 4, 7, 6])

    assert np.array_equal(removed[0].points, correct_removed_0_points)
    assert np.array_equal(removed[0].faces, correct_removed_0_faces)
    assert np.array_equal(removed[1].points, correct_removed_1_points)
    assert np.array_equal(removed[1].faces, correct_removed_1_faces)


def test_remove_shared_faces_with_ray_trace_angled():
    a = pv.Box(quads=False)
    b = pv.Box(quads=False).rotate_z(45, inplace=False).translate([2.5, 0.5, 0], inplace=False)

    removed = remove_shared_faces_with_ray_trace([a, b], ray_length=5)

    # p = pv.Plotter()
    # for mesh in [a, b]:
    #     p.add_mesh(mesh, style="wireframe")
    # p.show()

    # Even though the long rays definitely hit, nothing is removed because the angle is not within tolerance
    assert np.array_equal(removed[0].points, a.points)
    assert np.array_equal(removed[0].faces, a.faces)
    assert np.array_equal(removed[1].points, b.points)
    assert np.array_equal(removed[1].faces, b.faces)


def test_pyvista_faces_by_dimension():
    # mesh points
    vertices = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    # mesh faces
    faces = np.hstack([[4, 0, 1, 2, 3],  # square
                       [3, 0, 1, 4],  # triangle
                       [3, 1, 2, 4]])  # triangle

    mesh = pv.PolyData(vertices, faces)

    faces_by_dim = pyvista_faces_by_dimension(mesh.faces)
    assert np.array_equal(faces_by_dim[3], np.array([3, 0, 1, 4, 3, 1, 2, 4]))
    assert np.array_equal(faces_by_dim[4], np.array([4, 0, 1, 2, 3]))


def test_extract_faces_edges_lines():
    b = pv.Box()
    b = b.remove_cells(0)
    edges = b.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                    feature_edges=False, manifold_edges=False)

    faces = select_faces_using_edges(b, edges)

    # b["points"] = np.array((range(0, len(b.points))))
    # p = pv.Plotter()
    # p.add_mesh(b)
    # p.add_point_labels(b, "points")
    # p.show()

    assert np.array_equal(faces, [1, 2, 3, 4])


def test_extract_faces_with_edges_duplicates():
    b = pv.Box()
    b = b.remove_cells([0, 1, 2])
    edges = b.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                    feature_edges=False, manifold_edges=False)

    faces = select_faces_using_edges(b, edges)

    # b["points"] = np.array((range(0, len(b.points))))
    # p = pv.Plotter()
    # p.add_mesh(b)
    # p.add_point_labels(b, "points")
    # p.show()

    three_sides = [item for item, count in collections.Counter(faces).items() if count > 2]

    assert np.array_equal(three_sides, [1, 2])


def test_extract_faces_with_edges_flap():
    b = pv.Box(quads=False)
    b = b.remove_cells(0)
    b.points = np.vstack((b.points, [0, 0, 0]))
    b.faces = pyvista_faces_to_1d(np.vstack((pyvista_faces_to_2d(b.faces), [0, 1, 8])))

    edges = b.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                    feature_edges=False, manifold_edges=False)

    faces = select_faces_using_edges(b, edges)
    two_sides = [item for item, count in collections.Counter(faces).items() if count > 1]

    # b["points"] = np.array((range(0, len(b.points))))
    #
    # bf = np.array([0 if not np.any(np.isin(faces, i)) else 1 for i in range(b.n_faces)])
    # double_bf = np.array([0 if not np.any(np.isin(two_sides, i)) else 1 for i in range(b.n_faces)])
    #
    # b["boundary_faces"] = bf+double_bf
    # c = b.remove_cells(two_sides)
    # p = pv.Plotter(shape=(1, 2))
    # p.add_mesh(b, show_edges=True, scalars="boundary_faces", cmap=cm.get_cmap("Set1_r"))
    # p.add_point_labels(b, "points")
    # p.add_mesh(edges, color="red", label="Boundary Edges", line_width=2)
    # p.add_legend()
    # p.subplot(0, 1)
    # p.add_mesh(c, show_edges=True, scalars="boundary_faces", cmap=cm.get_cmap("Set1_r"))
    # p.add_mesh(edges, color="red", label="Boundary Edges", line_width=2)
    # p.show()

    assert (np.array_equal(two_sides, [11]))


# def test_extract_faces_with_edges_non_manifold_plus_flap():
#     b = pv.Box(quads=False)
#     b = b.remove_cells(0)
#     b.points = np.vstack((b.points, [0, 0, 0]))
#     b.faces = pyvista_faces_to_1d(np.vstack((pyvista_faces_to_2d(b.faces), [0, 1, 8])))
#     b.points = np.vstack((b.points, [0.5, 0, 0]))
#     b.faces = pyvista_faces_to_1d(np.vstack((pyvista_faces_to_2d(b.faces), [3, 5, 9])))
#
#     edges = b.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
#                                     feature_edges=False, manifold_edges=False)
#
#     faces = extract_faces_with_edges(b, edges)
#     two_sides = [item for item, count in collections.Counter(faces).items() if count > 1]
#
#     b["points"] = np.array((range(0, len(b.points))))
#     b["boundary_faces"] = np.array([0 if not np.any(np.isin(faces, i)) else 1 for i in range(b.n_faces)])
#     b["double_boundary_faces"] = np.array([0 if not np.any(np.isin(two_sides, i)) else 1 for i in range(b.n_faces)])
#     p = pv.Plotter()
#     p.add_mesh(b, show_edges=True, scalars="double_boundary_faces", cmap=cm.get_cmap("Set1_r"))
#     p.add_point_labels(b, "points")
#     p.add_mesh(edges, color="red", label="Boundary Edges", line_width=2)
#     p.add_legend()
#     p.show()
#
#     assert(np.array_equal(two_sides, [11]))


def test_find_sequence():
    a = [1, 2, 3, 4]
    b = [1, 2]
    c = [2, 3]
    d = [1, 3]

    ab = find_sequence(a, b)
    ac = find_sequence(a, c)
    ad = find_sequence(a, d)

    assert (ab == 0)
    assert (ac == 1)
    assert (ad == -1)


def test_find_sequence_reverse():
    a = [1, 2, 3, 4]
    b = [2, 1]
    c = [3, 1]

    abr = find_sequence(a, b, check_reverse=True)
    acr = find_sequence(a, c, check_reverse=True)
    assert abr == 0
    assert acr == -1


def test_find_loops_and_chains():
    lines = [[1, 2], [2, 3], [3, 1], [5, 6], [6, 7]]

    correct_loops = [[(1, 2), (2, 3), (3, 1)]]
    correct_chains = [[(5, 6), (6, 7)]]

    loops, chains = find_loops_and_chains(lines)

    assert np.array_equal(loops, correct_loops)
    assert np.array_equal(chains, correct_chains)


def test_triangulate_loop():
    loop = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1)]
    correct_faces = [[8, 1, 2], [3, 8, 2], [7, 8, 3], [4, 7, 3], [6, 7, 4], [5, 6, 4]]

    faces = triangulate_loop_with_stitch(loop)

    assert np.array_equal(faces, correct_faces)


def test_triangulate_loop_with_nearest_neighbors_boundary_m():
    points = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 1, 0], [0.5, 1, 0], [1, 0.5, 0], [2, 1, 0], [3, 1, 0]])
    loop = np.array([[1, 2], [2, 7], [7, 6], [6, 5], [5, 4], [4, 3], [3, 0], [0, 1]])
    correct_faces = [[1, 5, 0], [0, 5, 3], [3, 5, 4], [5, 1, 6], [6, 1, 7], [7, 1, 2]]

    faces = triangulate_loop_with_nearest_neighbors(loop, points)

    # boundary = pv.PolyData(points, lines=pyvista_faces_to_1d(loop))
    # mesh = pv.PolyData(boundary.points, pyvista_faces_to_1d(faces))
    # p = pv.Plotter()
    # p.add_mesh(mesh, style="wireframe")
    # p.add_mesh(boundary, color="red")
    # p.show()

    assert np.array_equal(faces, correct_faces)


def test_triangulate_loop_with_nearest_neighbors_boundary_square():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]])
    loop = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]])
    correct_faces = [[0, 1, 7], [7, 1, 6], [6, 1, 5], [5, 1, 4], [4, 1, 3], [3, 1, 2]]

    faces = triangulate_loop_with_nearest_neighbors(loop, points)

    # boundary = pv.PolyData(points, lines=pyvista_faces_to_1d(loop))
    # mesh = pv.PolyData(boundary.points, pyvista_faces_to_1d(faces))
    # p = pv.Plotter()
    # p.add_mesh(mesh, style="wireframe")
    # p.add_mesh(boundary, color="red")
    # p.show()

    assert np.array_equal(faces, correct_faces)


def test_triangulate_loop_with_nearest_neighbors_boundary_3d():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]])
    loop = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]])

    faces = triangulate_loop_with_nearest_neighbors(loop, points)
    mesh = pv.PolyData(points, pyvista_faces_to_1d(faces))

    intersecting_triangles = select_intersecting_triangles(mesh, justproper=True)

    # boundary = pv.PolyData(points, lines=pyvista_faces_to_1d(loop))
    # p = pv.Plotter()
    # p.add_mesh(mesh, style="wireframe")
    # p.add_mesh(boundary, color="red")
    # p.show()

    assert np.empty(intersecting_triangles)


def test_dihedral_angle():
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 0.5, 0.5], [0.5, 0, 0.5], [0, -0.5, 0.5], [-0.5, 0, 0.5]])
    faces = np.array([[0, 1, 2], [0, 1, 5], [0, 1, 4], [0, 1, 3]])
    normals = [compute_normal(points[face]) for face in faces]
    plane_normal = points[1] - points[0]

    # surface = pv.PolyData(points, pyvista_faces_to_1d(faces))
    # face_points = surface.cell_centers()
    # face_points["Normals"] = np.array(normals)
    # arrows = face_points.glyph(orient="Normals", geom=pv.Arrow())
    # surface["point_labels"] = [f"Point {i}" for i in range(surface.n_points)]
    # p = pv.Plotter()
    # p.add_mesh(arrows, color="black")
    # p.add_mesh(surface, scalars=None, color="White")
    # p.add_point_labels(surface, "point_labels")
    # p.add_axes()
    # p.show()

    angle_01 = dihedral_angle(normals[0], normals[1], plane_normal, degrees=True)
    angle_02 = dihedral_angle(normals[0], normals[2], plane_normal, degrees=True)
    angle_03 = dihedral_angle(normals[0], normals[3], plane_normal, degrees=True)

    assert angle_01 == 90
    assert angle_02 == 180
    assert angle_03 == 270


def test_refine_surface():
    surface = pv.Box(quads=False)
    surface.points = np.vstack((surface.points, [0., 0., 0.]))
    surface.faces = pyvista_faces_to_1d(np.vstack(([0, 1, 8], pyvista_faces_to_2d(surface.faces))))
    surface_refined = extract_outer_surface(surface)

    # surface.plot(style="wireframe")
    # surface_refined.plot(style="wireframe")

    assert find_sequence(surface.faces, [0, 1, 8]) == 1
    assert find_sequence(surface_refined.faces, [0, 1, 8]) == -1


def test_identify_neighbors():
    surface = pv.Cone(resolution=3)
    surface.points = np.vstack((surface.points, [0., 0., 0.]))
    surface.faces = pyvista_faces_to_1d(np.vstack(([0, 1, 4], pyvista_faces_to_2d(surface.faces))))

    correct_neighbors_dict = {0: {(0, 1): [2, 4], (1, 4): [], (4, 0): []},
                              2: {(0, 1): [0, 4], (2, 1): [1], (2, 0): [3]},
                              4: {(0, 1): [0, 2], (1, 3): [1], (3, 0): [3]},
                              1: {(3, 2): [3], (2, 1): [2], (1, 3): [4]},
                              3: {(3, 2): [1], (2, 0): [2], (3, 0): [4]}}

    correct_lines_dict = {(0, 1): [0, 2, 4],
                          (1, 4): [0],
                          (4, 0): [0],
                          (3, 2): [1, 3],
                          (2, 1): [1, 2],
                          (1, 3): [1, 4],
                          (2, 0): [2, 3],
                          (3, 0): [3, 4]}

    # surface.plot(style="wireframe")

    neighbors_dict, lines_dict = identify_neighbors(surface)

    assert correct_neighbors_dict == neighbors_dict
    assert correct_lines_dict == lines_dict


def test_rewind_face():
    surface = pv.Box()

    correct_original_face = [0, 4, 6, 2]
    correct_rewound_face = [0, 2, 6, 4]

    faces = pyvista_faces_to_2d(surface.faces)
    face_0_original = np.copy(faces[0])
    rewind_face(surface, 0)
    face_0_new = faces[0]

    np.testing.assert_array_equal(correct_original_face, face_0_original)
    np.testing.assert_array_equal(correct_rewound_face, face_0_new)


def test_rewind_neighbor():
    surface = pv.Box()

    face_2 = np.copy(pyvista_faces_to_2d(surface.faces)[2])

    rewind_neighbor(surface, 0, 2, (0, 4), wind_opposite=False)
    face_2_not_rewound = np.copy(pyvista_faces_to_2d(surface.faces)[2])

    rewind_neighbor(surface, 0, 2, (0, 4))
    face_2_rewound = np.copy(pyvista_faces_to_2d(surface.faces)[2])

    correct_face_2 = [0, 1, 5, 4]
    correct_face_2_not_rewound = [0, 1, 5, 4]
    correct_face_2_rewound = [0, 4, 5, 1]

    np.testing.assert_array_equal(face_2, correct_face_2)
    np.testing.assert_array_equal(face_2_not_rewound, correct_face_2_not_rewound)
    np.testing.assert_array_equal(face_2_rewound, correct_face_2_rewound)


def test_compute_neighbor_angles_box():
    surface = pv.Box()
    surface.faces = pyvista_faces_to_1d(np.vstack((pyvista_faces_to_2d(surface.faces), [0, 4, 7, 3])))

    box_outer_angle = 2 * np.pi * (3 / 4)
    box_angle_to_inner = 2 * np.pi * (7 / 8)

    # p = pv.Plotter()
    # p.add_mesh(surface, style="wireframe")
    # p.add_point_labels(surface.cell_centers().points, list(range(surface.n_cells)))
    # p.add_point_labels(surface.points, list(range(surface.n_points)))
    # p.show()

    neighbor_angles = compute_neighbor_angles(surface, known_face=0, neighbors=[2, 6], shared_line=(0, 4))
    np.testing.assert_almost_equal(neighbor_angles[0], box_outer_angle)
    np.testing.assert_almost_equal(neighbor_angles[1], box_angle_to_inner)

    rewind_face(surface, 0)
    box_outer_angle_rewound = 2 * np.pi * (1 / 4)
    box_inner_angle_rewound = 2 * np.pi * (1 / 8)

    neighbor_angles_rewound = compute_neighbor_angles(surface, known_face=0, neighbors=[2, 6], shared_line=(0, 4),
                                                      use_winding_order_normal=True)

    np.testing.assert_almost_equal(neighbor_angles_rewound[0], box_outer_angle_rewound)
    np.testing.assert_almost_equal(neighbor_angles_rewound[1], box_inner_angle_rewound)


def test_identify_neighbors_square():
    surface = pv.Box()
    correct_n = {0: {(0, 4): [2], (4, 6): [5], (6, 2): [3], (2, 0): [4]},
                 2: {(0, 4): [0], (5, 1): [1], (0, 1): [4], (5, 4): [5]},
                 5: {(4, 6): [0], (7, 5): [1], (5, 4): [2], (6, 7): [3]},
                 3: {(6, 2): [0], (3, 7): [1], (6, 7): [5], (3, 2): [4]},
                 4: {(2, 0): [0], (1, 3): [1], (0, 1): [2], (3, 2): [3]},
                 1: {(5, 1): [2], (1, 3): [4], (3, 7): [3], (7, 5): [5]}}

    correct_l = {(0, 4): [0, 2],
                 (4, 6): [0, 5],
                 (6, 2): [0, 3],
                 (2, 0): [0, 4],
                 (5, 1): [1, 2],
                 (1, 3): [1, 4],
                 (3, 7): [1, 3],
                 (7, 5): [1, 5],
                 (0, 1): [2, 4],
                 (5, 4): [2, 5],
                 (6, 7): [3, 5],
                 (3, 2): [3, 4]}
    n, l = identify_neighbors(surface)

    assert n == correct_n
    assert l == correct_l


def test_remove_boundary_faces_recursively():
    surface = pv.Cone(resolution=3)
    surface.points = np.vstack((surface.points, [[0., 0., 0.5], [-0.5, 0., 0.5], [0.5, 0., 0.5]]))
    surface.faces = pyvista_faces_to_1d(
        np.vstack(([[0, 1, 4], [1, 4, 5], [0, 4, 6]], pyvista_faces_to_2d(surface.faces))))

    correct_faces = np.array([3, 0, 1, 2, 3, 3, 2, 1, 3, 3, 1, 0, 3, 3, 0, 2])

    surface_r = remove_boundary_faces_recursively(surface)

    # p = pv.Plotter()
    # p.add_mesh(surface, style="wireframe")
    # p.add_point_labels(surface.points, surface.points.tolist())
    # p.show()
    #
    # surface_r.plot()

    assert np.array_equal(surface_r.faces, correct_faces)


def test_remove_boundary_edges_recursively_2():
    resolution = 10
    half_sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution).clip()
    full_sphere = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution, center=(-0.5, 0, 0))
    union = half_sphere.boolean_union(full_sphere)
    intersection = half_sphere.boolean_intersection(full_sphere)
    example_mesh = union.merge(intersection)
    example_mesh = pv.PolyData(example_mesh.points, example_mesh.faces)  # Why is this necessary?

    # p = pv.Plotter()
    # p.add_mesh(example_mesh, style="wireframe")
    # # p.add_point_labels(example_mesh.cell_centers().points, list(range(example_mesh.n_cells)))
    # p.add_points(example_mesh.cell_centers().points[[12, 121]])
    # p.show()

    boundary_removed = remove_boundary_faces_recursively(example_mesh)

    boundary_edges = boundary_removed.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                            non_manifold_edges=False, manifold_edges=False)

    assert boundary_edges.n_faces == 0


def test_extract_enclosed_regions():
    a = pv.Box().translate([-2, 0, 0], inplace=False)
    b = pv.Box()
    c = a.merge(b)
    c = c.remove_cells([1])

    # p = pv.Plotter()
    # p.add_mesh(c, style="wireframe")
    # p.add_point_labels(c.cell_centers().points, list(range(c.n_cells)))
    # p.show()

    regions = extract_enclosed_regions(c)

    # p = pv.Plotter()
    # p.add_mesh(regions[0], style="wireframe", color="red")
    # p.add_mesh(regions[1], style="wireframe", color="blue")
    # p.show()

    region_0_correct_points = np.array([[-3., -1., -1.],
                                        [-3., -1., 1.],
                                        [-3., 1., 1.],
                                        [-3., 1., -1.],
                                        [-1., -1., -1.],
                                        [-1., -1., 1.],
                                        [-1., 1., 1.],
                                        [-1., 1., -1.]])

    region_1_correct_points = np.array([[-1., -1., -1.],
                                        [-1., -1., 1.],
                                        [-1., 1., 1.],
                                        [-1., 1., -1.],
                                        [1., -1., 1.],
                                        [1., -1., -1.],
                                        [1., 1., -1.],
                                        [1., 1., 1.]])

    assert np.array_equal(regions[0].points, region_0_correct_points)
    assert np.array_equal(regions[1].points, region_1_correct_points)


def test_extract_enclosed_regions_2():
    resolution = 30  # Resolution 40 gives 6872 faces. Highest we can go before stack overflow
    sphere_a = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution)
    sphere_b = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution, center=(0.5, 0, 0))
    union_result = sphere_a.boolean_union(sphere_b)
    intersection_result = sphere_a.boolean_intersection(sphere_b)
    merge_result = union_result.merge(intersection_result)

    # p = pv.Plotter()
    # p.add_mesh(merge_result, style="wireframe")
    # # p.add_point_labels(merge_result.cell_centers().points, list(range(merge_result.n_cells)))
    # p.add_points(merge_result.cell_centers().points[0])
    # p.show()

    regions = extract_enclosed_regions(merge_result)
    # cmap = cm.get_cmap("Set1")
    # p = pv.Plotter()
    # for i, region in enumerate(regions):
    #     p.add_mesh(region, style="wireframe", color=cmap(i), label=f"Region {i}")
    #
    # p.add_legend()
    # p.show()

    assert len(regions) == 3


def main():
    p = pv.Plotter()
    p.add_mesh(mesh_a, style="wireframe")
    p.add_mesh(mesh_b, style="wireframe")
    p.add_mesh(mesh_c, style="wireframe")
    p.show()


if __name__ == "__main__":
    main()
