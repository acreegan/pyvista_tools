================
Module Reference
================

PyVista Features
----------------
pyvista_features is a module that provides high level features for working with pyvista objects. These features should
transform or create pyvista objects. They are intended to be useful on their own as part of a workflow that makes use
of pyvista objects.

.. automodule:: pyvista_tools.pyvista_features
    :members: extract_enclosed_regions, extract_outer_surface, remove_boundary_faces_recursively,
                remove_shared_faces_with_merge, rewind_faces_to_normals

PyVista Tools
-------------
pyvista_tools is a module that provides tools for working with pyvista objects. This module should contain functions
that are not necessarily useful on their own, but can be used to support development of pyvista features.

.. automodule:: pyvista_tools.pyvista_tools
    :members: pyvista_faces_to_2d, pyvista_faces_to_1d, select_shared_faces, select_faces_using_points,
        select_points_used_by_faces, select_faces_using_edges, select_intersecting_triangles, identify_neighbors,
        find_face_on_outer_surface


Geometry Tools
--------------
geometry_tools is a module that provides functions for making geometric calculations in support of pyvista_tools. These
functions should not rely on pyvista specific types or data structures.

.. automodule:: pyvista_tools.geometry_tools
    :members: find_sequence, compute_normal, dihedral_angle

