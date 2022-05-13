=====================
Remove Boundary Faces
=====================

We can use remove_boundary_faces_recursively to remove faces in a surface mesh that are not part
of an enclosed region.

To demonstrate, we create an example mesh consisting of a sphere with a half-sphere attached.

.. code-block:: python

    half_sphere = pv.Sphere().clip()
    full_sphere = pv.Sphere(center=(-0.5, 0, 0))
    union = half_sphere.boolean_union(full_sphere)
    intersection = half_sphere.boolean_intersection(full_sphere)
    example_mesh = union.merge(intersection)
    example_mesh = pv.PolyData(example_mesh.points, example_mesh.faces)

    example_mesh.plot(show_edges=True)

.. image:: ../_static/mesh_with_boundary_edges.png

We then remove the boundary faces:

.. code-block:: python

    boundary_removed = remove_boundary_faces_recursively(example_mesh)

and plot the result

.. code-block:: python

    shared_faces_removed = remove_shared_faces_with_merge([example_mesh, boundary_removed])
    p = pv.Plotter()
    p.add_mesh(boundary_removed, show_edges=True, label="Boundary Faces Removed")
    p.add_mesh(shared_faces_removed, color="red", show_edges=True, opacity=.2, label="Removed Faces")
    p.add_legend()
    p.add_title("Mesh with boundary\nfaces removed")
    p.show()

.. image:: ../_static/mesh_with_boundary_edges_removed.png