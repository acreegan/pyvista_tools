=====================
Extract Outer Surface
=====================

**extract_outer_surface** is a function used to select only the elements of a mesh that lie on
the outer surface.

To demonstrate this, first we load a surface mesh that contains some unwanted inner faces:

.. code-block:: python

    mesh_filename = "meshes/mock_lung/upper_lobe_of_left_lung_surface_unrefined.stl"
    mesh = pv.read(mesh_filename)

and view it:

.. code-block:: python

    p = pv.Plotter()
    p.add_mesh(mesh, show_edges=True, opacity=0.2)
    p.add_title("Mesh with unwanted\ninner faces")
    p.show()

.. image:: ../_static/mesh_with_unwanted_inner_faces.png

We then extract the outer surface of the mesh

.. code-block:: python

    refined, removed_faces = extract_outer_surface(mesh, return_removed_faces=True)

and view the result:

.. code-block:: python

    p = pv.Plotter()
    removed = pv.PolyData(mesh.points, pyvista_faces_to_1d(pyvista_faces_to_2d(mesh.faces)[removed_faces]))
    p.add_mesh(refined, style="wireframe", label="Refined mesh")
    p.add_mesh(removed, style="wireframe", color="red", opacity=0.2, label="Removed faces")
    p.add_title("Refined Mesh")
    p.add_legend()
    p.show()

.. image:: ../_static/refined_mesh.png