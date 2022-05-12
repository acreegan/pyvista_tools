=============
PyVista Tools
=============

PyVista Tools is a python package that provides extended functionality to PyVista including
new features and new tools for working with PyVista meshes.

Documentation can be found at: https://acreegan.github.io/pyvista_tools/

Source code can be found at: https://github.com/acreegan/pyvista_tools

Features
--------
- Extract enclosed regions of a mesh
- Extract outer surface of a mesh
- Remove surfaces not part of an enclosed region
- Remove faces shared by a list of meshes

**Tools**

- Convert representation of PyVista cells between 2d and 1d
- Select points or faces using faces, points, or edges
- Identify face neighbors
- Find face on mesh outer surface

**Geometry Tools**

- Dihedral angle between faces (0 to 2pi)
- Compute face normal

Installation
------------
To install PyVista Tools, run this command in your terminal:

.. code-block:: console

    $ pip install git+https://github.com/acreegan/pyvista_tools.git

Credits
-------
* Development Lead: Andrew Creegan <andrew.s.creegan@gmail.com>
