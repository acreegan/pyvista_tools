import pyvista as pv
from pyvista_tools import extract_outer_surface, pyvista_faces_to_1d, pyvista_faces_to_2d

mesh_filename = "meshes/mock_lung/upper_lobe_of_left_lung_surface_unrefined.stl"


def main():
    # Load mesh
    mesh = pv.read(mesh_filename)

    # Extract outer surface
    refined, removed_faces = extract_outer_surface(mesh, return_removed_faces=True)

    # Plot result
    removed = pv.PolyData(mesh.points, pyvista_faces_to_1d(pyvista_faces_to_2d(mesh.faces)[removed_faces]))
    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(mesh, style="wireframe")
    p.add_title("Original Mesh")
    p.subplot(0, 1)
    p.add_mesh(refined, style="wireframe", label="Refined mesh")
    p.add_mesh(removed, style="wireframe", color="red", opacity=0.2, label="Removed faces")
    p.add_title("Refined Mesh")
    p.add_legend()
    p.link_views()
    p.show()


if __name__ == "__main__":
    main()
