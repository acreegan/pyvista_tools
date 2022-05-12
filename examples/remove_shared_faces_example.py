import pyvista as pv
from pyvista_tools import remove_shared_faces_with_merge
from matplotlib import cm

mesh_filenames = ["meshes/mock_lung/lower_lobe_of_left_lung_surface.stl",
                  "meshes/mock_lung/upper_lobe_of_left_lung_surface.stl"]


def main():
    # Load meshes
    meshes = [pv.read(filename) for filename in mesh_filenames]

    # Merge and remove shared faces
    merged = remove_shared_faces_with_merge(meshes)

    # Plot results
    p = pv.Plotter(shape=(1, 2))
    cmap = cm.get_cmap("Set1")
    for i, mesh in enumerate(meshes):
        p.add_mesh(mesh, style="wireframe", color=cmap(i), label=f"Mesh {i}")
        p.add_title("Meshes with\nshared faces")
    p.subplot(0, 1)
    p.add_mesh(merged, style="wireframe")
    p.add_title("Shared faces\nremoved")
    p.link_views()
    p.show()


if __name__ == "__main__":
    main()
