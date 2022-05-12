import pyvista as pv
from pyvista_tools import remove_shared_faces_with_merge, extract_enclosed_regions
from matplotlib import cm

mesh_filenames = ["meshes/mock_lung/lower_lobe_of_left_lung_surface.stl",
                  "meshes/mock_lung/upper_lobe_of_left_lung_surface.stl"]


def main():
    # Prepare example mesh
    meshes = [pv.read(filename) for filename in mesh_filenames]
    merged = remove_shared_faces_with_merge(meshes, keep_one=True)

    # Extract enclosed regions
    regions = extract_enclosed_regions(merged)

    # Plot results

    cmap = cm.get_cmap("Set1")
    p = pv.Plotter(shape=(1, 2))
    p.add_mesh(merged, style="wireframe")
    p.add_title("Mesh with\ninner wall")
    p.subplot(0, 1)
    for i, mesh in enumerate(regions):
        p.add_mesh(mesh, style="wireframe", color=cmap(i), label=f"Region {i}")
    p.add_title("Enclosed regions")
    p.add_legend()
    p.link_views()
    p.show()


if __name__ == "__main__":
    main()
