import getfem as gf
import numpy as np
import os

def verify_regions(mesh, base_filename="mesh_regions"):

    """
    Export each mesh region to individual VTK files in a dedicated folder.
    
    Parameters:
    -----------
    mesh : gf.Mesh
        Any GetFEM mesh object
    base_filename : str
        Base name for output folder (default: "mesh_regions")
    
    Returns:
    --------
    tuple : (folder_path, list of created VTK filenames)
    """
    
    # Create output directory
    output_dir = f"{base_filename}_analysis"
    os.makedirs(output_dir, exist_ok=True)

    
    files_created = []
    regions = mesh.regions()
    
    print(f"Found {len(regions)} regions: {sorted(regions)}")
    
    # Export individual region files
    for region_id in sorted(regions):
        try:
            # Create mesh function
            mf = gf.MeshFem(mesh, 1)
            mf.set_classical_fem(1)
            
            # Create field that's 1 on this region, 0 elsewhere
            field = np.zeros(mf.nbdof())
            
            # Get DOFs on this region
            region_dofs = mf.basic_dof_on_region(region_id)
          
            field[region_dofs] = 1.0
            
            # Export to VTK
            filename = os.path.join(output_dir, f"region_{region_id}.vtk")
            mf.export_to_vtk(filename, field, f"Region_{region_id}")
            files_created.append(filename)
            print(f" Created: {filename}")
            
        except Exception as e:
            print(f" Region {region_id}: Failed - {e}")
    
    print(f"\n Done! Created {len(files_created)} VTK files in '{output_dir}'")
    
    return output_dir, files_created


def mesh_statistics_corrected(mesh, name="Mesh"):
    """
    Correct mesh statistics using actual GetFEM methods
    """
    print(f"\n=== {name} Statistics ===")
    
    # Basic info
    print(f"Dimension: {mesh.dim()}D")
    print(f"Number of points: {mesh.nbpts()}")
    print(f"Number of elements: {mesh.nbcvs()}")
    
    # Get point coordinates for bounding box
    points = mesh.pts()
    bbox_min = np.min(points, axis=1)
    bbox_max = np.max(points, axis=1)
    
    print(f"\nBounding box:")
    for i in range(mesh.dim()):
        length = bbox_max[i] - bbox_min[i]
        print(f"  Dimension {i}: [{bbox_min[i]:.6f}, {bbox_max[i]:.6f}] (length: {length:.6f})")
    
    # Element areas/volumes (this gives actual area in 2D, volume in 3D, length in 1D)
    element_areas = mesh.convex_area()
    print(f"\nElement areas/volumes:")
    print(f"  Min: {np.min(element_areas):.6f}")
    print(f"  Max: {np.max(element_areas):.6f}")
    print(f"  Mean: {np.mean(element_areas):.6f}")
    print(f"  Std: {np.std(element_areas):.6f}")
    
    # Element characteristic sizes (derived from area/volume)
    if mesh.dim() == 1:
        element_sizes = element_areas  # For 1D, area is actually length
    elif mesh.dim() == 2:
        element_sizes = np.sqrt(element_areas)  # sqrt(area) for 2D
    elif mesh.dim() == 3:
        element_sizes = np.power(element_areas, 1/3)  # cbrt(volume) for 3D
    
    print(f"\nCharacteristic element sizes:")
    print(f"  Min: {np.min(element_sizes):.6f}")
    print(f"  Max: {np.max(element_sizes):.6f}")
    print(f"  Mean: {np.mean(element_sizes):.6f}")
    print(f"  Ratio (max/min): {np.max(element_sizes)/np.min(element_sizes):.2f}")
    
    # Element radius estimates
    element_radii = mesh.convex_radius()
    print(f"\nElement radius estimates:")
    print(f"  Min: {np.min(element_radii):.6f}")
    print(f"  Max: {np.max(element_radii):.6f}")
    print(f"  Mean: {np.mean(element_radii):.6f}")
    
    # Mesh quality
    qualities = mesh.quality()
    print(f"\nMesh quality (0-1, higher is better):")
    print(f"  Min: {np.min(qualities):.3f}")
    print(f"  Mean: {np.mean(qualities):.3f}")
    print(f"  Elements with quality < 0.3: {np.sum(qualities < 0.3)}")
    print(f"  Elements with quality < 0.1: {np.sum(qualities < 0.1)}")
    
    return {
        'dim': mesh.dim(),
        'nb_points': mesh.nbpts(),
        'nb_elements': mesh.nbcvs(),
        'element_areas': element_areas,
        'element_sizes': element_sizes,
        'element_radii': element_radii,
        'qualities': qualities,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'domain_size': bbox_max - bbox_min
    }

