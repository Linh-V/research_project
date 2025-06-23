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