// DFG 2D-3 Benchmark: Cylinder in Channel
// Equivalent to Python FEniCSx mesh generation
SetFactory("OpenCASCADE");

// Geometry parameters (exact match to Python)
L = 2.2;      // Channel length
H = 0.41;     // Channel height
c_x = 0.2;    // Cylinder center x
c_y = 0.2;    // Cylinder center y
r = 0.05;     // Cylinder radius

// Mesh size parameters (exact match to Python)
res_min = r / 3;           // 0.01666... - fine mesh near cylinder
lc_max = 0.25 * H;         // 0.1025 - coarse mesh far field

// Create rectangle (channel)
Rectangle(1) = {0, 0, 0, L, H};

// Create disk (cylinder obstacle)
Disk(2) = {c_x, c_y, 0, r, r};

// Subtract cylinder from channel
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

// Define physical groups (exact match to Python markers)
Physical Surface("Fluid", 1) = {1};

// Note: Boundary curve numbers need to be identified after Boolean operation
// In Gmsh GUI, you can check these with Tools > Visibility
// Typically after BooleanDifference:
// - Curve 5 is the cylinder boundary
// - Curves 6,7,8,9 are the rectangle boundaries

// Physical boundary markers (matching Python)
Physical Curve("Inlet", 2) = {7};      // Left boundary (x=0)
Physical Curve("Outlet", 3) = {8};     // Right boundary (x=L)
Physical Curve("Walls", 4) = {6, 9};   // Top and bottom boundaries
Physical Curve("Obstacle", 5) = {5};   // Cylinder boundary

// Mesh size fields (exact match to Python algorithm)
// Field 1: Distance to cylinder
Field[1] = Distance;
Field[1].CurvesList = {5};  // Cylinder boundary (obstacle)
Field[1].Sampling = 100;

// Field 2: Threshold - equivalent to Python's threshold_field
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = res_min;      // LcMin = r/3
Field[2].SizeMax = lc_max;       // LcMax = 0.25*H
Field[2].DistMin = r;            // DistMin = r
Field[2].DistMax = 2 * H;        // DistMax = 2*H

// Set background mesh
Background Field = 2;

// Mesh algorithm options (exact match to Python)
Mesh.Algorithm = 8;                    // Frontal-Delaunay for quads
Mesh.RecombinationAlgorithm = 2;       // Simple full-quad recombination
Mesh.RecombineAll = 1;                 // Recombine all triangles
Mesh.SubdivisionAlgorithm = 1;         // All quadrangles
Mesh.ElementOrder = 2;                 // Second order elements

// Generate mesh
Mesh 2;
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;               // Equivalent to Python's gmsh.model.mesh.optimize("Netgen")

// Save mesh
Save "cylinder_channel_quad_fenics.msh";