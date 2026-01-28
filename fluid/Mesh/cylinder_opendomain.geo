// Cylinder in Channel open domain
SetFactory("OpenCASCADE");

// Geometry parameters
L = 2.2;      // Channel length
H = 0.41;     // Channel height
c_x = 0.2;    // Cylinder center x
c_y = 0.2;    // Cylinder center y
r = 0.05;     // Cylinder radius

// Mesh size parameters
res_min = r / 3;           
lc_max = 0.5 * H;         

// Create rectangle: L behind cylinder, L/2 in front, H above and below
Rectangle(1) = {c_x - L/2, c_y - H, 0, L + L/2, 2*H};

// Create disk (cylinder obstacle)
Disk(2) = {c_x, c_y, 0, r, r};

// Subtract cylinder from channel
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }

// Physical groups
Physical Surface("Fluid", 1) = {1};
Physical Curve("Inlet", 2) = {7};      
Physical Curve("Outlet", 3) = {8};     
Physical Curve("Walls", 4) = {6, 9};   
Physical Curve("Obstacle", 5) = {5};   

// Mesh size fields
Field[1] = Distance;
Field[1].CurvesList = {5};
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = res_min;
Field[2].SizeMax = lc_max;
Field[2].DistMin = r;
Field[2].DistMax = 2 * H;

Background Field = 2;

Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
Mesh.SubdivisionAlgorithm = 1;

Mesh 2;
Mesh.ElementOrder = 2;
SetOrder 2;
Mesh.Optimize = 1;
OptimizeMesh "Netgen";

Save "cylinder_opendomain.msh";
