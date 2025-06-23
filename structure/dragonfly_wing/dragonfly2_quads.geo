// Definition of the Geometry with Structured Quad Mesh
////////////////////////////////////////////////////////////

lc = 1e-2;
L = 10; // (cm)
thickness = 0.005*L;

// Number of elements through thickness (important for thin structures)
n_thickness = 8;  // At least 2 elements through thickness
n_length = 100;   // Elements along length
n = 10;
// Define centerline points first
Point(1000) = {0, 0, 0, lc};
Point(1001) = {0.09*L, 0, 0, lc};
Point(1002) = {0.14*L, 0.055*L, 0, lc};
Point(1003) = {0.19*L, 0, 0, lc};
Point(1004) = {0.24*L, 0.055*L, 0, lc};
Point(1005) = {0.29*L, 0, 0, lc};
Point(1006) = {0.34*L, 0, 0, lc};
Point(1007) = {0.5*L, 0.02*L, 0, lc};
Point(1008) = {0.6*L, 0.06*L, 0, lc};
Point(1009) = {0.7*L, 0.08*L, 0, lc};
Point(1010) = {0.75*L, 0.085*L, 0, lc};
Point(1011) = {0.8*L, 0.08*L, 0, lc};
Point(1012) = {0.85*L, 0.07*L, 0, lc};
Point(1013) = {0.9*L, 0.055*L, 0, lc};
Point(1014) = {0.95*L, 0.04*L, 0, lc};
Point(1015) = {1*L, 0.025*L, 0, lc};

// Bottom layer points
Point(1) = {0, -thickness/2, 0, lc};
Point(2) = {0.09*L, -thickness/2, 0, lc};
Point(3) = {0.14*L, 0.055*L - thickness/2, 0, lc};
Point(4) = {0.19*L, -thickness/2, 0, lc};
Point(5) = {0.24*L, 0.055*L - thickness/2, 0, lc};
Point(6) = {0.29*L, -thickness/2, 0, lc};
Point(7) = {0.34*L, -thickness/2, 0, lc};
Point(8) = {0.5*L, 0.02*L - thickness/2, 0, lc};
Point(9) = {0.6*L, 0.06*L - thickness/2, 0, lc};
Point(10) = {0.7*L, 0.08*L - thickness/2, 0, lc};
Point(11) = {0.75*L, 0.085*L - thickness/2, 0, lc};
Point(12) = {0.8*L, 0.08*L - thickness/2, 0, lc};
Point(13) = {0.85*L, 0.07*L - thickness/2, 0, lc};
Point(14) = {0.9*L, 0.055*L - thickness/2, 0, lc};
Point(15) = {0.95*L, 0.04*L - thickness/2, 0, lc};
Point(16) = {1*L, 0.025*L - thickness/2, 0, lc};

// Top layer points
Point(17) = {0, thickness/2, 0, lc};
Point(18) = {0.09*L, thickness/2, 0, lc};
Point(19) = {0.14*L, 0.055*L + thickness/2, 0, lc};
Point(20) = {0.19*L, thickness/2, 0, lc};
Point(21) = {0.24*L, 0.055*L + thickness/2, 0, lc};
Point(22) = {0.29*L, thickness/2, 0, lc};
Point(23) = {0.34*L, thickness/2, 0, lc};
Point(24) = {0.5*L, 0.02*L + thickness/2, 0, lc};
Point(25) = {0.6*L, 0.06*L + thickness/2, 0, lc};
Point(26) = {0.7*L, 0.08*L + thickness/2, 0, lc};
Point(27) = {0.75*L, 0.085*L + thickness/2, 0, lc};
Point(28) = {0.8*L, 0.08*L + thickness/2, 0, lc};
Point(29) = {0.85*L, 0.07*L + thickness/2, 0, lc};
Point(30) = {0.9*L, 0.055*L + thickness/2, 0, lc};
Point(31) = {0.95*L, 0.04*L + thickness/2, 0, lc};
Point(32) = {1*L, 0.025*L + thickness/2, 0, lc};

// Create lines for structured mesh
// Bottom contour
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 15};
Line(15) = {15, 16};

// Top contour
Line(16) = {17, 18};
Line(17) = {18, 19};
Line(18) = {19, 20};
Line(19) = {20, 21};
Line(20) = {21, 22};
Line(21) = {22, 23};
Line(22) = {23, 24};
Line(23) = {24, 25};
Line(24) = {25, 26};
Line(25) = {26, 27};
Line(26) = {27, 28};
Line(27) = {28, 29};
Line(28) = {29, 30};
Line(29) = {30, 31};
Line(30) = {31, 32};

// Transverse lines (connecting bottom to top)
Line(31) = {1, 17};   // Leading edge
Line(32) = {2, 18};
Line(33) = {3, 19};
Line(34) = {4, 20};
Line(35) = {5, 21};
Line(36) = {6, 22};
Line(37) = {7, 23};
Line(38) = {8, 24};
Line(39) = {9, 25};
Line(40) = {10, 26};
Line(41) = {11, 27};
Line(42) = {12, 28};
Line(43) = {13, 29};
Line(44) = {14, 30};
Line(45) = {15, 31};
Line(46) = {16, 32};  // Trailing edge

// Create surfaces between each segment for structured mesh
// This creates individual quads that can be meshed with structured algorithm

// Segment 1: Points 1-2-18-17
Line Loop(1) = {1, 32, -16, -31};
Plane Surface(1) = {1};
Transfinite Line {1, 16} = 10;  // Along length
Transfinite Line {31, 32} = n_thickness;  // Through thickness
Transfinite Surface {1};
Recombine Surface {1};

// Segment 2: Points 2-3-19-18
Line Loop(2) = {2, 33, -17, -32};
Plane Surface(2) = {2};
Transfinite Line {2, 17} = 6;  // Fewer elements on short segment
Transfinite Line {32, 33} = n_thickness;
Transfinite Surface {2};
Recombine Surface {2};

// Continue for all segments...
// I'll show a more automated approach:

// Define all surfaces
surf_count = 1;
For i In {1:15}
    Line Loop(surf_count) = {i, i+31, -(i+15), -(i+30)};
    Plane Surface(surf_count) = {surf_count};
    Transfinite Surface {surf_count};
    Recombine Surface {surf_count};
    surf_count += 1;
EndFor

// Set transfinite constraints on all transverse lines
For i In {31:46}
    Transfinite Line {i} = n_thickness;
EndFor

// Set appropriate divisions for longitudinal lines based on segment length
// Adjust these based on your needs
Transfinite Line {1, 16} = 20*n;
Transfinite Line {2, 17} = 15*n;
Transfinite Line {3, 18} = 15*n;
Transfinite Line {4, 19} = 15*n;
Transfinite Line {5, 20} = 15*n;
Transfinite Line {6, 21} = 15*n;
Transfinite Line {7, 22} = 18*n;
Transfinite Line {8, 23} = 20*n;
Transfinite Line {9, 24} = 20*n;
Transfinite Line {10, 25} = 15*n;
Transfinite Line {11, 26} = 15*n;
Transfinite Line {12, 27} = 15*n;
Transfinite Line {13, 28} = 15*n;
Transfinite Line {14, 29} = 15*n;
Transfinite Line {15, 30} = 15*n;

// Physical groups
Physical Surface("Wing") = {1:15};

// Boundary tags
Physical Line("Leading") = {31};
Physical Line("Trailing") = {46};
Physical Line("Bottom") = {1:15};
Physical Line("Top") = {16:30};

// Mesh settings
Mesh.Algorithm = 8;  // Frontal-Delaunay for quads
Mesh.RecombineAll = 1;  // Force quad elements
Mesh.RecombinationAlgorithm = 1;  // Blossom
Mesh.ElementOrder = 2;  // Quadratic elements (better for bending)

Mesh 2;
Save "dragonfly_quad.msh";
