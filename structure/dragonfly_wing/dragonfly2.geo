// Definition of the Geometry 
////////////////////////////////////////////////////////////

lc = 1e-2;
L = 10; // (cm)
// Bottom layer points
Point(1) = {0, 0, 0, lc};
Point(2) = {0.09*L, 0, 0, lc};
Point(3) = {0.14*L, 0.055*L, 0, lc};
Point(4) = {0.19*L, 0, 0, lc};
Point(5) = {0.24*L, 0.055*L, 0, lc};
Point(6) = {0.29*L, 0, 0, lc};
Point(7) = {0.34*L, 0, 0, lc};
Point(8) = {0.5*L, 0.02*L, 0, lc};
Point(9) = {0.6*L, 0.06*L, 0, lc};
Point(10) = {0.7*L, 0.08*L, 0, lc};
Point(11) = {0.75*L, 0.085*L, 0, lc};
Point(12) = {0.8*L, 0.08*L, 0, lc};
Point(13) = {0.85*L, 0.07*L, 0, lc};
Point(14) = {0.9*L, 0.055*L, 0, lc};
Point(15) = {0.95*L, 0.04*L, 0, lc};
Point(16) = {1*L, 0.025*L, 0, lc};

// Top layer points (y + 0.005)
Point(17) = {0, 0.005*L, 0, lc};
Point(18) = {0.09*L, 0.005*L, 0, lc};
Point(19) = {0.14*L, 0.06*L, 0, lc};
Point(20) = {0.19*L, 0.005*L, 0, lc};
Point(21) = {0.24*L, 0.06*L, 0, lc};
Point(22) = {0.29*L, 0.005*L, 0, lc};
Point(23) = {0.34*L, 0.005*L, 0, lc};
Point(24) = {0.5*L, 0.025*L, 0, lc};
Point(25) = {0.6*L, 0.065*L, 0, lc};
Point(26) = {0.7*L, 0.085*L, 0, lc};
Point(27) = {0.75*L, 0.09*L, 0, lc};
Point(28) = {0.8*L, 0.085*L, 0, lc};
Point(29) = {0.85*L, 0.075*L, 0, lc};
Point(30) = {0.9*L, 0.06*L, 0, lc};
Point(31) = {0.95*L, 0.045*L, 0, lc};
Point(32) = {1*L, 0.03*L, 0, lc};

// Leading edge
Line(1) = {1, 17};

// Bottom surface
Line(2) = {1, 2};
Line(3) = {2, 3};
Line(4) = {3, 4};
Line(5) = {4, 5};
Line(6) = {5, 6};
Line(7) = {6, 7};
Line(8) = {7, 8};
Line(9) = {8, 9};
Line(10) = {9, 10};
Line(11) = {10, 11};
Line(12) = {11, 12};
Line(13) = {12, 13};
Line(14) = {13, 14};
Line(15) = {14, 15};
Line(16) = {15, 16};

// Trailing edge
Line(17) = {16, 32};

// Top surface (reverse order)
Line(18) = {32, 31};
Line(19) = {31, 30};
Line(20) = {30, 29};
Line(21) = {29, 28};
Line(22) = {28, 27};
Line(23) = {27, 26};
Line(24) = {26, 25};
Line(25) = {25, 24};
Line(26) = {24, 23};
Line(27) = {23, 22};
Line(28) = {22, 21};
Line(29) = {21, 20};
Line(30) = {20, 19};
Line(31) = {19, 18};
Line(32) = {18, 17};

// Define the closed loop (corrected order)
Line Loop(1) = {
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,   // bottom surface
  17,                                                  // trailing edge
  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, // top surface
  -1                                                   // leading edge (backwards)
};

// Define the surface
Plane Surface(1) = {1};

//Domain used for assembling step
Physical Surface(1) = {1};

// Boundary tags
LEADING  = 10;
TRAILING = 20;
TOP      = 30;
BOTTOM   = 40;

Physical Line(LEADING)  = {1};
Physical Line(BOTTOM)   = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
Physical Line(TRAILING) = {17};
Physical Line(TOP)      = {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};

// Mesh controls (if needed)
//Recombine Surface{1}; 
Mesh.Algorithm = 6;
Mesh 2;
Save "dragonfly2.msh";

