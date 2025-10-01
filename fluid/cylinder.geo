// Cylinder Mesh

Nx1 = 41; Rx1 = 1.00;
Nx2 = 41; Rx2 = 1.035;
Ny = 91; Ry = 4.50;
Nb = 101; Rb= 0.965;
Nc = 41; Rc= 1.00;
// Domain points
//+
Point(1) = {-10, -10, 0, 1.0};
//+
Point(2) = {10, -10, 0, 1.0};
//+
Point(3) = {40, -10, 0, 1.0};
//+
Point(4) = {-10, 10, 0, 1.0};
//+
Point(5) = {10, 10, 0, 1.0};
//+
Point(6) = {40, 10, 0, 1.0};
// Cylinder points
//+
Point(7) = {-0.35, -0.35, 0, 1.0};
//+
Point(8) = {0.35, -0.35, 0, 1.0};
//+
Point(9) = {-0.35, 0.35, 0, 1.0};
//+
Point(10) = {0.35, 0.35, 0, 1.0};
//+
Point(11) = {0, -0, 0, 1.0};
//+

// Domain Lines
Line(1) = {1, 2}; Transfinite Curve {1} = Nx1 Using Progression Rx1;
//+
Line(2) = {2, 3}; Transfinite Curve {2} = Nx2 Using Progression Rx2;
//+
Line(3) = {4, 5}; Transfinite Curve {3} = Nx1 Using Progression Rx1;
//+
Line(4) = {5, 6}; Transfinite Curve {4} = Nx2 Using Progression Rx2;
//+
Line(5) = {1, 4}; Transfinite Curve {5} = Ny Using Bump Ry;
//+
Line(6) = {2, 5}; Transfinite Curve {6} = Ny Using Bump Ry;
//+
Line(7) = {3, 6}; Transfinite Curve {7} = Ny Using Bump Ry;

// Cylinder arc
//+
Circle(8) = {7, 11, 8}; Transfinite Curve {8} = Nc Using Progression Rc;
//+
Circle(9) = {8, 11, 10}; Transfinite Curve {9} = Ny Using Progression Rc;
//+
Circle(10) = {10, 11, 9}; Transfinite Curve {10} = Nc Using Progression Rc;
//+
Circle(11) = {9, 11, 7}; Transfinite Curve {11} = Ny Using Progression Rc;
//+

// diagonals block lines 
Line(12) = {1, 7}; Transfinite Curve {12} = Nb Using Progression Rb;
//+
Line(13) = {2, 8}; Transfinite Curve {13} = Nb Using Progression Rb;
//+
Line(14) = {5, 10}; Transfinite Curve {14} = Nb Using Progression Rb;
//+
Line(15) = {4, 9}; Transfinite Curve {15} = Nb Using Progression Rb;

// Surfaces
//+
Curve Loop(1) = {12, 8, -13, -1};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {13, 9, -14, -6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {14, 10, -15, 3};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {15, 11, -12, 5};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {7, -4, -6, 2};
//+
Plane Surface(5) = {5};
//+
Transfinite Surface {1};
//+
Transfinite Surface {2};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {5};
//+
Recombine Surface {1};
//+
Recombine Surface {2};
//+
Recombine Surface {3};
//+
Recombine Surface {4};
//+
Recombine Surface {5};
//+
Physical Surface("Fluid", 101) = {4, 1, 2, 3, 5};
//+
Physical Curve("Inlet", 1001) = {5};
//+
Physical Curve("Outlet", 1002) = {7};
//+
Physical Curve("Wall", 1003) = {4, 3, 1, 2};
