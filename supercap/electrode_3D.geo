size = 0.00905;
//+
Point(1) = {0, 0, 0, size};
//+
Point(2) = {0.75, 0, 0, size};
//+
Point(3) = {1, 0, 0, size};
//+
Point(4) = {1, 1, 0, size};
//+
Point(5) = {0, 1, 0, size};
//+
Point(6) = {0.75, 0.75, 0, size};
//+
Point(7) = {0, 0.75, 0.0, size};
//+
Point(8) = {0, 0, 1.0, size};
//+
Point(9) = {0.75, 0, 1.0, size};
//+
Point(10) = {1, 0, 1.0, size};
//+
Point(11) = {1, 1, 1.0, size};
//+
Point(12) = {0, 1, 1.0, size};
//+
Point(13) = {0.75, 0.75, 1.0, size};
//+
Point(14) = {0, 0.75, 1.0, size};
//+

//+
Line(1) = {1, 2};
//+
Line(2) = {3, 3};
//+
Line(3) = {4, 4};
//+
Line(4) = {5, 4};
//+
Line(5) = {4, 3};
//+
Line(6) = {3, 2};
//+
Line(7) = {2, 6};
//+
Line(8) = {6, 7};
//+
Line(9) = {7, 1};
//+
Line(10) = {1, 8};
//+
Line(11) = {8, 14};
//+
Line(12) = {14, 12};
//+
Line(13) = {12, 11};
//+
Line(14) = {11, 10};
//+
Line(15) = {10, 9};
//+
Line(16) = {9, 8};
//+
Line(17) = {14, 13};
//+
Line(18) = {13, 9};
//+
Line(19) = {7, 5};
//+
Line(20) = {5, 12};
//+
Line(21) = {11, 4};
//+
Line(22) = {10, 3};
//+
Curve Loop(1) = {8, 9, 1, 7};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {4, 5, 6, 7, 8, 19};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {18, 16, 11, 17};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, 14, 15, -18, -17, 12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {22, -5, -21, 14};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {9, 10, 11, 12, -20, -19};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {1, -6, -22, 15, 16, -10};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {4, -21, -13, -20};
//+
Plane Surface(8) = {8};
//+
Surface Loop(1) = {6, 1, 2, 8, 5, 7, 4, 3};
//+
Volume(1) = {1};
//+
Physical Surface(1) = {1};
//+
Physical Surface(2) = {3};
//+
Physical Volume(0) = {1};
