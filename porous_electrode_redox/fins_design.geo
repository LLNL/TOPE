SetFactory("OpenCASCADE");
fingers_length = 0.8;
base_height = 0.1;
fingers_width = 0.15;
n_fingers = 3;
space_width = (1.0 - (n_fingers + 1.0/2.0) * fingers_width )/ n_fingers;
size = 0.01;

Point(1) = {0.0, 0.0, 0.0, size};
Point(2) = {1.0, 0.0, 0.0, size};
Point(3) = {1.0, 1.0, 0.0, size};
Point(4) = {0.0, 1.0, 0.0, size};
Point(5) = {0.75, 1.0, 0.0, size};
Point(6) = {0.75, 0.0, 0.0, size};
j = 7;

For k In {0:n_fingers - 1}
    Point(j) = {fingers_width / 2.0 + k * (fingers_width +  space_width), base_height, 0, size};
    j = j + 1;
    Point(j) = {fingers_width / 2.0 + space_width + k*(fingers_width +  space_width) , base_height, 0, size};
    j = j + 1;
EndFor

For k In {0:n_fingers - 1}
    Point(j) = {fingers_width / 2.0 + k * (fingers_width +  space_width), 1.0, 0, size};
    j = j + 1;
    Point(j) = {fingers_width / 2.0 + space_width + k*(fingers_width +  space_width) , 1.0, 0, size};
    j = j + 1;
EndFor//+
//+
Line(1) = {4, 13};
//+
Line(2) = {13, 14};
//+
Line(3) = {14, 15};
//+
Line(4) = {15, 16};
//+
Line(5) = {16, 17};
//+
Line(6) = {17, 5};
//+
Line(7) = {5, 18};
//+
Line(8) = {18, 3};
//+
Line(9) = {3, 2};
//+
Line(10) = {2, 6};
//+
Line(11) = {6, 1};
//+
Line(12) = {1, 4};
//+
Line(13) = {13, 7};
//+
Line(14) = {7, 8};
//+
Line(15) = {8, 14};
//+
Line(16) = {15, 9};
//+
Line(17) = {9, 10};
//+
Line(18) = {10, 16};
//+
Line(19) = {17, 11};
//+
Line(20) = {11, 12};
//+
Line(21) = {12, 18};
//+
Curve Loop(1) = {12, 1, 13, 14, 15, 3, 16, 17, 18, 5, 19, 20, 21, 8, 9, 10, 11};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {2, -15, -14, -13};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, -18, -17, -16};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {6, 7, -21, -20, -19};
//+
Plane Surface(4) = {4};
//+
Physical Surface(1) = {1};
//+
Physical Surface(2) = {2, 3, 4};
//+
Physical Curve(2) = {1, 2, 3, 4, 5, 6};
//+
Physical Curve(1) = {11};
