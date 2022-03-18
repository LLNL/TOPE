import pygmsh as pg
import sys
import gmsh
from parameters import DIRICHLET_1, DIRICHLET_2

def electrode_mesh(Lm, Lw, Ly, size=0.1, filename="electrode_mesh.msh"):
    BOTTOM = DIRICHLET_1
    TOP = DIRICHLET_2
    gmsh.initialize(sys.argv)
    gmsh.model.add("t10")

    # Let's create a simple rectangular geometry:
    lc = size
    gmsh.model.geo.addPoint(0.0, 0.0, 0, lc, 1)
    gmsh.model.geo.addPoint(Lm, 0.0, 0, lc, 2)
    gmsh.model.geo.addPoint(Lm + Lw, 0.0, 0, lc, 3)
    gmsh.model.geo.addPoint(Lm + Lw, Ly, 0, lc, 4)
    gmsh.model.geo.addPoint(Lm, Ly, 0, lc, 5)
    gmsh.model.geo.addPoint(0.0, Ly, 0, lc, 6)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 5, 4)
    gmsh.model.geo.addLine(5, 6, 5)
    gmsh.model.geo.addLine(6, 1, 6)

    gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6], 7)
    gmsh.model.geo.addPlaneSurface([7], 8)

    gmsh.model.geo.addPhysicalGroup(1, [1], BOTTOM)
    gmsh.model.geo.addPhysicalGroup(1, [5], TOP)
    gmsh.model.geo.addPhysicalGroup(2, [8], 1)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("electrode_mesh.geo_unrolled")
    gmsh.write(filename)
    gmsh.finalize()

if __name__ == "__main__":
    Lm = 1.5
    Lw = 0.25
    Ly = 1.
    electrode_mesh(Lm, Lw, Ly, size=Lw / 20)
