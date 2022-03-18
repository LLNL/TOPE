import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD
from pyop2.datatypes import IntType

from firedrake import VectorFunctionSpace, Function, Constant, \
    par_loop, dx, WRITE, READ, interpolate, FiniteElement, interval, tetrahedron
from firedrake.cython import dmcommon
from firedrake import mesh
from firedrake import function
from firedrake import functionspace

from pyadjoint.tape import no_annotations

def RectangleMeshLeft(nx, ny, Lx, Lm, Lw, quadrilateral=False, reorder=None,
                  diagonal="left", distribution_parameters=None, comm=COMM_WORLD):
    """Generate a rectangular mesh
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).
    The boundary edges in this mesh are numbered as follows:
    * 1: Lw < y < Lm+Lw, x==0
    * 2: everything else
    """
    Ly = Lm + 2.*Lw

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [i*(ny+1) + j,
                 i*(ny+1) + j+1,
                 (i+1)*(ny+1) + j,
                 (i+1)*(ny+1) + j+1,
                 (nx+1)*(ny+1) + i*ny + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)
    else:
        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[1]-(Lw+Lm/2.)) < Lm/2.+ytol and abs(face_coords[3]-(Lw+Lm/2.)) < Lm/2.+ytol and \
                    abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)

def RectangleMesh1(nx, ny, Lm, Lw, Ly, quadrilateral=False, reorder=None,
                  diagonal="left", distribution_parameters=None, comm=COMM_WORLD):
    """Generate a rectangular mesh
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).
    The boundary edges in this mesh are numbered as follows:
    * 1: Lw < x < Lm+Lw, y==0
    * 2: everything else
    """
    Lx = Lm + 2.*Lw

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [i*(ny+1) + j,
                 i*(ny+1) + j+1,
                 (i+1)*(ny+1) + j,
                 (i+1)*(ny+1) + j+1,
                 (nx+1)*(ny+1) + i*ny + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)
    else:
        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)


def RectangleMesh2(nx, ny, Lm, Lw, Ly, quadrilateral=False, reorder=None,
                  diagonal="left", distribution_parameters=None, comm=COMM_WORLD):
    """Generate a rectangular mesh
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).
    The boundary edges in this mesh are numbered as follows:
    * 1: Lw < x < Lm+Lw, y==0
    * 2: Lw < x < Lm+Lw, y==Ly
    * 3: everything else
    """
    Lx = Lm + 2.*Lw

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [i*(ny+1) + j,
                 i*(ny+1) + j+1,
                 (i+1)*(ny+1) + j,
                 (i+1)*(ny+1) + j+1,
                 (nx+1)*(ny+1) + i*ny + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)
    else:
        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            elif abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[1]-Ly) < ytol and abs(face_coords[3]-Ly) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)

def RectangleMesh3(nx, ny, Lm, Lw, Ly, quadrilateral=False, reorder=None,
                  diagonal="left", distribution_parameters=None, comm=COMM_WORLD):
    """Generate a rectangular mesh
    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg diagonal: For triangular meshes, should the diagonal got
        from bottom left to top right (``"right"``), or top left to
        bottom right (``"left"``), or put in both diagonals (``"crossed"``).
    The boundary edges in this mesh are numbered as follows:
    * 1: plane x == 0
    * 2: plane x == Lx
    * 3: Lw < x < Lm+Lw, y==0
    * 4: Lw < x < Lm+Lw, y==Ly
    * 5: everything else
    """
    Lx = Lm + 2.*Lw

    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [i*(ny+1) + j,
                 i*(ny+1) + j+1,
                 (i+1)*(ny+1) + j,
                 (i+1)*(ny+1) + j+1,
                 (nx+1)*(ny+1) + i*ny + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)
    else:
        cells = [i*(ny+1) + j, i*(ny+1) + j+1, (i+1)*(ny+1) + j+1, (i+1)*(ny+1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

    plex = mesh._from_cell_list(2, cells, coords, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            elif abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            elif abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
            elif abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[1]-Ly) < ytol and abs(face_coords[3]-Ly) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 5)
    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)


def BoxMesh2(nx, ny, nz, Lm, Lw, Ly, reorder=None, distribution_parameters=None, diagonal="default", comm=COMM_WORLD):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :arg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary surfaces are numbered as follows:

    * 1: Lw < x < Lm+Lw, y==0
    * 2: Lw < x < Lm+Lw, y==Ly
    * 3: everything else
    """
    Lx = Lm + 2.*Lw
    Lz = Lm + 2.*Lw
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
    zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
    # X moves fastest, then Y, then Z
    coords = np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nx, dtype=np.int32),
                          np.arange(ny, dtype=np.int32),
                          np.arange(nz, dtype=np.int32))
    if diagonal == "default":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        cells = [v0, v1, v3, v7,
                 v0, v1, v7, v5,
                 v0, v5, v7, v4,
                 v0, v3, v2, v7,
                 v0, v6, v4, v7,
                 v0, v2, v6, v7]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
    elif diagonal == "crossed":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        # There are only five tetrahedra in this cutting of hexahedra
        cells = [v0, v1, v2, v4,
                 v1, v7, v5, v4,
                 v1, v2, v3, v7,
                 v2, v4, v6, v7,
                 v1, v2, v7, v4]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
    else:
        raise ValueError("Unrecognised value for diagonal '%r'", diagonal)

    plex = mesh._from_cell_list(3, cells, coords, comm)

    # Apply boundary IDs
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        ztol = Lz/(2*nz)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[3]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[6]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+ztol and \
                    abs(face_coords[5]-(Lw+Lm/2.)) < Lm/2.+ztol and abs(face_coords[8]-(Lw+Lm/2.)) < Lm/2.+ztol and \
                    abs(face_coords[1]) < ytol and abs(face_coords[4]) < ytol and abs(face_coords[7]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            elif abs(face_coords[0]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[3]-(Lw+Lm/2.)) < Lm/2.+xtol and \
                    abs(face_coords[6]-(Lw+Lm/2.)) < Lm/2.+xtol and abs(face_coords[2]-(Lw+Lm/2.)) < Lm/2.+ztol and \
                    abs(face_coords[5]-(Lw+Lm/2.)) < Lm/2.+ztol and abs(face_coords[8]-(Lw+Lm/2.)) < Lm/2.+ztol and \
                    abs(face_coords[1]-Ly) < ytol and abs(face_coords[4]-Ly) < ytol and abs(face_coords[7]-Ly) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)

    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)



def BoxMesh3(nx, ny, nz, Lm, Lw, Ly, reorder=None, distribution_parameters=None, diagonal="default", comm=COMM_WORLD):
    """Generate a mesh of a 3D box.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg nz: The number of cells in the z direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :arg Lz: The extent in the z direction
    :arg diagonal: Two ways of cutting hexadra, should be cut into 6
        tetrahedra (``"default"``), or 5 tetrahedra thus less biased
        (``"crossed"``)
    :kwarg reorder: (optional), should the mesh be reordered?
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).

    The boundary surfaces are numbered as follows:

    * 1: Lw < x < Lm+Lw, y==0
    * 2: Lw < x < Lm+Lw, y==Ly
    * 3: everything else
    """
    Lx = Lm + Lw
    Lz = Lm + Lw
    for n in (nx, ny, nz):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a postive integer")

    xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
    zcoords = np.linspace(0, Lz, nz + 1, dtype=np.double)
    # X moves fastest, then Y, then Z
    coords = np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
    i, j, k = np.meshgrid(np.arange(nx, dtype=np.int32),
                          np.arange(ny, dtype=np.int32),
                          np.arange(nz, dtype=np.int32))
    if diagonal == "default":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        cells = [v0, v1, v3, v7,
                 v0, v1, v7, v5,
                 v0, v5, v7, v4,
                 v0, v3, v2, v7,
                 v0, v6, v4, v7,
                 v0, v2, v6, v7]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
    elif diagonal == "crossed":
        v0 = k*(nx + 1)*(ny + 1) + j*(nx + 1) + i
        v1 = v0 + 1
        v2 = v0 + (nx + 1)
        v3 = v1 + (nx + 1)
        v4 = v0 + (nx + 1)*(ny + 1)
        v5 = v1 + (nx + 1)*(ny + 1)
        v6 = v2 + (nx + 1)*(ny + 1)
        v7 = v3 + (nx + 1)*(ny + 1)

        # There are only five tetrahedra in this cutting of hexahedra
        cells = [v0, v1, v2, v4,
                 v1, v7, v5, v4,
                 v1, v2, v3, v7,
                 v2, v4, v6, v7,
                 v1, v2, v7, v4]
        cells = np.asarray(cells).swapaxes(0, 3).reshape(-1, 4)
    else:
        raise ValueError("Unrecognised value for diagonal '%r'", diagonal)

    plex = mesh._from_cell_list(3, cells, coords, comm)

    # Apply boundary IDs
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx/(2*nx)
        ytol = Ly/(2*ny)
        ztol = Lz/(2*nz)
        for face in boundary_faces:
            face_coords = plex.vecGetClosure(coord_sec, coords, face)
            if abs(face_coords[0]) < Lm+xtol and abs(face_coords[3]) < Lm+xtol and abs(face_coords[6]) < Lm+xtol and \
                    abs(face_coords[2]) < Lm+xtol and abs(face_coords[5]) < Lm+xtol and abs(face_coords[8]) < Lm+xtol and \
                    abs(face_coords[1]) < ytol and abs(face_coords[4]) < ytol and abs(face_coords[7]) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
            elif abs(face_coords[0]) < Lm+xtol and abs(face_coords[3]) < Lm+xtol and abs(face_coords[6]) < Lm+xtol and \
                    abs(face_coords[2]) < Lm+xtol and abs(face_coords[5]) < Lm+xtol and abs(face_coords[8]) < Lm+xtol and\
                    abs(face_coords[1]-Ly) < ytol and abs(face_coords[4]-Ly) < ytol and abs(face_coords[7]-Ly) < ytol:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
            else:
                plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)

    return mesh.Mesh(plex, reorder=reorder, distribution_parameters=distribution_parameters)
