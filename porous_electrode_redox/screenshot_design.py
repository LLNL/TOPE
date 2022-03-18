# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

pv_version = GetParaViewVersion()
assert pv_version >= 5.7, "Wrong paraview version, we need 5.7.0."

import argparse
parser = argparse.ArgumentParser(description='Plotting optimized design')
parser.add_argument('--results_dir', dest="results_dir", type=str, help='Results directory', default='./', required=True)
parser.add_argument('--parameters', dest="parameters", type=str, help='String with all parameters of the job', default="", required=True)
parser.add_argument('--filename', dest="filename", type=str, help='filename for the screenshot', default="optimized_design", required=True)

args = parser.parse_args()

results_dir = args.results_dir
parameters = args.parameters
filename = args.filename

LoadPalette(paletteName='WhiteBackground')
# create a new 'PVD Reader'
gammafpvd = PVDReader(FileName=results_dir + '/gammaf.pvd')
gammafpvd.PointArrays = ['gamma']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [860, 522]

# get layout
layout1 = GetLayout()

# show data in view
gammafpvdDisplay = Show(gammafpvd, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'gamma'
gammaLUT = GetColorTransferFunction('gamma')

# get opacity transfer function/opacity map for 'gamma'
gammaPWF = GetOpacityTransferFunction('gamma')

# create a new 'Reflect'
reflect1 = Reflect(Input=gammafpvd)

# show data in view
reflect1Display = Show(reflect1, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
reflect1Display.Representation = 'Surface'
reflect1Display.ColorArrayName = ['POINTS', 'gamma']
reflect1Display.LookupTable = gammaLUT
reflect1Display.OSPRayScaleArray = 'gamma'
reflect1Display.OSPRayScaleFunction = 'PiecewiseFunction'
reflect1Display.SelectOrientationVectors = 'None'
reflect1Display.ScaleFactor = 0.2
reflect1Display.SelectScaleArray = 'gamma'
reflect1Display.GlyphType = 'Arrow'
reflect1Display.GlyphTableIndexArray = 'gamma'
reflect1Display.GaussianRadius = 0.01
reflect1Display.SetScaleArray = ['POINTS', 'gamma']
reflect1Display.ScaleTransferFunction = 'PiecewiseFunction'
reflect1Display.OpacityArray = ['POINTS', 'gamma']
reflect1Display.OpacityTransferFunction = 'PiecewiseFunction'
reflect1Display.DataAxesGrid = 'GridAxesRepresentation'
reflect1Display.PolarAxes = 'PolarAxesRepresentation'
reflect1Display.ScalarOpacityFunction = gammaPWF
reflect1Display.ScalarOpacityUnitDistance = 0.07232841898872833

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
reflect1Display.ScaleTransferFunction.Points = [0.00029853484799686103, 0.0, 0.5, 0.0, 0.9999412240367448, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
reflect1Display.OpacityTransferFunction.Points = [0.00029853484799686103, 0.0, 0.5, 0.0, 0.9999412240367448, 1.0, 0.5, 0.0]

# hide data in view
Hide(gammafpvd, renderView1)

# show color bar/color legend
reflect1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# trace defaults for the display properties.
gammafpvdDisplay.Representation = 'Surface'
gammafpvdDisplay.ColorArrayName = ['POINTS', 'gamma']
gammafpvdDisplay.LookupTable = gammaLUT
gammafpvdDisplay.OSPRayScaleArray = 'gamma'
gammafpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
gammafpvdDisplay.SelectOrientationVectors = 'None'
gammafpvdDisplay.ScaleFactor = 0.2
gammafpvdDisplay.SelectScaleArray = 'gamma'
gammafpvdDisplay.GlyphType = 'Arrow'
gammafpvdDisplay.GlyphTableIndexArray = 'gamma'
gammafpvdDisplay.GaussianRadius = 0.01
gammafpvdDisplay.SetScaleArray = ['POINTS', 'gamma']
gammafpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
gammafpvdDisplay.OpacityArray = ['POINTS', 'gamma']
gammafpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
gammafpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
gammafpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
gammafpvdDisplay.ScalarOpacityFunction = gammaPWF
gammafpvdDisplay.ScalarOpacityUnitDistance = 0.08803965980774502

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
gammafpvdDisplay.ScaleTransferFunction.Points = [0.00027729603023151375, 0.0, 0.5, 0.0, 0.9999438746110206, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
gammafpvdDisplay.OpacityTransferFunction.Points = [0.00027729603023151375, 0.0, 0.5, 0.0, 0.9999438746110206, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [0.0, 0.5, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
gammafpvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()


# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# hide color bar/color legend
gammafpvdDisplay.SetScalarBarVisibility(renderView1, False)

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
gammaLUT.ApplyPreset('X Ray', True)

# Properties modified on gammaLUT
gammaLUT.EnableOpacityMapping = 1

# create a new 'Box'
box1 = Box()

# Properties modified on box1
box1.XLength = 2.0
box1.Center = [0.0, 0.5, 0.0]

# show data in view
box1Display = Show(box1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
box1Display.Representation = 'Surface'
box1Display.ColorArrayName = [None, '']
box1Display.OSPRayScaleArray = 'Normals'
box1Display.OSPRayScaleFunction = 'PiecewiseFunction'
box1Display.SelectOrientationVectors = 'None'
box1Display.ScaleFactor = 0.2
box1Display.SelectScaleArray = 'None'
box1Display.GlyphType = 'Arrow'
box1Display.GlyphTableIndexArray = 'None'
box1Display.GaussianRadius = 0.01
box1Display.SetScaleArray = ['POINTS', 'Normals']
box1Display.ScaleTransferFunction = 'PiecewiseFunction'
box1Display.OpacityArray = ['POINTS', 'Normals']
box1Display.OpacityTransferFunction = 'PiecewiseFunction'
box1Display.DataAxesGrid = 'GridAxesRepresentation'
box1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
box1Display.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
box1Display.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# change representation type
box1Display.SetRepresentationType('Outline')

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [0.0, 0.5, 0.0]
renderView1.CameraParallelScale = 1.118033988749895

# save screenshot
SaveScreenshot(results_dir + '/' + filename +'.png', renderView1, ImageResolution=[1720, 1044], OverrideColorPalette='WhiteBackground', TransparentBackground=1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0, 0.5, 10000.0]
renderView1.CameraFocalPoint = [0.0, 0.5, 0.0]
renderView1.CameraParallelScale = 1.118033988749895

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
