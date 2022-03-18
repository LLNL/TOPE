# trace generated using paraview version 5.8.0
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
parser.add_argument('--initial_design', dest="initial_design", type=str, help='initial design', default="uniform", required=True)

args = parser.parse_args()

results_dir = args.results_dir
parameters = args.parameters
filename = args.filename
initial_design = args.initial_design

# get the time-keeper
timeKeeper1 = GetTimeKeeper()


# get color transfer function/color map for 'gamma'
gammaLUT = GetColorTransferFunction('gamma')

# Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
gammaLUT.ApplyPreset('X Ray', True)

# get opacity transfer function/opacity map for 'gamma'
gammaPWF = GetOpacityTransferFunction('gamma')

# Properties modified on gammaLUT
gammaLUT.EnableOpacityMapping = 1

LoadPalette(paletteName='WhiteBackground')
# create a new 'PVD Reader'
geometry_uniformpvd = PVDReader(FileName=results_dir + '/geometry_' + initial_design + '.pvd')
geometry_uniformpvd.PointArrays = ['gamma']

# get animation scene
animationScene1 = GetAnimationScene()
animationScene1.GoToLast()

# create a new 'Reflect'
reflect1 = Reflect(Input=geometry_uniformpvd)

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1627, 880]

# get layout
layout1 = GetLayout()

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
reflect1Display.ScalarOpacityUnitDistance = 0.07231536861738841

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
reflect1Display.ScaleTransferFunction.Points = [2.2077453022765e-05, 0.0, 0.5, 0.0, 0.9832517795116531, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
reflect1Display.OpacityTransferFunction.Points = [2.2077453022765e-05, 0.0, 0.5, 0.0, 0.9832517795116531, 1.0, 0.5, 0.0]

# hide data in view
Hide(geometry_uniformpvd, renderView1)

# show color bar/color legend
reflect1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# hide color bar/color legend
reflect1Display.SetScalarBarVisibility(renderView1, False)

# Show orientation axes
renderView1.OrientationAxesVisibility = 1

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Hide orientation axes
renderView1.OrientationAxesVisibility = 0

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.14483954854039316, 0.46946584353967186, 10000.0]
renderView1.CameraFocalPoint = [0.14483954854039316, 0.46946584353967186, 0.0]
renderView1.CameraParallelScale = 0.7071067811865476


# Properties modified on box1
# create a new 'Box'
box1 = Box()
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

# save screenshot
SaveScreenshot(results_dir + '/' + filename + '.png', renderView1, ImageResolution=[1804, 871], OverrideColorPalette='WhiteBackground', TransparentBackground=1)
