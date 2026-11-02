"""
Evaluating the Warsaw pipeline on Articulate3D.

Articulate3D annotates the parts of articulated objects on the meshes of ScanNet++
scenes. A scene is converted into the input the pipeline reads, its ground truth is
built from a protocol written by hand before any run is judged, and a finished run is
scored against that ground truth face by face.
"""
