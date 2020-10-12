# Test for internal wave atlas 

**Final Goal**: Draw a line on a map and gather all of the inputs for a KdV run

 - Load an unstructured grid atlas file (requires its own class not in soda)
    - Test class methods:
        - __init__
        - interpolate
        - plot
        - contour
        - find_nearest
 - Interpolate SSH amplitudes onto a point
 - Interpolate time-series of slowly varying amplitude onto a point
 - Interpolate a time-series of slowly varying amplitude onto a line
 - Directional Fourier filter of a single amplitude array (with a box of choice)
 - Generate a DFF time-series at a point (w/ a specified cutoff angle range): used as a KdV BC
 - Generate a DFF time-series along a line: used as a regional model BC
 - Interpolate rho onto a point and calculate mode-function
