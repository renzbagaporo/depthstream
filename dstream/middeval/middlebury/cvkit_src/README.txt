
Computer Vision Toolkit (cvkit)
-------------------------------

cvkit is available for Linux as well as for Windows. It offers useful tools
for viewing and analyzing images and 3d models.

sv is a simple / scientific image viewer that can display monochrome and color
images with 8 and 16 bit integer as well as 32 bit float values as data types
per color channel. Functions include showing monochrome images with color
encoding, defining radiometric ranges, zooming and automatically reloading
images (Linux only). For image comparison, settings like zoom, radiometric
range, etc, can be kept while switching between images. Depth images (full or
parts) with associated camera parameter files can be visualized on the fly in
3D. sv natively supports the PGM, PPM and PFM image formats as well as TIFF
with 8 and 16 bit integer and 32 bit float values. TIFF, JPG, PNG, GIF and many
other raster data formats are supported through optional libraries like GDAL.

plyv is a simple but pretty fast viewer for colored point clouds and meshes
with per vertex coloring, shading and texture images. It also supports
on-the-fly conversion and visualization of depth images and cameras. plyv is
based on OpenGL and can cope with big data sets that consist of many million
vertices and triangles. Mainly the PLY format is supported, which has been
invented at Stanford University as an extendable format for storing vertices
and polygons together with additional information. It is especially useful for
scanned real-world data.

See INSTALL.txt and USAGE.txt for more information.

Acknowledgments
---------------

I would like to thank Daniel Scharstein for testing the tools, giving me
feedback and motivating me to make the tools publicly available. Thanks to
Daniel Scharstein also for the code for the jet and rainbow color coding.

This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory.

Author
------

Dr. Heiko Hirschmueller
www.robotic.dlr.de/Heiko.Hirschmueller

German Aerospace Center (DLR)
Institute of Robotics and Mechatronics
Department of Perception and Cognition
Oberpfaffenhofen, Germany
