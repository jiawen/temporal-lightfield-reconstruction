***********************************************************************
*** An implementation of
***
*** Lehtinen, J., Aila, T., Chen, J., Laine, S., and Durand, F. 2011,
*** Temporal Light Field Reconstruction for Rendering Distribution Effects,
*** ACM Transactions on Graphics 30(4) (Proc. ACM SIGGRAPH 2011), article 55.
***
*** http://research.nvidia.com/publication/temporal-light-field-reconstruction-rendering-distribution-effects
*** http://groups.csail.mit.edu/graphics/tlfr
*** http://dx.doi.org/10.1145/1964921.1964950
***********************************************************************

System requirements
===================

- Microsoft Windows XP, Vista, or 7. Developed and tested only on Windows 7 x64.

- For filtering large sample sets, several gigabytes of memory and a
  64-bit operating system.

- For GPU reconstruction:

  * NVIDIA CUDA-compatible GPU with compute capability 2.0 and at least 512
    megabytes of RAM. GeForce GTX 480 is recommended.
  
  * NVIDIA CUDA 4.0 or later
    (see http://developer.nvidia.com/cuda-toolkit-archive)

  * Microsoft Visual Studio 2008. Required even if you do not plan to
    build the source code, as CUDA compilation, which happens at
    runtime, requires it.

- This software runs and compiles only on Windows and Visual Studio
  2008. We welcome contributions of ports to other versions of Visual
  Studio and other OSs.


Instructions
============

General use
-----------

Launching reconstruction_app.exe will start the viewer application,
which by default loads a sample buffer from the data/ directory. The
default view (F1) shows the input samples with box filtering, which
results in a noisy image.

Pressing F3 (or clicking the corresponding button in the GUI) will run
our reconstruction algorithm and display the result.

Pressing space will toggle between CPU and GPU reconstruction. This
only works provided that you have a CUDA-capable GPU with compute
capability 2.0 or over.

Pressing F5 will run the reconstruction algorithm to produce a
non-blurry pinhole image at the end of the time interval (t=1). This
is for debugging purposes. Shading is in general not identical to the
ground truth pinhole image (F4): If, for example, the scene has
motion, the pinhole reconstruction is done from samples that include
motion blurred shadows, whereas the ground truth pinhole image is
rendered from a static setup.

If available, the app loads in a ground truth image from the same
directory as the sample buffer. F2 allows you to view it. File name
must be exactly as shown in the provided example. The two single
digits in the filename specify a gamma the images were rendered with,
so the app can apply the same to its own output (this just sets the
gamma slider on the right to the specified value).  The gamma is only
read from the non-pinhole reference, the pinhole reference must match.



Sample Buffer Format
--------------------

Sample buffers are stored in two files: the main file that contains
the sample data itself, and a second header file, whose name must
match the main file. For example,

	samplebuffer.txt
	samplebuffer.txt.header

form a valid pair.

The header specifies all kinds of useful information. It looks like this:

	Version 1.3
	Width 962
	Height 543
	Samples per pixel 16
	Motion model: perspective
	CoC coefficients (coc radius = C0/w+C1): -15.242497,19.053122
	Encoding = text
	x,y,z/w,w,u,v,t,r,g,b,a,mv_x,mv_y,mv_w,dwdx,dwdy

Width and Height specify the dimensions of the image, in
pixels. Samples per pixel says how many samples to expect. Motion
model is there for historical reasons; only value currently supported
is "perspective". The CoC coefficients give formulas for computing the
slope dx/du and dy/dv given the camera space depth (w) for a sample
(see below). The last row serves as a reminder on how to interpret the
numbers in the actual sample file.

The actual sample data file can be either text or binary. We recommend
generating your sample buffers in text format and using the
functionality in UVTSampleBuffer to convert it to binary; this can be
done by loading in the text version and serializing back to disk with
the binary flag turned on.

The "CoC coefficients" C0,C1 are constants that are used for computing
the circle of confusion for a given depth, assuming a thin lens
model. The CoC corresponds directly to the slopes dx/du and dy/dv for
a given depth w. This is how to compute C0 and C1, illustrated using
PBRT's perspective camera API:

	float f = 1.f / ( 1 + 1.f / pCamera->getFocalDistance() );
	float sensorSize = 2 * tan( 0.5f * pCamera->getFoVRadians() );
	float cocCoeff1 = pCamera->getLensRadius() * f / ( pCamera->getFocalDistance() - f );
	cocCoeff1 *= min( camera->film->xResolution, camera->film->yResolution ) / sensorSize;
	float cocCoeff0 = -cocCoeff1 * pCamera->getFocalDistance();

If you use another model, you must derive the constants C0 and C1 yourself.
	
In the main sample file, each line describes one sample specified by
16 floating point numbers.

	x and y are the sample's pixel coordinates, including
	fractional subpixel offsets.

	z/w is projected z, which is currently unused.
	
	w is the camera-space depth.
	
	u and v are the lens coordinates at which the sample was
	taken, in the range [-1,1].
	
	t is the time coordinate in the range [0,1] denoting the
	instant the sample was taken.
	
	r,g,b,a is the sample's radiance, in linear RGB. alpha is
	currently unused.
	
	mv_x, mv_y and mv_w are the sample's motion vector. They
	encode the difference of the camera space (homogeneous)
	position of the sample at the end of the shutter interval
	(t=1) and the beginning of the shutter interval (t=0). In
	other words, it must satisfy

	(xy*w)(T=0) = xy(T=t)*w(T=t) - t*V
	(xy*w)(T=1) = xy(T=t)*w(T=t) + (1-t)*V
	
	Where xy(T=t) and w(T=t) mean the sample's original screen
	position and depth at the time it was taken. Notice that the
	screen position is converted to homogeneous coordinates by
	multiplication by w, but there is no scale and bias to [-1,1]
	clip space coordinates; dividing by w yields pixel coordinates
	directly.
	
	NOTE that pbrt's default motion model is not world-affine as
	it performs interpolation of rotation matrices. As mentioned
	in the paper, we changed this in our version. A pbrt patch
	that outputs sample buffers with world-affine motion will be
	released separately.
	
	See Sec. 3.1 of the paper for a more thorough explanation.


CUDA
----

The pre-built 64-bit binary is naturally built with CUDA support, but
it is by default disabled in the code to enable building without the
CUDA SDK. To enable CUDA support, find the line

	#define FW_USE_CUDA 0
  
in src/framework/base/DLLImports.hpp and change it to

	#define FW_USE_CUDA 1
  
Provided you have a working install of CUDA Toolkit 4.0, this will
enable the GPU code path (toggled by Spacebar in the application). The
first time it is run, the application will compile and cache the .cu
file containing the reconstruction kernels. This may take a few
seconds.

If you get an error during initialization, the most probable
explanation is that the application is unable to launch nvcc.exe
contained in the CUDA Toolkit. In this case, you should:

   - Set CUDA_BIN_PATH to point to the CUDA Toolkit "bin" directory, e.g.
     "set CUDA_BIN_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\bin".

   - Set CUDA_INC_PATH to point to the CUDA Toolkit "include" directory, e.g.
     "set CUDA_INC_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include".

   - Run vcvars32.bat to setup Visual Studio paths, e.g.
     "C:\Program Files\Microsoft Visual Studio 9.0\VC\bin\vcvars32.bat".


Release notes
=============

- The soft shadow filtering code is included for reference, but not
directly callable from the example application. This is because it
requires a significant amount of support code for generating the light
samples from camera samples, etc. We apologize for the inconvenience.

- DEVIATION FROM THE PAPER: Hieararchy consumes less memory than
reported due to code cleanup.


Known issues
------------

Fast motion with large depth differences:

Objects that undergo fast motion during the shutter interval, such
that they move significantly towards or away from the camera, cause
the samples' apparent screen trajectories x(t) and y(t) to deviate
from straight lines in XYT. Because of this curvature, the BVH nodes'
bounds become looser -- this is because we use linear XYUVT
hyperplanes we use as the bounds. This in turn may lead to somewhat
reduced efficiency, particularly in the CUDA implementation which has
strict bounds on the number of tree nodes it can handle during the
filtering. Exceeding this bound causes it to revert back to the CPU
implementation. We have not observed this behavior in anything but in
test cases where the motion in the Z direction is large.

Using bounds that better adapt to the perspective-induced curvature in
the trajectories should fix this problem if it becomes a practical
issue. One such formulation can be found in our paper "Clipless
Dual-Space Bounds for Faster Stochastic Rasterization", ACM TOG 30(4)
(Proc. SIGGRAPH 2011).
