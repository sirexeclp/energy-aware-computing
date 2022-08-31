- Automatic Generation of 1D Recursive Filter Code for GPUs http://on-demand.gputechconf.com/gtc/2018/presentation/s8189-automatic-generation-of-1d-recursive-filter-code-for-gpus.pdf
- The Scientist and Engineer's Guide to
Digital Signal Processing
http://www.dspguide.com
- Digital Signal Processing Filtering with GPU 
https://pdfs.semanticscholar.org/cb5e/776fab1cbbb6d1b16c23c1781989b7a9c105.pdf
- CUDA DSP Filter for ECG Signals (FIR) https://www.researchgate.net/publication/304918949_CUDA_DSP_Filter_for_ECG_Signals

- https://github.com/andmax/gpufilter 

> The main goal of this project is to provide the baseline code in  C/C++/CUDA for computing the fastest boundary-aware recursive filtering (paper 3) running on the GPU (graphics processing unit), i.e. a massively-parallel processor (paper 2). The fastest means fastest to date (as the last published paper) and boundary awareness means closed-form exact (i.e. no approximations). The idea of boundary awareness is to compute the exact initial feedbacks needed for recursive filtering infinite input extensions (paper 3).
Please keep in mind that this code is just to check the performance and accuracy of the recursive filtering algorithms on a 2D random image of 32bit floats. Nevertheless, the code can be specialized for 1D or 3D inputs, and for reading any given input and data type.

- resampling using the frequency domain https://dsp.stackexchange.com/questions/36284/can-you-decimate-downsample-a-signal-in-frequency-domain-just-like-you-can-int

- python resampling cpu (ist bei langen ekgs teilweise sehr langsam) https://dsp.stackexchange.com/questions/52941/scipy-resample-fourier-method-explanation?noredirect=1&lq=1

- Signal Processing on a Graphics Card (Thesis) http://www.rrsg.ee.uct.ac.za/theses/ug_projects/radhakrishnan_ugthesis.pdf

- An Implementation of a FIR Filter on a GPU http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.329.2844&rep=rep1&type=pdf

- Assessment of Graphic Processing Units (GPUs) for Department ofDefense (DoD) Digital Signal Processing (DSP) Applications https://pdfs.semanticscholar.org/a2d6/afcdd094688691ecba8a924e15245372b348.pdf

- Statistical Power Consumption Analysis and Modelingfor GPU-based Computing http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9557&rep=rep1&type=pdf

- FFT vs. convolution:  https://stackoverflow.com/a/18406854/7997186 und https://stackoverflow.com/a/18385421/7997186 

Zum vergleich der Filter den ich bräuchte hat 9 koeffizienten jweils in zähler und nenner.

https://www.sciencedirect.com/science/article/pii/B9780125921015500075
https://pdfs.semanticscholar.org/adaf/b3c81cdabc746439ce86b4346074b194d4a6.pdf
http://www.fftw.org/benchfft/ffts.html
https://www.sciencedirect.com/science/article/pii/S0167819184904137?via%3Dihub
http://www.fftw.org/pldi99.pdf
https://cnx.org/contents/4kChocHM@6/Efficient-FFT-Algorithm-and-Programming-Tricks
http://www.dsp-book.narod.ru/DSPMW/07.PDF


TODO: split radix
TODO: fix glitches in current fft-implementation
better bit reversal
std::complex vs c-style
std::vector vs. double
double vs. float
complex vs. real input (should be factor 2)
radix/base 4/8 << simd, loop unroling?
loop unroling in general

phasen (warmup/run) erkennen
ernergie states 
cores abschalten ?

MAX-Q 

https://images.nvidia.com/content/tesla/pdf/Tesla-V100-PCIe-Product-Brief.pdf
https://blogs.nvidia.com/blog/2018/12/14/what-is-max-q/

Persistence Mode
       A  flag that indicates whether persistence mode is enabled for the GPU.
       Value is either "Enabled" or  "Disabled".   When  persistence  mode  is
       enabled  the  NVIDIA driver remains loaded even when no active clients,
       such as X11 or nvidia-smi,  exist.   This  minimizes  the  driver  load
       latency  associated with running dependent apps, such as CUDA programs.
       For all CUDA-capable products.  Linux only.

   Accounting Mode
       A flag that indicates whether accounting mode is enabled  for  the  GPU
       Value  is  either  When accounting is enabled statistics are calculated
       for each compute process running on the GPU.  Statistics can be queried
       during  the lifetime or after termination of the process. The execution
       time of process is reported as 0 while the process is in running  state
       and  updated to actual execution time after the process has terminated.
       See --help-query-accounted-apps for more info.


       List of valid properties to query for the switch "--query-accounted-apps=":

Section about Accounted Graphics/Compute Processes properties
List of accounted processes having had a graphics/compute context on the device.

Format options and one or more properties to be queried need to be provided as comma separated values for this switch.
For example:
nvidia-smi --query-accounted-apps=gpu_name,pid,time,gpu_util,mem_util,max_memory_usage --format=csv

List of properties that can be queried are:

"timestamp"
The timestamp of where the query was made in format "YYYY/MM/DD HH:MM:SS.msec".

"gpu_name"
The official product name of the GPU. This is an alphanumeric string. For all products.

"gpu_bus_id"
PCI bus id as "domain:bus:device.function", in hex.

"gpu_serial"
This number matches the serial number physically printed on each board. It is a globally unique immutable alphanumeric value.

"gpu_uuid"
This value is the globally unique immutable alphanumeric identifier of the GPU. It does not correspond to any physical label on the board.

"vgpu_instance"
vGPU instance

"pid"
Process ID of the compute application

"gpu_utilization" or "gpu_util"
GPU Utilization

"mem_utilization" or "mem_util"
Percentage of GPU memory utilized on the device by the context.

"max_memory_usage"
Maximum amount memory used on the device by the context.

"time"
Amount of time in ms during which the compute context was active.
