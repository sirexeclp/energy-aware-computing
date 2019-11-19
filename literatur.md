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