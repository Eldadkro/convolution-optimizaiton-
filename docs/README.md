# Setup 
You are given a 1024×2048 uint8 difference image.

You need to apply two 5×5 filters to the image:

Filter 1: An LPF (low-pass filter), as described during the interview.
Filter 2: An HPF (high-pass filter) of your choice.

Prepare C++ code that can perform both tasks together with maximum speed and efficiency.

Please include a README file with build/run instructions and a basic explanation of the optimizations you chose to use.

You may, and are encouraged to, use as many techniques as possible to improve the program’s runtime, provided that you can explain what you did and why it improves performance.

You may use CUDA, but you must also provide an implementation that does not rely on CUDA.

Bonus: Generate or simulate an example image and provide an environment in which the filters can be run.

Good luck!


## Assumptions 

1) Images comes in a stream, and we optimize for latancy, meaning a single image should finish as fast as possible.
    * no batching as images should have small latancy
    * memcpy in and out should be considered as it is only a small part of the general pipeline
2) I generated a simple image for example perpuses. 
3) I used codex to guide research and code generation, however I validated each step and directed the architectrue 
4) I used opencv 4.6 for image reads and CPU pipeline and CUDA with NPP for the GPU pipeline.
5) Because I could use any HPF I could simply subtract the lowpassed image from the regular image and get the rest of the spectrum, so I used that for the HPF.

## Architecture

The FilterEngine lib provide an abstraction for the computation and dessosiate the dependecies from the app such that only basic and naitive types are passed between the user and the lib.
I used CMAKE in order to handle the build process and a python env for the resize of the original image to 1024x2048.
By passing a backend enum to the filterEngine the used can choose between CPU and GPU such that only that type of pipeline is supported by the instance of FilterEngine.
Then the flow is pretty simple, in the init (setup) everything is loaded, the buffers and kernels if nessesery.
then in the run function the minimum possible operation are ran. In order to see the optimization tech and upgrade path used, check the Optimization section.

## Optimizations
First, for the sake of simplicity I started with the CPU workside, which is farther expalined in [cpu-optimizations.md](cpu-optimizations.md).
For details on the CUDA-side optimizations and measured results, see [cuda-optimizations.md](cuda-optimizations.md).



