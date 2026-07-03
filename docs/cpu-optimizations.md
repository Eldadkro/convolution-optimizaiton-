# CPU Optimization Summary

This document records the CPU-side optimization steps applied to the 5x5 LPF/HPF benchmark and the measured impact during local runs on `data/image.png` with 100 iterations.

Test system:

| Item | Value |
| --- | --- |
| CPU | 13th Gen Intel Core i9-13900K |
| CPU SIMD target | `-O3 -march=native` |
| CUDA build | `build-release-cuda` |
| Input | `data/image.png` |
| Benchmark command | `build-release-cuda/kopt_bench data/image.png 100 cuda` |
| OpenCV threads | 1 |


# Preview 
I started with a generalized convolution function for the LPF for the sake of generality, but the optimization took into account that we use box filter, which is both seperable, and could be computed using the integral of the image by adding and subtracting differnt parts to create the sum of the 5x5 elements, then I only needed to divide the results by 25. however it isn't as memory efficiant as simply runnig the seperated kernel, even though it is more flops and reads, the memory locality and vectorization provided the fastest result. (using opencv becuase of the lack of time, and working with well supported liberaries).
for the sake of maintainabiltiy I scaned the image and casted the elements to float, however more performance could have been gained by running on the 8u image instead cause of caching.


I started 
## Changes

| Step | Upgrade | What Changed | Why It Helped | Measured CPU Avg |
| --- | --- | --- | --- | --- |
| 1 | Baseline cpu convolution | nothing | Correct CPU implementation, but didn't take into account the box LPF. | ~ ms 24.27|
| 2 | baseline opencv 2dfilter | linked against opencv 4.6 | used opencv for general fitler, but didn't take into account the box LPF. | ~ ms 13.44|
| 3 | naive integral solution | implemented custom integral and calculated the box | it didn't  | ~ ms 15.17|
| 4 | opencv integral solution | used opencv integral and calculated the box | it didn't  | ~ ms 15.14|
| 5 | used opencv seperable API for the box | used the seperable filter function | we did more flops but memory locality and SIMD proved the best  | ~ ms 13.12|



