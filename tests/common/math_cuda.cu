#include <cuda_runtime.h>
#include <stdio.h>

#include "../helpers.h"
#include "tinyrend/common/math.h"
#include "tinyrend/common/vec.h"

using namespace tinyrend;

__global__ void test_rsqrt_kernel(float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    y[idx] = rsqrt(x[idx]);
}

int test_rsqrt() {
    int fails = 0;

    {
        float h_x[2] = {4.0f, 1.0f};
        float h_y[2];
        float h_y_expected[2] = {0.5f, 1.0f};
        float *d_x, *d_y;
        cudaMalloc((void **)&d_x, 2 * sizeof(float));
        cudaMalloc((void **)&d_y, 2 * sizeof(float));
        cudaMemcpy(d_x, h_x, 2 * sizeof(float), cudaMemcpyHostToDevice);
        test_rsqrt_kernel<<<1, 2>>>(d_x, d_y);
        cudaMemcpy(h_y, d_y, 2 * sizeof(float), cudaMemcpyDeviceToHost);
        CHECK(h_y[0] == h_y_expected[0], "");
        CHECK(h_y[1] == h_y_expected[1], "");
    }

    return fails;
}

int main() {
    int fails = 0;

    fails += test_rsqrt();

    if (fails > 0) {
        printf("[common/math.cpp] %d tests failed!\n", fails);
    } else {
        printf("[common/math.cpp] All tests passed!\n");
    }

    return fails;
}