#include "headers/Color.cuh"

__device__ Color::Color() {}
__device__ Color::~Color() {}

__device__ Color::Color(int I_r, int I_g, int I_b)
{
    r = I_r;
    g = I_g;
    b = I_b;
}
