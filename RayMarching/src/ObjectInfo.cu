#include "headers/ObjectInfo.cuh"

__device__ ObjectInfo::ObjectInfo(float d)
{
    distance = d;
}

__device__ ObjectInfo::ObjectInfo(float d, Color c)
{
    distance = d;
    color = c;
}
