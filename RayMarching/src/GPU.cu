#include "headers/GPU.cuh"

template<class F1, class F2>
__device__ void GPU::getFrame(int *pixels, Vector *rays, int *time,
    F1 SceneSDF_Func, F2 LightingFunc )
{

    // Get the index of the thread
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    int TIME = time[0];

    // Make sure the index has a job
    if(index < N_THREADS)
    {
        // Get the first and last pixel the thread will process
        int start = index * ( (float)N_PIXELS / N_THREADS );
        int end =  ( index + 1 ) * ( (float)N_PIXELS / N_THREADS );

        // Process all the pixels and get there color
        for(int i = start; i < end; ++i)
        {
            pixels[i] = GPU::getPixel(rays[i], TIME, SceneSDF_Func, LightingFunc );
        }
    }
}

template<class F1, class F2>
__device__ int GPU::getPixel(Vector ray, int time, F1 SceneSDF_Func, F2 LightingFunc )
{

    int loopDepth = 0;
    float depth = 0;
    ObjectInfo obj;

    while( loopDepth <= MAX_DEPTH && ray.magnitude() <= MAX_DISTANCE )
    {
        Vector mRay = ray*depth;
        obj = SceneSDF_Func( mRay, time );

        if( obj.distance <= 1 )
        {
            float e = 0.001;
            Vector normal = Vector(
                SceneSDF_Func( Vector( ray.x+e, ray.y, ray.z )*depth, time ).distance - SceneSDF_Func( Vector( ray.x-e, ray.y, ray.z )*depth, time ).distance,
                SceneSDF_Func( Vector( ray.x, ray.y+e, ray.z )*depth, time ).distance - SceneSDF_Func( Vector( ray.x, ray.y-e, ray.z )*depth, time ).distance,
                SceneSDF_Func( Vector( ray.x, ray.y, ray.z+e )*depth, time ).distance - SceneSDF_Func( Vector( ray.x, ray.y, ray.z-e )*depth, time ).distance
            );
            normal.normalize();

            return Color(abs(normal.x)*255, abs(normal.y)*255, abs(normal.z)*255);
        }

        depth += obj.distance;
        ++loopDepth;
    }

    return Color(230, 230, 230);
}

template<class F1, class F2>
__global__ void getFrame(int *pixels, Vector *rays, int *time, F1 SceneSDF_Func, F2 LightingFunc )
{
    // Call the corresponding getFrame function
    GPU::getFrame(pixels, rays, time, SceneSDF_Func, LightingFunc);
}
