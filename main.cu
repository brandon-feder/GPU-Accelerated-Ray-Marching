#include "RayMarching/Settings.cu"
#include "RayMarching/RayMarching.cu"
#include "RayMarching/Helpers.cu"

__device__ float SphereA_SDF(Vector ray, float time);
__device__ Vector SphereA_Normal(Vector ray, float dist);

int main()
{

    auto Scene = [] __device__ (Vector ray, float time)
    {
        float SphereA_Dist = SphereA_SDF( ray, time );

        if( SphereA_Dist <= 1 )
        {
            return ObjectInfo( SphereA_Dist, Color(255, 0, 0) );
        } else
        {
            return ObjectInfo( SphereA_Dist );
        }
    };

    auto Lighting = [] __device__ (Vector ray)
    {
        return 0;
    };

    RayMarching::init(WIDTH, HEIGHT, N_THREADS, Scene, Lighting);
}


__device__ float SphereA_SDF(Vector ray, float time)
{
    return SphereSDF( Vector( 0, 1000, 0 ), 100, ray );
}

__device__ Vector SphereA_Normal(Vector ray, float dist)
{
    return isClose( dist ) ? SphereSDF_Normal( Vector( 0, 1000, 0 ), 100, ray ) : Vector(0, 0, 0);
}
