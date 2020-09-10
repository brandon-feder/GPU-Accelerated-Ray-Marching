template<class F>
__device__ int getPixel(int i, Vector ray, F func);

template<class F>
__global__ void getFrame( int *pixels, Vector *rays, F func )
{
    // Get the prcess index
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    // If index has a job to do
    if( index <  N_THREADS )
    {
        // Get the first and last pixels the thread needs to get the valu eof
        int start = index * ( (float)N_PIXELS / N_THREADS );
        int end =  ( index + 1 ) * ( (float)N_PIXELS / N_THREADS );

        // Compute the value between [start, end)
        for(int i = start; i < end; ++i)
        {
            pixels[i] = getPixel(i, rays[i], func);
        }
    }
}

__device__ int phongReflection(ObjectInfo obj, Vector ray)
{
    float i_s = 10;
    float i_d = 5;
    float i_a = i_s + i_d;

    float k_s = 0.5;
    float k_d = 0.2;
    float k_a = 0.4;
    float a = 3;

    Vector light = Vector(-100, 0, 0, false);
    Vector L_m = light - ray;
    Vector N = obj.normal;
    Vector R_m = N*(-2)*( L_m * N ) - L_m;
    Vector V = Vector(0, 0, 0, false) - obj.objPosition;

    float I_p = k_a*i_a + k_d * ( L_m * N ) * i_d + k_s * ( R_m * V ) * ( R_m * V ) * ( R_m * V ) * i_s;

    return (int)I_p;

}

template<class F>
__device__ int getPixel(int i, Vector ray, F func)
{
    int loopDepth = 0;
    float depth = 0;
    ObjectInfo obj;
    while( loopDepth <= MAX_DEPTH && ray.magnitude() <= MAX_DISTANCE )
    {
        obj = func( ray*depth );

        if( obj.isRayClose )
        {

            // return obj.color;
            return phongReflection(obj, ray);
        }

        depth += obj.distance;
        ++loopDepth;
    }

    return color(230, 230, 230);
}
