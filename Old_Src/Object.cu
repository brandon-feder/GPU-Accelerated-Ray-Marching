struct ObjectInfo
{
    float distance;
    int color;
    Vector objPosition;
    Vector normal;
    bool isRayClose;

    __device__ ObjectInfo(float d, int c, Vector n, bool isClose, Vector pos)
    {
        distance = d;
        color = c;
        normal = n;
        isRayClose = isClose;
        objPosition = pos;
    }

    __device__ ObjectInfo() {}
};

class Sphere
{
    public:
        Vector position;
        float radious;

        __device__ Sphere(Vector pos, float r)
        {
            position = pos;
            radious = r;
        }

        __device__ float SDF(Vector ray)
        {
            return ( position - ray ).magnitude() - radious;
        }

        __device__ Vector Normal( Vector ray )
        {
            float x = ray.x;
            float y = ray.y;
            float z = ray.y;
            float e = NORMAL_EPSILON;

            return Vector(
                SDF(Vector( x + e, y, z, false)) - SDF(Vector( x - e, y, z, false)),
                SDF(Vector( x, y + e, z, false)) - SDF(Vector( x, y - e, z, false)),
                SDF(Vector( x, y, z + e, false)) - SDF(Vector( x, y, z - e, false)),
                true
            );
        }
};
