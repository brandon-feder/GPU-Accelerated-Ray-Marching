class ObjectInfo
{
    public:
        __device__ ObjectInfo() {}
        __device__ ObjectInfo(float d, Color c);
        __device__ ObjectInfo(float d);
        Color color;
        float distance;
};
