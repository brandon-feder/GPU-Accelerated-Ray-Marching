
// ==========================================
// ============ Helper Functions ============
// ==========================================

__device__ bool isClose(float distance)
{
    return distance <= 1;
}


// ==========================================
// ================== SDFs ==================
// ==========================================

__device__ float SphereSDF(Vector spherePos, float radious, Vector ray)
{
    return (spherePos - ray).magnitude() - radious;
}

__device__ Vector SphereSDF_Normal(Vector spherePos, float radious, Vector ray)
{
    Vector normal = Vector(
        SphereSDF( spherePos, radious, ray + Vector(EPSILON, 0, 0) ) - SphereSDF( spherePos, radious, ray - Vector(EPSILON, 0, 0) ),
        SphereSDF( spherePos, radious, ray + Vector(0, EPSILON, 0) ) - SphereSDF( spherePos, radious, ray - Vector(0, EPSILON, 0) ),
        SphereSDF( spherePos, radious, ray + Vector(0, 0, EPSILON) ) - SphereSDF( spherePos, radious, ray - Vector(0, 0, EPSILON) )
    );

    normal.normalize();
    return normal;
}
