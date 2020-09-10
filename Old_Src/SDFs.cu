/*
    There must be the distFunc(Vector) as it is the startin point for getting the value of the SDF.
    You must use __device__ before the type in order to tell the nvcc compiler that the code will be run
    on the GPU.
*/

ObjectInfo SceneSDF(Vector ray, Vector rotation, Vector translation)
{
    // // Create a Sphere
    // Vector SpherePos = Vector(0, 16, 0, false);
    // SpherePos = SpherePos.mapTransformation( theta, phi );
    // Sphere SphereA = Sphere( SpherePos, 4 );
    //
    // // Create an ObjectInfo object describing the object the sphere hit
    // ObjectInfo sphereAInfo = ObjectInfo(
    //     SphereA.SDF(ray),
    //     color(255, 0, 0),
    //     SphereA.Normal(ray),
    //     SphereA.SDF(ray) < MIN_DISTANCE,
    //     SpherePos
    // );
    //
    // return sphereAInfo;
};
