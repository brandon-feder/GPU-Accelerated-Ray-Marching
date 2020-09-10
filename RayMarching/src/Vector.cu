#ifdef __CUDACC__
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

#include "headers/Vector.cuh"

// ====== Methods ======

// Default constructor
CUDA_HOSTDEV Vector::Vector() {}

// Other constructor
CUDA_HOSTDEV Vector::Vector( float I_x, float I_y, float I_z )
{
    x = I_x;
    y = I_y;
    z = I_z;
}

// Destructor
CUDA_HOSTDEV Vector::~Vector() {}

// Clones the Vector
CUDA_HOSTDEV Vector Vector::clone()
{
    return Vector(
        x, y, z
    );
}

// Gets the magnitude of the Vector
CUDA_HOSTDEV float Vector::magnitude()
{
    return sqrt( x*x + y*y + z*z );
}

// Normalizes the vectors components
CUDA_HOSTDEV void Vector::normalize()
{
    float mag = magnitude();

    if(mag != 0)
    {
        x /= mag;
        y /= mag;
        z /= mag;
    }
}

// Converts the Vector to spherical cordinates
// Note* Does not take into account whether the cordinates
//      already representing spherical cordinates
CUDA_HOSTDEV void Vector::toSpherical()
{
    float r = magnitude();
    float p = atan2( y, x );
    float t = ( r == 0 ) ? 0 : acos( z / r );

    x = r;
    y = p;
    z = t;
}

// Converts the Vector to cartesain cordinates
// Note* Does not take into account whether the cordinates
//      already representing cartesian cordinates
CUDA_HOSTDEV void Vector::toCartesian()
{
    // printf("in 1: <%f, %f, %f>\n",  x, y, z);
    float r = x;
    float p = y;
    float t = z;

    x = r * sin( t ) * cos( p );
    y = r * sin( t ) * sin( p );
    z = r * cos( t );
    // printf("in 2: <%f, %f, %f>\n",  x, y, z);
}

// ====== Operators ======

// Multiplitaction Operator
CUDA_HOSTDEV Vector operator*(const float &constant, const Vector &vec)
{
    return Vector(
        vec.x*constant,
        vec.y*constant,
        vec.z*constant
    );
}

CUDA_HOSTDEV Vector operator*(const Vector &vec, const float &constant)
{
    return Vector(
        vec.x*constant,
        vec.y*constant,
        vec.z*constant
    );
}

CUDA_HOSTDEV void Vector::operator*=(const Vector &vec)
{
    x *= vec.x;
    y *= vec.y;
    z *= vec.z;
}


// Dot Product
CUDA_HOSTDEV float operator*(const Vector &vec1, const Vector &vec2)
{
    return vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z;
}

// Addition And Subtraction
CUDA_HOSTDEV Vector operator+(const Vector &vec1, const Vector &vec2)
{
    return Vector(
        vec1.x + vec2.x,
        vec1.y + vec2.y,
        vec1.z + vec2.z
    );
}

CUDA_HOSTDEV Vector operator-(const Vector &vec1, const Vector &vec2)
{
    return Vector(
        vec1.x - vec2.x,
        vec1.y - vec2.y,
        vec1.z - vec2.z
    );
}

CUDA_HOSTDEV void Vector::operator+=(const Vector &vec)
{
    x += vec.x;
    y += vec.y;
    z += vec.z;
}

CUDA_HOSTDEV void Vector::operator-=(const Vector &vec)
{
    x -= vec.x;
    y -= vec.y;
    z -= vec.z;
}
