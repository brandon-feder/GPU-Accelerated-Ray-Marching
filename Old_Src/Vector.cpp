class Vector
{
    public:
        float x, y, z;

        __host__ __device__
        Vector() {}

        __host__ __device__
        Vector(float Xi, float Yi, float Zi, bool normalize)
        {
            x = Xi; y = Yi; z = Zi;

            if( normalize )
            {
                int mag = sqrt( x*x + y*y + z*z );
                x /= mag;
                y /= mag;
                z /= mag;
            }
        }

        __host__ __device__
        Vector operator-(const Vector& vec) {
            return Vector(
                vec.x - x,
                vec.y - y,
                vec.z - z,
                false
            );
        }

        __host__ __device__
        Vector operator+(const Vector& vec) {
            return Vector(
                vec.x + x,
                vec.y + y,
                vec.z + z,
                false
            );
        }


        __host__ __device__
        float operator*(const Vector& vec) {
            return vec.x*x + vec.y*y + vec.z*z;
        }

        __host__ __device__ // Todo: Add float * Vector as well
        Vector operator*(const float& constant) {
            return Vector(
                constant * x,
                constant * y,
                constant * z,
                false
            );
        }

        __host__ __device__ // Todo: Add float * Vector as well
        void operator+=(const Vector& Vector) {
            x += Vector.x;
            y += Vector.y;
            z += Vector.z;
        }

        __host__ __device__ // Todo: Add float * Vector as well
        void operator-=(const Vector& Vector) {
            x += Vector.x;
            y += Vector.y;
            z += Vector.z;
        }

        __host__ __device__
        float magnitude()
        {
            return sqrt( x*x + y*y + z*z );
        }

        __device__ __host__
        Vector mapTransformation(Vector rotation, Vector translation)
        {
            /*
            float r = magnitude(); // Gets the magnitude of the vector describing the position witch equals the radious
            float p = ((x != 0) ? atanf( y / x ) : 0); // Calculate theta
            float t = acosf( z / r ); // Calculate phi

            // Tale infoa ccount the qudrant of t
            if(x < 0 || y < 0)
                t *= -1;

            // Add angles changes
            t += theta;
            p += M_PI/2 - phi;

            // Bacl to cartesian cordinates
            float newX = r * sin( t ) * cos( p );
            float newY = r * sin( t ) * sin( p );
            float newZ = r * cos( t );


            // Return the new vector
            return Vector( newX, newY, newZ, false);
            */
        }

        __host__ __device__
        Vector toSpherical()
        {
            float r = magnitude();
            return Vector(
                r,
                acosf( z / r ),
                ((x != 0) ? atanf( y / x ) : 0),
                false
            );
        }

        __host__ __device__
        Vector toCartesian()
        {
            return Vector(
                x * sin( y ) * cos( z ),
                x * sin( y ) * sin( z ),
                x * cos( y ),
                false
            );
        }
};
