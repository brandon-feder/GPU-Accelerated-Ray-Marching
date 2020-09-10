class Vector
{
    public:
        float x;
        float y;
        float z;

        // Constructors and destructors
        CUDA_HOSTDEV Vector(float I_x, float I_y, float I_z);
        CUDA_HOSTDEV Vector();

        CUDA_HOSTDEV ~Vector();

        // Methods
        CUDA_HOSTDEV Vector clone();
        CUDA_HOSTDEV float magnitude();

        CUDA_HOSTDEV void normalize();
        CUDA_HOSTDEV void toSpherical();
        CUDA_HOSTDEV void toCartesian();

        // Operators
        CUDA_HOSTDEV void operator+=(const Vector &vec);
        CUDA_HOSTDEV void operator-=(const Vector &vec);
        CUDA_HOSTDEV void operator*=(const Vector &vec);
};
