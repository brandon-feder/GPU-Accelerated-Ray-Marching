class Color
{
    public:
        int r, g, b;

        CUDA_HOSTDEV Color();
        CUDA_HOSTDEV Color(int I_r, int I_g, int I_b);
        CUDA_HOSTDEV ~Color();

        CUDA_HOSTDEV operator int()
        {
            int rgba = r;
            rgba = (rgba<<8) + g;
            rgba = (rgba<<8) + b;
            rgba = (rgba<<8) + 255;
            return rgba;
        }
};
