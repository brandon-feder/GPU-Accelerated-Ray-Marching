class GPU
{
    private:
        // Gets the rgba value of a single pixel
        template<class F1, class F2>
        __device__ static int getPixel(Vector ray, int time, F1 SceneSDF_Func, F2 LightingFunc );

    public:
        // Sets pixels[] to contain all the pixel rgba values
        template<class F1, class F2>
        __device__ static void getFrame(int *pixels, Vector *rays, int *time, F1 SceneSDF_Func, F2 LightingFunc );
};
