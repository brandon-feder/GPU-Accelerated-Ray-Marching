class RayMarching
{
    private:
        static void initSDL();
        static void initMemory();
        static void initRays();

        template <class F1, class F2>
        static void mainFunc(F1 SceneSDF_Func, F2 LightingFunc);

        static void updateFrame();
        static bool handleEvent( SDL_Event event );
        static void quit();

        static int *TIME_MEM;

        static Vector *rays;
        static int *pixels;

        static SDL_Window *window;
        static SDL_Renderer *renderer;
        static SDL_Texture *tex;
        
        static std::chrono::steady_clock::time_point start;
    public:
        template <class F1, class F2>
        static void init(int width, int height, int nThreads, F1 SceneSDF_Func, F2 LightingFunc);
};
