#include <iostream>
#include <math.h>
#include <chrono>
#include <SDL2/SDL.h>
#include <cuda_runtime.h>
#include <math.h>

__managed__ int N_CORES_PER_BLOCK;
__managed__ int N_BLOCKS;
__managed__ int N_PIXELS;

#include "src/Vector.cu"
#include "src/Color.cu"
#include "src/ObjectInfo.cu"
#include "src/GPU.cu"

#include "RayMarching.cuh"

// ====== Setup Static Variables ======
int *RayMarching::TIME_MEM;

Vector *RayMarching::rays;
int *RayMarching::pixels;

SDL_Window *RayMarching::window;
SDL_Renderer *RayMarching::renderer;
SDL_Texture *RayMarching::tex;

std::chrono::steady_clock::time_point RayMarching::start;

// ===== Define Static Functions ======
template <class F1, class F2>
void RayMarching::init(int width, int height, int nThreads, F1 SceneSDF_Func, F2 LightingFunc)
{
    // Pre-awsomeness setup stuff
    initSDL();
    initMemory();
    initRays();

    // Get the start of the program
    start = std::chrono::steady_clock::now();

    N_CORES_PER_BLOCK =  std::min( 1024, N_THREADS );
    N_BLOCKS = N_THREADS / N_CORES_PER_BLOCK + (N_THREADS % N_CORES_PER_BLOCK != 0);
    N_PIXELS = WIDTH*HEIGHT;

    // The main loop
    mainFunc(SceneSDF_Func, LightingFunc);

    // Cleanup then quit
    quit();
}

void RayMarching::initSDL()
{
    // Setup SDL
    SDL_Init(SDL_INIT_VIDEO);
    window = SDL_CreateWindow("Real Time Ray Marching", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);
    renderer = SDL_CreateRenderer(window, -1, 0);
    tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);

    // Hold the users cursor at the center of the screen
    if( HOLD_CURSOR )
    {
        SDL_WarpMouseInWindow(window, WIDTH / 2, HEIGHT / 2);
        SDL_SetRelativeMouseMode( SDL_TRUE );
    }

}

void RayMarching::initMemory()
{
    // Setup shared memory
    cudaMallocManaged( &rays, sizeof(Vector)*WIDTH*HEIGHT );
    cudaMallocManaged( &pixels, sizeof(int)*WIDTH*HEIGHT );
    cudaMallocManaged( &TIME_MEM, sizeof(int) );
}

void RayMarching::initRays()
{
    int i = 0;
    for(int z = -HEIGHT/2; z < HEIGHT/2; z++)
    {
        for(int x = -WIDTH/2; x < WIDTH/2; x++)
        {
            rays[i] = Vector( x, CAMERA_DEPTH, z );
            rays[i].normalize();
            ++i;
        }
    }
}

template <class F1, class F2>
void RayMarching::mainFunc(F1 SceneSDF_Func, F2 LightingFunc)
{
    bool quit = false; // Whether to quit or not
    int frame = 0;
    while(!quit)
    {
        // Get the time at the start of the frame
        std::chrono::steady_clock::time_point frameStart = std::chrono::steady_clock::now();

        // Get the time elapsed since the start of the program
        int timeSinceStart = std::chrono::duration_cast<std::chrono::milliseconds>(frameStart - start).count();

        // Set the shared memory TIME_MEM variable to contain the current time
        TIME_MEM[0] = timeSinceStart;

        getFrame<<<N_BLOCKS, N_CORES_PER_BLOCK>>>( pixels, rays, TIME_MEM, SceneSDF_Func, LightingFunc );

        // Wait for the getFrame function to finish on the GPU
        cudaError_t error = cudaDeviceSynchronize();

        if(error != 0)
        {
            std::cout << cudaGetErrorString(error) << std::endl;
            quit = true;
        } else
        {
            // Update the pixels
            updateFrame();

            SDL_Event event; // Buffer to contain the current event being processed
            while ( SDL_PollEvent( &event ) ) // While there are vents, get the next event
            {
                if(handleEvent( event )) // Handle the event. If true, quit
                {
                    quit = true;
                    break;
                }
            }

            if( PRINT_FPS && frame % 100 == 0)
            {
                // Get the time at the end of the frame
                std::chrono::steady_clock::time_point frameEnd = std::chrono::steady_clock::now();

                // Calculate the FPS
                int frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart).count();
                int FPS = 1000/frameTime;

                // Print the FPS
                std::cout << "FPS: " << FPS << std::endl;
            }

            ++frame;
        }

    }
}

void RayMarching::updateFrame()
{
    // Update the texture with all the pixels
    SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, tex, NULL, NULL);
    SDL_RenderPresent(renderer);
}

bool RayMarching::handleEvent(SDL_Event event)
{
    // Switch statement to get possible events
    switch( event.type )
    {
        case SDL_QUIT:
            return true; // Tell the main loop to quit
    }


    return false; // Tell the main loop to countinue
}

void RayMarching::quit()
{
    // Clean up SDL
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    // Delete shared memmory
    cudaFree(pixels);
    cudaFree(rays);
    cudaFree(TIME_MEM);
}
