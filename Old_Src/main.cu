#include <iostream>
#include <SDL2/SDL.h>
#include <algorithm>    // std::max

#include "Color.cpp"
#include "Settings.cpp"
#include "Vector.cpp"
#include "Object.cu"
#include "Ray.cpp"
#include "Camera.cu"

#include "SDFs_helpers.cu"


Vector *rays;
int *pixels;

// Variables for SDL
SDL_Window * window;
SDL_Renderer * renderer;
SDL_Texture * tex;

__managed__  Vector cameraRotation;
__managed__  Vector cameraPosition;

void initSDL();
void initRays();
void cleanUpSDL();
void updateFrame();

int main( void )
{
    // Setup vectors to store the rotation and translation of the camera
    cameraRotation = Vector(0, 0, 0, false);
    cameraPosition = Vector(0, 0, 0, false);

    // Initialize SDL
    initSDL();

    // Create variables accesible by CPU and GPU
    cudaMallocManaged( &rays, sizeof(Vector)*WIDTH*HEIGHT );
    cudaMallocManaged( &pixels, sizeof(int)*WIDTH*HEIGHT );

    initRays();

    // Main drawing loop
    bool quit = false; // Whether to keep drawing new frames or not
    while(!quit)
    {
        // Inclide The Device Scene SDF lambda
        #include "SDFs.cu"

        // Get the next frame of the scene N_THREADS
        getFrame<<<N_BLOCKS, N_CORES>>>( pixels, rays );

        // Wait for the getFrame function to finish on the GPU
        cudaError_t error = cudaDeviceSynchronize();

        if(error != 0)
        {
            std::cout << cudaGetErrorString(error) << "\n";
            exit(1);
        }
        // Update the texture that shows the pixels
        updateFrame();

        // Handle events such as quit
        SDL_Event event;
        while (SDL_PollEvent(&event)) { // For every event in queue
            switch (event.type) // If event is a quit event
            {
                case SDL_QUIT: // If a request to quit was made
                    quit = true; // quit
                    break;

                case SDL_MOUSEWHEEL: // If the mouse wheel was moved
                    if(event.wheel.y > 0) // If the mouse wheel was increased
                    {
                        CAMERA_DISTANCE += 10; // Increase the camera depth
                    } else if(event.wheel.y < 0) // If the mouse wheel was decreased
                    {
                        CAMERA_DISTANCE -= 10; // Decrease the camera deoth
                    }

                    // Make sure the camera distance is at least 10
                    CAMERA_DISTANCE = std::max( CAMERA_DISTANCE, 10 );
                    break;

                case SDL_MOUSEMOTION:
                    cameraRotation.y += (float)event.motion.yrel/1000;
                    cameraRotation.z -= (float)event.motion.xrel/1000;
                    break;

                case SDL_KEYDOWN:
                    switch (event.key.keysym.sym) {
                        case SDLK_w:
                            cameraPosition += cameraRotation.toCartesian();
                            break;
                        case SDLK_s:
                            cameraPosition -= cameraRotation.toCartesian();
                            break;
                    }
                    break;
            }
        }
    }

    // Free CPU and GPU accesible variables
    cudaFree(pixels);
    cudaFree(rays);
}

void initSDL()
{
    SDL_Init(SDL_INIT_VIDEO);

    // Creat SDL Window
    window = SDL_CreateWindow("Real Time DFT", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);

    if( STATIC_POINTER )
    {
        SDL_WarpMouseInWindow(window, WIDTH / 2, HEIGHT / 2);
        SDL_SetRelativeMouseMode( SDL_TRUE );
    }
    // Setup the render and texture
    renderer = SDL_CreateRenderer(window, -1, 0);
    tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);
}

void initRays()
{

    // Create the rays that will be used with for the casting
    int i = 0;
    for(int x = -WIDTH/2; x < WIDTH/2; x++)
    {
        for(int z = -HEIGHT/2; z < HEIGHT/2; z++)
        {
            Vector ray = Vector(x, CAMERA_DISTANCE, z, true);
            rays[i] = ray;
            ++i;
        }
    }
}

void updateFrame()
{
    // Update the texture with pixels
    SDL_UpdateTexture(tex, NULL, pixels, WIDTH * sizeof(Uint32));
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, tex, NULL, NULL);
    SDL_RenderPresent(renderer);
}

void cleanUpSDL()
{
    // Clean up SDL and quit
    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
