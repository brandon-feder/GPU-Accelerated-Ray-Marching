// ==== User Configurations ===
const int WIDTH = 1000;
const int HEIGHT = 1000;

const int N_THREADS = WIDTH*HEIGHT; // Should Not Be More Than the # Of Pixels
const int N_OBJECTS = 1;

const int EXAMPLE = 1;

const int MIN_DISTANCE = 1;
const int MAX_DISTANCE = 1000;
const int MAX_DEPTH = 100;

int CAMERA_DISTANCE = (WIDTH+HEIGHT)/2;

const int NORMAL_EPSILON = WIDTH*HEIGHT;

const bool STATIC_POINTER = false;

// ==== System Configurations ====
const int N_PIXELS = WIDTH*HEIGHT;
const int N_CORES = std::min( 1024, N_THREADS );
const int N_BLOCKS = ceil(((float)N_THREADS) / N_CORES);
