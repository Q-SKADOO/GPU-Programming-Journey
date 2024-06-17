







![image](https://github.com/Q-SKADOO/GPU-Programming-Journey/assets/112571800/08bf9bab-a62a-4666-b3dc-adbc57fc31fe)
Concurrent


![image](https://github.com/Q-SKADOO/GPU-Programming-Journey/assets/112571800/31d0fe03-2025-4d43-8760-411e2a41f6c1)
Threadorder


![image](https://github.com/Q-SKADOO/GPU-Programming-Journey/assets/112571800/ab1528aa-b922-48c7-8581-63235702be2f)
Inorderkernels

'''c++
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib> // Include for random number generation
#include <ctime>   // Include for seeding random number generator


__global__ void player_one(int *array, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int pos = i + j * nx;

    printf("Thread ID: (%d, %d), Block ID: (%d, %d), Global ID: (%d, %d), Position: (%2d), Value: %d \n",
            (int) threadIdx.x,
            (int) threadIdx.y,
            (int) blockIdx.x,
            (int) blockIdx.y,
            i,
            j,
            pos,
            array[pos]);

     if (i < nx && j < ny) {
        // Calculate the linear index
        int linearIdx = pos;

        // Ensure only one thread updates the array element
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            // Check if the randomly chosen cell is empty (contains a 0)
            if (array[linearIdx] == 0) {
                // Place an "X" (or 1) at the randomly chosen index
                atomicExch(&array[linearIdx], 1); // Assuming 1 represents "X"
            }
        }
    }


     //
        printf("Thread ID: (%d, %d), Block ID: (%d, %d), Global ID: (%d, %d), Position: (%2d), Value: %d \n",
            (int) threadIdx.x,
            (int) threadIdx.y,
            (int) blockIdx.x,
            (int) blockIdx.y,
            i,
            j,
            pos,
            array[pos]);

}

__global__ void player_two(int *array, int nx, int ny) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int pos = i + j * nx;

    printf("Thread ID: (%d, %d), Block ID: (%d, %d), Global ID: (%d, %d), Position: (%2d), Value: %d \n",
            (int) threadIdx.x,
            (int) threadIdx.y,
            (int) blockIdx.x,
            (int) blockIdx.y,
            i,
            j,
            pos,
            array[pos]);


        if (i < nx && j < ny) {
        // Initialize random number generator
        srand(clock64() * threadIdx.x + threadIdx.y);

        // Keep trying to find a random thread with corresponding array element 0
        bool updated = false;
        while (!updated) {
            // Randomly select a thread within the block
            int randomIdxX = rand() % blockDim.x;
            int randomIdxY = rand() % blockDim.y;
            int linearIdx = randomIdxX + randomIdxY * nx;

            // Ensure only one thread updates the array element
            if (threadIdx.x == randomIdxX && threadIdx.y == randomIdxY) {
                // Check if the cell is empty (contains a 0) and not already marked by player_one (1)
                if (array[pos] == 0) {
                    // Place an "O" (or 2) at the randomly chosen index
                    atomicExch(&array[pos], 2); // Assuming 2 represents "O"
                    updated = true;
                    printf("Updated array[%d] to %d\n", pos, array[pos]);
                } else {
                    printf("Skipping update of array[%d] with value %d\n", pos, array[pos]);
                }
            }

            __syncthreads(); // Synchronize threads within the block before retrying
        }
    }


            printf("Thread ID: (%d, %d), Block ID: (%d, %d), Global ID: (%d, %d), Position: (%2d), Value: %d \n",
            (int) threadIdx.x,
            (int) threadIdx.y,
            (int) blockIdx.x,
            (int) blockIdx.y,
            i,
            j,
            pos,
            array[pos]);


}


int main() {

    // Create a 3 x 3 tic tac toe board
    int nx = 3;
    int ny = nx;
    int *h_I = new int[nx * ny];

    for (int i = 0; i < nx * ny; ++i)
        h_I[i] = 0;

    //for (int j = 0; j < nx; ++j)
    //    h_I[j + nx * j] = 1;

    printf("Indices:\n");
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j)
            printf("%2d ", i + nx * j);
    printf("\n");
    }

    printf("Values:\n");
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j)
            printf("%d ", h_I[i + nx * j]);
    printf("\n");
    }


    // Decompose the identity matrix into 1 x 1 blocks and each block of size 3 x 3
    dim3 grid(1, 1);
    dim3 block(3, 3);

    // Allocate memory on the device and copy the identity matrix over from the host to the device
    int *d_I;
    size_t numbytes = nx * ny * sizeof(int);
    hipMalloc(&d_I, numbytes);
    hipMemcpy(d_I, h_I, numbytes, hipMemcpyHostToDevice);

    // Launch the kernel that prints information about how the identity matrix is mapped onto the
    // chosen decomposition
    player_one<<<grid, block>>>(d_I, nx, ny);
    hipDeviceSynchronize();

    // Launch the kernel to update the board by player_two
    player_two<<<grid, block>>>(d_I, nx, ny);
    hipDeviceSynchronize();

    // Copy the updated board back to the host
    hipMemcpy(h_I, d_I, numbytes, hipMemcpyDeviceToHost);

    // Print the updated board
    std::cout << "Updated Board:\n";
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j)
            std::cout << h_I[i + nx * j] << " ";
        std::cout << std::endl;
    }


    hipFree(d_I);

    // Free host memory
    delete[] h_I;

    return 0;
}
'''

![image](https://github.com/Q-SKADOO/GPU-Programming-Journey/assets/112571800/4498472d-24ae-4fc9-923a-2fbd953281e7)
Threadblock

![image](https://github.com/Q-SKADOO/GPU-Programming-Journey/assets/112571800/2c80ebe8-c53a-4ffe-a5a4-66ea4852b58a)

Delayloop
