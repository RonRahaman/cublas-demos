#define N 16
#define threadsPerBlock 4
#define blocksPerGrid 4

__global__ void dot(float *a, float *b, float *c, int N) {
    __shared__ float cache[32]; // Careful!
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    
    // set the cache values
    cache[cacheIndex] = temp;
    
    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

extern "C" void f_dot(float *a, float*b, float *c, int N, int blocksPerGrid, int threadsPerBlock)  {
  return dot<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);
}

