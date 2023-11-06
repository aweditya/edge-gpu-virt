#include <parboil.h>
#include <pthread.h>
#include "client.h"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float> &v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float> &);

extern "C" void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

// Parameters of tile sizes
#define TILE_N 16
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N * TILE_TB_HEIGHT)

__global__ void mysgemmNT(const float *A, int lda, const float *B, int ldb, float *C, int ldc, int k, float alpha, float beta, int blockOffsetx, int blockOffsety);

class SGEMMClient : public Client
{
public:
    SGEMMClient(WorkloadManager *manager, cudaStream_t *offeredStream,
                struct pb_Parameters *params,
                float alpha, float beta) : Client(manager, offeredStream)
    {
        /* Read in data */
        // load A
        readColMajorMatrixFile(params->inpFiles[0], m, k, matA);

        // load B^T
        readColMajorMatrixFile(params->inpFiles[2], n, k, matBT);

        // allocate space for C
        A_sz = m * k * sizeof(float);
        B_sz = k * n * sizeof(float);
        C_sz = m * n * sizeof(float);
        matC = std::vector<float>(m * n);

        this->alpha = alpha;
        this->beta = beta;
        this->lda = m * k;
        this->ldb = k * n;
        this->ldc = m * n;

        kernel.blockConf.x = TILE_N;
        kernel.blockConf.y = TILE_TB_HEIGHT;

        kernel.gridConf.x = m / TILE_M;
        kernel.gridConf.y = n / TILE_N;

        blockOffsetx = 0;
        blockOffsety = 0;

        kernel.kernelFunction = (void *)mysgemmNT;
        kernel.arguments[0] = &matA.front();
        kernel.arguments[1] = &lda;
        kernel.arguments[2] = &matBT.front();
        kernel.arguments[3] = &ldb;
        kernel.arguments[4] = &matC.front();
        kernel.arguments[5] = &ldc;
        kernel.arguments[6] = &alpha;
        kernel.arguments[7] = &beta;
        kernel.arguments[8] = &blockOffsetx;
        kernel.arguments[9] = &blockOffsety;

        kernel.sharedMem = 0;
        pthread_create(&clientThread, NULL, SGEMMClient::threadFunction, this);
    }

    ~SGEMMClient()
    {
        pthread_join(clientThread, NULL);
    }

    void setupKernel();
    void finishKernel();

private:
    int m, n, k;
    float alpha, beta;
    std::vector<float> matA;
    size_t A_sz;
    int lda;
    std::vector<float> matBT;
    size_t B_sz;
    int ldb;
    std::vector<float> matC;
    size_t C_sz;
    int ldc;
    float *dA, *dB, *dC;
};