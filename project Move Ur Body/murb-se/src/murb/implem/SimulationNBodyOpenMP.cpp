#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <string>

#include "SimulationNBodyOpenMP.hpp"

/*
    Command test = ./bin/murb -n 1000 -i 100 -v --im cpu+omp
    OMP_NUM_THREADS=8 OMP_SCHEDULE="dynamic,16"
*/
SimulationNBodyOpenMP::SimulationNBodyOpenMP(const unsigned long nBodies, const std::string &scheme, const float soft,
                                             const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    softSquared = this->soft * this->soft;
    mG.resize(this->getBodies().getN());
    /*
    omp_sched_t kind;
    int chunk_size;

    printf("OMP_NUM_THREADS = %d\n", omp_get_max_threads());

    omp_get_schedule(&kind, &chunk_size);
    printf("OMP_SCHEDULE = %s, chunk size = %d\n",
           (kind == omp_sched_static) ? "static" :
           (kind == omp_sched_dynamic) ? "dynamic" :
           (kind == omp_sched_guided) ? "guided" : "auto",
           chunk_size);
    */
}

void SimulationNBodyOpenMP::initIteration()
{
      const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;

        mG[iBody] = this->G * d[iBody].m;
    }
}

// OMP with the old optim version (but still the fastest)
void SimulationNBodyOpenMP::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const float softSquared = soft * soft;
    std::vector<float> ms(this->getBodies().getN(), 0);

#pragma omp parallel 
    {
        #pragma omp for schedule(runtime)
        for (unsigned long i = 0; i < this->getBodies().getN(); i++) {
            ms[i] = this->G * d[i].m;
        }
    }

    float sumAx = 0.f;
    float sumAy = 0.f;
    float sumAz = 0.f;

    #pragma omp parallel firstprivate(sumAx,sumAy,sumAz)
    {
        #pragma omp for schedule(runtime)
        for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
            const float x = d[iBody].qx;
            const float y = d[iBody].qy;
            const float z = d[iBody].qz;
            
            sumAx = 0.f;
            sumAy = 0.f;
            sumAz = 0.f;

            for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
                const float rijx = d[jBody].qx - x;
                const float rijy = d[jBody].qy - y;
                const float rijz = d[jBody].qz - z;

                const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
                float powR = rijSquared + softSquared;
                powR *= std::sqrt(powR);
                const float ai = ms[jBody] / powR;    

                sumAx += ai * rijx;
                sumAy += ai * rijy;
                sumAz += ai * rijz;
            }
            this->accelerations[iBody].ax += sumAx;
            this->accelerations[iBody].ay += sumAy;
            this->accelerations[iBody].az += sumAz;
        }
    }
}


// Optim Critical version (much more memory consumming than above version) Run faster on other machine than jetson ??!
/*
void SimulationNBodyOpenMP::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    std::vector<std::vector<float>> thread_i_accelerationX;
    std::vector<std::vector<float>> thread_i_accelerationY;
    std::vector<std::vector<float>> thread_i_accelerationZ;

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp single
        {
            int thread_NUMBER = omp_get_num_threads();

            thread_i_accelerationX.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
            thread_i_accelerationY.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
            thread_i_accelerationZ.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
        }

        #pragma omp for schedule(static) nowait
        for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
            // flops = n * 20
            const float x = d[iBody].qx;
            const float y = d[iBody].qy;
            const float z = d[iBody].qz;
            const float imG = -mG[iBody];

            for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody++) {
                const float rijx = d[jBody].qx - x; // 1 flop
                const float rijy = d[jBody].qy - y; // 1 flop
                const float rijz = d[jBody].qz - z; // 1 flop


                // pow function more costly
                const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; 

                float powR = rijSquared + softSquared;
                powR *= std::sqrt(powR);
                const float ai = this->G * d[jBody].m / powR;                     

                const float ai2 = imG / powR;

                thread_i_accelerationX[thread_id][iBody] += ai * rijx; // 2 flops
                thread_i_accelerationY[thread_id][iBody] += ai * rijy; // 2 flops
                thread_i_accelerationZ[thread_id][iBody] += ai * rijz; // 2 flops

                //better for cache and fps
                thread_i_accelerationX[thread_id][jBody] += ai2 * (rijx); // 2 flops
                thread_i_accelerationY[thread_id][jBody] += ai2 * (rijy); // 2 flops
                thread_i_accelerationZ[thread_id][jBody] += ai2 * (rijz); // 2 flops

            }
            
        }
        #pragma omp critical
        for ( long unsigned int i = 0; i < this->getBodies().getN(); i++ ) {
           this->accelerations[i].ax += thread_i_accelerationX[thread_id][i];
           this->accelerations[i].ay += thread_i_accelerationY[thread_id][i];
           this->accelerations[i].az += thread_i_accelerationZ[thread_id][i];

        }
    }
}
*/

// Linearized Version (slowest version)
/*
void SimulationNBodyOpenMP::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    std::vector<int> vi, vj;
    // initialising linearised index vectors
    for (long unsigned int i = 0; i < this->getBodies().getN(); i++) {
        for (long unsigned int j = i; j < this->getBodies().getN(); j++) {
            vi.push_back(i);
            vj.push_back(j);
        }
    }
    
    std::vector<std::vector<float>> thread_i_accelerationX;
    std::vector<std::vector<float>> thread_i_accelerationY;
    std::vector<std::vector<float>> thread_i_accelerationZ;

// flops = n² * 20
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        #pragma omp single
        {
            int thread_NUMBER = omp_get_num_threads();

            thread_i_accelerationX.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
            thread_i_accelerationY.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
            thread_i_accelerationZ.resize(thread_NUMBER, std::vector<float>(this->getBodies().getN(), 0.0f));
        }

        #pragma omp for schedule(static) nowait
        for (long unsigned int k = 0; k < vi.size(); k++) {
            int i = vi[k];
            int j = vj[k];

            const float rijx = d[j].qx - d[i].qx; // 1 flop
            const float rijy = d[j].qy - d[i].qy; // 1 flop
            const float rijz = d[j].qz - d[i].qz; // 1 flop

            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
            const float powR = std::pow(rijSquared + softSquared, 3.f / 2.f);

            const float ai = this->G * d[j].m / powR;
            const float ai2 = -this->G * d[i].m / powR;

            // printf("i : %d j : %d , %.16f %.16f\n",i,j,ai,ai2);

            thread_i_accelerationX[thread_id][i] += ai * rijx; // 2 flops
            thread_i_accelerationY[thread_id][i] += ai * rijy; // 2 flops
            thread_i_accelerationZ[thread_id][i] += ai * rijz; // 2 flops

            //better for cache and fps
            thread_i_accelerationX[thread_id][j] += ai2 * (rijx); // 2 flops
            thread_i_accelerationY[thread_id][j] += ai2 * (rijy); // 2 flops
            thread_i_accelerationZ[thread_id][j] += ai2 * (rijz); // 2 flops

            // printf("i ax %.16f j ax %.16f \n",  this->accelerations[i].ax ,this->accelerations[j].ax);
        }

        #pragma omp critical
        for ( long unsigned int i = 0; i < this->getBodies().getN(); i++ ) {
           this->accelerations[i].ax += thread_i_accelerationX[thread_id][i];
           this->accelerations[i].ay += thread_i_accelerationY[thread_id][i];
           this->accelerations[i].az += thread_i_accelerationZ[thread_id][i];

        }

    }
}
*/

void SimulationNBodyOpenMP::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
