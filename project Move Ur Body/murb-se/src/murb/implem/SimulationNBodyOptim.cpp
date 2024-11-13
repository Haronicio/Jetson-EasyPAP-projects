#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyOptim.hpp"

/*
    Command test = ./bin/murb -n 1000 -i 100 -v --im cpu+optim

    starting complexity = O(n²)

    We can reach a lower complexity transforming the algorithm in some triangle instead of a cube, and
    getting a complexity of O(n²/2), but not too sure how do that

    In fact the complexity will be more like n+n-1+n-2 but it becomes something n²/2 so it's more or less
    what I was looking for.

    To make it happens, I saw some operations was more or less the same, especially this one
    d[jBody] - d[iBody], the result of the operation we will make further will be d[iBody] - d[jBody]
    but in fact the second has the same result of the negation of the first
    -(d[jBody] - d[iBody]) =  d[iBody] - d[jBody].
    With this idea we can while we are looping on the iBody keep for each other Body their x, y and z
    calcul without having to recalculate everything. To stay simple we in fact for each loop compute two
    points x, y, z and not just one, but we didn't have to redo every operations, only two for this second computation
    the ai computation because here it depends of iBody and no more of jBody.

    We can remove some operations who are too much times :
    - first one would be the calcul of the softSquared variable who is just a constant in the function
   computeBodiesAcceleration and so who can be compute only once at the start of the function.

    - Secondly we can store in variables x,y and z the values of d[iBody].qx, d[iBody].qy, d[iBody].qz to not have to
   look for them each loop of j because they depend only of iBody.

    - Thirdly with the same idea instead of directly adding the acceleration valueinto the acceleration vector we create
   intermediate variables sumAx, sumAy, sumAz.

    - Remove the costly operation pow and replace it with * when it's simply a pow like pow(x, 2) and else replace the 
    pow(x, 3.f / 2.f) with a sqrt and * operation

    - Calculate this->G * d.m for each body only once at each start of iteration, use one more vector but less cpu operation.

    It made a *2.2 on the jetson.
*/
SimulationNBodyOptim::SimulationNBodyOptim(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
    softSquared = this->soft * this->soft; // 1 flops
    mG.resize(this->getBodies().getN());
}

void SimulationNBodyOptim::initIteration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;

        mG[iBody] = this->G * d[iBody].m;
    }
}

// Optim V1 ok
/*
void SimulationNBodyOptim::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        const float x = d[iBody].qx;
        const float y = d[iBody].qy;
        const float z = d[iBody].qz;

        float sumAx = 0.f;
        float sumAy = 0.f;
        float sumAz = 0.f;
        // flops = n * 19
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - x; // 1 flop
            const float rijy = d[jBody].qy - y; // 1 flop
            const float rijz = d[jBody].qz - z; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            float powR = rijSquared + softSquared;
            powR *= std::sqrt(powR);
            // powR = std::pow(rijSquared + softSquared, 3.f / 2.f); 
            // ai = this->G * d[jBody].m / powR; 
            const float ai = this->G * d[jBody].m / powR;                 
            // const float ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            sumAx += ai * rijx; // 2 flops
            sumAy += ai * rijy; // 2 flops
            sumAz += ai * rijz; // 2 flops
        }
        this->accelerations[iBody].ax += sumAx;
        this->accelerations[iBody].ay += sumAy;
        this->accelerations[iBody].az += sumAz;
    }
}
*/

// Optim OK v2
/*
void SimulationNBodyOptim::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {

        const float x = d[iBody].qx;
        const float y = d[iBody].qy;
        const float z = d[iBody].qz;
        const float m = d[iBody].m;

        for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - x; 
            const float rijy = d[jBody].qy - y; 
            const float rijz = d[jBody].qz - z; 

            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); 

            const float powR = std::pow(rijSquared + softSquared, 3.f / 2.f); 
            const float ai = this->G * d[jBody].m / powR;                     

            const float ai2 = -this->G * m / powR; 

            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;

            //better for cache and fps
            this->accelerations[jBody].ax += ai2 * (rijx);
            this->accelerations[jBody].ay += ai2 * (rijy);
            this->accelerations[jBody].az += ai2 * (rijz);
        }
    }
}
*/

// Optim OK v3 (best version)
void SimulationNBodyOptim::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    
    float x, y, z, rijx, rijy, rijz, rijSquared, powR, ai, ai2, imG;

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {

        x = d[iBody].qx;
        y = d[iBody].qy;
        z = d[iBody].qz;
        imG = -mG[iBody];

        for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody++) {
            rijx = d[jBody].qx - x; 
            rijy = d[jBody].qy - y; 
            rijz = d[jBody].qz - z; 

            rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; 

            powR = rijSquared + softSquared;
            powR *= std::sqrt(powR);
            ai = mG[jBody] / powR;                    

            ai2 = imG / powR; 
            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;

            this->accelerations[jBody].ax += ai2 * (rijx);
            this->accelerations[jBody].ay += ai2 * (rijy);
            this->accelerations[jBody].az += ai2 * (rijz);
        }
    }
}

// Optim v4
/*
void SimulationNBodyOptim::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    
    std::vector<float> ms(this->getBodies().getN(), 0);
    // float m;
    float x, y, z, rijx, rijy, rijz, rijSquared, powR, ai, ai2;

    for (unsigned long i = 0; i < this->getBodies().getN(); i++) {
        ms[i] = this->G * d[i].m;
    }

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {

        x = d[iBody].qx;
        y = d[iBody].qy;
        z = d[iBody].qz;
        // m = -this->G * d[iBody].m;

        for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody+=2) {
            // loop 1
            rijx = d[jBody].qx - x; 
            rijy = d[jBody].qy - y; 
            rijz = d[jBody].qz - z; 

            // pow function more costly
            rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; 

            powR = rijSquared + softSquared;
            powR *= std::sqrt(powR);
            // powR = std::pow(rijSquared + softSquared, 3.f / 2.f); 
            // ai = this->G * d[jBody].m / powR; 
            ai = ms[jBody] / powR;                    

            ai2 = (-ms[iBody]) / powR; 
            // ai2 = m * powR;
            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;

            //better for cache and fps
            this->accelerations[jBody].ax += ai2 * (rijx);
            this->accelerations[jBody].ay += ai2 * (rijy);
            this->accelerations[jBody].az += ai2 * (rijz);


            // loop 2
            rijx = d[jBody + 1].qx - x; 
            rijy = d[jBody + 1].qy - y; 
            rijz = d[jBody + 1].qz - z; 

            // pow function more costly
            rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; 

            powR = rijSquared + softSquared;
            powR *= std::sqrt(powR);
            // powR = std::pow(rijSquared + softSquared, 3.f / 2.f); 
            // ai = this->G * d[jBody].m / powR; 
            ai = ms[jBody + 1] / powR;                    

            ai2 = (-ms[iBody]) / powR; 
            // ai2 = m * powR;
            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;

            //better for cache and fps
            this->accelerations[jBody + 1].ax += ai2 * (rijx);
            this->accelerations[jBody + 1].ay += ai2 * (rijy);
            this->accelerations[jBody + 1].az += ai2 * (rijz);
        }
    }
}
*/

void SimulationNBodyOptim::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
