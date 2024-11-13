#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include <mipp.h>

#include "SimulationNBodySIMD.hpp"

/*
    Command test = ./bin/murb -n 1000 -i 100 -v --im cpu+simd

    Differents ameliorations of this version :

    We choose to use the data structure of array for the bodies and accelerations 
    because with this structure data are contiguous and we can then load and store direcly
    the vector registers than we use. 

    We have used as much as as possible the opti operations like the fused multiplication. 
    
    The pow function didn't currently exist with mipp so we decided to calculate the 
    pow(n, 1.5) using the operation n *= sqrt(n).

    We have limited the number of load, the only one made are to get the values of bodies.

    We have choose to only use vector register to get the element of iBody and not jBody because 
    every element need to make operations with each other so every iBody must make the operations
    with every jBody. If we load the jBody in a vector register we will need in fact to 
    make the calcul for every jBody contain in the vector register. It just become the same than 
    making it sequential for us, we then choose to give up this idea.

    Result with ./bin/murb -n 1000 -i 100 -v --im cpu+simd :
    
    We earn some fps compared to naive version but not so much ...

*/
SimulationNBodySIMD::SimulationNBodySIMD(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    // this->accelerations = (accSoA_t<float>*)malloc(this->getBodies().getN() * sizeof(accSoA_t<float>));
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());

    // compute e²
    softSquared = soft * soft; // 1 flops
    mG.resize(this->getBodies().getN());
}

void SimulationNBodySIMD::initIteration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;

        mG[iBody] = this->G * d[iBody].m;
    }
}

// V1 SIMD ok
/*
void SimulationNBodySIMD::computeBodiesAcceleration()
{
    // const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    mipp::Reg<float> rrijxIbody;
    mipp::Reg<float> rrijyIbody;
    mipp::Reg<float> rrijzIbody;

    mipp::Reg<float> rrijx;
    mipp::Reg<float> rrijy;
    mipp::Reg<float> rrijz;

    mipp::Reg<float> rrijSquared = 0.f;

    mipp::Reg<float> rai = 0.f;

    mipp::Reg<float> raccx = 0.f;
    mipp::Reg<float> raccy = 0.f;
    mipp::Reg<float> raccz = 0.f;

    mipp::Reg<float> sumx = 0.f;
    mipp::Reg<float> sumy = 0.f;
    mipp::Reg<float> sumz = 0.f;

    // flops = n² * 20
    
    unsigned long iBody;
    std::vector<float> vsumx(this->getBodies().getN());
    std::vector<float> vsumy(this->getBodies().getN());
    std::vector<float> vsumz(this->getBodies().getN());

    // Version with test of using vector register for i 
    for (iBody = 0; (iBody + mipp::N<float>()) < this->getBodies().getN(); iBody += mipp::N<float>()) {
        // flops = n * 20

        rrijxIbody.load(&d.qx[iBody]);
        rrijyIbody.load(&d.qy[iBody]);
        rrijzIbody.load(&d.qz[iBody]);
 
        rrijxIbody = -rrijxIbody;
        rrijyIbody = -rrijyIbody;
        rrijzIbody = -rrijzIbody;

        sumx = 0.f;
        sumy = 0.f;
        sumz = 0.f;
    
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {

            rrijx = rrijxIbody + d.qx[jBody];
            rrijy = rrijyIbody + d.qy[jBody];
            rrijz = rrijzIbody + d.qz[jBody];
            
            rrijSquared = mipp::fmadd(rrijx, rrijx, mipp::fmadd(rrijy, rrijy, mipp::fmadd(rrijz, rrijz, mipp::Reg<float>(softSquared))));
            rrijSquared *= mipp::sqrt(rrijSquared);
            
            rai = d.m[jBody] * this->G;
            rai /= rrijSquared;

            raccx = rai * rrijx;
            raccy = rai * rrijy;
            raccz = rai * rrijz;

            sumx += raccx;
            sumy += raccy;
            sumz += raccz;
        }

        sumx.store(&this->accelerations.ax[iBody]);
        sumy.store(&this->accelerations.ay[iBody]);
        sumz.store(&this->accelerations.az[iBody]);
    }

    for (int i = iBody; i < this->getBodies().getN(); i++) {
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            // compute e²
            const float softSquared = std::pow(this->soft, 2); // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d.m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}
*/

// V2 SIMD ok
void SimulationNBodySIMD::computeBodiesAcceleration()
{
    // const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    mipp::Reg<float> rrijxIbody;
    mipp::Reg<float> rrijyIbody;
    mipp::Reg<float> rrijzIbody;

    mipp::Reg<float> rrijx;
    mipp::Reg<float> rrijy;
    mipp::Reg<float> rrijz;

    mipp::Reg<float> rrijSquared = 0.f;

    mipp::Reg<float> rai = 0.f;

    mipp::Reg<float> raccx = 0.f;
    mipp::Reg<float> raccy = 0.f;
    mipp::Reg<float> raccz = 0.f;

    mipp::Reg<float> sumx = 0.f;
    mipp::Reg<float> sumy = 0.f;
    mipp::Reg<float> sumz = 0.f;

    // flops = n² * 20
    
    unsigned long iBody;
    std::vector<float> vsumx(this->getBodies().getN());
    std::vector<float> vsumy(this->getBodies().getN());
    std::vector<float> vsumz(this->getBodies().getN());

    // Version with test of using vector register for i 
    for (iBody = 0; (iBody + mipp::N<float>()) < this->getBodies().getN(); iBody += mipp::N<float>()) {
        // flops = n * 20

        rrijxIbody.load(&d.qx[iBody]);
        rrijyIbody.load(&d.qy[iBody]);
        rrijzIbody.load(&d.qz[iBody]);
 
        rrijxIbody = -rrijxIbody;
        rrijyIbody = -rrijyIbody;
        rrijzIbody = -rrijzIbody;

        sumx = 0.f;
        sumy = 0.f;
        sumz = 0.f;
    
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {

            rrijx = rrijxIbody + d.qx[jBody];
            rrijy = rrijyIbody + d.qy[jBody];
            rrijz = rrijzIbody + d.qz[jBody];
            
            rrijSquared = mipp::fmadd(rrijx, rrijx, mipp::fmadd(rrijy, rrijy, mipp::fmadd(rrijz, rrijz, mipp::Reg<float>(softSquared))));
            rrijSquared *= mipp::sqrt(rrijSquared);
            
            rai = mG[jBody];
            rai /= rrijSquared;

            raccx = rai * rrijx;
            raccy = rai * rrijy;
            raccz = rai * rrijz;

            sumx += raccx;
            sumy += raccy;
            sumz += raccz;
        }

        sumx.store(&this->accelerations.ax[iBody]);
        sumy.store(&this->accelerations.ay[iBody]);
        sumz.store(&this->accelerations.az[iBody]);
    }

    for (int i = iBody; i < this->getBodies().getN(); i++) {
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - d.qx[iBody];
            const float rijy = d.qy[jBody] - d.qy[iBody];
            const float rijz = d.qz[jBody] - d.qz[iBody];

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

            float powR = rijSquared + softSquared;
            powR *= std::sqrt(powR);
            // const float ai = mG[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f);
            const float ai = mG[jBody] / powR;

            this->accelerations.ax[iBody] += ai * rijx;
            this->accelerations.ay[iBody] += ai * rijy;
            this->accelerations.az[iBody] += ai * rijz;
        }
    }
}

// V2 SIMD (not ok)
/*
void SimulationNBodySIMD::computeBodiesAcceleration()
{
    // const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    mipp::Reg<float> rrijxIbody;
    mipp::Reg<float> rrijyIbody;
    mipp::Reg<float> rrijzIbody;
    mipp::Reg<float> rmIbody;

    mipp::Reg<float> rrijx;
    mipp::Reg<float> rrijy;
    mipp::Reg<float> rrijz;

    mipp::Reg<float> rrijSquared = 0.f;

    mipp::Reg<float> rai = 0.f;
    mipp::Reg<float> rai2 = 0.f;

    mipp::Reg<float> raccx = 0.f;
    mipp::Reg<float> raccy = 0.f;
    mipp::Reg<float> raccz = 0.f;

    mipp::Reg<float> sumx = 0.f;
    mipp::Reg<float> sumy = 0.f;
    mipp::Reg<float> sumz = 0.f;

    // std::vector<mipp::Reg<float>> vrsumx(this->getBodies().getN() / mipp::N<float>() + 1, 0.f);
    // std::vector<mipp::Reg<float>> vrsumy(this->getBodies().getN() / mipp::N<float>() + 1, 0.f);
    // std::vector<mipp::Reg<float>> vrsumz(this->getBodies().getN() / mipp::N<float>() + 1, 0.f);

    std::vector<mipp::Reg<float>> vrsumx(this->getBodies().getN(), 0.f);
    std::vector<mipp::Reg<float>> vrsumy(this->getBodies().getN(), 0.f);
    std::vector<mipp::Reg<float>> vrsumz(this->getBodies().getN(), 0.f);

    mipp::Reg<float> sumYx = 0.f;
    mipp::Reg<float> sumYy = 0.f;
    mipp::Reg<float> sumYz = 0.f;

    // flops = n² * 20
    
    unsigned long iBody;
    std::vector<float> vsumx(this->getBodies().getN());
    std::vector<float> vsumy(this->getBodies().getN());
    std::vector<float> vsumz(this->getBodies().getN());

    // Version with test of using vector register for i 
    for (iBody = 0; (iBody + mipp::N<float>()) < this->getBodies().getN(); iBody += mipp::N<float>()) {
        // flops = n * 20

        rrijxIbody.load(&d.qx[iBody]);
        rrijyIbody.load(&d.qy[iBody]);
        rrijzIbody.load(&d.qz[iBody]);
        rmIbody.load(&d.m[iBody]);
 
        rrijxIbody = -rrijxIbody;
        rrijyIbody = -rrijyIbody;
        rrijzIbody = -rrijzIbody;

        sumx = 0.f;
        sumy = 0.f;
        sumz = 0.f;
    
        for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody++) {

            rrijx = rrijxIbody + d.qx[jBody];
            rrijy = rrijyIbody + d.qy[jBody];
            rrijz = rrijzIbody + d.qz[jBody];
            
            rrijSquared = mipp::fmadd(rrijx, rrijx, mipp::fmadd(rrijy, rrijy, mipp::fmadd(rrijz, rrijz, mipp::Reg<float>(softSquared))));
            rrijSquared *= mipp::sqrt(rrijSquared);
            
            rai = d.m[jBody] * this->G;
            rai /= rrijSquared;

            rai2 = rmIbody * this->G;
            rai2 /= rrijSquared;

            sumx = mipp::fmadd(rai, rrijx, sumx);
            sumy = mipp::fmadd(rai, rrijy, sumy);
            sumz = mipp::fmadd(rai, rrijz, sumz);
            
            vrsumx[jBody] += rai2 * (-rrijx);
            vrsumy[jBody] += rai2 * (-rrijy);
            vrsumz[jBody] += rai2 * (-rrijx);
        }

        if (iBody != 0) {
            sumYx = {vrsumx[iBody].sum(), vrsumx[iBody + 1].sum(), vrsumx[iBody + 2].sum(), vrsumx[iBody + 3].sum()};
            sumYy = {vrsumy[iBody].sum(), vrsumy[iBody + 1].sum(), vrsumy[iBody + 2].sum(), vrsumy[iBody + 3].sum()};
            sumYz = {vrsumz[iBody].sum(), vrsumz[iBody + 1].sum(), vrsumz[iBody + 2].sum(), vrsumz[iBody + 3].sum()};
            sumx += sumYx;
            sumy += sumYy;
            sumz += sumYz;
        }

        sumx.store(&this->accelerations.ax[iBody]);
        sumy.store(&this->accelerations.ay[iBody]);
        sumz.store(&this->accelerations.az[iBody]);
    }
    */
    /*
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // flops = n * 20
        const float x = d[iBody].qx;
        const float y = d[iBody].qy;
        const float z = d[iBody].qz;
        const float m = d[iBody].m;

        if (iBody != 0) {
            this->accelerations[iBody].ax += xSave[iBody]; // 2 flops
            this->accelerations[iBody].ay += ySave[iBody]; // 2 flops
            this->accelerations[iBody].az += zSave[iBody]; // 2 flops
        }
        for (unsigned long jBody = iBody; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d[jBody].qx - x; // 1 flop
            const float rijy = d[jBody].qy - y; // 1 flop
            const float rijz = d[jBody].qz - z; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}

            const float powR = std::pow(rijSquared + softSquared, 3.f / 2.f); // 2 flops
            const float ai = this->G * d[jBody].m / powR; // 3 flops
            const float ai2 = this->G * m / powR; // 3 flops

            this->accelerations[iBody].ax += ai * rijx; // 2 flops
            this->accelerations[iBody].ay += ai * rijy; // 2 flops
            this->accelerations[iBody].az += ai * rijz; // 2 flops

            xSave[jBody] += ai2 * (-rijx);
            ySave[jBody] += ai2 * (-rijy);
            zSave[jBody] += ai2 * (-rijz);
        }
    }
    */
    
    /*
    for (int i = iBody; i < this->getBodies().getN(); i++) {
        for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
            const float rijx = d.qx[jBody] - d.qx[iBody]; // 1 flop
            const float rijy = d.qy[jBody] - d.qy[iBody]; // 1 flop
            const float rijz = d.qz[jBody] - d.qz[iBody]; // 1 flop

            // compute the || rij ||² distance between body i and body j
            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
            // compute e²
            const float softSquared = std::pow(this->soft, 2); // 1 flops
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            const float ai = this->G * d.m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            this->accelerations.ax[iBody] += ai * rijx; // 2 flops
            this->accelerations.ay[iBody] += ai * rijy; // 2 flops
            this->accelerations.az[iBody] += ai * rijz; // 2 flops
        }
    }
}
*/

void SimulationNBodySIMD::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
