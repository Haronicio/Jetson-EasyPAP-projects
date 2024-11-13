// First ok version
__kernel void computeBodiesAccelerationOcl(const float G, const float softSquared, __global float *qx,
    __global float *qy, __global float *qz, __global float *m, __global float *ax, __global float *ay,
    __global float *az) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    const float rijx = qx[y] - qx[x];
    const float rijy = qy[y] - qy[x];
    const float rijz = qz[y] - qz[x];

    const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

    float powR = rijSquared + softSquared;
    powR *= rsqrt(powR) * powR;

    const float ai = G * m[y] / powR;

    ax[x] += ai * rijx;
    ay[x] += ai * rijy;
    az[x] += ai * rijz;
}

// Using local memory
/*
__kernel void computeBodiesAccelerationOcl(const float G, const float softSquared, __global float *qx,
    __global float *qy, __global float *qz, __global float *m, __global float *ax, __global float *ay,
    __global float *az) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t xl = get_local_id(0);
    size_t yl = get_local_id(1);

    // if (yl == 0) {
    //     printf("x = %d, y = %d, xl = %d, yl = %d\n", x, y, xl, yl);
    // }
    // size_t size = get_global_size(0);

    __local float total_x[1000];
    __local float total_y[1000];
    __local float total_z[1000];

    total_x[x] = qx[x];
    total_y[x] = qy[x];
    total_z[x] = qz[x];

    barrier(CLK_LOCAL_MEM_FENCE);

    float rijx = qx[y] - total_x[x]; // 1 flop
    float rijy = qy[y] - total_y[x]; // 1 flop
    float rijz = qz[y] - total_z[x]; // 1 flop

    // compute the || rij ||² distance between body i and body j
    float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
    // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
    float ai = G * m[y] / pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

    // total_x[xl] += ai * rijx;
    // total_y[xl] += ai * rijy;
    // total_z[xl] += ai * rijz;
 
    // barrier(CLK_LOCAL_MEM_FENCE);
 
    // ax[x] += total_x[xl];
    // ay[x] += total_y[xl];
    // az[x] += total_z[xl];
 
    // barrier(CLK_LOCAL_MEM_FENCE);

    ax[x] += ai * rijx; // 2 flops
    ay[x] += ai * rijy; // 2 flops
    az[x] += ai * rijz; // 2 flops

    // trying to put only half of x index
 
    // rijx = qx[y + size] - qx[x + size]; // 1 flop
    // rijy = qy[y + size] - qy[x + size]; // 1 flop
    // rijz = qz[y + size] - qz[x + size]; // 1 flop
 
    // // compute the || rij ||² distance between body i and body j
    // rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
    // // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
    // ai = G * m[y + size] / pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops
 
    // // add the acceleration value into the acceleration vector: ai += || ai ||.rij
    // ax[x + size] += ai * rijx; // 2 flops
    // ay[x + size] += ai * rijy; // 2 flops
    // az[x + size] += ai * rijz; // 2 flops
}
*/

// On dim 1
/*
 __kernel void computeBodiesAccelerationOcl(const float G, const float softSquared, __global float *qx,
     __global float *qy, __global float *qz, __global float *m, __global float *ax, __global float *ay,
     __global float *az, __global int *test) {
     
     size_t x = get_global_id(0);
     size_t size = get_global_size(0);
     size_t xl = get_local_id(0);
     size_t xg = get_group_id(0);
     // size_t y = get_global_id(1);
     // size_t size = get_global_size(0);
     // size_t xg = get_group_id(0);
     // size_t yg = get_group_id(1);
 
     const float rijx = qx[xg] - qx[xl]; // 1 flop
     const float rijy = qy[xg] - qy[xl]; // 1 flop
     const float rijz = qz[xg] - qz[xl]; // 1 flop
 
     // compute the || rij ||² distance between body i and body j
     const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz; // 5 flops
     // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
     const float ai = G * m[xg] / pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops
 
     // add the acceleration value into the acceleration vector: ai += || ai ||.rij
     ax[xl] += ai * rijx; // 2 flops
     ay[xl] += ai * rijy; // 2 flops
     az[xl] += ai * rijz; // 2 flops
}
*/