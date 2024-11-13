#include "kernel/ocl/common.cl"

/*
  ./run -k transpose -l images/shibuya.png --gpu --variant ocl_naif -n -i 10
  ./run --load-image images/shibuya.png --kernel transpose -n -i 10 
  ./run -l images/shibuya.png -k transpose -v tiled -ts 16 -n -i 10

  With 10 iterations :

  gpu naif version = 437.445
  cpu seq version = 723.611
  cpu tile version = 1536.093

  with 1 iteration :
  
  gpu naif version = 415.478
  cpu seq version = 69.780
  cpu tile version = 160.195

  The cpu version is faster with not so much iteration, it can be explain by the 
  fact the writing on the out are not done in a coalescent way.
*/

__kernel void transpose_ocl_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}

/*
  ./run -k transpose -l images/shibuya.png --gpu --variant ocl_tiled -n -i 1
*/
__kernel void transpose_ocl_tiled (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile[TILE_H][TILE_W];

  int x = get_global_id(0);
  int y = get_global_id(1);
  int xl = get_local_id(0);
  int yl = get_local_id(1);

  tile[yl][xl] = in[y * DIM + x];

  barrier(CLK_LOCAL_MEM_FENCE);

  out[x * DIM + y] = tile[xl][yl];
}
