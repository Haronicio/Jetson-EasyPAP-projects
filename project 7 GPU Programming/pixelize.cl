/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include "kernel/ocl/common.cl"

__kernel void pixelize_ocl (__global unsigned *in)
{
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xl = get_local_id(0);
  int yl = get_local_id(1);

  __local unsigned tile[TILE_H][TILE_W];

  tile[yl][xl] = in [y * DIM + x];
  barrier(CLK_LOCAL_MEM_FENCE);

  in[y * DIM + x] = tile[0][0];
}

/*
  My barriers seems off ? The code :
  if (xl == 0 && yl < 1) { 
    tile[yl][xl] /= (TILE_H * TILE_W);
  }
  works just fine but if I had : 
  if (xl == 0 && yl < 1) { 
    tile[0][0] /= (TILE_H * TILE_W);
  }

  The image become multi color, it really seems because in a case every threads 
  are waiting, and in the other absolutely not, I'm putting everywhere barriers 
  and it didn't change a thing really strange. 

  The same way the version generic with loop didn't seems to work with similar reason ...
*/
__kernel void pixelize_ocl_red (__global unsigned *in)
{
  __local int4 tile[TILE_H][TILE_W];

  int x = get_global_id(0);
  int y = get_global_id(1);
  int xl = get_local_id(0);
  int yl = get_local_id(1);

  tile[yl][xl] = color_to_int4(in[y * DIM + x]);
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (int i = TILE_W; i > 1; i /= 2) {
    if (xl < (i/2)) {
      tile[yl][xl] += tile[yl][xl+(i/2)];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  /*
  if (xl == 0) {
    for (int i = TILE_H; i > 1; i /= 2) {
      if (yl < (i/2)) {
        tile[yl][xl] += tile[yl + (i/2)][0];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (xl == 0 && yl == 0) { 
    tile[yl][xl] /= (TILE_H * TILE_W);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  */
  
  if (xl == 0) {
    if (yl < 8) {
      tile[yl][0] += tile[yl + 8][0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (yl < 4) {
      tile[yl][0] += tile[yl + 4][0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (yl < 2) {
      tile[yl][0] += tile[yl + 2][0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (xl == 0 && yl < 1) {
      tile[0][0] += tile[yl + 1][0];
      tile[0][0] /= (TILE_H * TILE_W);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  
  in[y * DIM + x] = int4_to_color(tile[0][0]);
}
