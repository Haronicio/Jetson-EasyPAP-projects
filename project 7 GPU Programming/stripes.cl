/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

/*
  ./run -k stripes -l images/1024.png -n --gpu -tw 128 -th 2 -i 1000 --arg 4 : 4048.801
  ./run -k stripes -l images/1024.png -n --gpu -tw 128 -th 2 -i 1000 --arg 5 : 2175.327

  2⁴ = 16, 2⁵ = 32
  It seems the size of a warp is of 32 threads because the execution time fall at the 
  moment where we get strides of 2⁵ pixels.
*/

#include "kernel/ocl/common.cl"


static unsigned scale_component (unsigned c, unsigned percentage)
{
  unsigned coul;

  coul = c * percentage / 100;
  if (coul > 255)
    coul = 255;

  return coul;
}

static unsigned scale_color (unsigned c, unsigned percentage)
{
  int4 v = color_to_int4 (c);

  v.s1 = scale_component (v.s1, percentage); // Blue
  v.s2 = scale_component (v.s2, percentage); // Green
  v.s3 = scale_component (v.s3, percentage); // Red

  return int4_to_color (v);
}

static unsigned brighten (unsigned c)
{
  for (int i = 0; i < 15; i++)
    c = scale_color (c, 101);

  return c;
}

static unsigned darken (unsigned c)
{
  for (int i = 0; i < 15; i++)
    c = scale_color (c, 99);

  return c;
}

__kernel void stripes_ocl (__global unsigned *in, __global unsigned *out)
{
  int y = get_global_id (1);
  int x = get_global_id (0);

  #ifdef PARAM
    unsigned mask = (1 << PARAM);
  #else
    unsigned mask = 1;
  #endif
  
  if (x & mask)
    out [y * DIM + x] = brighten (in [y * DIM + x]);
  else
    out [y * DIM + x] = darken (in [y * DIM + x]);
}

// We assume that GPU_TILE_W is an even multiple of 32 (64, 128, 192, ...)
__kernel void stripes_ocl_opt (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILE_H][TILE_W];
  unsigned y = get_global_id (1), yloc = get_local_id (1);
  unsigned x = get_global_id (0), xloc = get_local_id (0);
  unsigned index = 2 * xloc;

  // We fetch a tile by reading global memory contiguously 
  tile[yloc][xloc] = in [y * DIM + x];

  // We make sure the tile is completely fetched
  barrier (CLK_LOCAL_MEM_FENCE);

  // The first half of buddies executes "darken" on even x-positions
  // The second half executes "brighten" on odd x-positions
  // Accesses to "tile" are not contiguous at all, but it's harmless
  // Because (GPU_TILE_W/2) is a multiple of 32, there is no divergence inside warps
  if (index < get_local_size (0)) {
    tile [yloc][index] = darken(tile [yloc][index]);
  } else {
    index += - get_local_size (0) + 1;
    tile [yloc][index] = brighten (tile [yloc][index]);
  }

  barrier (CLK_LOCAL_MEM_FENCE);

  // After a second barrier, the tile is written back to memory
  out [y * DIM + x] = tile [yloc][xloc];
}
