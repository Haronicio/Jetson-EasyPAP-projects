/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/
#include "easypap.h"

#include <omp.h>

/*
///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur -v seq -si
//
// exec time in ms

//        DEN     A57

//-O0     9200   27019

//-O1     3860    3824

//-O2     4069    3324

//-O3     3481    3784

Each time the denver cores can get very different result fall to 9s
can up to 16s, I think something it succesfully apply his simplifactions
with his hardware code translator but something it juste fail ?

*/
int blur_do_tile_default (int x, int y, int width, int height)
{
  //for every pixel on the image
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 
      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      //define the limit of the convolution (ie tile dimension)
      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;

      //for every pixel in the tile
      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++) {
          //extract pixel channels
          unsigned c = cur_img (yloc, xloc);
          //add 
          r += extract_red (c);
          g += extract_green (c);
          b += extract_blue (c);
          a += extract_alpha (c);
          n += 1;
        }
      //compute the average of the tile for each channels
      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = rgba (r, g, b, a);
    }
    
  return 0;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur -v seq
//
unsigned blur_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM, 0);

    swap_images ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur -v tiled -ts 32 -m si
//
unsigned blur_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0);

    swap_images ();
  }

  return 0;
}

/*
no border version
exec time in ms
   DEN     A57

   9000    26364

A57 a little bit faster, but Denver like the precedent question can fall
to 9s and got until 16s (probably for the same reason than for the 
default version)
*/
int blur_do_tile_default_nb(int x, int y, int width, int height)
{
  //for every pixel on the image
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 
	
      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        next_img(i,j) = cur_img(i,j);
        continue;
      }

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      //for every pixel nearest the computed pixel
      for (int yloc = i - 1; yloc <= i + 1; yloc++) 
        for (int xloc = j - 1; xloc <= j + 1; xloc++) {
          //extract pixel channels
          unsigned c = cur_img (yloc, xloc);
          //add 
          r += extract_red (c);
          g += extract_green (c);
          b += extract_blue (c);
          a += extract_alpha (c);
          n += 1;
        }
      //compute the average of the tile for each channels
      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = rgba (r, g, b, a);
    }
    
  return 0;
}

/*
optim1 version (unrolling X)
exec time in ms
   DEN     A57

   8000    24826

We gain time with the two kind of cores

Hmmm
But I think Denver cores was supposed to already make this simplification
with his hardware code translator but sometines it doesn't 
*/
int blur_do_tile_optim1(int x, int y, int width, int height)
{
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        next_img(i,j) = cur_img(i,j);
        continue;
      }

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      //for every pixel nearest the computed pixel
      for (int yloc = i - 1; yloc <= i + 1; yloc++) {
         //extract pixel channels
        unsigned c = cur_img (yloc, j-1);
        //add 
        r += extract_red (c);
        g += extract_green (c);
        b += extract_blue (c);
        a += extract_alpha (c);
        n += 1;

        c = cur_img (yloc, j);
        r += extract_red (c);
        g += extract_green (c);
        b += extract_blue (c);
        a += extract_alpha (c);
        n += 1;

        c = cur_img (yloc, j+1);
        r += extract_red (c);
        g += extract_green (c);
        b += extract_blue (c);
        a += extract_alpha (c);
        n += 1;
      }
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = rgba (r, g, b, a);
    }
    
  return 0;
}

/*
optim2 version (unrolling X and Y)
exec time in ms
   DEN     A57

   8000    23806

We gain time with the A57
*/
int blur_do_tile_optim2(int x, int y, int width, int height)
{

  short n = 9;
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        next_img(i,j) = cur_img(i,j);
        continue;
      }

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0;

      //for every pixel nearest the computed pixel
      
      //extract pixel channels
      unsigned c = cur_img (i-1, j-1);
      //add 
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i-1, j);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i-1, j+1);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i, j-1);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i, j);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i, j+1);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i+1, j-1);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i+1, j);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);

      c = cur_img (i+1, j+1);
      r += extract_red (c);
      g += extract_green (c);
      b += extract_blue (c);
      a += extract_alpha (c);
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = rgba (r, g, b, a);
    }
    
  return 0;
}


/*
optim3 version (unlining)
exec time in ms
   DEN     A57

   5367    17000

Big improvement for the two kind of cores
*/
int blur_do_tile_optim3(int x, int y, int width, int height)
{

  short n = 9;
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++) { 

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        next_img(i,j) = cur_img(i,j);
        continue;
      }

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0;

      //for every pixel nearest the computed pixel
      
      //extract pixel channels
      unsigned c = cur_img (i-1, j-1);
      //add 
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i-1, j);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i-1, j+1);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i, j-1);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i, j);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i, j+1);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i+1, j-1);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i+1, j);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = cur_img (i+1, j+1);
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = (r << 24) | (g << 16) | (b << 8) | a;
    }
    
  return 0;
}

/*
optim4 version (variable rotation)
exec time in ms
   DEN     A57

   3200    12900

Pretty good ameloration for the two kind of core
*/
int blur_do_tile_optim4(int x, int y, int width, int height)
{
  short n = 9;
  unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0;
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++) {
    for (int j = x; j < x + width; j++) { 

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        next_img(i,j) = cur_img(i,j);
        continue;
      }

      if (j == x || j == x+1) {
        c0 = cur_img(i-1, x);
        c1 = cur_img(i-1, x+1);
        c3 = cur_img(i, x);
        c4 = cur_img(i, x+1);
        c6 = cur_img(i+1, x);
        c7 = cur_img(i+1, x+1);
      }

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0;

      //for every pixel nearest the computed pixel
      
      //extract pixel channels
      unsigned c = c0;
      //add 
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c1;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c2 = cur_img (i-1, j+1);
      c = c2;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c3;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c4;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c5 = cur_img (i, j+1);
      c = c5;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c6;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c7;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c8 = cur_img (i+1, j+1);
      c = c8;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = (r << 24) | (g << 16) | (b << 8) | a;

      c0 = c1;
      c1 = c2;
      c3 = c4;
      c4 = c5;
      c6 = c7;
      c7 = c8;
    }
  }
    
  return 0;
}

/*
optim5 version (edge treatment)
exec time in ms
   DEN     A57

  3550    13000

It takes juste a little more time, normal because we add some 
loops and operations without apply the precedent opti to the 
new operations to compute border pixels ...
*/
int blur_do_tile_optim5(int x, int y, int width, int height)
{

  unsigned short n = 9;
  unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0;
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++) {
     for (int j = x; j < x + width; j++) { 

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0;

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1) {
        unsigned short n2 = 0;
        //define the limit of the convolution (ie tile dimension)
        
        int i_d = (i > 0) ? i - 1 : i;
        int i_f = (i < DIM - 1) ? i + 1 : i;
        int j_d = (j > 0) ? j - 1 : j;
        int j_f = (j < DIM - 1) ? j + 1 : j;
        
        //for every pixel in the tile
        for (int yloc = i_d; yloc <= i_f; yloc++) {
          for (int xloc = j_d; xloc <= j_f; xloc++) {
            //extract pixel channels
            unsigned c = cur_img (yloc, xloc);
            //add 
            r += extract_red (c);
            g += extract_green (c);
            b += extract_blue (c);
            a += extract_alpha (c);
            n2 += 1;
          }
        }
        
        //compute the average of the tile for each channels
        r /= n2;
        g /= n2;
        b /= n2;
        a /= n2;

        //save the new pixel channels
        next_img (i, j) = rgba (r, g, b, a);
      
        //next_img(i,j) = cur_img(i,j);
        continue;
      }

      //for every pixel nearest the computed pixel
      
      //extract pixel channels
      unsigned c = c0;
      //add 
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c1;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c2 = cur_img (i-1, j+1);
      c = c2;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c3;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c4;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c5 = cur_img (i, j+1);
      c = c5;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c6;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c7;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c8 = cur_img (i+1, j+1);
      c = c8;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = (r << 24) | (g << 16) | (b << 8) | a;

      c0 = c1;
      c1 = c2;
      c3 = c4;
      c4 = c5;
      c6 = c7;
      c7 = c8;
    }
  }
  return 0;
}

/*
optim6 version (reduction)
exec time in ms
   DEN     A57

  3400    12500

it takes nearly the same time, hmmmm but maybe it's normal
*/
int blur_do_tile_optim6(int x, int y, int width, int height)
{

  unsigned short n = 9;
  unsigned c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0, c8 = 0;
  
  //for every pixel on the image
  for (int i = y; i < y + height; i++) {
     for (int j = x; j < x + width; j++) { 

      //channels of the image (rgba) number of pixel extract (n)
      unsigned r = 0, g = 0, b = 0, a = 0;

      // border treatment
      if (i == 0 || i == DIM-1 || j == 0 || j == DIM-1)
      {
        unsigned short n2 = 0;
        //define the limit of the convolution (ie tile dimension)
        
        int i_d = (i > 0) ? i - 1 : i;
        int i_f = (i < DIM - 1) ? i + 1 : i;
        int j_d = (j > 0) ? j - 1 : j;
        int j_f = (j < DIM - 1) ? j + 1 : j;
        
        //for every pixel in the tile
        for (int yloc = i_d; yloc <= i_f; yloc++) {
          for (int xloc = j_d; xloc <= j_f; xloc++) {
            //extract pixel channels
            unsigned c = cur_img (yloc, xloc);
            //add 
            r += extract_red (c);
            g += extract_green (c);
            b += extract_blue (c);
            a += extract_alpha (c);
            n2 += 1;
          }
        }
        
        //compute the average of the tile for each channels
        r /= n2;
        g /= n2;
        b /= n2;
        a /= n2;

        //save the new pixel channels
        next_img (i, j) = rgba (r, g, b, a);
      
        //next_img(i,j) = cur_img(i,j);
        continue;
      }

      //for every pixel nearest the computed pixel

      int j1 = j+1;
      
      //extract pixel channels
      unsigned c = c0;
      //add 
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c1;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c2 = cur_img (i-1, j1);
      c = c2;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c3;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c4;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c5 = cur_img (i, j1);
      c = c5;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c6;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c = c7;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;

      c8 = cur_img (i+1, j1);
      c = c8;
      r += c >> 24;
      g += (c >> 16) & 255;
      b += (c >> 8) & 255;
      a += c & 255;
      //compute the average of the tile for each channels

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      //save the new pixel channels
      next_img (i, j) = (r << 24) | (g << 16) | (b << 8) | a;

      c0 = c1;
      c1 = c2;
      c3 = c4;
      c4 = c5;
      c6 = c7;
      c7 = c8;
    }
  }
  return 0;
}


/*

Last question and conclusion

      Denver    A57

-00    3400    12500

-01    2189    1800

-02    1736    2045

-03    1673    2100

The optimized code is faster by so much,
for the denver cores it is about 2.5* faster
but with the A57 it is about more than 10* faster.

Otherwise we can see the optimizations we have made even
with the different level of optimization accelerate the execution,
with the default version with the -O3 flags, time was about 3480ms  
but here only 1673ms, about the same difference between default et 
optim6 version for the times with A57. 

So we can say than our optimizations are usefull even using at the same the opti
flags (O?) of the compilater.

We can also see than the flag -O3 compare to -02 or -01 for the A57 doesn't change anything 
to the execution time and even can make it slower ...
*/