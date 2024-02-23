/*Mandelbrot.cu
CUDA C program that utilises the nvidia GPU to compute the mandelbrot set.

The GPU is used to compute the mandelbrot set before copying the data back over to the host for producing the bitmap.

Author: Bryce Watson
Student ID: 220199390

Parameters:
  1. The desired width of the mandelbrot image
  2. The desired height of the mandelbrot image

Returns:
  0 on success,
  1 on failure

To build and run this program you must be on bourbaki

To build it use: make

To run:
  make run (this will run the program with width = 1920 and height = 1080, producing a 1920x1080 mandelbrot image)

OR

  ./mandelbrot <width> <height>

e.g ./mandelbrot 1920 1080

to clean/remove files:
  make clean

*/

/****** Included libaries ******/
#include "bmpfile.h"
#include <stdio.h>
#include <cuda_runtime.h>

/*Mandelbrot values*/
#define RESOLUTION 971500.0
#define XCENTER -0.77099513
#define YCENTER 0.10872488389
#define MAX_ITER  11150

/*Colour Values*/
#define COLOUR_DEPTH 225
#define COLOUR_MAX 190.0
#define GRADIENT_COLOUR_MAX 230.0

#define FILENAME "my_mandelbrot_fractal.bmp"

/* Mandelbrot Cuda Kernel function */
__global__ void
mandelbrot (double *results, int width, int height, int xoffset, int yoffset)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Determine where in the mandelbrot set, the pixel is referencing
  double x = XCENTER + ((double) xoffset + (double) col) / RESOLUTION;
  double y = YCENTER + ((double) yoffset - (double) row) / RESOLUTION;

  // declare mandelbrot values
  double a = 0;
  double b = 0;
  double aold = 0;
  double bold = 0;
  double zmagsqr = 0;
  int iter = 0;

  // Check if the x,y coord are part of the mendelbrot set - refer to the
  // algorithm
  while (iter < MAX_ITER && zmagsqr <= 4.0)
    {
      ++iter;
      a = (aold * aold) - (bold * bold) + x;
      b = 2.0 * aold * bold + y;

      zmagsqr = a * a + b * b;

      aold = a;
      bold = b;
    }

  // Add the iter value to the results array
  int index = row * width + col;
  results[index] = iter;
}


/**
 * Computes the color gradiant
 * color: the output vector
 * x: the gradiant (beetween 0 and 360)
 * min and max: variation of the RGB channels (Move3D 0 -> 1)
 * Check wiki for more details on the colour science:
 * en.wikipedia.org/wiki/HSL_and_HSV
 */
void
GroundColorMix (double *color, double x, double min, double max)
{
  /*
   * Red = 0
   * Green = 1
   * Blue = 2
   */
  double posSlope = (max - min) / 60;
  double negSlope = (min - max) / 60;

  if (x < 60)
    {
      color[0] = max;
      color[1] = posSlope * x + min;
      color[2] = min;
      return;
    }
  else if (x < 120)
    {
      color[0] = negSlope * x + 2.0 * max + min;
      color[1] = max;
      color[2] = min;
      return;
    }
  else if (x < 180)
    {
      color[0] = min;
      color[1] = max;
      color[2] = posSlope * x - 2.0 * max + min;
      return;
    }
  else if (x < 240)
    {
      color[0] = min;
      color[1] = negSlope * x + 4.0 * max + min;
      color[2] = max;
      return;
    }
  else if (x < 300)
    {
      color[0] = posSlope * x - 4.0 * max + min;
      color[1] = min;
      color[2] = max;
      return;
    }
  else
    {
      color[0] = max;
      color[1] = min;
      color[2] = negSlope * x + 6 * max;
      return;
    }
}

/**
 * @brief Simple function to parse command line arguments
 *
 * This function ensures that the correct command line arguments are given.
 * This function expects the following command line arguments:
 * 	argv[1]: Desired width of the mandelbrot image
 * 	argv[2]: Desired height of the mandelbrot image
 *
 * @param argc An integer representing the number of command line arguments
 * @param *argv[] An array of pointers to strings representing the command line arguments
 * @param *width A pointer to a float representing the desired width of the mandelbrot image
 * input file
 * @param *height A pointer to a float representing the desired height of the mandelbrot image
 *
 * @return 0 on success, -1 on failure
 */
int
parse_args (int argc, char *argv[], float *width, float *height)
{
  if ((argc != 3) || ((*width = atoi (argv[1])) <= 0)
      || ((*height = atoi (argv[2])) <= 0))
    {
      fprintf (stderr, "Usage: %s <width> <height>\n", argv[0]);
      return (-1);
    }
  return 0;
}

/**
 * @brief Simple cleanup function to free any allocated memory and other cleanup
 *
 * @param *d_results A pointer to the device's copy of the results array
 * @param *results A pointer to the host's copy of the results array
 * @param err A variable representing the Cuda errors
 *
 * @return nothing
 */
void
cleanup (double *d_results, double *results, cudaError_t err)
{

  if (results)
    {
      free (results);
      results = NULL;
    }

  if (results)
    {
      fprintf (stderr, "Failed to free host results array");
      exit (EXIT_FAILURE);
    }

  err = cudaFree (d_results);

  if (err != cudaSuccess)
    {
      fprintf (stderr,
	       "Failed to free device results vector (error code %s)\n",
	       cudaGetErrorString (err));
      exit (EXIT_FAILURE);
    }

  err = cudaDeviceReset ();

  if (err != cudaSuccess)
    {
      fprintf (stderr, "Failed to denitialize the device, error: %s\n",
	       cudaGetErrorString (err));
      exit (EXIT_FAILURE);
    }

}

/**
 * @brief Main function: Sets up CUDA environment to Compute the mandelbrot set on a GPU before producing a corresponding bitmap to a file on the host machine.
 *
 * @param argc An integer representing the number of command line arguments
 * @param *argv[] An array of pointers to the command line arguments
 *
 * @return 0 on success, 1 on failure
 */
int
main (int argc, char *argv[])
{

  /* Get width and height from command line args */
  float width;
  float height;

  if (parse_args (argc, argv, &width, &height) < 0)
    exit (EXIT_FAILURE);

  /* Declare variables */
  bmpfile_t *bmp;
  rgb_pixel_t pixel = { 0, 0, 0, 0 };
  int xoffset = -(width - 1) / 2;
  int yoffset = (height - 1) / 2;
  bmp = bmp_create (width, height, 32);
  int col = 0;
  int row = 0;
  cudaError_t err = cudaSuccess;
  size_t size = width * height * sizeof (double);


  /* Allocate memory */
  double *results = (double *) malloc (size);	//results array (data for each pixel)

  /* Allocate device memory */
  double *d_results = NULL;
  err = cudaMalloc ((void **) &d_results, size);

  //check if results vector allocation was successful
  if (results == NULL)
    {
      fprintf (stderr, "Failed to allocate host results vector");
      cleanup (d_results, results, err);
      exit (EXIT_FAILURE);
    }

  //check if device results vector allocation was successful
  if (err != cudaSuccess)
    {
      fprintf (stderr, "Failed to allocate device results vector");
      cleanup (d_results, results, err);
      exit (EXIT_FAILURE);
    }

  //Determine threads per block (16x16 = 256 threads)
  dim3 threadsPerBlock (16, 16);
  //Determine blocks per grid
  dim3 blocksPerGrid ((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
		      (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  /* Launch Cuda Kernel */
  mandelbrot <<< blocksPerGrid, threadsPerBlock >>> (d_results, width, height,
						     xoffset, yoffset);

  /* check if kernel launched correctly */
  err = cudaGetLastError ();

  if (err != cudaSuccess)
    {
      fprintf (stderr, "Failed to launch mandelbrot kernel (error code %s)\n",
	       cudaGetErrorString (err));
      cleanup (d_results, results, err);
      exit (EXIT_FAILURE);
    }

  /* Copy mandelbrot set data (device result vector) back from the GPU to the CPU (host result vector) */
  err = cudaMemcpy (results, d_results, size, cudaMemcpyDeviceToHost);

  // check if the copy was successful
  if (err != cudaSuccess)
    {
      fprintf (stderr,
	       "Failed to copy results vector from device to host (error code %s)\n",
	       cudaGetErrorString (err));
      cleanup (d_results, results, err);
      exit (EXIT_FAILURE);
    }

  /* Color mapping to map mandelbrot set data to pixel colours */
  for (col = 0; col < width; col++)
    {
      for (row = 0; row < height; row++)
	{

	  //get the iter value from the results array
	  int iter = results[(int) (row * width + col)];


	  // Mandelbrot stuff
	  double x_col;
	  double color[3];


	  /* Generate the colour of the pixel from the iter value */
	  x_col = (COLOUR_MAX -
		   ((((float) iter / ((float) MAX_ITER) *
		      GRADIENT_COLOUR_MAX))));
	  GroundColorMix (color, x_col, 1, COLOUR_DEPTH);
	  pixel.red = color[0];
	  pixel.green = color[1];
	  pixel.blue = color[2];
	  bmp_set_pixel (bmp, col, row, pixel);
	}
    }

  //Create BMP File
  bmp_save (bmp, FILENAME);

  //Cleanup/exit
  bmp_destroy (bmp);

  cleanup (d_results, results, err);

  exit (EXIT_SUCCESS);
}
