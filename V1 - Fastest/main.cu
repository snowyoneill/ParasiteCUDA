#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cutil.h> // For GPU timer

#  define CUSTOM_CUDA_SAFE_CALL(call) {                                      \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
		getchar();                                                           \
		exit(EXIT_FAILURE);                                                  \
		}																	 \
	}

texture<unsigned char, 1, cudaReadModeElementType> texRef;
texture<unsigned char, 1, cudaReadModeElementType> minTex;
texture<unsigned char, 1, cudaReadModeElementType> maxTex;

texture<unsigned char, 1, cudaReadModeElementType> texRef_R;

#include "parasite.h"
#include "preProcessing.cu"
#include "postProcessing.cu"

//#include <conio.h>;
//#include <iostream>; // used for C++

////////////////////////////////////////////////////////////////////////////////
/* Custom HPC Counter                                                         */
////////////////////////////////////////////////////////////////////////////////

//#define TEST_WRITE_TIME

//#define BENCHMARKING
//#define ELAPSED
//#define NUM_OF_PASSES 40
//#define DEBUG_OUTPUT

extern "C" {
   #include "hr_time.h"
}

extern "C" {
   #include "cputime.h"
}

////////////////////////////////////////////////////////////////////////////////
/* End Custom HPC Counter                                                     */
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/* Timeb headers                                                              */
////////////////////////////////////////////////////////////////////////////////
//#include <sys/timeb.h>
//#include <time.h>  // standard time functions

//// to get portable code we need this
//#ifdef WIN32 // _WIN32 is defined by all Windows 32 compilers, but not by others.
//#define timeb _timeb
//#define ftime _ftime
//#endif
////////////////////////////////////////////////////////////////////////////////
/* End timeb headers                                                          */
////////////////////////////////////////////////////////////////////////////////

// Global Variables
#ifdef CAL_SIZE
	short* parasiteSize;
#endif
double* parasiteCompactness;
char* imagePath;

int main(int argc, unsigned char* argv[])
//int main(int argc, unsigned char** argv)
{
	//short* histogram;

	// Create a int to store the total number of fluke images.
	int numberOfInputImages;

	// get the index of the image we want and make a unsigned char array of it then concat it to imageFile.
	// Set the size of the variable 'imageFileName' to 255 characters
	imagePath = (char*)malloc(sizeof(char)*255);
	memset(imagePath, 0, (sizeof(char)*255));
	
	// get the index of the image we want and make a unsigned char array of it then concat it to imageFile.
	// Set the size of the variable 'imageFileName' to 255 characters
	char* imageFileName = (char*)malloc(sizeof(char)*255);
	memset(imageFileName, 0, sizeof(char)*255);

	char imageIndexString[10];
	// Construct the file name
	int imageIndex = 1;
	itoa(imageIndex,imageIndexString,10);

	//strcpy(imageFileName, "images/colour");
	//strcat(imageFileName, imageIndexString);
	//strcat(imageFileName, ".bmp");

	char* videoName = (char*)malloc(sizeof(char)*30);
	if (argc != 3) /* argc should be 3 for correct execution */
	{
	    printf("No folder or video name specified.\n");
		getchar();
		return 1;
	}
	else
	{
		//printf("path: %s\n", argv[1]);
		// copy the first argument to the 'imagePath' variable.
		strcpy(imagePath, (char*) argv[1]);
		#ifndef RELEASE_BUILD
			strcat(imagePath, "\images");
		#endif
		// copy the second argument to the 'videoName' variable.
		strcpy(videoName, (char*) argv[2]);
		//massterImageFileName = (char*) argv[1];
	}
	//printf("--------\n");
	//printf("imagePath: %s\n", imagePath);
	
	// Set the integer 'numberOfInputImages'.
	numberOfInputImages = getNumberOfImages((char*) imagePath);
	//numberOfInputImages = 10;
	//numberOfInputImages = 10;


    printf("////////////////////////////////////////\n");
	printf("/       CUDA PARASITE ANALYSIS         /\n");
	printf("////////////////////////////////////////\n");
	printf("Starting program...\n\n");
	printf("Image Path: %s\n", imagePath);
	printf("#Input Images: %d\n", numberOfInputImages);
	printf("___________________\n\n");


	sprintf(imageFileName,"%s/colour%d.bmp", imagePath, imageIndex);
	//printf("%s/colour%d.bmp", imagePath, imageIndex);

	stopWatch executionTimer;
	#ifdef BENCHMARKING
		#ifdef ELAPSED
			stopWatch s;
		#endif
	#endif

	// Set up all the necessary variables
	// Create a character image array for stoing the inout fluke image.
	unsigned char* imageArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
	// unsigned char *imageArray2 = malloc(sizeof(*imageArray2) * MAX_DATA_SIZE);  // You can do this instead if you use a C compiler - better practice.
	
	// Define pointer to the CUDA device array (d_imageArray) and assign it to the global device memory.
	unsigned char* d_imageArray;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc((void**)&d_imageArray, sizeof(unsigned char) * MAX_DATA_SIZE) );

	unsigned char* r_imageArray;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&r_imageArray, sizeof(unsigned char) * MAX_DATA_SIZE) );

	// Create a variable img of type t_bmp (defined in imageio) to store the input image and initialise its size to the size of a t_bmp structure.
	t_bmp* img = (t_bmp*)malloc(sizeof(t_bmp));

	#ifdef CAL_SIZE
		parasiteSize = (short *)malloc(sizeof(short) * numberOfInputImages);
	#endif
	parasiteCompactness = (double *)malloc(sizeof(double) * numberOfInputImages);

	//printf("Path: %s\nNo %d", imageFileName, numberOfInputImages);

	#ifdef BENCHMARKING
		#ifdef ELAPSED
			FILE *benchMarkFileElapsed;
			benchMarkFileElapsed = fopen("benchMarkFileElapsed.txt", "w");
			fprintf(benchMarkFileElapsed, "------------Benchmarking Algorithms (Elapsed)------------\n");
			fprintf(benchMarkFileElapsed, "Average elapsed times in ms.\n");
			fprintf(benchMarkFileElapsed, "Number of Passes:%d\n\n", NUM_OF_PASSES);
			double aggregatedElapsedTime = 0;
		#else
			FILE *benchMarkFileCPU;
			benchMarkFileCPU = fopen("benchMarkFileCPU.txt", "w");
			fprintf(benchMarkFileCPU, "------------Benchmarking Algorithms (CPU)------------\n");
			fprintf(benchMarkFileCPU, "Average CPU times in ms.\n");
			fprintf(benchMarkFileCPU, "Number of Passes:%d\n\n", NUM_OF_PASSES);
			double aggregatedCPUTime = 0;
		#endif
	#endif

	#ifdef RUNS
		FILE *runOutput;
		runOutput = fopen("Run_Results.txt", "w");
		fprintf(runOutput, "Pass\tTime\n");

		double totalRuntime = 0;
	for(int k=0; k<NUM_RUNS; k++)
	{
	#endif

		if(initialiseTimer(&executionTimer)==0)
			startTimer(&executionTimer);
		else
		{
			printf("Timer failed to initialise.\n");
			getchar();
			exit(EXIT_FAILURE);
		}
		// While there are no more images to load.
		while((imageIndex<=numberOfInputImages) && (libbmp_load(imageFileName, img) != 1))
		{
			#ifdef TITLES
			printf("------ Processing Frame ------\n");
			#endif

			//printf("imageFileName: %s\n", imageFileName);

			#ifdef BENCHMARKING
				printf("Number Of Passes: %d\n\n", NUM_OF_PASSES);
			#endif

			// Convert the loaded input image into a 1 dimensional array inorder to pass the array to the CUDA
			// device
 			imageToArray(img, imageArray);

			// Create a histgram for the initial input image
			#ifdef DEBUG_OUTPUT
				outputCSV(imageArray, "initialImage");
			#endif

			//float *outPixelValue = (float*)malloc(sizeof(float));
			//float *d_outPixelValue;
			//CUSTOM_CUDA_SAFE_CALL(cudaMalloc((void**)&d_outPixelValue, (sizeof(float))) );
			//test<<<1, 1>>>(217, 1.268657, -20.298508, d_outPixelValue);
			//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(outPixelValue, d_outPixelValue, sizeof(float), cudaMemcpyDeviceToHost) );
			//printf("\n");
			//printf("Test Value: %f\n", *outPixelValue);
			//printf("Test Value: %f\n", (unsigned char) *outPixelValue);
			//printf("\n");

			/* Test to see where image is data is written: ANS - bottom left corner to top right

				//for (int i = 0; i < MAX_DATA_SIZE; i++)
				//	imageArray[i] = 0;

				//imageArray[0] = 1;
				//imageArray[1] = 1;
				//imageArray[2] = 1;
				//imageArray[3] = 1;
				//imageArray[4] = 1;

				//imageArray[307200-1] = 3;
				//imageArray[307200-2] = 3;
				//imageArray[307200-3] = 3;
				//imageArray[307200-4] = 3;
				//imageArray[307200-5] = 3;

				//createBinaryImage(img, imageArray);
				//libbmp_write("imageTest.bmp", img);
			*/
			
			/*
				// Rewrite the array back to an img to test it is identical
				// imageArray[0] = bottom left pixel - after modify imageio methods.
				for (int i=0; i<10; i++)
					imageArray[i]=0;

				createImageFomArray(img, imageArray);
				libbmp_write("imageCUDA.bmp", img);
			*/

			/* 
				// Check that both arrays are identical - host and device

				outputArray(imageArray, "a");
				unsigned char* tempImageArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
				cudaMemcpy(tempImageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost);
				outputArray(tempImageArray, "b");
			*/

			// Moy is black and white intensity
			// Moy of 0 is black
			// Moy of 255 is white
			// height goes from bottom up not top down!!!!
			// initialize all necessary variables


			// Create a histogram. Useful in sequential processing for lookups.
			//////////////////////////////////////////////////////////////////////////////////
			///* Create Histogram                                                           */
			//////////////////////////////////////////////////////////////////////////////////

			/*
				//short *h_lut = (short*) malloc(sizeof(short) * 256);

				//short* d_lut;
				//cudaMalloc((void**)&d_lut, (sizeof(short)) * 256);
				//cudaMemset(d_lut, 0, (sizeof(short)) * 256);

				// Used to create a histogram using the atomicAdd function - requires type int (even though the original implementation uses type short.
				// Histogram lut may not be needed for this CUDA application as each thread will only calculate its output value once and continue -- REQUIRES EXPERIMENTATION!!!
				int *h_lut = (int*) malloc(sizeof(int) * 256);

				int* d_lut;
				cudaMalloc((void**)&d_lut, (sizeof(int)) * 256);
				cudaMemset(d_lut, 0, (sizeof(int)) * 256);

				createHistogramCUDA<<<gridDim, blockDim>>>(d_imageArray, d_lut);
				cudaMemcpy(h_lut, d_lut, (sizeof(int) * 256), cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_lut, d_lut, (sizeof(short) * 256), cudaMemcpyDeviceToHost);

				// create a histogram using the standard sequential method.
				histogram = createHistogram(img);

				// print the values for comparsion.
				for(int i = 0; i < 256; i++)
					//printf("Moy: %d : %d\n", i, h_lut[i]);
					printf("%d \t %d\n", histogram[i], h_lut[i]);

				// Output histogram of the gray levels
				FILE *lutHistogram;
				lutHistogram = fopen("lutHistogramCUDA.csv", "w");

				for (int i = 0; i < 256; i++)
				{
					fprintf(lutHistogram, "%d, %d\n", i, h_lut[i]);
					printf("Moy: %d : %d\n", i, h_lut[i]);
				}

				fclose(lutHistogram);
				// Output histogram of the gray levels - END
			*/

			//histogram = createHistogram(img);
			////libbmp_write("image.bmp", img);
			//libbmp_write("image.bmp", img);
			//////////////////////////////////////////////////////////////////////////////////
			///* End Create Histogram                                                       */
			//////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Enhance Contrast                                                           */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Enhance Contrast\n");
			#endif

			// SETUP ENVIRONMENT

			/* This code is now depreciated! See below

				//unsigned char* h_I1_I2 = (unsigned char*) malloc(sizeof(unsigned char) * 2);
				//h_I1_I2[0] = 255;	// Set the lower bound I1 (smallest pixel value to max unsigned char) to 255
				//h_I1_I2[1] = 0;		// Set the lower bound I2 (largest pixel value to min unsigned char) to 0
				//unsigned char* d_I1_I2;
				//CUSTOM_CUDA_SAFE_CALL(cudaMalloc((void**)&d_I1_I2, (sizeof(unsigned char) * 2)) )
				//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_I1_I2, h_I1_I2, (sizeof(unsigned char) * 2), cudaMemcpyHostToDevice));
			*/
			
			/* This code is now depreciated!
			   Do not need to process the minimum and maximum arrays on the host any more.

				//unsigned char* h_resultMin;
				//unsigned char* h_resultMax;
				//h_resultMin = (unsigned char *)malloc(sizeof(unsigned char) * minMemoryRequired);
				//h_resultMax = (unsigned char *)malloc(sizeof(unsigned char) * minMemoryRequired);
			*/

			int minMemoryRequired = MAX_DATA_SIZE / THREADS_PER_BLOCK / 2; /* the current calculateI1AndI2 GPU algorithm only requires half the amount of blocks other reduction algorithms
																			   need.
																			   MAX_DATA_SIZE = 640*480 = 307200
																			   THREADS_PER_BLOCK = 32
																			   
																			   therefore we need
																			   307200/32 = 9600 blocks

																			   but we only execute half of these = 4800 blocks
																			*/
			//printf("minBlocksRequired: %d\n", minMemoryRequired);
			unsigned char* d_resultMin;
			unsigned char* d_resultMax;
    		CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultMin, sizeof(unsigned char) * minMemoryRequired) );
			CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultMax, sizeof(unsigned char) * minMemoryRequired) );


			int minBlocksRequiredAtStage2 = (int) ceil(((float)minMemoryRequired / THREADS_PER_BLOCK));
			//printf("minBlocksRequiredAtStage2: %d\n", minBlocksRequiredAtStage2);
			unsigned char* d_resultMinS2;
			unsigned char* d_resultMaxS2;
    		CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultMinS2, sizeof(unsigned char) * minBlocksRequiredAtStage2) );
			CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_resultMaxS2, sizeof(unsigned char) * minBlocksRequiredAtStage2) );

			int sharedMemSize = 2 * minBlocksRequiredAtStage2 * sizeof(unsigned char);
			//printf("minSharedMemoryRequired * sizeof(char): %d\n", sharedMemSize);

			/* This code is now depreciated!

				//unsigned char* d_I1;
				//unsigned char* d_I2;
				//CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_I1, sizeof(unsigned char)) );
				//CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_I2, sizeof(unsigned char)) );
			*/

			float* d_M_C;
			CUSTOM_CUDA_SAFE_CALL(cudaMalloc((void**)&d_M_C, (sizeof(float) * 2)))

			// END - SETUP ENVIRONMENT

			// Copy the initial input image to the device.
			CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

			cudaBindTexture(NULL, texRef, d_imageArray, MAX_DATA_SIZE);
			cudaBindTexture(NULL, minTex, d_resultMin, sizeof(unsigned char) * minMemoryRequired);
			cudaBindTexture(NULL, maxTex, d_resultMax, sizeof(unsigned char) * minMemoryRequired);

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES+1; i++) { // plus one to ignore warm up time

					// This memory copy is only needed when benchmarking because by default the device image array will contain the correct data from the previous operation.
					CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif
							
							//printf("minMemoryRequired: %d", minMemoryRequired);
							
							//calculateI1AndI2<<<BLOCKS_PER_GRID_ROW/2, THREADS_PER_BLOCK>>>(d_imageArray, d_resultMin, d_resultMax);
							calculateI1AndI2Texture<<<BLOCKS_PER_GRID_ROW/2, THREADS_PER_BLOCK>>>(d_resultMin, d_resultMax);

							//printf("calculateI1AndI2 Kernel Return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMin, d_resultMin, sizeof(unsigned char) * minMemoryRequired, cudaMemcpyDeviceToHost) );
							//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMax, d_resultMax, sizeof(unsigned char) * minMemoryRequired, cudaMemcpyDeviceToHost) );						

							//calculateI1AndI2Stage2<<<minBlocksRequiredAtStage2, THREADS_PER_BLOCK>>>(d_resultMin, d_resultMax, d_resultMinS2, d_resultMaxS2);
							calculateI1AndI2Stage2Texture<<<minBlocksRequiredAtStage2, THREADS_PER_BLOCK>>>(d_resultMinS2, d_resultMaxS2);

							
							//printf("calculateI1AndI2Stage2 Kernel Return: %s\n", cudaGetErrorString( cudaGetLastError() ));

							/* Check results

								//unsigned char* h_resultMinS2 = (unsigned char*) malloc(sizeof(unsigned char) * minBlocksRequiredAtStage2);
								//unsigned char* h_resultMaxS2 = (unsigned char*) malloc(sizeof(unsigned char) * minBlocksRequiredAtStage2);
								//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMinS2, d_resultMinS2, sizeof(unsigned char) * minBlocksRequiredAtStage2, cudaMemcpyDeviceToHost) );
								//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMaxS2, d_resultMaxS2, sizeof(unsigned char) * minBlocksRequiredAtStage2, cudaMemcpyDeviceToHost) );
								//printf("h_resultMinS2[1], [2], [3]: %d, %d, %d\n", h_resultMinS2[0], h_resultMinS2[1], h_resultMinS2[2]);
								//printf("h_resultMaxS2[1], [2], [3]: %d, %d, %d\n", h_resultMaxS2[0], h_resultMaxS2[1], h_resultMaxS2[2]);
							*/
							

							//calculateI1AndI2Final<<<1, 1, (3*sizeof(int))>>>(d_resultMinS2, d_resultMaxS2, d_M_C); // will only work for THREADS_PER_BLOCK = 256 - only 3 blocks remaining to process.
							//printf("calculateI1AndI2Final Kernel Return: %s\n", cudaGetErrorString( cudaGetLastError() ));

							// Dynamically calculate the amount of shared memory needed for the next stage and call the kernel
							//<<< dimGrid, dimBlock, sharedMemSize >>>
							calculateI1AndI2Final<<<1, 1, sharedMemSize>>>(d_resultMinS2, d_resultMaxS2, minBlocksRequiredAtStage2, d_M_C); // At this the number of elements left to process
																																			// should be mininal - one block and one thread
																																			// should be sufficient to find I1 and I2 - 
																																			// if there are more than 512 elements to process this
																																			// will not work as you can only have 512 threads per
																																			// block.

							//calculateI1AndI2Final<<<1, 1, sharedMemSize>>>(d_resultMinS2, d_resultMaxS2, minBlocksRequiredAtStage2, d_M_C); // will only work for THREADS_PER_BLOCK = 256 - only 3 blocks remaining to process.
							//printf("calculateI1AndI2Final Kernel Return: %s\n", cudaGetErrorString( cudaGetLastError() ));

							//// Host gradient and intercept
							//float* h_M_C = (float*) malloc(sizeof(float)*2);
							//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_M_C, d_M_C, (sizeof(float) * 2), cudaMemcpyDeviceToHost) );

							//printf("h_M_C[0]: %f\n", h_M_C[0]);
							//printf("h_M_C[1]: %f\n", h_M_C[1]);

	//----------------------------------------------------------------------------
						
							/* This code is now depreciated!
							   
							   This code was used to copy the intermediate results of the initial calculateI1AndI2 back to the host.  However this memory transfer requires a lot of 
							   time, therefore i have decided to keep the entire process of calculating I1, I2, the gradient and intercept on the device.

								//// Copy the data back to the host
								//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMin, d_resultMin, sizeof(unsigned char) * minMemoryRequired, cudaMemcpyDeviceToHost) );
								//CUSTOM_CUDA_SAFE_CALL( cudaMemcpy(h_resultMax, d_resultMax, sizeof(unsigned char) * minMemoryRequired, cudaMemcpyDeviceToHost) );

								//// Each block returned one result, so lets finish this off with the cpu.
								//// By using CUDA, we basically reduced how much the CPU would have to work by about 256 times.
								//unsigned char I1, I2;
								//I1 = 255;
								//I2 = 0;
								//for (int i=0 ; i < minMemoryRequired; i++)
								//{
								//	if (h_resultMin[i] < I1) I1 = h_resultMin[i];
								//	if (h_resultMax[i] > I2) I2 = h_resultMax[i];
								//}
								////printf("\n\n\n");
								//printf("GPU --> Min: %d -- Max %d\n", I1, I2);

								//// Calculate the gradient and intercept
								//h_M_C[0] = 255 / ((float) I2 - (float)I1);
								//h_M_C[1] = -h_M_C[0] * (float) I1;
								//
								//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_M_C, h_M_C, (sizeof(float) * 2), cudaMemcpyHostToDevice));

								////calEnhancementCUDA<<<gridDim, blockDim>>>(d_imageArray, d_I1_I2, d_M_C);
								////CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_M_C, d_M_C, (sizeof(float) * 2), cudaMemcpyDeviceToHost));
								////CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_I1_I2, d_I1_I2, (sizeof(unsigned char) * 2), cudaMemcpyDeviceToHost));

								////printf("CUDA I1        : %d\n", h_I1_I2[0]); // 16
								////printf("CUDA I2        : %d\n", h_I1_I2[1]); // 217
								////printf("CUDA Gradient  : %f\n", h_M_C[0]); // 1.27
								////printf("CUDA Intercept : %f\n", h_M_C[1]); // -20.3
							*/

	//----------------------------------------------------------------------------

							
							enhanceContrastCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, d_M_C);
							cudaThreadSynchronize();
					
			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						if (i!=0 && NUM_OF_PASSES>1) // ignore warm up time
						{
							printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
							aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						}
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Enhance constrast timer didnt initialise\n"); }
				#endif

				// Print results to file and screen and clean up
				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Enhance Contrast       : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Enhance Contrast       : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif
			#endif

			CUSTOM_CUDA_SAFE_CALL(cudaFree(d_resultMin));
			CUSTOM_CUDA_SAFE_CALL(cudaFree(d_resultMax));

			// unbind textures
			cudaUnbindTexture(minTex);
			cudaUnbindTexture(maxTex);

			CUSTOM_CUDA_SAFE_CALL(cudaFree(d_resultMinS2));
			CUSTOM_CUDA_SAFE_CALL(cudaFree(d_resultMaxS2));
			CUSTOM_CUDA_SAFE_CALL(cudaFree(d_M_C));

			#ifdef DEBUG_OUTPUT
				// Copy data back of the device to write to an image.
				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));

				//unsigned char* test = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
				//histogram = createHistogram(img);
				//enhanceContrast(img, histogram);
				//imageToArray(img, test);

				//imageArray[224957] = 255;			// If the project is set to debug (i.e. run on GPU) imageArray[224957] & imageArray[235826] gets set to 254 instead of 255
				//imageArray[235826] = 255;

				//outputArrays(test, "a", imageArray, "b");

				outputCSV(imageArray, "enhanceContrast");
				createImageFomArray(img, imageArray);
				libbmp_write("imageStage1CUDA.bmp", img);
			#endif

			////////////////////////////////////////////////////////////////////////////////
			/* End Enhance Contrast                                                       */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Filtering                                                                  */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Filtering\n");
			#endif

			//unsigned char* r_imageArray;
			//CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&r_imageArray, sizeof(unsigned char) * MAX_DATA_SIZE) );
			//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(r_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));	// for testing outside pixels

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES; i++) {

						CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif

							lowPassFilterCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
							//lowPassFilterCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
							cudaThreadSynchronize();

			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
						aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Filtering timer didnt initialise\n"); }
				#endif

				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Filtering              : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Filtering              : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif

			#endif

			#ifdef DEBUG_OUTPUT

				//lowPassFilterImage(img);
				//medianFilterImage(img);
				//libbmp_write("imageMedianFilter.bmp", img);
				//libbmp_write("imageStage2.bmp", img);

				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, r_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));
				//CUSTOM_CUDA_SAFE_CALL(cudaFree(r_imageArray));  // shouldn't be technically freed - but becuase of benchmarking the array is copied to d_imageArray so there is no need for
																// r_imageArray


				//imageToArray(img, imageArray);

				//for (int i = 1; i < (img->height - 1); i++)
				//{
				//	for (int j = 1; j < (img->width - 1); j++)
				//	{
				//		img->data[i][j].moy = 0;
				//		img->data[i][j].r = 0;
				//		img->data[i][j].g = 0;
				//		img->data[i][j].b = 0;
				//		libbmp_write("imageStage2ii.bmp", img);
				//	}
				//}


				/* Updated the low pass filter sequential to take into account the border pixels.
				   Also removed a major bug where the convolution operation output the results of each pixel calculation to the same image instead of a new image.

					t_bmp* outputImg = (t_bmp*)malloc(sizeof(t_bmp));
					libbmp_copyAndCreateImg(img, outputImg);		// initialise outputImage by copying
					lowPassFilterImage(img, outputImg);

					// Switch pointers so img points to outputImg (the result of the low pass filter operation) and outputImg points to the old data (img), which is then removed.
					t_bmp* temp;
					//printf("img: %x OutputImage: %x temp: %xd\n\n", img, outputImage, temp);
					temp = outputImg;
					//printf("temp = outputImage\n");
					//printf("img: %x OutputImage: %x temp: %xd\n", img, outputImage, temp);
					outputImg = img;
					//printf("outputImage = img\n");
					//printf("img: %x OutputImage: %x temp: %xd\n", img, outputImage, temp);
					img = temp;
					//printf("img = temp\n");
					//printf("img: %x OutputImage: %x temp: %xd\n", img, outputImg, temp);
					for(int i = 0; i < outputImg->height; i++)
						free(outputImg->data[i]);

					free(outputImg->data);
					free(outputImg);
					//printf("free(outputImage); img = temp\n");
					//printf("img: %x OutputImage: %x temp: %xd\n", img, outputImage, temp);

					//unsigned char* testArray = (unsigned char *) malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
					//imageToArray(img, testArray);
					//libbmp_write("imageStage2i.bmp", img);
					//outputArray(testArray, "c");
					//
					//createImageFomArray(img, imageArray);
					//outputArray(imageArray, "d");
				*/

				outputCSV(imageArray, "filter");
				createImageFomArray(img, imageArray);
				libbmp_write("imageStage2CUDA.bmp", img);
			#endif
			
			////////////////////////////////////////////////////////////////////////////////
			/* End Filtering                                                              */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Segment Image                                                              */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Segment Image\n");
			#endif

			CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, r_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToDevice));

			unsigned char threshold;

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES; i++) {

						CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif

						//imageToArray(img, imageArray);
						//threshold = iterativeCalculateThresholdUsingArray(imageArray);
						//printf("Threshold value Sequential = %d\n", threshold);

						//threshold = iterativeCalculateThresholdCUDA(d_imageArray);
						threshold = iterativeCalculateThresholdCUDATexture();
						//printf("Threshold value CUDA = %d\n", threshold);

						//segmentImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, threshold);
						//segmentImageCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, threshold);
						segmentImageCoalescedCUDA<<<BLOCKS_PER_GRID_ROW/4, THREADS_PER_BLOCK>>>((uchar4*) d_imageArray, threshold);

						//segmentImageCoalescedCUDATest<<<BLOCKS_PER_GRID_ROW/4, THREADS_PER_BLOCK>>>(d_imageArray, threshold);
						cudaThreadSynchronize();

			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
						aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Segment timer didnt initialise\n"); }
				#endif

				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Segment Image          : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Segment Image          : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));
				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif
			#endif

			
			//segmentImage(img);
			//libbmp_write("imageSegmented.bmp", img);

			#ifdef DEBUG_OUTPUT
				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));

				createBinaryImage(img, imageArray);
				libbmp_write("imageStage3CUDA.bmp", img);
			#endif

			////////////////////////////////////////////////////////////////////////////////
			/* End Segment Image                                                          */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			#ifdef DEBUG_OUTPUT
				//Convert image to a 1D array
				binaryImageToArray(img, imageArray);
				createBinaryImage(img, imageArray);
				//libbmp_write("imageWith1DArray.bmp", img);
				libbmp_write("imageStage4CUDA.bmp", img);
			#endif

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Dilate Image three times                                                   */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Dilate Image\n");
			#endif

			cudaBindTexture(NULL, texRef_R, r_imageArray, MAX_DATA_SIZE);

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES; i++) {

						CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));
						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif

							//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
							////CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));
							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
							//	//printf("dilateImageCUDA 1 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
							//	//printf("dilateImageCUDA 2 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
							//	//printf("dilateImageCUDA 3 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
							//	//printf("dilateImageCUDA 4 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
							//	//printf("dilateImageCUDA 4 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
							//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

							//dilateImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
							//	//printf("dilateImageCUDA 5 return: %s\n", cudaGetErrorString( cudaGetLastError() ));


							CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
							dilateImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
							dilateImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);
							dilateImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
							dilateImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);
							dilateImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
							dilateImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);

							
							cudaThreadSynchronize();

							//dilateImage(img->height, img->width, imageArray);
							//dilateImage(img->height, img->width, imageArray);
							//dilateImage(img->height, img->width, imageArray);
							//dilateImage(img->height, img->width, imageArray);
							//dilateImage(img->height, img->width, imageArray);
							//dilateImage(img->height, img->width, imageArray);

			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
						aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Dilate timer didnt initialise\n"); }
				#endif
				
				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Dilate Image           : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Dilate Image           : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));
				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif
			#endif

			

			#ifdef DEBUG_OUTPUT
				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));

				createBinaryImage(img, imageArray);
				//libbmp_write("imageDilated.bmp", img);
				libbmp_write("imageStage5CUDA.bmp", img);
			#endif

			////////////////////////////////////////////////////////////////////////////////
			/* End Dilate Image three times                                               */
			////////////////////////////////////////////////////////////////////////////////

			// The following code was a template for creating a method to fill in the black gaps
			// in the worm segmentation.

			//int *fillDetails = findLargestArea(img->height, img->width, imageArray);
			////fill(imageArray, fillDetails[1], fillDetails[2]);

			//createBinaryImage(img, imageArray);
			//libbmp_write("imageStage5i.bmp", img);

			//free(fillDetails);

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Erode Image                                                                */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Erode Image\n");
			#endif

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES; i++) {

						CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));
						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif

						////CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));  // Not required if only passing once.
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
						//	//printf("erodeImageCUDA 1 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
						//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

						//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
						//	//printf("erodeImageCUDA 2 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
						//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

						//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
						//	//printf("erodeImageCUDA 3 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
						//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

						//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
						//	//printf("erodeImageCUDA 4 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
						//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

						//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray);
						//	//printf("erodeImageCUDA 5 return: %s\n", cudaGetErrorString( cudaGetLastError() ));
						//	//CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );

						//CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						//erodeImageCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray);
						//	//printf("erodeImageCUDA 6 return: %s\n", cudaGetErrorString( cudaGetLastError() ));



						CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
						CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);
						CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
						CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);
						CUSTOM_CUDA_SAFE_CALL( cudaMemset(r_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_D<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray);
						CUSTOM_CUDA_SAFE_CALL( cudaMemset(d_imageArray, 0, sizeof(unsigned char) * MAX_DATA_SIZE) );
						erodeImageCUDA_R<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray);



						cudaThreadSynchronize();

						//erodeImage(img->height, img->width, imageArray);
						//erodeImage(img->height, img->width, imageArray);
						//erodeImage(img->height, img->width, imageArray);
						//erodeImage(img->height, img->width, imageArray);
						//erodeImage(img->height, img->width, imageArray);
						//erodeImage(img->height, img->width, imageArray);

			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
						aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Erode timer didnt initialise\n"); }
				#endif

				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Erode Image            : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Erode Image            : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif
			#endif

			//erodeImage(img->height, img->width, imageArray);
			//erodeImage(img->height, img->width, imageArray);
			//erodeImage(img->height, img->width, imageArray);
			//erodeImage(img->height, img->width, imageArray);
			//erodeImage(img->height, img->width, imageArray);
			//erodeImage(img->height, img->width, imageArray);

			#ifdef DEBUG_OUTPUT
				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));
				
				createBinaryImage(img, imageArray);
				//libbmp_write("imageEroded.bmp", img);
				libbmp_write("imageStage6CUDA.bmp", img);
			#endif
			////////////////////////////////////////////////////////////////////////////////
			/* Erode Image                                                                */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Remove Noise                                                               */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef TITLES
			printf("-------------------Remove Noise\n");
			#endif

			int maskSize = 3;

			#ifdef BENCHMARKING
				#ifdef ELAPSED
				if(initialiseTimer(&s)==0) {
				#endif
					for (int i = 0; i < NUM_OF_PASSES; i++) {
						CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

						#ifdef ELAPSED
						startTimer(&s);
						#else
						initCPUTime();
						#endif
			#endif

						//removeUnwantedNoiseHeightCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray, maskSize);
						//removeUnwantedNoiseWidthCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray, maskSize);
						//removeUnwantedNoiseHeightCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, r_imageArray, maskSize);
						//removeUnwantedNoiseWidthCUDA<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, d_imageArray, maskSize);

						removeUnwantedNoiseHeightCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, maskSize);
						removeUnwantedNoiseWidthCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, maskSize);
						removeUnwantedNoiseHeightCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(r_imageArray, maskSize);
						removeUnwantedNoiseWidthCUDATexture<<<BLOCKS_PER_GRID_ROW, THREADS_PER_BLOCK>>>(d_imageArray, maskSize);

						cudaThreadSynchronize();

						//removeUnwantedNoiseHeight(img->height, img->width, imageArray, maskSize);
						//removeUnwantedNoiseWidth(img->height, img->width, imageArray, maskSize);
						//removeUnwantedNoiseHeight(img->height, img->width, imageArray, maskSize);
						//removeUnwantedNoiseWidth(img->height, img->width, imageArray, maskSize);

			#ifdef BENCHMARKING
						#ifdef ELAPSED
						stopTimer(&s);
						#else
						double endTime = getCPUTimeSinceStart();
						#endif

						#ifdef ELAPSED
						printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
						aggregatedElapsedTime += getElapsedTimeInMilli(&s);
						#else
						printf("CPU Time: %f(ms)\n", endTime);
						aggregatedCPUTime += endTime;
						#endif
					}
				#ifdef ELAPSED
				} else { printf("Unwanted noise timer didnt initialise\n"); }
				#endif

				#ifdef ELAPSED
				fprintf(benchMarkFileElapsed, "Remove Noise           : ");
				fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

				printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
				aggregatedElapsedTime = 0.0;
				#else
				fprintf(benchMarkFileCPU, "Remove Noise           : ");
				fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

				printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
				aggregatedCPUTime = 0.0;
				#endif
			#endif

			//removeUnwantedNoiseHeight(img->height, img->width, imageArray, maskSize);
			//removeUnwantedNoiseWidth(img->height, img->width, imageArray, maskSize);
			//removeUnwantedNoiseHeight(img->height, img->width, imageArray, maskSize);
			//removeUnwantedNoiseWidth(img->height, img->width, imageArray, maskSize);

			#ifdef DEBUG_OUTPUT
				CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));
				
				createBinaryImage(img, imageArray);
				//libbmp_write("imageNoiseRemoval.bmp", img);
				libbmp_write("imageStage7CUDA.bmp", img);
			#endif
			////////////////////////////////////////////////////////////////////////////////
			/* End Remove Noise                                                           */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Find Centre Points                                                         */
			////////////////////////////////////////////////////////////////////////////////

			//getLineSizeWidth(img->height, img->width, imageArray);
			//getLineSizeHeight(img->height, img->width, imageArray);
			//createBinaryImage(img, imageArray);
			//libbmp_write("imageCentrePoints.bmp", img);

			////////////////////////////////////////////////////////////////////////////////
			/* End Find Centre Points                                                     */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Calculate Area                                                             */
			////////////////////////////////////////////////////////////////////////////////

			// Need to copy the results to the host to carry out the final steps.
			CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));

			#ifdef COMPACTNESS
				//libbmp_load("imageStage7CUDAi.bmp", img);
				//binaryImageToArray(img, imageArray);
				////imageToArray(img, imageArray);
				//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(d_imageArray, imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyHostToDevice));

				#ifdef TITLES
				printf("-------------------Find Pixels\n");
				#endif

				int *details = (int*) malloc(sizeof(int) * 3);

				#ifdef BENCHMARKING
					//t_bmp* initialImg;
					unsigned char* initialImgArray;

					//libbmp_load("images/colourprocessed1CircleTest.bmp", img);
					//binaryImageToArray(img, imageArray);

					initialImgArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
					copyArray(imageArray, initialImgArray);


					#ifdef ELAPSED
					if(initialiseTimer(&s)==0) {
					#endif
						for (int i = 0; i < NUM_OF_PASSES; i++) {
							copyArray(initialImgArray, imageArray);
				
							#ifdef ELAPSED
							startTimer(&s);
							#else
							initCPUTime();
							#endif
				#endif

							free(details); // details is allocated in findPixels
							details = findPixels(img->height, img->width, imageArray);

				#ifdef BENCHMARKING
							#ifdef ELAPSED
							stopTimer(&s);
							#else
							double endTime = getCPUTimeSinceStart();
							#endif

							#ifdef ELAPSED
							printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
							aggregatedElapsedTime += getElapsedTimeInMilli(&s);
							#else
							printf("CPU Time: %f(ms)\n", endTime);
							aggregatedCPUTime += endTime;
							#endif
						}
					#ifdef ELAPSED
					} else { printf("Find pixels timer didnt initialise\n"); }
					#endif

					#ifdef ELAPSED
					fprintf(benchMarkFileElapsed, "Find Pixels            : ");
					fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

					printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
					aggregatedElapsedTime = 0.0;
					#else
					fprintf(benchMarkFileCPU, "Find Pixels            : ");
					fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

					printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
					aggregatedCPUTime = 0.0;
					#endif

					free(initialImgArray);
				#endif

				//details = findPixels(img->height, img->width, imageArray);
				//printf("details[0] %d : details[1] %d : details[2] %d\n", details[0], details[1], details[2]); // largestArea, blobCoordI, blobCoordJ

				#ifdef TITLES
				printf("-------------------Find and Replace Pixels\n");
				#endif

				#ifdef BENCHMARKING
					initialImgArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
					copyArray(imageArray, initialImgArray);

					#ifdef ELAPSED
					if(initialiseTimer(&s)==0) {
					#endif
						for (int i = 0; i < NUM_OF_PASSES; i++) {
							copyArray(initialImgArray, imageArray);

							#ifdef ELAPSED
							startTimer(&s);
							#else
							initCPUTime();
							#endif
				#endif
							
							findAndReplace(imageArray, details[1], details[2], 4, 1);

				#ifdef BENCHMARKING
							#ifdef ELAPSED
							stopTimer(&s);
							#else
							double endTime = getCPUTimeSinceStart();
							#endif

							#ifdef ELAPSED
							printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
							aggregatedElapsedTime += getElapsedTimeInMilli(&s);

							//fprintf(benchMarkFileElapsed, "%f\n", getElapsedTimeInMilli(&s));
							#else
							printf("CPU Time: %f(ms)\n", endTime);
							aggregatedCPUTime += endTime;
							#endif
						}
					#ifdef ELAPSED
					} else { printf("Find and replace timer didnt initialise\n"); }
					#endif

					//outputArrays(imageArray, "e", initialImgArray, "f");

					#ifdef ELAPSED
					fprintf(benchMarkFileElapsed, "Find and Replace Pixels: ");
					fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

					printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
					aggregatedElapsedTime = 0.0;
					#else
					fprintf(benchMarkFileCPU, "Find and Replace Pixels: ");
					fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

					printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
					aggregatedCPUTime = 0.0;
					#endif


					free(initialImgArray);
					//checkPixels(imageArray, details[1], details[2], 4, 1);
					free(details);
				#endif

				double area;
				double perimiter;
				double compactness;

				#ifdef BENCHMARKING
					initialImgArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
					copyArray(imageArray, initialImgArray);

					#ifdef ELAPSED
					if(initialiseTimer(&s)==0) {
					#endif
						for (int i = 0; i < NUM_OF_PASSES; i++) {
							copyArray(initialImgArray, imageArray);

							#ifdef ELAPSED
							startTimer(&s);
							#else
							initCPUTime();
							#endif
				#endif

								area = calculateArea(imageArray);
								//double perimiter = calculatePerimeter("imageNoiseRemoval.bmp", img, imageArray);
								//perimiter = calculatePerimeter(img, imageArray);
								perimiter = calculatePerimeterOptimised(img, imageArray, area);
								compactness = calculateCompactness(perimiter, area);
								//printf("Area: %g \n", area);
								//printf("Perimiter: %g \n", perimiter);
								//printf("Compactness: %g \n", (double) compactness);

				#ifdef BENCHMARKING
							#ifdef ELAPSED
							stopTimer(&s);
							#else
							double endTime = getCPUTimeSinceStart();
							#endif

							#ifdef ELAPSED
							printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
							aggregatedElapsedTime += getElapsedTimeInMilli(&s);

							//fprintf(benchMarkFileElapsed, "%f\n", getElapsedTimeInMilli(&s));
							#else
							printf("CPU Time: %f(ms)\n", endTime);
							aggregatedCPUTime += endTime;
							#endif
						}
					#ifdef ELAPSED
					} else { printf("Compactness timer didnt initialise\n"); }
					#endif

					#ifdef ELAPSED
					fprintf(benchMarkFileElapsed, "Calculate Compactness: ");
					fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

					printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
					aggregatedElapsedTime = 0.0;
					#else
					fprintf(benchMarkFileCPU, "Calculate Compactness: ");
					fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

					printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
					aggregatedCPUTime = 0.0;
					#endif

					free(initialImgArray);
					//checkPixels(imageArray, details[1], details[2], 4, 1);
					

					parasiteCompactness[imageIndex - 1] = compactness;
				#endif
				
				//// Calculate bounding box
				//findBoundingBox(img->height, img->width, imageArray);
				//createBinaryImage(img, imageArray);
				//char* boundingImageFileName = (char*)malloc(sizeof(char)*19);
				//strcpy(boundingImageFileName, "images/imageNoiseBounding");
				//strcat(boundingImageFileName, imageIndexString);
				//strcat(boundingImageFileName, ".bmp");
				//libbmp_write(boundingImageFileName, img);
			#endif

			//fclose(benchMarkFileElapsed);
			//printf("End of test.\n");

			////////////////////////////////////////////////////////////////////////////////
			/* End Calculate Area                                                         */
			////////////////////////////////////////////////////////////////////////////////

			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Find Parasite Size                                                         */
			////////////////////////////////////////////////////////////////////////////////

			//// need 480 threads
			//dim3 widthBlocks(15, 1, 1);
			//dim3 widthBlock_size(32, 1, 1);
			//getLineSizeWidthCUDA<<< widthBlocks, widthBlock_size>>>(WIDTH, d_imageArray);

			//// need 640 threads
			//dim3 heightBlocks(20, 1, 1);
			//dim3 heightBlock_size(32, 1, 1);
			//getLineSizeHeightCUDA<<< heightBlocks, heightBlock_size>>>(HEIGHT, WIDTH, d_imageArray);




			//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(imageArray, d_imageArray, (sizeof(unsigned char) * MAX_DATA_SIZE), cudaMemcpyDeviceToHost));

			//createBinaryImage(img, imageArray);
			//sprintf(imageFileName, "colourprocessed%d.bmp", imageIndex);
			//libbmp_write(imageFileName, img);

			#ifdef CAL_SIZE
				// This is a CUDA kernel to implement the same operation the sequential host version
				//parasiteSize[imageIndex - 1] = (short) getSizeOfBlobCUDA(HEIGHT, WIDTH, imageArray);


			/* These methods are used to find the intercept between the width and height midpoints.
			   The getSizeOfBlobCircleTest then uses this intercept when determining where the centre of the
			   test circle should be drawn this however does not always lead to best case output.

				//getLineSizeWidth(img->height, img->width, imageArray);
				//getLineSizeHeight(img->height, img->width, imageArray);
			*/

				#ifdef BENCHMARKING
					unsigned char* initialImgArray;
					initialImgArray = (unsigned char *)malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
					copyArray(imageArray, initialImgArray);

					#ifdef ELAPSED
					if(initialiseTimer(&s)==0) {
					#endif
						for (int i = 0; i < NUM_OF_PASSES; i++) {
							copyArray(initialImgArray, imageArray);

							#ifdef ELAPSED
							startTimer(&s);
							#else
							initCPUTime();
							#endif
				#endif

							parasiteSize[imageIndex - 1] = (short) getSizeOfBlobCircleTest(img->height, img->width, imageArray);

				#ifdef BENCHMARKING
							#ifdef ELAPSED
							stopTimer(&s);
							#else
							double endTime = getCPUTimeSinceStart();
							#endif

							#ifdef ELAPSED
							printf("Elapsed Time: %f(s), %f(ms)\n", getElapsedTime(&s), getElapsedTimeInMilli(&s));
							aggregatedElapsedTime += getElapsedTimeInMilli(&s);

							//fprintf(benchMarkFileElapsed, "%f\n", getElapsedTimeInMilli(&s));
							#else
							printf("CPU Time: %f(ms)\n", endTime);
							aggregatedCPUTime += endTime;
							#endif
						}
					#ifdef ELAPSED
					} else { printf("Find Blob size timer didnt initialise\n"); }
					#endif

					#ifdef ELAPSED
					fprintf(benchMarkFileElapsed, "Find blob size         : ");
					fprintf(benchMarkFileElapsed, "%f\n", (aggregatedElapsedTime/NUM_OF_PASSES));

					printf("\nAverage Elapsed Time: %f(ms)\n\n", (aggregatedElapsedTime/NUM_OF_PASSES));
					aggregatedElapsedTime = 0.0;
					#else
					fprintf(benchMarkFileCPU, "Find blob size         : ");
					fprintf(benchMarkFileCPU, "%f\n", (aggregatedCPUTime/NUM_OF_PASSES));

					printf("\nAverage CPU Time: %f(ms)\n\n", (aggregatedCPUTime/NUM_OF_PASSES));
					aggregatedCPUTime = 0.0;
					#endif

					free(initialImgArray);
				#endif


			//fclose(benchMarkFileElapsed);
			//printf("End of test.\n");

			#endif

			////////////////////////////////////////////////////////////////////////////////
			/* End Find Parasite Size                                                     */
			////////////////////////////////////////////////////////////////////////////////


			#ifdef DEBUG_OUTPUT
				// Redraw the noise removal image once the noise and fluke have been distinguished.
				createBinaryImage(img, imageArray);
				//libbmp_write("imageNoiseRemoval.bmp", img);
				libbmp_write("imageStage8.bmp", img);
			#endif
			

			/**************************************************************************************************************/

			createBinaryImage(img, imageArray);

			//strcpy(imageFileName, (char*) argv[1]);
			//strcat(imageFileName, "\\colour");
			//strcat(imageFileName, "processed");
			//strcat(imageFileName, imageIndexString);
			//strcat(imageFileName, ".bmp");

			//sprintf(imageFileName, "images/colourprocessed%d.bmp", imageIndex);
			sprintf(imageFileName, "%s/colourprocessed%d.bmp", imagePath, imageIndex);
			libbmp_write(imageFileName, img);

			
			#ifdef TEST_WRITE_TIME
				stopWatch imageTimer;
				int passes = 40;
				double elapsedTime;
				if(initialiseTimer(&imageTimer)==0) {
					for(int i=0; i<passes; i++)
					{
						startTimer(&imageTimer);
						libbmp_write(imageFileName, img);
						stopTimer(&imageTimer);
						printf("%f\n", getElapsedTimeInMilli(&imageTimer));
						elapsedTime += getElapsedTimeInMilli(&imageTimer);
					}
				}
				printf("\nAverage Write Time: %f(ms)\n\n", (elapsedTime/passes));
			#endif


			/**************************************************************************************************************/

			////////////////////////////////////////////////////////////////////////////////
			/* Create new image name for next image                                       */
			////////////////////////////////////////////////////////////////////////////////
			
			//printf("image index: %d \n", imageIndex);
			imageIndex++;
			//itoa(imageIndex,imageIndexString,10); // void itoa(int input, char *buffer, int radix) - base radix: 10 (decimal)

			//strcpy(imageFileName, "images/colour");
			//strcat(imageFileName, imageIndexString);
			//strcat(imageFileName, ".bmp");

			//sprintf(imageFileName,"images/colour%d.bmp", imageIndex);
			sprintf(imageFileName,"%s/colour%d.bmp", imagePath, imageIndex);

			//strcpy(imageFileName, (char*) argv[1]);
			//strcat(imageFileName, "\\colour");
			//strcat(imageFileName, imageIndexString);
			//strcat(imageFileName, ".bmp");

			////////////////////////////////////////////////////////////////////////////////
			/* End Create new image name for next image                                   */
			////////////////////////////////////////////////////////////////////////////////

			#ifdef BENCHMARKING
				#ifdef ELAPSED
					fprintf(benchMarkFileElapsed, "\n");
				#else
					fprintf(benchMarkFileCPU, "\n");
				#endif
			#endif

			for(int i = 0; i < img->height; i++)
				free(img->data[i]);
			free(img->data);
		}

	stopTimer(&executionTimer);

	#ifdef RUNS
		totalRuntime += getElapsedTime(&executionTimer);
		fprintf(runOutput, "%d\t%f\n", k, getElapsedTime(&executionTimer));
		imageIndex = 1;
		sprintf(imageFileName,"%s/colour%d.bmp", imagePath, imageIndex);
	}
	#endif

	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_imageArray));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(r_imageArray));

	cudaUnbindTexture(texRef);
	cudaUnbindTexture(texRef_R);

	#ifndef RUNS
		printf("Total Run Time: %f(s)\n\n", getElapsedTime(&executionTimer));
	#endif

	#ifdef BENCHMARKING
		#ifdef ELAPSED
			fclose(benchMarkFileElapsed);
		#else
			fclose(benchMarkFileCPU);
		#endif
	#endif

	// Free all malloc memory
	//free(histogram);
	free(img);
	free(imageArray);

	#ifdef RUNS
		printf("Run results appended to file.\n");
		fprintf(runOutput, "Average:\t%f\n", totalRuntime/NUM_RUNS);
		fclose(runOutput);
	#else
		#ifdef TXT
			FILE *txtOutput;
			//txtOutput = fopen("Results.txt", "w");
			txtOutput = fopen("Results.txt", "a"); // append to the current results file
			fprintf(txtOutput, "%s\n", videoName);
			fprintf(txtOutput, "Total Elapsed Time(s):\t%f\n", getElapsedTime(&executionTimer));
			#ifdef CAL_SIZE
				fprintf(txtOutput, "Image No\tSize\tCompactness\n");
			#else
				fprintf(txtOutput, "Image No\tCompactness\n");
			#endif
		#endif

		#ifdef CSV
			FILE *csvOutput;
			csvOutput = fopen("Results.csv", "a"); // append to the current results file
			fprintf(csvOutput, "%s\n", videoName);
			fprintf(csvOutput, "Total Elapsed Time(s):,%f\n", getElapsedTime(&executionTimer));
			#ifdef CAL_SIZE
				fprintf(csvOutput, "Image No,Size,Compactness\n");
			#else
				fprintf(csvOutput, "Image No,Compactness\n");
			#endif
		#endif

		#ifdef CAL_SIZE
			//Compensate for being too small
			compensateForSmallSize(parasiteSize, numberOfInputImages);
		#endif

		#ifndef TXT
			printf("--------------- Flux Unsorted ---------------\n");
		#endif
		for(int i = 0; i < numberOfInputImages; i++)
		{
			#ifndef TXT
			#ifndef CSV
				if (i%imagesPerThread == 0)
					printf("++++++++++++++++\n");
				printf("Array index %d: size = %d \t compactness = %f\n" , i, parasiteSize[i], parasiteCompactness[i]);
			#endif
			#endif

			#ifdef TXT
				#ifdef CAL_SIZE
					fprintf(txtOutput, "%d\t%d\t%f\n" , i, parasiteSize[i], parasiteCompactness[i]);
				#else
					fprintf(txtOutput, "%d\t%f\n" , i, parasiteCompactness[i]);
				#endif
			#endif

			#ifdef CSV
				#ifdef CAL_SIZE
					fprintf(csvOutput, "%d,%d,%f\n" , i, parasiteSize[i], parasiteCompactness[i]);
				#else
					fprintf(csvOutput, "%d,%f\n" , i, parasiteCompactness[i]);
				#endif
			#endif
		}

		#ifdef TXT
			fprintf(txtOutput, "\n");
		#endif

		#ifdef CSV
			fprintf(csvOutput, "\n");
		#endif

		// Use this one.
		int flux;
		if (numberOfInputImages >= 3) {
			#ifdef TXT
				flux = findParasiteFluxAlt(parasiteCompactness, numberOfInputImages, txtOutput);
			#else
				flux = findParasiteFluxAlt(parasiteCompactness, numberOfInputImages, csvOutput);
			#endif
			//flux = findParasiteFluxCompactness(numberOfInputImages, parasiteCompactness);
			//flux = findParasiteFlux3(numberOfInputImages, parasiteSize);
			#ifndef TXT
			#ifndef CSV
				printf("___________________________________\n");
				printf("Number of head projections: %d\n", flux);
			#endif
			#endif

			#ifdef TXT
				fprintf(txtOutput, "___________________________________\n");
				fprintf(txtOutput, "Number of head projections\t%d\n", flux);
			#endif

			#ifdef CSV
				fprintf(csvOutput, "___________________________________\n");
				fprintf(csvOutput, "Number of head projections,%d\n", flux);
			#endif
		}
		else
			printf("Not enough images to analyse.\n");

		#ifdef TXT
			printf("Results appended to file.\n");
			fprintf(txtOutput, "\n");
			fclose(txtOutput);
		#endif

		#ifdef CSV
			printf("Results appended to file.\n");
			fprintf(csvOutput, "\n");
			fclose(csvOutput);
		#endif
	#endif

	#ifdef CAL_SIZE
		free(parasiteSize);
	#endif
	free(parasiteCompactness);
	free(imagePath);
	
	printf("___________________\n");
	printf("PROGRAM COMPLETED\n\n");
	printf("PRESS ANY KEY TO EXIT\n");

	fflush(stdout);
	//getchar();
	return EXIT_SUCCESS;
}