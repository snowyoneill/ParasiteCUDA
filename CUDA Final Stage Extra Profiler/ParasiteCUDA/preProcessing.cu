/*
 * This file contains all methods related to image pre-processing and segmentation
 */

/// <summary>
/// Create a histogram of the image
/// </summary>
//__global__ void createHistogramCUDA(unsigned char *d_imageArray, short *lut )// to use atomicAdd we need an int
__global__ void createHistogramCUDA(unsigned char *d_imageArray, int *lut )
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int moy = d_imageArray[idx];
	// We require an atomicAdd instruction because of the problem of race conditions.
	// This method of creating a histogram does not fully utilse the potential of the GPU as every thread is calling an atomic
	// add instruction which means the required value in global memory must be read, modified and rewritten back to global memory which
	// has a large latency.
	atomicAdd(&(lut[moy]), 1);
}

/// <summary>
/// Find the largest and smallest values in the global array - Stage 1 reduction
/// </summary>
//__global__ void calEnhancementCUDA(unsigned char *d_imageArray, unsigned char *d_I1AndI2, float *d_M_C)
//__global__ void calLutCUDA(short *d_imageArray, unsigned char *d_I1AndI2, float *d_M_C)
__global__ void calculateI1AndI2(unsigned char *d_imageArray, unsigned char *pMinResults, unsigned char *pMaxResults)
{
	// Declare arrays to be in shared memory.
	// If THREADS_PER_BLOCK=32
	// THREADS_PER_BLOCK elements * (1 byte per element) * 2 = 64B
	// so each SM can have 16KB / (64B / 1024) = 256 blocks
	__shared__ unsigned char min[THREADS_PER_BLOCK];
	__shared__ unsigned char max[THREADS_PER_BLOCK];


	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    unsigned char val1 = d_imageArray[i];
	unsigned char val2 = d_imageArray[i + THREADS_PER_BLOCK];
	min[tid] = (val1 < val2) ? val1 : val2;
	max[tid] = (val1 > val2) ? val1 : val2;

    __syncthreads();

    // do reduction in shared mem
    //for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    for(unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s) 
        {
			unsigned char temp = min[tid + s];

			if (temp < min[tid])
				min[tid] = temp;

			temp = max[tid + s];

			if (temp > max[tid])
					max[tid] = temp;

        }
        __syncthreads();
    }

	// unroll the last warp
	if (tid < 32)
    {
		{		
			min[tid] = (min[tid] < min[tid + 32]) ? min[tid] : min[tid + 32];
			max[tid] = (max[tid] > max[tid + 32]) ? max[tid] : max[tid + 32];
		}
		{		
			min[tid] = (min[tid] < min[tid + 16]) ? min[tid] : min[tid + 16];
			max[tid] = (max[tid] > max[tid + 16]) ? max[tid] : max[tid + 16];
		}
		{		
			min[tid] = (min[tid] < min[tid + 8]) ? min[tid] : min[tid + 8];
			max[tid] = (max[tid] > max[tid + 8]) ? max[tid] : max[tid + 8];
		}
		{		
			min[tid] = (min[tid] < min[tid + 4]) ? min[tid] : min[tid + 4];
			max[tid] = (max[tid] > max[tid + 4]) ? max[tid] : max[tid + 4];
		}
		{		
			min[tid] = (min[tid] < min[tid + 2]) ? min[tid] : min[tid + 2];
			max[tid] = (max[tid] > max[tid + 2]) ? max[tid] : max[tid + 2];
		}
		{		
			min[tid] = (min[tid] < min[tid + 1]) ? min[tid] : min[tid + 1];
			max[tid] = (max[tid] > max[tid + 1]) ? max[tid] : max[tid + 1];
		}
    }

	// write result for this block to global memory
    if (tid == 0)
	{
		pMinResults[blockIdx.x] = min[0];
		pMaxResults[blockIdx.x] = max[0];
	}

	/* From a previous version of calculateI1AndI2 (calEnhancementCUDA)

		////// Set only one single thread to carry out the gradient and intercept calculation.
		////if (idx == 640*480-100000)
		////{
		//	// Gradient
		//	d_M_C[0] = 255 / ((float)d_I1AndI2[1] - (float)d_I1AndI2[0]); // 255 / (I2 - I1)
		//	// Intercept
		//	d_M_C[1] = -d_M_C[0] * (float)d_1AndI2[0];
		////}
	*/
}

/// <summary>
/// Find the largest and smallest values in the global array - Stage 1 reduction
/// Uses texture memory
/// </summary>
__global__ void calculateI1AndI2Texture(unsigned char *pMinResults, unsigned char *pMaxResults)
{
	__shared__ unsigned char min[THREADS_PER_BLOCK];
	__shared__ unsigned char max[THREADS_PER_BLOCK];

	unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    unsigned char val1 = tex1Dfetch(d_imageArrayTex, i);
	unsigned char val2 = tex1Dfetch(d_imageArrayTex, i + THREADS_PER_BLOCK);
	min[tid] = (val1 < val2) ? val1 : val2;
	max[tid] = (val1 > val2) ? val1 : val2;

    __syncthreads();

    for(unsigned int s=blockDim.x/2; s>32; s>>=1)
    {
        if (tid < s) 
        {
			unsigned char temp = min[tid + s];

			if (temp < min[tid])
				min[tid] = temp;

			temp = max[tid + s];

			if (temp > max[tid])
					max[tid] = temp;

        }
        __syncthreads();
    }

	// unroll the last warp
	if (tid < 32)
    {
		{		
			min[tid] = (min[tid] < min[tid + 32]) ? min[tid] : min[tid + 32];
			max[tid] = (max[tid] > max[tid + 32]) ? max[tid] : max[tid + 32];
		}
		{		
			min[tid] = (min[tid] < min[tid + 16]) ? min[tid] : min[tid + 16];
			max[tid] = (max[tid] > max[tid + 16]) ? max[tid] : max[tid + 16];
		}
		{		
			min[tid] = (min[tid] < min[tid + 8]) ? min[tid] : min[tid + 8];
			max[tid] = (max[tid] > max[tid + 8]) ? max[tid] : max[tid + 8];
		}
		{		
			min[tid] = (min[tid] < min[tid + 4]) ? min[tid] : min[tid + 4];
			max[tid] = (max[tid] > max[tid + 4]) ? max[tid] : max[tid + 4];
		}
		{		
			min[tid] = (min[tid] < min[tid + 2]) ? min[tid] : min[tid + 2];
			max[tid] = (max[tid] > max[tid + 2]) ? max[tid] : max[tid + 2];
		}
		{		
			min[tid] = (min[tid] < min[tid + 1]) ? min[tid] : min[tid + 1];
			max[tid] = (max[tid] > max[tid + 1]) ? max[tid] : max[tid + 1];
		}
    }

	// write result for this block to global mem
    if (tid == 0)
	{
		pMinResults[blockIdx.x] = min[0];
		pMaxResults[blockIdx.x] = max[0];
	}
}

/// <summary>
/// Find the largest and smallest values in the global array - Stage 2 reduction
/// Takes the result of the previous kernel and further reduces them
/// </summary>
__global__ void calculateI1AndI2Stage2(unsigned char *pMinResults, unsigned char *pMaxResults, unsigned char *d_smallestValues, unsigned char *d_largestValues)
{
	__shared__ unsigned char min[THREADS_PER_BLOCK];
	__shared__ unsigned char max[THREADS_PER_BLOCK];


	unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x) + threadIdx.x;

	//unsigned char minVal1 = pMinResults[i];
	//unsigned char minVal2 = pMinResults[i + THREADS_PER_BLOCK];
	//min[tid] = (minVal1 < minVal2) ? minVal1 : minVal2;

	//unsigned char maxVal1 = pMaxResults[i];
	//unsigned char maxVal2 = pMaxResults[i + THREADS_PER_BLOCK];
	//max[tid] = (maxVal1 > maxVal2) ? maxVal1 : maxVal2;

	//printf("minTex: %d", tex1Dfetch(minTex, i));
	//printf("maxTex: %d", tex1Dfetch(maxTex, i));
	min[tid] = pMinResults[i];
	max[tid] = pMaxResults[i];

	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>32; s>>=1)
	{
		if ((tid < s) && ((i+s) < (MAX_DATA_SIZE / THREADS_PER_BLOCK / 2)))
		{
			unsigned char temp = min[tid + s];

			if (temp < min[tid])
				min[tid] = temp;

			temp = max[tid + s];

			if (temp > max[tid])
					max[tid] = temp;

		}
		__syncthreads();
	}

	// unroll the last warp
	if (tid < 32)
	{
		{		
			min[tid] = (min[tid] < min[tid + 32]) ? min[tid] : min[tid + 32];
			max[tid] = (max[tid] > max[tid + 32]) ? max[tid] : max[tid + 32];
		}
		{		
			min[tid] = (min[tid] < min[tid + 16]) ? min[tid] : min[tid + 16];
			max[tid] = (max[tid] > max[tid + 16]) ? max[tid] : max[tid + 16];
		}
		{		
			min[tid] = (min[tid] < min[tid + 8]) ? min[tid] : min[tid + 8];
			max[tid] = (max[tid] > max[tid + 8]) ? max[tid] : max[tid + 8];
		}
		{		
			min[tid] = (min[tid] < min[tid + 4]) ? min[tid] : min[tid + 4];
			max[tid] = (max[tid] > max[tid + 4]) ? max[tid] : max[tid + 4];
		}
		{		
			min[tid] = (min[tid] < min[tid + 2]) ? min[tid] : min[tid + 2];
			max[tid] = (max[tid] > max[tid + 2]) ? max[tid] : max[tid + 2];
		}
		{		
			min[tid] = (min[tid] < min[tid + 1]) ? min[tid] : min[tid + 1];
			max[tid] = (max[tid] > max[tid + 1]) ? max[tid] : max[tid + 1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		d_smallestValues[blockIdx.x] = min[0];
		d_largestValues[blockIdx.x] = max[0];
	}
}

/// <summary>
/// Find the largest and smallest values in the global array - Stage 2 reduction
/// Takes the result of the previous kernel and further reduces them
/// Uses texture memory
/// </summary>
__global__ void calculateI1AndI2Stage2Texture(unsigned char *d_smallestValues, unsigned char *d_largestValues)
{
	__shared__ unsigned char min[THREADS_PER_BLOCK];
	__shared__ unsigned char max[THREADS_PER_BLOCK];


	unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x) + threadIdx.x;

	//printf("minTex: %d", tex1Dfetch(minTex, i));
	//printf("maxTex: %d", tex1Dfetch(maxTex, i));
	min[tid] = tex1Dfetch(minTex, i);
	max[tid] = tex1Dfetch(maxTex, i);

	__syncthreads();

	for(unsigned int s=blockDim.x/2; s>32; s>>=1)
	{
		if ((tid < s) && ((i+s) < (MAX_DATA_SIZE / THREADS_PER_BLOCK / 2)))
		{
			unsigned char temp = min[tid + s];

			if (temp < min[tid])
				min[tid] = temp;

			temp = max[tid + s];

			if (temp > max[tid])
					max[tid] = temp;

		}
		__syncthreads();
	}

	// unroll the last warp
	if (tid < 32)
	{
		{		
			min[tid] = (min[tid] < min[tid + 32]) ? min[tid] : min[tid + 32];
			max[tid] = (max[tid] > max[tid + 32]) ? max[tid] : max[tid + 32];
		}
		{		
			min[tid] = (min[tid] < min[tid + 16]) ? min[tid] : min[tid + 16];
			max[tid] = (max[tid] > max[tid + 16]) ? max[tid] : max[tid + 16];
		}
		{		
			min[tid] = (min[tid] < min[tid + 8]) ? min[tid] : min[tid + 8];
			max[tid] = (max[tid] > max[tid + 8]) ? max[tid] : max[tid + 8];
		}
		{		
			min[tid] = (min[tid] < min[tid + 4]) ? min[tid] : min[tid + 4];
			max[tid] = (max[tid] > max[tid + 4]) ? max[tid] : max[tid + 4];
		}
		{		
			min[tid] = (min[tid] < min[tid + 2]) ? min[tid] : min[tid + 2];
			max[tid] = (max[tid] > max[tid + 2]) ? max[tid] : max[tid + 2];
		}
		{		
			min[tid] = (min[tid] < min[tid + 1]) ? min[tid] : min[tid + 1];
			max[tid] = (max[tid] > max[tid + 1]) ? max[tid] : max[tid + 1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		d_smallestValues[blockIdx.x] = min[0];
		d_largestValues[blockIdx.x] = max[0];
	}
}

/* Doesnt dynamically allocate shared memory - only works for THREADS_PER_BLOCK = 256

	//__global__ void calculateI1AndI2Final(unsigned char *pMinResults, unsigned char *pMaxResults, float *d_M_C)
	//{
	//	extern __shared__ unsigned char s_min[];
	//	extern __shared__ unsigned char s_max[];
	//
	//	//printf("pMinResults[0]: %d\n", pMinResults[0]);
	//	//printf("pMinResults[1]: %d\n", pMinResults[1]);
	//	//printf("pMinResults[2]: %d\n", pMinResults[2]);
	//
	//	unsigned char I1=255;
	//	s_min[0] = pMinResults[0];
	//	s_min[1] = pMinResults[1];
	//	s_min[2] = pMinResults[2];
	//
	//	//printf("s_min[0]: %d\n", s_min[0]);
	//	//printf("s_min[1]: %d\n", s_min[1]);
	//	//printf("s_min[2]: %d\n", s_min[2]);
	//
	//	unsigned char I2=0;
	//	s_max[0] = pMaxResults[0];
	//	s_max[1] = pMaxResults[1];
	//	s_max[2] = pMaxResults[2];
	//
	//
	//	for(unsigned int i=0; i<3; i++)
	//	{
	//		if(s_min[i] < I1)
	//			I1 = s_min[i];
	//
	//		if(s_max[i] > I2)
	//			I2 = s_max[i];
	//	}
	//
	//	//printf("I1: %d", I1);
	//	//printf("I2: %d", I2);
	//
	//	// Gradient
	//	d_M_C[0] = 255 / ((float)I2 - (float)I1); // 255 / (I2 - I1)
	//	// Intercept
	//	d_M_C[1] = -d_M_C[0] * ((float)I1);
	//}
*/

/// <summary>
/// Finds the largest and smallest gray values in the remaining max and min arrays
/// Should be called with 1 block and 1 thread
/// </summary>
__global__ void calculateI1AndI2Final(unsigned char *pMinResults, unsigned char *pMaxResults, unsigned char minBlocksRequiredAtStage2, float *d_M_C)
{
	extern __shared__ unsigned char s_minAndMax[];


	unsigned char* minArray = &s_minAndMax[0];
	unsigned char* maxArray = &s_minAndMax[minBlocksRequiredAtStage2];

	unsigned char I1=255;
	for(unsigned int i=0; i<minBlocksRequiredAtStage2; i++)
		minArray[i] = pMinResults[i];

	unsigned char I2=0;
	for(unsigned int i=0; i<minBlocksRequiredAtStage2; i++)
		maxArray[i] = pMaxResults[i];

	for(unsigned int i=0; i<minBlocksRequiredAtStage2; i++)
	{
		if(minArray[i] < I1)
			I1 = minArray[i];

		if(maxArray[i] > I2)
			I2 = maxArray[i];
	}

	// Gradient
	d_M_C[0] = 255 / ((float)I2 - (float)I1); // 255 / (I2 - I1)
	// Intercept
	d_M_C[1] = -d_M_C[0] * ((float)I1);
}

//__global__ void test(float I_PixelValue, float gradient, float intercept, float *outPixelValue)
//{
//
//
//	unsigned char pixelValue = (unsigned char)I_PixelValue;
//
//	//printf("pixelValue: %d\n", pixelValue);
//	//printf("gradient: %f\n", gradient);
//	//printf("intercept: %f\n", intercept);
//	//printf("new pixelValue: %f\n", (gradient * pixelValue));
//	//printf("new pixelValue: %f\n", ((gradient * pixelValue) + intercept));
//
//	////outPixelValue[0] = ((gradient * pixelValue) + intercept);
//	outPixelValue[0] = (unsigned char) ((gradient * pixelValue) + intercept);
//}

/// <summary>
/// Carries out contrast enhancement on the given input image array using I1 and I2 calculated from the
/// reduction kernels
/// </summary>
__global__ void enhanceContrastCUDA(unsigned char *d_imageArray, float *d_M_C)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	unsigned char pixelValue = d_imageArray[idx];
	float gradient = d_M_C[0];
	float intercept = d_M_C[1];

	if (pixelValue < (-intercept)/gradient)
		d_imageArray[idx] = 0;
	else if (pixelValue > (255 - intercept)/gradient)
		d_imageArray[idx] = 255;
	else
	{
		//if(idx==235826 || idx==224957)
		//{
		//	//printf("idx: %d\n", idx);
		//	//printf("idx value: %d\n", d_imageArray[idx]);
		//	//printf("gradient: %f\n", gradient);
		//	//printf("intercept: %f\n", intercept);
		//	//printf("new pixelValue: %f\n", (gradient * pixelValue));
		//	//printf("new pixelValue: %f\n", ((gradient * pixelValue) + intercept));

		//	//pixelValue: 217
		//	//gradient: 1.268657
		//	//intercept: -20.298508
		//	//new pixelValue: 255
		//}

		//if(idx==235826 || idx==224957) {
		//	d_imageArray[idx] = 255;
		//} 
		//else
			d_imageArray[idx] = (unsigned char) ((gradient * (float)pixelValue) + intercept);
	}
}

/* Not utilised - creating a histogram is not required

	//__global__ void enhanceContrastCUDAUsingLut(unsigned char *d_imageArray, float *d_M_C, short* lut)
	//{
	//	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	//
	//	int pixelValue = d_imageArray[idx];
	//
	//	if (pixelValue < -d_M_C[1]/d_M_C[0])
	//		d_imageArray[idx] = 0;
	//	else if (pixelValue > (255 - d_M_C[1])/d_M_C[0])
	//		d_imageArray[idx] = 255;
	//	else
	//		d_imageArray[idx] = (int) (lut[pixelValue]);
	//}
*/

/// <summary>
/// Carries out a low pass filter on the given image
/// Uses texture memory to improve performance as many array elements are inspected more than once
/// </summary>
__global__ void lowPassFilterCUDATexture(unsigned char *r_imageArray)
{

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
		short sum = 0;
		short count = 0;

		if (idx >= 640 )							{ sum += tex1Dfetch(d_imageArrayTex, idx-640);		count++; }		// Pixel above
		if (idx <= 306559 )							{ sum += tex1Dfetch(d_imageArrayTex, idx+640);		count++; }		// Pixel below
		if (idx % 640 != 0 )						{ sum += tex1Dfetch(d_imageArrayTex, idx-1); 		count++; }		// Pixel left
		if (idx % 640 != 639 )						{ sum += tex1Dfetch(d_imageArrayTex, idx+1); 		count++; }		// Pixel right
		if ((idx >= 640) && (idx % 640 != 639))		{ sum += tex1Dfetch(d_imageArrayTex, idx-640+1); 	count++; }		// Top right
		if ((idx <= 306559) && (idx % 640 != 639))	{ sum += tex1Dfetch(d_imageArrayTex, idx+640+1); 	count++; }		// Bottom right
		if ((idx >= 640) && (idx % 640 != 0))		{ sum += tex1Dfetch(d_imageArrayTex, idx-640-1); 	count++; }		// Top left
		if ((idx <= 306559) && (idx % 640 != 0))	{ sum += tex1Dfetch(d_imageArrayTex, idx+640-1); 	count++; }		// Bottom left

		sum += tex1Dfetch(d_imageArrayTex, idx); count++;

		r_imageArray[idx] = (unsigned char) (sum/count);
}

/// <summary>
/// Carries out a low pass filter on the given image
/// </summary>
__global__ void lowPassFilterCUDA(unsigned char *d_imageArray, unsigned char *r_imageArray)
{

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	//if ((idx >= 640) && (idx <= 306559) && (idx % 640 != 0) && (idx % 640 != 639)) // Ignore all the border pixels - bad implementation
	//{
		// Mask size of 9.
		short sum = 0;
		short count = 0;

		// 640 = WIDTH
		// 306559 = MAX_DATA_SIZE - WIDTH - 1

		if (idx >= 640 )							{ sum += d_imageArray[idx-640];		count++; }		// Pixel above
		if (idx <= 306559 )							{ sum += d_imageArray[idx+640];		count++; }		// Pixel below
		if (idx % 640 != 0 )						{ sum += d_imageArray[idx-1];		count++; }		// Pixel left
		if (idx % 640 != 639 )						{ sum += d_imageArray[idx+1];		count++; }		// Pixel right
		if ((idx >= 640) && (idx % 640 != 639))		{ sum += d_imageArray[idx-640+1];	count++; }		// Top right
		if ((idx <= 306559) && (idx % 640 != 639))	{ sum += d_imageArray[idx+640+1];	count++; }		// Bottom right
		if ((idx >= 640) && (idx % 640 != 0))		{ sum += d_imageArray[idx-640-1];	count++; }		// Top left
		if ((idx <= 306559) && (idx % 640 != 0))	{ sum += d_imageArray[idx+640-1];	count++; }		// Bottom left

		sum += d_imageArray[idx]; count++;

			r_imageArray[idx] = (unsigned char) (sum/count);
	//}
}

/*

Isodata algorithm - This iterative technique for choosing a threshold was developed by Ridler and Calvard . The histogram is initially segmented into two parts using a starting 
threshold value such as 0 = 2B-1, half the maximum dynamic range. The sample mean (mf,0) of the gray values associated with the foreground pixels and the sample mean (mb,0) of the 
gray values associated with the background pixels are computed. A new threshold value 1 is now computed as the average of these two sample means. The process is repeated, based 
upon the new threshold, until the threshold value does not change any more.

*/
/// <summary>
/// Calculates the appropriate threshold using reduction - Stage 1
/// Uses global memory
/// </summary>
__global__ void calculateThresholdCUDA(unsigned char *d_imageArray, unsigned char d_threshold, unsigned int *d_greaterThanSum , unsigned int *d_greaterThanCount, unsigned int *d_lessThanSum)
{
	// 6144KB of shared memory - for 512 threads
	__shared__ unsigned int greaterThanThreshold[THREADS_PER_BLOCK];
	__shared__ unsigned int greaterThanThresholdCount[THREADS_PER_BLOCK];
	__shared__ unsigned int lessThanThreshold[THREADS_PER_BLOCK];

	unsigned char threshold = d_threshold;
	//printf("d_threshold: %d\n", threshold);

	unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (d_imageArray[i] > threshold)
	{
		greaterThanThreshold[tid] = d_imageArray[i];
		greaterThanThresholdCount[tid] = 1;
		lessThanThreshold[tid] = 0;

		//printf("greaterThanThreshold[%d]: %d\n", tid, greaterThanThreshold[tid]);
		//printf("greaterThanThresholdCount[%d]: %d\n", tid, greaterThanThresholdCount[tid]);
		//printf("lessThanThreshold: %d\n", lessThanThreshold[0]);
	}
	else
	{
		lessThanThreshold[tid] = d_imageArray[i];

		greaterThanThreshold[tid] = 0;
		greaterThanThresholdCount[tid] = 0;
	}

	__syncthreads();

	
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
			//if(tid == 0)
			//{
			//	printf("tid: %d\n", tid);
			//	printf("s: %d\n", s);
			//	printf("greaterThanThreshold[tid]: %d\n", greaterThanThreshold[tid]);
			//	printf("greaterThanThreshold[tid + s]: %d\n", greaterThanThreshold[tid + s]);
			//	printf("greaterThanThreshold[tid]: %d\n", greaterThanThresholdCount[tid]);
			//	printf("greaterThanThresholdCount[tid + s]: %d\n", greaterThanThresholdCount[tid + s]);
			//	printf("lessThanThreshold[tid]: %d\n", lessThanThreshold[tid]);
			//	printf("lessThanThreshold[tid + s]: %d\n", lessThanThreshold[tid + s]);
			//	printf("------------------\n");
			//}

			greaterThanThreshold[tid] += greaterThanThreshold[tid + s];
			greaterThanThresholdCount[tid] += greaterThanThresholdCount[tid + s];
			lessThanThreshold[tid] += lessThanThreshold[tid + s];
        }
        __syncthreads();
    }



    if (tid == 0)
	{
 		d_greaterThanSum[blockIdx.x] = greaterThanThreshold[0];
		d_greaterThanCount[blockIdx.x] = greaterThanThresholdCount[0];
		d_lessThanSum[blockIdx.x] = lessThanThreshold[0];
	}
}

/// <summary>
/// Calculates the appropriate threshold using reduction - Stage 1
/// Uses texture memory to fetch data
/// </summary>
__global__ void calculateThresholdCUDATexture(unsigned char d_threshold, unsigned int *d_greaterThanSum , unsigned int *d_greaterThanCount, unsigned int *d_lessThanSum)
{
	// 3072KB of shared memory.
	__shared__ unsigned int greaterThanThreshold[THREADS_PER_BLOCK];
	__shared__ unsigned int greaterThanThresholdCount[THREADS_PER_BLOCK];
	__shared__ unsigned int lessThanThreshold[THREADS_PER_BLOCK];

	unsigned char threshold = d_threshold;

	unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x) + threadIdx.x;

	if (tex1Dfetch(d_imageArrayTex, i) > threshold)
	{
		greaterThanThreshold[tid] = tex1Dfetch(d_imageArrayTex, i);
		greaterThanThresholdCount[tid] = 1;
		lessThanThreshold[tid] = 0;
	}
	else
	{
		lessThanThreshold[tid] = tex1Dfetch(d_imageArrayTex, i);

		greaterThanThreshold[tid] = 0;
		greaterThanThresholdCount[tid] = 0;
	}

	__syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
			greaterThanThreshold[tid] += greaterThanThreshold[tid + s];
			greaterThanThresholdCount[tid] += greaterThanThresholdCount[tid + s];
			lessThanThreshold[tid] += lessThanThreshold[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
	{
 		d_greaterThanSum[blockIdx.x] = greaterThanThreshold[0];
		d_greaterThanCount[blockIdx.x] = greaterThanThresholdCount[0];
		d_lessThanSum[blockIdx.x] = lessThanThreshold[0];
	}
}

/// <summary>
/// Stage 2 of threshold reduction
/// Uses global memory
/// </summary>
__global__ void calculateThresholdCUDAStage2(unsigned int *d_greaterThanSum , unsigned int *d_greaterThanCount, unsigned int *d_lessThanSum, \
											 unsigned int *d_greaterThanSumFinalValues, unsigned int *d_greaterThanCountFinalValues, unsigned int *d_lessThanSumFinalValues)
{
	// 3072KB of shared memory.
	__shared__ unsigned int greaterThanThreshold[THREADS_PER_BLOCK];
	__shared__ unsigned int greaterThanThresholdCount[THREADS_PER_BLOCK];
	__shared__ unsigned int lessThanThreshold[THREADS_PER_BLOCK];

	unsigned int tid = threadIdx.x;
    unsigned int i = (blockIdx.x*blockDim.x) + threadIdx.x;

	greaterThanThreshold[tid] = d_greaterThanSum[i];
	greaterThanThresholdCount[tid] = d_greaterThanCount[i];
	lessThanThreshold[tid] = d_lessThanSum[i];


	__syncthreads();

    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if ((tid < s) && ((i+s) < (MAX_DATA_SIZE / THREADS_PER_BLOCK)))
        {
			greaterThanThreshold[tid] += greaterThanThreshold[tid + s];
			greaterThanThresholdCount[tid] += greaterThanThresholdCount[tid + s];
			lessThanThreshold[tid] += lessThanThreshold[tid + s];
        }
        __syncthreads();
    }


    if (tid == 0)
	{
 		d_greaterThanSumFinalValues[blockIdx.x] = greaterThanThreshold[0];
		d_greaterThanCountFinalValues[blockIdx.x] = greaterThanThresholdCount[0];
		d_lessThanSumFinalValues[blockIdx.x] = lessThanThreshold[0];
	}
}

/// <summary>
/// Final stage - calculates the appropriate threshold value
/// Should be called with 1 block and 1 thread
/// </summary>
__global__ void calculateThresholdCUDAFinal(unsigned int *d_greaterThanSumFinalValues, unsigned int *d_greaterThanCountFinalValues, unsigned int *d_lessThanSumFinalValues, \
											unsigned int minBlocksRequiredForThresholdingAtStage2, unsigned char *newThresholdValue)
{
	extern __shared__ unsigned int s_greaterThan_Count_lessThan[];

	unsigned int* greaterThanSumArray = &s_greaterThan_Count_lessThan[0];
	unsigned int* greaterThanCountArray = &s_greaterThan_Count_lessThan[minBlocksRequiredForThresholdingAtStage2];
	unsigned int* lessThanArray = &s_greaterThan_Count_lessThan[minBlocksRequiredForThresholdingAtStage2 * 2];

	unsigned int backgroundSetValue = 0;
	for(unsigned int i=0; i<minBlocksRequiredForThresholdingAtStage2; i++)
		greaterThanSumArray[i] = d_greaterThanSumFinalValues[i];

	unsigned int backgroundSetCount = 0;
	for(unsigned int i=0; i<minBlocksRequiredForThresholdingAtStage2; i++)
		greaterThanCountArray[i] = d_greaterThanCountFinalValues[i];

	unsigned int objectSetValue = 0;
	for(unsigned int i=0; i<minBlocksRequiredForThresholdingAtStage2; i++)
		lessThanArray[i] = d_lessThanSumFinalValues[i];

	for(unsigned int i=0; i<minBlocksRequiredForThresholdingAtStage2; i++)
	{
		backgroundSetValue += greaterThanSumArray[i];
		backgroundSetCount += greaterThanCountArray[i];
		objectSetValue += lessThanArray[i];
	}

	backgroundSetCount = (backgroundSetCount == 0) ? 1 : backgroundSetCount;
	unsigned int backgroundSetAverage = backgroundSetValue / backgroundSetCount;
	//printf("backgroundSetAverage: %d\n", backgroundSetAverage);
	unsigned int objectSetAverage = objectSetValue / (MAX_DATA_SIZE - backgroundSetCount);
	//printf("objectSetAverage: %d\n", objectSetAverage);

	*newThresholdValue = (unsigned char) ((objectSetAverage + backgroundSetAverage) / 2);
	//printf("newThresholdValue: %d\n",*newThresholdValue);
}

/// <summary>
/// Calculates an automatic threshold based on the ISODATA approach
/// Uses only global device memory
/// </summary>
int iterativeCalculateThresholdCUDA(unsigned char *d_imageArray)
//int iterativeCalculateThresholdCUDA(unsigned char *imageArray)
{
	int thresholdFound;

	unsigned char thresholdValue, newThresholdValue;
	newThresholdValue = 255;  //Max dynamic range -1
	thresholdFound = 0;

	int minMemoryRequiredForThresholding = MAX_DATA_SIZE / THREADS_PER_BLOCK;
	
	//unsigned int* h_greaterThanSum = (unsigned int*)malloc(sizeof(unsigned int) * minMemoryRequiredForThresholding);
	//unsigned int* h_greaterThanCount = (unsigned int*)malloc(sizeof(unsigned int) * minMemoryRequiredForThresholding);
	//unsigned int* h_lessThanSum = (unsigned int*)malloc(sizeof(unsigned int) * minMemoryRequiredForThresholding);

	unsigned int* d_greaterThanSum;
	unsigned int* d_greaterThanCount;
	unsigned int* d_lessThanSum;
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanSum, sizeof(unsigned int) * minMemoryRequiredForThresholding) );
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanCount, sizeof(unsigned int) * minMemoryRequiredForThresholding) );
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_lessThanSum, sizeof(unsigned int) * minMemoryRequiredForThresholding) );


	int minBlocksRequiredForThresholdingAtStage2 = (int) ceil(((float)minMemoryRequiredForThresholding / THREADS_PER_BLOCK));
	

	//unsigned int* h_greaterThanSum = (unsigned int*)malloc(sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2);
	//unsigned int* h_greaterThanCount = (unsigned int*)malloc(sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2);
	//unsigned int* h_lessThanSum = (unsigned int*)malloc(sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2);

	unsigned int* d_greaterThanSumS2;
	unsigned int* d_greaterThanCountS2;
	unsigned int* d_lessThanSumS2;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanSumS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanCountS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_lessThanSumS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );

	unsigned char* h_threshold = (unsigned char*) malloc(sizeof(unsigned char));
	unsigned char* d_threshold;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_threshold, sizeof(unsigned char)) );

	unsigned int sharedMemSize = 3 * minBlocksRequiredForThresholdingAtStage2 * sizeof(unsigned int);
	


	while(thresholdFound != 1)
	{
		//printf("\n****************\n");
		//printf("thresholdValue - Before: %d\n", thresholdValue);
		thresholdValue = newThresholdValue;
		//printf("thresholdValue - After: %d\n", thresholdValue);
		//printf("+++++++++++++++++++\n");

		//calculateThresholdCUDA<<<minMemoryRequired, 32>>>(d_imageArray, thresholdValue, d_greaterThanSum, d_greaterThanCount, d_lessThanSum);
		//printf("minMemoryRequiredForThresholding: %d\n", minMemoryRequiredForThresholding);
		calculateThresholdCUDA<<<minMemoryRequiredForThresholding, THREADS_PER_BLOCK>>>(d_imageArray, thresholdValue, d_greaterThanSum, d_greaterThanCount, d_lessThanSum);
			//printf("calculateThresholdCUDA Return: %s\n", cudaGetErrorString( cudaGetLastError() ));

		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanSum, d_greaterThanSum, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanCount, d_greaterThanCount, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_lessThanSum, d_lessThanSum, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));
		//
		//unsigned int backgroundSetValue, backgroundSetCount, objectSetValue;
		//backgroundSetValue = backgroundSetCount = objectSetValue = 0;
		//for (int i=0 ; i < minMemoryRequiredForThresholding; i++)
		//{
		//	backgroundSetValue += h_greaterThanSum[i];
		//	backgroundSetCount += h_greaterThanCount[i];
		//	objectSetValue += h_lessThanSum[i];
		//}

		//printf("------------RESULTS---------------\n");
		//printf("minMemoryRequired: %d\n", minMemoryRequiredForThresholding);
		//printf("objectSetValue: %d\n", objectSetValue);
		//printf("objectSetCount: %d\n", (307200 - backgroundSetCount));
		//
		//printf("backgroundSetValue: %d\n", backgroundSetValue);
		//printf("backgroundSetCount: %d\n", backgroundSetCount);
		//printf("\n");

		//printf("minBlocksRequiredForThresholdingAtStage2: %d\n", minBlocksRequiredForThresholdingAtStage2);
		calculateThresholdCUDAStage2<<<minBlocksRequiredForThresholdingAtStage2, THREADS_PER_BLOCK>>>(d_greaterThanSum, d_greaterThanCount, d_lessThanSum, \
																									d_greaterThanSumS2, d_greaterThanCountS2, d_lessThanSumS2);


		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanSum, d_greaterThanSumS2, (sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanCount, d_greaterThanCountS2, (sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_lessThanSum, d_lessThanSumS2, (sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2), cudaMemcpyDeviceToHost));
		//
		//unsigned int backgroundSetValue, backgroundSetCount, objectSetValue;
		//backgroundSetValue = backgroundSetCount = objectSetValue = 0;
		//for (int i=0 ; i < minBlocksRequiredForThresholdingAtStage2; i++)
		//{
		//	backgroundSetValue += h_greaterThanSum[i];
		//	backgroundSetCount += h_greaterThanCount[i];
		//	objectSetValue += h_lessThanSum[i];
		//}

		//printf("------------RESULTS---------------\n");
		//printf("minMemoryRequired: %d\n", minBlocksRequiredForThresholdingAtStage2);
		//printf("objectSetValue: %d\n", objectSetValue);
		//printf("objectSetCount: %d\n", (307200 - backgroundSetCount));
		//
		//printf("backgroundSetValue: %d\n", backgroundSetValue);
		//printf("backgroundSetCount: %d\n", backgroundSetCount);
		//printf("\n");

		//printf("minSharedMemoryRequired * sizeof(int): %d\n", sharedMemSize);
		calculateThresholdCUDAFinal<<<1, 1, sharedMemSize>>>(d_greaterThanSumS2, d_greaterThanCountS2, d_lessThanSumS2, minBlocksRequiredForThresholdingAtStage2, d_threshold);
			//printf("calculateThresholdCUDA Return: %s\n", cudaGetErrorString( cudaGetLastError() ));
		CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_threshold, d_threshold, (sizeof(unsigned char)), cudaMemcpyDeviceToHost));

		newThresholdValue = *h_threshold;

		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanSum, d_greaterThanSum, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_greaterThanCount, d_greaterThanCount, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));
		//CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_lessThanSum, d_lessThanSum, (sizeof(unsigned int) * minMemoryRequiredForThresholding), cudaMemcpyDeviceToHost));


		//for (i = 0; i < MAX_DATA_SIZE; i++)
		//{
		//	if (imageArray[i] > thresholdValue)
		//	{
		//		backgroundSetSize++;
		//		backgroundSetValue += imageArray[i];
		//	}
		//	else
		//	{
		//		objectSetSize++;
		//		objectSetValue += imageArray[i];
		//	}
		//}

		//printf("------------RESULTS---------------\n");
		//printf("minMemoryRequired: %d\n", minMemoryRequired);
		//printf("objectSetValue: %d\n", objectSetValue);
		//printf("objectSetCount: %d\n", (307200 - backgroundSetCount));
		//
		//printf("backgroundSetValue: %d\n", backgroundSetValue);
		//printf("backgroundSetCount: %d\n", backgroundSetCount);
		//printf("\n");

		
		////backgroundSetCount = (backgroundSetCount == 0) ? 1 : backgroundSetCount;
		//backgroundSetAverage = backgroundSetValue / backgroundSetCount;
		////printf("backgroundSetAverage: %d\n", backgroundSetAverage);
		//objectSetAverage = objectSetValue / (MAX_DATA_SIZE - backgroundSetCount);
		////printf("objectSetAverage: %d\n", objectSetAverage);

		//newThresholdValue = (unsigned char) ((objectSetAverage + backgroundSetAverage) / 2);
		//printf("thresholdValue: %d\n", thresholdValue);
		//printf("newThresholdValue: %d\n", newThresholdValue);
		//printf("----------------\n");

		if (thresholdValue == newThresholdValue)
		{
			thresholdFound = 1;
			//printf("\nnewThresholdValue: %d\n",newThresholdValue);
		}

	}

	// cleans up by freeing all device and host memory no longer required
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanSum));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanCount));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_lessThanSum));

	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanSumS2));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanCountS2));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_lessThanSumS2));
	
	free(h_threshold);
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_threshold));

	return (int) newThresholdValue;
}


/// <summary>
/// Calculates an automatic threshold based on the ISODATA approach
/// Uses texture device memory for initial stage
/// </summary>
int iterativeCalculateThresholdCUDATexture()
{
	int thresholdFound;

	unsigned char thresholdValue, newThresholdValue;
	newThresholdValue = 255;  //Max dynamic range -1
	thresholdFound = 0;

	int minMemoryRequiredForThresholding = MAX_DATA_SIZE / THREADS_PER_BLOCK;

	unsigned int* d_greaterThanSum;
	unsigned int* d_greaterThanCount;
	unsigned int* d_lessThanSum;
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanSum, sizeof(unsigned int) * minMemoryRequiredForThresholding) );
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanCount, sizeof(unsigned int) * minMemoryRequiredForThresholding) );
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_lessThanSum, sizeof(unsigned int) * minMemoryRequiredForThresholding) );
	
	int minBlocksRequiredForThresholdingAtStage2 = (int) ceil(((float)minMemoryRequiredForThresholding / THREADS_PER_BLOCK));

	unsigned int* d_greaterThanSumS2;
	unsigned int* d_greaterThanCountS2;
	unsigned int* d_lessThanSumS2;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanSumS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );
    CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_greaterThanCountS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_lessThanSumS2, sizeof(unsigned int) * minBlocksRequiredForThresholdingAtStage2) );

	unsigned char* h_threshold = (unsigned char*) malloc(sizeof(unsigned char));
	unsigned char* d_threshold;
	CUSTOM_CUDA_SAFE_CALL( cudaMalloc( (void **)&d_threshold, sizeof(unsigned char)) );

	unsigned int sharedMemSize = 3 * minBlocksRequiredForThresholdingAtStage2 * sizeof(unsigned int);

	while(thresholdFound != 1)
	{
		thresholdValue = newThresholdValue;
		calculateThresholdCUDATexture<<<minMemoryRequiredForThresholding, THREADS_PER_BLOCK>>>(thresholdValue, d_greaterThanSum, d_greaterThanCount, d_lessThanSum);
		calculateThresholdCUDAStage2<<<minBlocksRequiredForThresholdingAtStage2, THREADS_PER_BLOCK>>>(d_greaterThanSum, d_greaterThanCount, d_lessThanSum, \
																									d_greaterThanSumS2, d_greaterThanCountS2, d_lessThanSumS2);
		calculateThresholdCUDAFinal<<<1, 1, sharedMemSize>>>(d_greaterThanSumS2, d_greaterThanCountS2, d_lessThanSumS2, minBlocksRequiredForThresholdingAtStage2, d_threshold);
		CUSTOM_CUDA_SAFE_CALL(cudaMemcpy(h_threshold, d_threshold, (sizeof(unsigned char)), cudaMemcpyDeviceToHost));

		newThresholdValue = *h_threshold;

		if (thresholdValue == newThresholdValue)
			thresholdFound = 1;
	}

	// cleans up by freeing all device and host memory no longer required
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanSum));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanCount));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_lessThanSum));

	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanSumS2));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_greaterThanCountS2));
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_lessThanSumS2));
	
	free(h_threshold);
	CUSTOM_CUDA_SAFE_CALL(cudaFree(d_threshold));

	return (int) newThresholdValue;
}


/// <summary>
/// Calculates an automatic threshold based on the ISODATA approach
/// Carried out by the host
/// </summary>
int iterativeCalculateThresholdUsingArray(unsigned char *imageArray)
{
	int i, thresholdValue, newThresholdValue, objectSetSize, backgroundSetSize, objectSetValue, backgroundSetValue,
		objectSetAverage, backgroundSetAverage, thresholdFound;

	thresholdValue = -1;
	newThresholdValue = 255;  //Max dynamic range -1
	thresholdFound = 0;

	//int initialAvgGrayLevel = 0;
	//for (int j = 0; j < (MAX_DATA_SIZE); j++)
	//	initialAvgGrayLevel += imageArray[j];

	//printf("TotalGrayLevel: %d\n", initialAvgGrayLevel);
	//printf("MAX_DATA_SIZE: %d\n", MAX_DATA_SIZE);
	//printf("initialAvgGrayLevel: %d\n", (initialAvgGrayLevel / (MAX_DATA_SIZE)));

	//newThresholdValue = (initialAvgGrayLevel / (MAX_DATA_SIZE));


	while(thresholdFound != 1)
	{
		//printf("\n****************\n");
		//printf("thresholdValue - Before: %d\n", thresholdValue);
		thresholdValue = newThresholdValue;
		//printf("thresholdValue - After: %d\n", thresholdValue);
		
		objectSetSize = 0;
		backgroundSetSize = 0;
		objectSetValue = 0;
		backgroundSetValue = 0;
		
		for (i = 0; i < MAX_DATA_SIZE; i++)
		{
			if (imageArray[i] > thresholdValue)
			{
				backgroundSetSize++;
				backgroundSetValue += imageArray[i];
			}
			else
			{
				objectSetSize++;
				objectSetValue += imageArray[i];
			}
		}

		//printf("objectSetSize: %d\n", objectSetSize);
		//printf("objectSetValue: %d\n", objectSetValue);
		//printf("backgroundSetSize: %d\n", backgroundSetSize);
		//printf("backgroundSetValue: %d\n", backgroundSetValue);
		//printf("\n");

		objectSetSize = (objectSetSize == 0) ? 1 : objectSetSize;
		// Get average of background and object sets
		objectSetAverage = objectSetValue / objectSetSize;
		//printf("objectSetAverage: %d\n", objectSetAverage);

		//backgroundSetSize = (backgroundSetSize == 0) ? 1 : backgroundSetSize;
		backgroundSetAverage = backgroundSetValue / backgroundSetSize;
		//printf("backgroundSetAverage: %d\n", backgroundSetAverage);

		newThresholdValue = (objectSetAverage + backgroundSetAverage) / 2;
		//printf("newThresholdValue: %d\n",newThresholdValue);
		//printf("----------------\n");

		if (thresholdValue == newThresholdValue)
		{
			thresholdFound = 1;
			//printf("\nnewThresholdValue: %d\n",newThresholdValue);
		}

	}

	return newThresholdValue;
}

/// <summary>
/// Segments the given image using the automatically calculated threshold.
/// Uses global memory - no optimisation
/// </summary>
__global__ void segmentImageCUDA(unsigned char *d_imageArray, unsigned char threshold)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (d_imageArray[idx] > threshold)
		d_imageArray[idx] = 0;
	else
		d_imageArray[idx] = 1;
}

/// <summary>
/// Segments the given image using the automatically calculated threshold.
/// Uses texture memory
/// </summary>
__global__ void segmentImageCUDATexture(unsigned char *d_imageArray, unsigned char threshold)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tex1Dfetch(d_imageArrayTex, idx) > threshold)
		d_imageArray[idx] = 0;
	else
		d_imageArray[idx] = 1;
}

/// <summary>
/// Segments the given image using the automatically calculated threshold.
/// Uses coalesced memory reads
/// </summary>
__global__ void segmentImageCoalescedCUDA(uchar4 *d_imageArray, unsigned char threshold)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	uchar4 data = d_imageArray[idx];

	data.x = (data.x > threshold ? 0 : 1);
	data.y = (data.y > threshold ? 0 : 1);
	data.z = (data.z > threshold ? 0 : 1);
	data.w = (data.w > threshold ? 0 : 1);

	d_imageArray[idx] = data;
}

/// <summary>
/// Segments the given image using the automatically calculated threshold.
/// Naive attempt to coalesce memory
/// </summary>
__global__ void segmentImageCoalescedCUDATest(unsigned char *d_imageArray, unsigned char threshold)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;


				//printf("threshold: %d\n", threshold);
				//printf("d_imageArray[0]: %d\n", *(d_imageArray + idx));
				//printf("d_imageArray[1]: %d\n", *(d_imageArray + idx + 1));
				//printf("d_imageArray[2]: %d\n", *(d_imageArray + idx + 2));
				//printf("d_imageArray[3]: %d\n", *(d_imageArray + idx + 3));

				//printf("idx: %d\n", idx);
				//printf("d_imageArray+idx: %d\n", d_imageArray+idx);
				//printf("d_imageArray+(idx*4): %d\n", d_imageArray+(idx*4));

				//unsigned int* iptr = (unsigned int*)(d_imageArray+(idx*4));
				//unsigned int i = *iptr;
				//unsigned char* cptr = (unsigned char*) &i;

	unsigned int* iptr = (unsigned int*)(d_imageArray+(idx*4));
	unsigned int i = *iptr;

	//unsigned char* cptr = (unsigned char*) &i;
	//unsigned char cArray[4];
	typedef __align__(4) unsigned char alignedArray;
	alignedArray cArray[4];

	unsigned int mask = 0x000000FF;

	//cArray[0] = (unsigned char) (i & mask);
	//for(int s=0; s<4; s++)
	//for(unsigned char s=0; s<4; s++)
	//{
	//	if (s==0)
	//		cArray[s] = (unsigned char) (i & mask);
	//	else
	//	{
	//		mask = mask << 8;
	//		cArray[s] = (unsigned char) ((i & mask) >> (8*s));
	//	}
	//}

	//uchar1  sresult;
	//sresult = make_uchar1(0);
	//for(sresult.x=0; sresult.x<4; sresult.x++)
	//{
	//		cArray[sresult.x] = ((i & mask) >> (8*sresult.x));
	//		mask = mask << 8;
	//}


	// Unaligned access
	//for(unsigned char s=0; s<4; s++)
	//{
	//		cArray[s] = (unsigned char) ((i & mask) >> (8*s));
	//		mask = mask << 8;
	//}


	// Unaligned access - however the compiler seems to correct the memory address access by masking and shifting
	// addr[n] replaced with ((*(addr & ~3)) >> (addr & 3)) & 0xff; -- from irc

	//mask = mask << 0;
	//cArray[0] = (unsigned char) (i & mask) >> 0;
	//mask = mask << 8;
	//cArray[1] = (unsigned char) ((i & mask) >> 8);
	//mask = mask << 8;
	//cArray[2] = (unsigned char) ((i & mask) >> 16);
	//mask = mask << 8;
	//cArray[3] = (unsigned char) ((i & mask) >> 24);

	//printf("i: %ud\n", i);

    for(int s=0; s<4; s++) 
    {
		//printf("cArray[%d] before: %d\n", s, cArray[s]);
		if (cArray[s] > threshold)
			cArray[s] = 0;
		else
			cArray[s] = 1;
		//printf("cArray[%d] after : %d\n", s, cArray[s]);
    }

	//printf("i: %ud\n", i);
	//printf("---------------------\n");
	unsigned int* cptr = (unsigned int*) cArray;
	*iptr = *cptr;
	//*(d_imageArray+idx) = *iptr;
}

/*	Cannot coalesce memory reads for unsigned char's in this fashion

	//// Image Size = 640*480
	//// Threads per block = 256
	//// Blocks per grid = 300
	//// segmentImageCoalescedCUDA<<<300, THREADS_PER_BLOCK>>>(d_imageArray, threshold);
	//__global__ void segmentImageCoalescedCUDA(unsigned char *d_imageArray, unsigned char threshold)
	//{
	//	unsigned char temp[4];
	//
	//	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	//
	//    for(int s=0; s<4; s++) 
	//    {
	//		if (d_imageArray[(4*idx)+s] > threshold)
	//			temp[s] = 0;
	//		else
	//			temp[s] = 1;
	//    }
	//
	//	//__syncthreads();
	//
	//	for(int s=0; s<4; s++) 
	//		d_imageArray[(4*idx)+s] = temp[s];
	//}
*/