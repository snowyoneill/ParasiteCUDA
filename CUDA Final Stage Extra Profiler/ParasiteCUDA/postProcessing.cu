
/*
 * This file contains all methods related to image post-processing using CUDA
 */

/// <summary>
/// Erodes a given image which stored in linear device memory 'd_imageArray'
/// Uses global memory to fetch array element and writes the result to the secondary memory space 
/// 'r_imageArray' 
/// </summary>
__global__ void erodeImageCUDA(unsigned char *d_imageArray, unsigned char *r_imageArray)
{
	// get current idx in array
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	// idx >= width -> ensures that the pixels 0-639 do not get operated on i.e. top row
	// idx <= (height * width) - width -> ensures that bottom row does not get operated on
    // width && idx % width != 0 -> ensures that left most column does not get operated on
	// idx % width != 639 -> ensures that right most column does not get operated on
	if (d_imageArray[idx] == 1)
	{
		// Pixel above
		// Pixel below
		// Pixel left
		// Pixel right
		// Top left
		// Top right
		// Bottom left
		// Bottom right
		if(		((idx - 640 < 0)									|| (d_imageArray[idx-640] == 1) )			&&		\
				((idx + 640 > 307199)								|| (d_imageArray[idx+640] == 1) )			&&		\
				((idx % 640 == 0)									|| (d_imageArray[idx-1] == 1) )				&&		\
				((idx % 640 == 639)									|| (d_imageArray[idx+1] == 1) )				&&		\
				(( ((idx % 640) == 0)	|| (idx - 640 < 0))			|| (d_imageArray[idx - 640 - 1] == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx - 640 < 0))			|| (d_imageArray[idx - 640 + 1] == 1) )		&&		\
				(( ((idx % 640) == 0)	|| (idx + 640 > 307199))	|| (d_imageArray[idx + 640 - 1] == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx + 640 > 307199))	|| (d_imageArray[idx + 640 + 1] == 1) )	
			)
			r_imageArray[idx] = 1;
	}
}

/// <summary>
/// Erodes a given image which stored in linear device memory 'd_imageArray'
/// Uses texture memory to fetch array element
/// Must be used in conjunction with erodeImageCUDA_R in order to prevent using 'cudaMemcpy' to place output results back
/// into the input stream/global address space 'd_imageArray' which 'd_imageArrayTex' is bound to - this is required to
/// utilise the texture cache
/// </summary>
__global__ void erodeImageCUDA_D(unsigned char *r_imageArray)
{
	// get current idx in array
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tex1Dfetch(d_imageArrayTex, idx) == 1)
	{
		// Pixel above
		// Pixel below
		// Pixel left
		// Pixel right
		// Top left
		// Top right
		// Bottom left
		// Bottom right
		if(		((idx - 640 < 0)									|| (tex1Dfetch(d_imageArrayTex, idx-640) == 1) )			&&		\
				((idx + 640 > 307199)								|| (tex1Dfetch(d_imageArrayTex, idx+640) == 1) )			&&		\
				((idx % 640 == 0)									|| (tex1Dfetch(d_imageArrayTex, idx-1) == 1) )				&&		\
				((idx % 640 == 639)									|| (tex1Dfetch(d_imageArrayTex, idx+1) == 1) )				&&		\
				(( ((idx % 640) == 0)	|| (idx - 640 < 0))			|| (tex1Dfetch(d_imageArrayTex, idx - 640 - 1) == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx - 640 < 0))			|| (tex1Dfetch(d_imageArrayTex, idx - 640 + 1) == 1) )		&&		\
				(( ((idx % 640) == 0)	|| (idx + 640 > 307199))	|| (tex1Dfetch(d_imageArrayTex, idx + 640 - 1) == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx + 640 > 307199))	|| (tex1Dfetch(d_imageArrayTex, idx + 640 + 1) == 1) )
			)
			r_imageArray[idx] = 1;
	}
}

/// <summary>
/// Erodes a given image which stored in linear device memory 'd_imageArray'
/// Uses texture memory to fetch array element
/// Writes results back to the primary global array 'd_imageArray'
/// </summary>
__global__ void erodeImageCUDA_R(unsigned char *d_imageArray)
{
	// get current idx in array
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tex1Dfetch(r_imageArrayTex, idx) == 1)
	{
		// Pixel above
		// Pixel below
		// Pixel left
		// Pixel right
		// Top left
		// Top right
		// Bottom left
		// Bottom right
		if(		((idx - 640 < 0)									|| (tex1Dfetch(r_imageArrayTex, idx-640) == 1) )			&&		\
				((idx + 640 > 307199)								|| (tex1Dfetch(r_imageArrayTex, idx+640) == 1) )			&&		\
				((idx % 640 == 0)									|| (tex1Dfetch(r_imageArrayTex, idx-1) == 1) )				&&		\
				((idx % 640 == 639)									|| (tex1Dfetch(r_imageArrayTex, idx+1) == 1) )				&&		\
				(( ((idx % 640) == 0)	|| (idx - 640 < 0))			|| (tex1Dfetch(r_imageArrayTex, idx - 640 - 1) == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx - 640 < 0))			|| (tex1Dfetch(r_imageArrayTex, idx - 640 + 1) == 1) )		&&		\
				(( ((idx % 640) == 0)	|| (idx + 640 > 307199))	|| (tex1Dfetch(r_imageArrayTex, idx + 640 - 1) == 1) )		&&		\
				(( ((idx % 640) == 639) || (idx + 640 > 307199))	|| (tex1Dfetch(r_imageArrayTex, idx + 640 + 1) == 1) )
			)
			d_imageArray[idx] = 1;
	}
}

/// <summary>
/// Dilates a given image which stored in linear device memory 'd_imageArray'
/// Uses global memory to fetch array element and writes the result to the secondary memory space 
/// 'r_imageArray' 
/// </summary>
__global__ void dilateImageCUDA(unsigned char *d_imageArray, unsigned char *r_imageArray)
{
	// get current idx in array
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (d_imageArray[idx] == 1)
	{
		r_imageArray[idx] = 1;
		
		if (idx >= 640 )							r_imageArray[idx-640] = 1;			// Pixel above
		if (idx <= 306559 )							r_imageArray[idx+640] = 1;			// Pixel below
		if (idx % 640 != 0 )						r_imageArray[idx-1] = 1;			// Pixel left
		if (idx % 640 != 639 )						r_imageArray[idx+1] = 1;			// Pixel right
		if ((idx >= 640) && (idx % 640 != 639))		r_imageArray[idx-640+1] = 1;		// Top right
		if ((idx <= 306559) && (idx % 640 != 639))	r_imageArray[idx+640+1] = 1;		// Bottom right
		if ((idx >= 640) && (idx % 640 != 0))		r_imageArray[idx-640-1] = 1;		// Top left
		if ((idx <= 306559) && (idx % 640 != 0))	r_imageArray[idx+640-1] = 1;		// Bottom left

		// Possibly faster??
		//if (idx >= 640 ) {
		//	r_imageArray[idx-640] = 1;								// Pixel above
		//	if (idx % 640 != 639) r_imageArray[idx-640+1] = 1;		// Top right
		//	if (idx % 640 != 0)   r_imageArray[idx-640-1] = 1;		// Top left
		//}
		//if (idx <= 306559 ) {
		//	r_imageArray[idx+640] = 1;								// Pixel below
		//	if (idx % 640 != 639) r_imageArray[idx+640+1] = 1;		// Bottom right
		//	if (idx % 640 != 0)   r_imageArray[idx+640-1] = 1;		// Bottom left	
		//}
		//if (idx % 640 != 0 )	r_imageArray[idx-1] = 1;			// Pixel left
		//if (idx % 640 != 639 )	r_imageArray[idx+1] = 1;		// Pixel right

	}
}

/// <summary>
/// Dilates a given image which stored in linear device memory 'd_imageArray'
/// Uses texture memory to fetch array element
/// Must be used in conjunction with dilateImageCUDA_R in order to prevent using 'cudaMemcpy' to place output results back
/// into the input stream/global address space 'd_imageArray' which 'd_imageArrayTex' is bound to - this is required to
/// utilise the texture cache
/// </summary>
__global__ void dilateImageCUDA_D(unsigned char *r_imageArray)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tex1Dfetch(d_imageArrayTex, idx) == 1)
	{
		r_imageArray[idx] = 1;
		
		if (idx >= 640 )							r_imageArray[idx-640] = 1;			// Pixel above
		if (idx <= 306559 )							r_imageArray[idx+640] = 1;			// Pixel below
		if (idx % 640 != 0 )						r_imageArray[idx-1] = 1;			// Pixel left
		if (idx % 640 != 639 )						r_imageArray[idx+1] = 1;			// Pixel right
		if ((idx >= 640) && (idx % 640 != 639))		r_imageArray[idx-640+1] = 1;		// Top right
		if ((idx <= 306559) && (idx % 640 != 639))	r_imageArray[idx+640+1] = 1;		// Bottom right
		if ((idx >= 640) && (idx % 640 != 0))		r_imageArray[idx-640-1] = 1;		// Top left
		if ((idx <= 306559) && (idx % 640 != 0))	r_imageArray[idx+640-1] = 1;		// Bottom left
	}
}

/// <summary>
/// Dilates a given image which stored in linear device memory 'd_imageArray'
/// Uses texture memory to fetch array element
/// Writes results back to the primary global array 'd_imageArray'
/// </summary>
__global__ void dilateImageCUDA_R(unsigned char *d_imageArray)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tex1Dfetch(r_imageArrayTex, idx) == 1)
	{
		d_imageArray[idx] = 1;
		
		if (idx >= 640 )							d_imageArray[idx-640] = 1;			// Pixel above
		if (idx <= 306559 )							d_imageArray[idx+640] = 1;			// Pixel below
		if (idx % 640 != 0 )						d_imageArray[idx-1] = 1;			// Pixel left
		if (idx % 640 != 639 )						d_imageArray[idx+1] = 1;			// Pixel right
		if ((idx >= 640) && (idx % 640 != 639))		d_imageArray[idx-640+1] = 1;		// Top right
		if ((idx <= 306559) && (idx % 640 != 639))	d_imageArray[idx+640+1] = 1;		// Bottom right
		if ((idx >= 640) && (idx % 640 != 0))		d_imageArray[idx-640-1] = 1;		// Top left
		if ((idx <= 306559) && (idx % 640 != 0))	d_imageArray[idx+640-1] = 1;		// Bottom left
	}
}

/// <summary>
/// This method is designed to remove remaining noise in the vertial direction after post-processing
/// Only removes groups of pixel below the defined mask size
/// Uses global memory
/// </summary>
__global__ void removeUnwantedNoiseHeightCUDA(unsigned char *d_imageArray, unsigned char *r_imageArray, int maskSize)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	int unwantedHeight = maskSize;
	int heightCount;

	r_imageArray[idx] = d_imageArray[idx];

	// greater than two rows from the bottom and less than two rows from the top (if masksize == 3)
	if ((idx >= WIDTH * (unwantedHeight-1)) && (idx <= (HEIGHT * WIDTH) - ((unwantedHeight-1) * WIDTH)))
	{
		heightCount = 0;
		//check pixels
		if (d_imageArray[idx] == 1)
		{
			heightCount++;
			
			//check above two pixels
			for (int k = 1; k < unwantedHeight; k++)
			{
				if (d_imageArray[idx + (k * WIDTH)] == 1)
					heightCount++;
			}
			
			//check below two pixels
			for (int k = 1; k < unwantedHeight; k++)
			{
					if (d_imageArray[idx - (k * WIDTH)] == 1)
						heightCount++;
			}

			if(heightCount < unwantedHeight)
				r_imageArray[idx] = 0;
		}
	}
}

/// <summary>
/// This method is designed to remove remaining noise in the vertial direction after post-processing
/// Only removes groups of pixel below the defined mask size
/// Uses texture memory
/// </summary>
__global__ void removeUnwantedNoiseHeightCUDATexture(unsigned char *r_imageArray, int maskSize)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	int unwantedHeight = maskSize;
	int heightCount;

	r_imageArray[idx] = tex1Dfetch(d_imageArrayTex, idx);

	if ((idx >= WIDTH * (unwantedHeight-1)) && (idx <= (HEIGHT * WIDTH) - ((unwantedHeight-1) * WIDTH)))
	{
		heightCount = 0;
		if (tex1Dfetch(d_imageArrayTex, idx) == 1)
		{
			heightCount++;
			
			for (int k = 1; k < unwantedHeight; k++)
			{
				if (tex1Dfetch(d_imageArrayTex, idx + (k * WIDTH)) == 1)
					heightCount++;
			}
			for (int k = 1; k < unwantedHeight; k++)
			{
				if (tex1Dfetch(d_imageArrayTex, idx - (k * WIDTH)) == 1)
					heightCount++;
			}

			if(heightCount < unwantedHeight)
				r_imageArray[idx] = 0;
		}
	}
}

/// <summary>
/// This method is designed to remove remaining noise in the horizontal direction after post-processing
/// Only removes groups of pixel below the defined mask size
/// Uses global memory
/// </summary>
__global__ void removeUnwantedNoiseWidthCUDA(unsigned char *d_imageArray, unsigned char *r_imageArray, int maskSize)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int unwantedWidth = maskSize;
	int widthCount;

	r_imageArray[idx] = d_imageArray[idx];

    if ((idx % WIDTH >= (unwantedWidth-1)) && (idx % WIDTH <= 639 - (unwantedWidth-1)))
	{
		widthCount = 0;
		//check pixels
		if (d_imageArray[idx] == 1)
		{
			widthCount++;

			//check right pixels
			for (int k = 1; k < unwantedWidth; k++)
			{
				if (d_imageArray[idx + (k * 1)] == 1)
					widthCount++;
			}
			
			//check left pixels
			for (int k = 1; k < unwantedWidth; k++)
			{
				if (d_imageArray[idx - (k * 1)] == 1)
					widthCount++;
			}

			//check pixel height and delete if necessary
			if(widthCount < unwantedWidth)
				r_imageArray[idx] = 0;
		}
	}
}

/// <summary>
/// This method is designed to remove remaining noise in the horizontal direction after post-processing
/// Only removes groups of pixel below the defined mask size
/// Uses texture memory
/// </summary>
__global__ void removeUnwantedNoiseWidthCUDATexture(unsigned char *d_imageArray, int maskSize)
{
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int unwantedWidth = maskSize;
	int widthCount;

	d_imageArray[idx] = tex1Dfetch(r_imageArrayTex, idx);

    if ((idx % WIDTH >= (unwantedWidth-1)) && (idx % WIDTH <= 639 - (unwantedWidth-1)))
	{
		widthCount = 0;
		//check pixels
		if (tex1Dfetch(r_imageArrayTex, idx) == 1)
		{
			widthCount++;

			//check right pixels
			for (int k = 1; k < unwantedWidth; k++)
			{
				if (tex1Dfetch(r_imageArrayTex, idx + (k * 1)) == 1)
					widthCount++;
			}
			
			//check left pixels
			for (int k = 1; k < unwantedWidth; k++)
			{
				if (tex1Dfetch(r_imageArrayTex, idx - (k * 1)) == 1)
					widthCount++;
			}

			//check pixel height and delete if necessary
			if(widthCount < unwantedWidth)
				d_imageArray[idx] = 0;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Sequential Methods																																		  */
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Erodes a given image which is represented by the linear array 'anArray'
/// </summary>
void erodeImage(int height, int width, unsigned char *anArray)
{
	unsigned char *tempArray = (unsigned char*)malloc(sizeof(unsigned char) * height * width);
    int i, j;

	for (i = 0; i < width * height; i++)
    {
        tempArray[i] = anArray[i];
    }

	// Perform logical and operation to erode image
    for (i = 1; i < height - 1; i++)
    {
        for (j = 1; j < width - 1; j++)
        {
			if (anArray[(i * width) + j] == 1)
			{
				if (anArray[((i - 1) * width) + (j - 1)] != 1 || anArray[((i - 1)* width) + j]  != 1 || 
					anArray[((i - 1) * width) + (j + 1)] != 1 || anArray[(i * width) + (j - 1)] != 1 ||
					anArray[(i * width) + (j + 1)] != 1 || anArray[((i + 1) * width) + (j - 1)] != 1 ||
					anArray[((i + 1) * width) + j] != 1 || anArray[((i + 1) * width) + (j + 1)] != 1)
					{
						tempArray[(i * width) + j] = 0;
					}
			}
        }
    }

	for (i = 0; i < width * height; i++)
    {
        anArray[i] = tempArray[i];
    }

	free(tempArray);
}