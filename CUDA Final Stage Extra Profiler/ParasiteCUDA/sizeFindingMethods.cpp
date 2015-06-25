#include "parasite.h"

/*
 * This file contains all methods related to size calculations and object classification
 */

/// <summary>
/// Calculates compactness of reference shape for given measurements.
/// </summary>
double calculateCompactness(double perimiter, double area)
{
    double compactness = (double) (perimiter * perimiter) / (double) area;
    return compactness;
}

/// <summary>
/// Calculates the area of the given array.
/// </summary>
// Import for _msize
double calculateArea(unsigned char *anArray)
{
	int length = (MAX_DATA_SIZE);
	int area = 0;

	// For each element in the area, if its value is 1 add 1 to the area count.
    for(int i = 0; i < length; i++)
        if (anArray[i] == 1)
			area++;

	return area;
}


/// <summary>
/// Calculates the perimeter of the given image
/// Unoptimised - recalculates the size of 'anArray' which is already known  
/// </summary>
double calculatePerimeter( t_bmp *img, unsigned char *anArray)
//double calculatePerimeter(char *filename, t_bmp *img, unsigned char *anArray)
{
	double perimiter = 0;
	double initialArea = calculateArea(anArray);

	unsigned char* tempPerimeterArray = (unsigned char *) malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
	if(tempPerimeterArray != NULL)
	{
		//for(int i = 0; i < (MAX_DATA_SIZE); i++)
		//	tempPerimeterArray[i] = anArray[i];
		memcpy (tempPerimeterArray, anArray, sizeof(unsigned char) * MAX_DATA_SIZE);


		//// I used this method to load the previously processed image into the new temporary image for erosion.
		//libbmp_load(filename, tempPerimeterImg);
		erodeImage(HEIGHT, WIDTH, tempPerimeterArray);

		// Perimeter equals the initial area minus the new area after the erosion.
		double newArea = calculateArea(tempPerimeterArray);
		perimiter = initialArea - newArea;

		// Free temp malloc memory
		//for(int i = 0; i < tempPerimeterImg->height; i++)
		//	free(tempPerimeterImg->data[i]);
		//free(tempPerimeterImg->data);
		//free(tempPerimeterImg);
		free(tempPerimeterArray);
	}
	else
		printf("Pointer is null: ");

	return perimiter;
}

/// <summary>
/// Calculates the perimeter of the given image
/// Optimised - simply calculates the size of the temporary array and substracts the value from the initial size
/// </summary>
double calculatePerimeterOptimised( t_bmp *img, unsigned char *anArray, double initialArea)
{
	double perimiter = 0;
	unsigned char* tempPerimeterArray = (unsigned char *) malloc(sizeof(unsigned char) * MAX_DATA_SIZE);
	if(tempPerimeterArray != NULL)
	{
		memcpy (tempPerimeterArray, anArray, sizeof(unsigned char) * MAX_DATA_SIZE);

		erodeImage(img->height, img->width, tempPerimeterArray);

		double newArea = calculateArea(tempPerimeterArray);
		perimiter = initialArea - newArea;
		free(tempPerimeterArray);
	}

	return perimiter;
}


/// <sumary>
/// Reset counters and call check pixels.
/// </sumary>
int findAndReplace(unsigned char *anArray, int i, int j, int findColour, int replaceColour)
{
	int pixelCount = 0;
	int count = 0;

	return checkPixels(anArray, i, j, findColour, replaceColour, count, pixelCount);
}

/// <sumary>
/// Recursively count all colours specified by the 'findColour' parameter and replace them with
/// the 'replaceColour' parameter.
/// </sumary>

//#include <setjmp.h>
//jmp_buf place;


/* This code only get executed once - bug occurs when calling 'checkPixel' more than once i.e. for benchmarking.
   Instead insert new method 'findAndReplace' to set 'pixelCount' and count to 0. 

//// Set the pixel count and the number of recursive iterations to 0.
//int pixelCount = 0;
//int count = 0;
*/
int checkPixels(unsigned char *anArray, int i, int j, int findColour, int replaceColour, int count, int pixelCount)
{
	count++;
	if (count > 10000)
		return 0;
		//longjmp(place, 4);
		//abort();
	// Check above previously marked pixel
	if (!(i-1 < 0))
	{
		if (anArray[((i-1)*WIDTH)+j] == findColour)
		{
			anArray[((i-1)*WIDTH)+j] = replaceColour;
			pixelCount = 1 + checkPixels(anArray, (i-1), j, findColour, replaceColour, count, pixelCount);
		}
	}
	// Check below previously marked pixel
	if (!(i+1 >= HEIGHT))
	{
		if (anArray[((i+1)*WIDTH)+j] == findColour)
		{
			anArray[((i+1)*WIDTH)+j] = replaceColour;
			pixelCount = 1 + checkPixels(anArray, (i+1), j, findColour, replaceColour, count, pixelCount);
		}
	}
	// Check to the right of previously marked pixel
	if (!(j+1 >= WIDTH))
	{
		if (anArray[(i*WIDTH)+(j+1)] == findColour)
		{
			anArray[(i*WIDTH)+(j+1)] = replaceColour;
			pixelCount = 1 + checkPixels(anArray, i, j+1, findColour, replaceColour, count, pixelCount);
		}
	}
	// Check to the left of previously marked pixel
	if (!(j-1 < 0))
	{
		if (anArray[(i*WIDTH)+(j-1)] == findColour)
		{
			anArray[(i*WIDTH)+(j-1)] = replaceColour;
			pixelCount = 1 + checkPixels(anArray, i, j-1, findColour, replaceColour, count, pixelCount);
		}
	}
	return pixelCount;
}

/// <sumary>
/// Find all white pixels in the image and set them to blue and also determine the largest white blob's area and
/// coordinates.
/// </sumary>
int* findPixels(int height, int width, unsigned char *anArray)
{
	int blobCoordI = 0;
	int blobCoordJ = 0;
	int largestArea = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (anArray[(i*width) + j] == 1)
			{
				//// Now done automatically by 'findAndReplace'
				//count = 0;
				//pixelCount = 0;

				//int area;
				////anArray[(i*width) + j] = 3;
				//if (setjmp(place) != 0) {
				//	printf("Returned using longjmp\n");
				//	system("pause"); }
				//else
				//{
				//	area = checkPixels(anArray, i, j, 1, 4);
				//}

				/* *Update* to use 'findAndReplace'
				//int area = checkPixels(anArray, i, j, 1, 4);
				*/
				int area = findAndReplace(anArray, i, j, 1, 4);
				//if(setjmp(place) != 0){
				//	printf("Returned using longjmp\n");
				//	system("pause");
				//}
				if (area > largestArea)
				{
					blobCoordI = i;
					blobCoordJ = j;
					largestArea = area;
				}
			}
		}
	}

	//// If you declare an array in this fashion then its scope is limited to this function. i.e. it is a local variable.
	//// Local variables are retained on the stack. The return statement in this function does not return the array but
	//// a pointer to the first element of the array on the stack. In the main method whenever i called printf the correct
	//// values for 'largestArea', 'blobCoordI' and 'blobCoordJ' are passed correctly, however once the call has been made
	//// the stack is modified to store details of the called function.  As a result the pointers to the values no longer 
	//// consistent and produce erroneous output.  Therefore in order to solve this problem the array should be declared on
	//// heap by using malloc.
	//int details[] = {largestArea, blobCoordI, blobCoordJ};

	int *details = (int*) malloc(sizeof(int) * 3);
	//details = { largestArea, blobCoordI, blobCoordJ};
	details[0] = largestArea;
	details[1] = blobCoordI;
	details[2] = blobCoordJ;
	//printf("LargestArea: %d \n", largestArea);
	//printf("==============\n");
	//printf("details[0] %d : details[1] %d : details[2] %d\n", details[0], details[1], details[2]);
	//printf("==============\n");

	/* Create a green dot at the largest detected white area

		//anArray[(blobCoordI * width) + blobCoordJ - 1] = 3;
		//anArray[(blobCoordI * width) + blobCoordJ] = 3;
		//anArray[(blobCoordI * width) + blobCoordJ + 1] = 3;
		//anArray[(blobCoordI * width) - width + blobCoordJ] = 3;
		//anArray[(blobCoordI * width) + width + blobCoordJ] = 3;
	*/
	return details;
}

/// <summary>
/// Tries to determine the best bounding box that will surround the worm entirely. Not very accurate, irregular worm shapes
/// throw off detection.
/// </summary>
void findBoundingBox(int heightP, int widthP, unsigned char *anArray)
{	
	int pixelCount = 0;
	int maskSize = 30;
	bool found = false;
	int width = widthP;
	int height = heightP;

	////////////////////////////////////////////////////////////////////////////////
	/* Calculate Top Edge                                                         */
	////////////////////////////////////////////////////////////////////////////////
	// Iterate from the top of the image to the bottom looking for maskSize consecutive pixels.
	// This method will start from the top right hand corner.
	int topX = 0;
	for (int i = 0; (i < height) && !(found); i++)
	{
		for (int j = 0; (j < width) && !(found); j++)
		{
			if (anArray[(i * width) + j] == 1)
			{
				pixelCount++;
				if (pixelCount >= maskSize)
				{
					found = true;
					topX = i;
				}
			}
			else
				pixelCount = 0;
		}
	}

	for (int j = 0; j < width; j++)
		//anArray[((topX - 20) * width) + j] = 3;
		anArray[((topX) * width) + j] = 3;

	////////////////////////////////////////////////////////////////////////////////
	/* End Calculate Top Edge                                                     */
	////////////////////////////////////////////////////////////////////////////////

	pixelCount = 0;
	found = false;

	////////////////////////////////////////////////////////////////////////////////
	/* Calculate Bottom Edge                                                      */
	////////////////////////////////////////////////////////////////////////////////
	// Iterate from the bottom of the image to the top looking for maskSize consecutive pixels.
	// This method will start from the bottom right hand corner.
	int bottomX = 0;
	for (int i = height; (i > 0) && !(found); i--)
	{
		for (int j = 0; (j < width) && !(found); j++)
		{
			if (anArray[(i * width) + j] == 1)
			{
				pixelCount++;
				if (pixelCount >= maskSize)
				{
					found = true;
					bottomX = i;
				}
			}
			else
				pixelCount = 0;
		}
	}

	for (int j = 0; j < width; j++)
		//anArray[((bottomX + 20) * width) + j] = 3;
		anArray[((bottomX) * width) + j] = 3;

	////////////////////////////////////////////////////////////////////////////////
	/* End Calculate Bottom Edge                                                  */
	////////////////////////////////////////////////////////////////////////////////

	pixelCount = 0;
	found = false;

	////////////////////////////////////////////////////////////////////////////////
	/* Calculate Right Edge                                                       */
	////////////////////////////////////////////////////////////////////////////////
	// Iterate from the right of the image to the left looking for maskSize consecutive pixels.
	// This method will start from the top right hand corner and move to the left.
	int rightY = 0;
	for (int j = 0; (j < height) && !(found); j++)
	{
		for (int i = 0; (i < width) && !(found); i++)
		{
			if (anArray[(i * width) + j] == 1)
			{
				pixelCount++;
				if (pixelCount >= maskSize)
				{
					found = true;
					rightY = j;
				}
			}
			else
				pixelCount = 0;
		}
	}

	for (int i = 0; i < height; i++)
		//anArray[(i * width) + rightY - 20] = 3;
		anArray[(i * width) + rightY] = 3;

	////////////////////////////////////////////////////////////////////////////////
	/* End Calculate Right Edge                                                   */
	////////////////////////////////////////////////////////////////////////////////

	pixelCount = 0;
	found = false;

	////////////////////////////////////////////////////////////////////////////////
	/* Calculate Left Edge                                                        */
	////////////////////////////////////////////////////////////////////////////////
	// Iterate from the left of the image to the right looking for maskSize consecutive pixels.
	// This method will start from the top left hand corner and move to the right.
	int leftY = 0;
	for (int j = height; (j > 0) && !(found); j--)
	{
		for (int i = 0; (i < width) && !(found); i++)
		{
			if (anArray[(i * width) + j] == 1)
			{
				pixelCount++;
				if (pixelCount >= maskSize)
				{
					found = true;
					leftY = j;
				}
			}
			else
				pixelCount = 0;
		}
	}

	for (int i = 0; i < height; i++)
		//anArray[(i * width) + leftY + 20] = 3;
		anArray[(i * width) + leftY] = 3;


	////////////////////////////////////////////////////////////////////////////////
	/* End Calculate Left Edge                                                    */
	////////////////////////////////////////////////////////////////////////////////

	//int borders[] = { topX, 600, 100, 400 };
	//drawBoundingBox(borders);
}


// find centre of width
void getLineSizeWidth(int height, int width, unsigned char *anArray)
{
	int i, j, pixelHeight, firstPixelWidth, secondPixelWidth, secondPixelFound, pixelMidPoint, offset;
	secondPixelWidth = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (i = 0; i < height; i++)
	{
		j = 0;
		while (j < width)
		{
		//for (j=0; j++ < width; j++)
		//{
			if (anArray[(i * width) + j] == 1)
			{
				pixelHeight = i;
				firstPixelWidth = j;
				secondPixelFound = 0;
				offset = 1;
				while (secondPixelFound != 1)
				{
					// if the next pixel is a 1 make it the new value of pixel 2, update offset to check next pixel
					if (anArray[(i * width) + (j + offset)] == 1)
					{
						secondPixelWidth = j + offset;
						offset++;
					}
					// else update the offset and go through the loop again
					else
					{
						secondPixelFound = 1;
					}
				}

				// We have now found the start and end point of a line
				// Find a mid point and place a 2 here to indicate so
				pixelMidPoint = firstPixelWidth + ((secondPixelWidth - firstPixelWidth) / 2);
				anArray[(pixelHeight * width) + pixelMidPoint] = 2;
				j = j + offset;
			}
			else
			{
				// increment j to find next pixel
				j++;
			}
		}
	}
}

/////////////////////
/////////////////////
void getLineSizeWidthMod(int height, int width, unsigned char *anArray)
{
	int i, j, pixelHeight, firstPixelWidth, secondPixelWidth, secondPixelFound, pixelMidPoint, offset;
	int longestI = 0, longestMidPoint = 0, longestLength = 0;
	secondPixelWidth = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (i = 0; i < height; i++)
	{
		j = 0;
		while (j < width)
		{
		//for (j=0; j++ < width; j++)
		//{
			if (anArray[(i * width) + j] == 1)
			{
				pixelHeight = i;
				firstPixelWidth = j;
				secondPixelFound = 0;
				offset = 1;
				while (secondPixelFound != 1)
				{
					// if the next pixel is a 1 make it the new value of pixel 2, update offset to check next pixel
					if (anArray[(i * width) + (j + offset)] == 1)
					{
						secondPixelWidth = j + offset;
						offset++;
					}
					// else update the offset and go through the loop again
					else
					{
						secondPixelFound = 1;
					}
				}

				// We have now found the start and end point of a line
				// Find a mid point and place a 2 here to indicate so
				if(secondPixelWidth - firstPixelWidth > 8)
				{
					pixelMidPoint = firstPixelWidth + ((secondPixelWidth - firstPixelWidth) / 2);
					anArray[(pixelHeight * width) + pixelMidPoint] = 4;
				}
				j = j + offset;
			}
			else
			{
				// increment j to find next pixel
				j++;
			}
		}
	}
}


// find centre of height
void getLineSizeHeight(int height, int width, unsigned char *anArray)
{
	int i, j, pixelWidth, firstPixelHeight, secondPixelHeight, secondPixelFound, pixelMidPoint, offset;
	secondPixelHeight = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (j = 0; j < width; j++)
	{
		i = 0;
		while (i < height)
		{
		//for (j=0; j++ < width; j++)
		//{
			if (anArray[(i * width) + j] == 1 || anArray[(i * width) + j] == 2)
			{
				pixelWidth = j;
				firstPixelHeight = i;
				secondPixelFound = 0;
				offset = 1;
				while (secondPixelFound != 1)
				{
					// if the next pixel is a 1 make it the new vale of pixel 2, update offset to check next pixel
					if (anArray[((i+offset) * width) + j] == 1 || anArray[((i+offset) * width) + j] == 2)
					{
						secondPixelHeight = i + offset;
						offset++;
					}
					// else update the offset and go through the loop again
					else
					{
						secondPixelFound = 1;
					}
				}

				// We have now found the start and end point of a line
				// Find a mid point and place a 3 here to indicate so
				pixelMidPoint = firstPixelHeight + ((secondPixelHeight - firstPixelHeight) / 2);
				if(anArray[(pixelMidPoint * width) + pixelWidth] == 2)
				{
					anArray[(pixelMidPoint * width) + pixelWidth] = 4;
				}
				else
				{
					anArray[(pixelMidPoint * width) + pixelWidth] = 3;
				}

				i = i + offset;
			}
			else
			{
				// increment j to find next pixel
				i++;
			}
		}
	}
}

/////////////////////
/////////////////////
void getLineSizeHeightMod(int height, int width, unsigned char *anArray)
{
	int i, j, pixelWidth, firstPixelHeight, secondPixelHeight, secondPixelFound, pixelMidPoint, offset;
	secondPixelHeight = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (j = 0; j < width; j++)
	{
		i = 0;
		while (i < height)
		{
		//for (j=0; j++ < width; j++)
		//{
			if (anArray[(i * width) + j] == 1 || anArray[(i * width) + j] == 2)
			{
				pixelWidth = j;
				firstPixelHeight = i;
				secondPixelFound = 0;
				offset = 1;
				while (secondPixelFound != 1)
				{
					// if the next pixel is a 1 make it the new vale of pixel 2, update offset to check next pixel
					if (anArray[((i+offset) * width) + j] == 1 || anArray[((i+offset) * width) + j] == 4)
					{
						secondPixelHeight = i + offset;
						offset++;
					}
					// else update the offset and go through the loop again
					else
					{
						secondPixelFound = 1;
					}
				}

				// We have now found the start and end point of a line
				// Find a mid point and place a 3 here to indicate so
				pixelMidPoint = firstPixelHeight + ((secondPixelHeight - firstPixelHeight) / 2);
				anArray[(pixelMidPoint * width) + pixelWidth] = 4;
				i = i + offset;
			}
			else
			{
				// increment j to find next pixel
				i++;
			}
		}
	}
}
/////////////////
/////////////////

// Draw a circle
// adapted from wikipedia midpoint circle algorithm 
void drawCircle(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour)
{
    int f = 1 - radius;
    int ddF_x = 1;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

	anArray[(startX * width) + (startY + radius)] = circleColour;
	anArray[(startX * width) + (startY - radius)] = circleColour;
	anArray[((startX + radius) * width) + startY] = circleColour;
	anArray[((startX - radius) * width) + startY] = circleColour;
 
    while(x < y)
    {
      //assert(ddF_x == 2 * x + 1);
      //assert(ddF_y == -2 * y);
      //assert(f == x*x + y*y - radius*radius + 2*x - y + 1);
      if(f >= 0) 
      {
        y--;
        ddF_y += 2;
        f += ddF_y;
      }
      x++;
      ddF_x += 2;
      f += ddF_x; 

	  anArray[((startX + x) * width) + (startY + y)] = circleColour;
	  anArray[((startX - x) * width) + (startY + y)] = circleColour;
	  anArray[((startX + x) * width) + (startY - y)] = circleColour;
	  anArray[((startX - x) * width) + (startY - y)] = circleColour;
	  anArray[((startX + y) * width) + (startY + x)] = circleColour;
	  anArray[((startX - y) * width) + (startY + x)] = circleColour;
	  anArray[((startX + y) * width) + (startY - x)] = circleColour;
	  anArray[((startX - y) * width) + (startY - x)] = circleColour;
    }
}

// Test to see if the unsigned charValue hits a 0 indicating blob size is max
int testMaxDilation(int height, int width, unsigned char *anArray, int unsigned charValue)
{
	int i, j;
	int maxDilation = 0;

	//printf("testing dilation////////////////////// \n");
	for (i = 1; i < height - 1; i++)
    {
        for (j = 1; j < width - 1; j++)
        {
            if(anArray[(i * width) + j] == charValue)
            {
                if (anArray[(i * width) + (j + 1)] == 0)
				{
					maxDilation = 1;
				}
                if (anArray[(i * width) + (j - 1)] == 0)
				{
					maxDilation = 1;
				}
                if (anArray[((i+1) * width) + (j)] == 0)
				{
					maxDilation = 1;
				}
                if (anArray[((i-1) * width) + (j)] == 0)
				{
					maxDilation = 1;
				}
            }
        }
    }

	/*
	if(maxDilation == 1)
	{
		printf("************MAX DILATION IS FOUND***********");
	}*/

	return maxDilation;
}


//////////////////
//////////////////

// Get blob size for any blue intersection, means unsigned char value of 4
// find size using circle method
int getSizeOfBlob(int height, int width, unsigned char *anArray)
{
	int i, j, charValue, radius, maxDilation, largestBlobI, largestBlobJ, largestBlobSize;
	charValue = 5;
	largestBlobI = 0; 
	largestBlobJ = 0;
	largestBlobSize = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (i = 1; i < height - 1; i++)
	{
		for (j = 1; j < width - 1; j++)
		{
			// Find blue pixel
			if (anArray[(i * width) + j] == 4)
			{
				//printf("blob has been found at %d %d \n", i, j);
				maxDilation = 0;
				radius = 1;
				// Do dilation using a colour which is distinct, unsigned char value of 5
				// make value 4 to value 5 and dilate as much as necessary until 5 contacts a 0
				
				// repeat this until the unsigned charValue meets a black pixel
				while(maxDilation != 1)
				{
					//printf("dilate image \n");
					//dilateImageunsigned char(height, width, anArray, unsigned charValue);
					//currentBlobSize =  currentBlobSize + 2;
					drawCircle(height, width, i, j, radius, anArray, charValue);
					radius++;
					//printf("test dilation \n");
					if(testMaxDilation(height, width, anArray, charValue) == 1)
					{
						maxDilation = 1;
					}

				}
				if (radius * 2 > largestBlobSize)
				{
					largestBlobSize = radius * 2;
					largestBlobI = i;
					largestBlobJ = j;
				}
				charValue++;
			}
		}
	}

	printf("Largest blob in pixels is: %d \n", largestBlobSize);
	printf("Largest blob I: %d \n", largestBlobI);
	printf("Largest blob J: %d \n", 639 - largestBlobJ);

	return largestBlobSize;
}


// Every time a point is drawn do this test
int testCircle(int width, int arrayPosition, unsigned char *anArray)
{
	int edgeDetected = 0;

	//if (!(arrayPosition < 0))
	//if (!((arrayPosition - 1 < 0) || (arrayPosition + 1 < 640) || (arrayPosition + width > 640) || (arrayPosition - width < 0)))
	if (!((arrayPosition - 1 < 0) || (arrayPosition - width < 0)))
	{
		if(anArray[arrayPosition+1] == 0)
		{
			edgeDetected = 1;
		}
		else if(anArray[arrayPosition - 1] == 0)
		{
			edgeDetected = 1;
		}
		else if(anArray[arrayPosition + width] == 0)
		{
			edgeDetected = 1;
		}
		else if(anArray[arrayPosition - width] == 0)
		{
			edgeDetected = 1;
		}
	}
	return edgeDetected;
}

// Draw a circle
// adapted from wikipedia midpoint circle algorithm 
void drawLargestCircle(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour)
{
    int f = 1 - radius;
    int ddF_x = 1;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

	anArray[(startX * width) + (startY + radius)] = circleColour;
	anArray[(startX * width) + (startY - radius)] = circleColour;
	anArray[((startX + radius) * width) + startY] = circleColour;
	//if( (startX - radius) > 0)
		anArray[((startX - radius) * width) + startY] = circleColour;
 
    while(x < y)
    {

      if(f >= 0) 
      {
        y--;
        ddF_y += 2;
        f += ddF_y;
      }
      x++;
      ddF_x += 2;
      f += ddF_x; 

	  anArray[((startX + x) * width) + (startY + y)] = circleColour;
	  anArray[((startX - x) * width) + (startY + y)] = circleColour;
	  anArray[((startX + x) * width) + (startY - y)] = circleColour;
	  anArray[((startX - x) * width) + (startY - y)] = circleColour;
	  anArray[((startX + y) * width) + (startY + x)] = circleColour;
	  //if( (startX - y) > 0)
		anArray[((startX - y) * width) + (startY + x)] = circleColour;
	  anArray[((startX + y) * width) + (startY - x)] = circleColour;
	  anArray[((startX - y) * width) + (startY - x)] = circleColour;
    }
}


// Draw a circle
// adapted from wikipedia midpoint circle algorithm 
int drawCircleAndTest(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour)
{
    int f = 1 - radius;
    int ddF_x = 1;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;
	int maxDilation = 0;

	//anArray[(startX * width) + (startY + radius)] = circleColour;
	//anArray[(startX * width) + (startY - radius)] = circleColour;
	//anArray[((startX + radius) * width) + startY] = circleColour;
	//anArray[((startX - radius) * width) + startY] = circleColour;
 
    while(x < y)
    {
      //assert(ddF_x == 2 * x + 1);
      //assert(ddF_y == -2 * y);
      //assert(f == x*x + y*y - radius*radius + 2*x - y + 1);
      if(f >= 0) 
      {
        y--;
        ddF_y += 2;
        f += ddF_y;
      }
      x++;
      ddF_x += 2;
      f += ddF_x; 

	  //anArray[((startX + x) * width) + (startY + y)] = circleColour;
	  maxDilation += testCircle(width, ((startX + x) * width) + (startY + y), anArray);

	  //anArray[((startX - x) * width) + (startY + y)] = circleColour;
 	  maxDilation += testCircle(width, ((startX - x) * width) + (startY + y), anArray);

	  //anArray[((startX + x) * width) + (startY - y)] = circleColour;
	  maxDilation += testCircle(width, ((startX + x) * width) + (startY - y), anArray);

	  //anArray[((startX - x) * width) + (startY - y)] = circleColour;
	  maxDilation += testCircle(width, ((startX - x) * width) + (startY - y), anArray);

	  //anArray[((startX + y) * width) + (startY + x)] = circleColour;
	  maxDilation += testCircle(width, ((startX + y) * width) + (startY + x), anArray);

	  //anArray[((startX - y) * width) + (startY + x)] = circleColour;
	  maxDilation += testCircle(width, ((startX - y) * width) + (startY + x), anArray);

	  //anArray[((startX + y) * width) + (startY - x)] = circleColour;
	  maxDilation += testCircle(width, ((startX + y) * width) + (startY - x), anArray);

	  //anArray[((startX - y) * width) + (startY - x)] = circleColour;
	  maxDilation += testCircle(width, ((startX - y) * width) + (startY - x), anArray);
    }

	return maxDilation;
}


// Get blob size for any blue intersection, means unsigned char value of 4
// find size using circle method
int getSizeOfBlobCircleTest(int height, int width, unsigned char *anArray)
{
	int i, j, charValue, radius, maxDilation, largestBlobI, largestBlobJ, largestBlobSize;
	charValue = 5;
	largestBlobI = 0; 
	largestBlobJ = 0;
	largestBlobSize = 0;

	// used to get the mid point in a horizontal line
	//iterate through array to find first pixel
	for (i = 0; i < height - 1; i++)
	{
		for (j = 0; j < width - 1; j++)
		{
			// Find white pixel
			if (anArray[(i * width) + j] == 1)
			{
				//printf("blob has been found at %d %d \n", i, j);
				maxDilation = 0;
				radius = 1;
				// Do dilation using a colour which is distinct, unsigned char value of 5
				// make value 4 to value 5 and dilate as much as necessary until 5 contacts a 0
				
				// repeat this until the unsigned charValue meets a black pixel
				while(maxDilation == 0)
				{
					//printf("dilate image \n");
					//dilateImageunsigned char(height, width, anArray, unsigned charValue);
					//currentBlobSize =  currentBlobSize + 2;
					maxDilation = drawCircleAndTest(height, width, i, j, radius, anArray, charValue);
					radius++;
				}
				if ((radius-1) * 2 > largestBlobSize)
				{
					largestBlobSize = (radius-1) * 2;
					largestBlobI = i;
					largestBlobJ = j;
				}
				charValue++;
			}
		}
	}
	
	drawLargestCircle(height, width, largestBlobI, largestBlobJ, largestBlobSize / 2, anArray, /* Green Circle */ 3);
	//printf("Largest blob in pixels is: %d \n", largestBlobSize);
	//printf("Largest blob I: %d \n", largestBlobI);
	//printf("Largest blob J: %d \n", 639 - largestBlobJ);

	return largestBlobSize;
}

int findParasiteFlux(int arraySize, short* parasiteSizeArray)
{
	printf("--------------- Calculate Flux ---------------\n");

	short* fluctuationArray = (short*)malloc(sizeof(short) * arraySize);
	short* lowerMedianArray = (short*)malloc(sizeof(short) * arraySize/2);
	short* upperMedianArray = (short*)malloc(sizeof(short) * arraySize - (arraySize/2));
	int lowerMedian, upperMedian, i, j, currentSize, fluctuations;
	fluctuations = 0;

	for (i = 0; i < arraySize; i++)
		fluctuationArray[i] = parasiteSizeArray[i];

	// Sort the parasite size array into order
	sortArray(parasiteSizeArray, arraySize);

	// Find the lower median
	for (i = 0; i < arraySize / 2; i++)
		lowerMedianArray[i] = parasiteSizeArray[i];

	sortArray(lowerMedianArray, arraySize / 2);
	lowerMedian = lowerMedianArray[(arraySize / 2) / 2];

	j = 0;
	// Find the upper median
	for (i = arraySize - (arraySize / 2); i < arraySize; i++)
	{
		upperMedianArray[j] = parasiteSizeArray[i];
		j++;
	}

	sortArray(upperMedianArray, arraySize - (arraySize/2));
	upperMedian = upperMedianArray[(arraySize-(arraySize / 2)) / 2];

	// Set Array to size indicators
	for (i = 0; i < arraySize; i++)
	{
		if (fluctuationArray[i] <= lowerMedian + 3)
			fluctuationArray[i] = 0;
		else if (fluctuationArray[i] >= upperMedian - 3)
			fluctuationArray[i] = 2;
		else
			fluctuationArray[i] = 1;
	}

	/////TEST THIS BAD BOY!!!!////
	//Remove impossible transitions i.e. 202
	i = 0;
	while(i < arraySize-2)
	{
		if(fluctuationArray[i] == 2 && fluctuationArray[i+1] == 0 && fluctuationArray[i+2] == 2)
		{
			fluctuationArray[i+1] = 2;
			i = i + 2;
		}
		else
			i++;
	}

	i = 0;
	currentSize = fluctuationArray[i];
	/*
	while (currentSize == 1)
	{
		i++;
		currentSize = fluctuationArray[i];
	}*/

	// count number of changes from big to small and vice versa
	/*
	for (i; i < arraySize; i++)
	{
		if (fluctuationArray[i] == 0 && currentSize == 2)
		{
			fluctuations++;
			currentSize = 0;
			printf("fluctuation at %d\n", i);
		}
		if (fluctuationArray[i] == 2 && currentSize == 0)
		{
			fluctuations++;
			currentSize = 2;
			printf("fluctuation at %d\n", i);
		}
	}*/
	
	while (currentSize != 2)
	{
		i++;
		currentSize = fluctuationArray[i];
	}

	while (i < arraySize)
	{
		if (fluctuationArray[i] == 0 && currentSize == 2)
		{
			fluctuations++;
			i++;
			currentSize = fluctuationArray[i];
			printf("fluctuation at %d\n", i);
		}
		else if (currentSize == 2)
		{
			i++;
		}
		else
		{
			i++;
			currentSize = fluctuationArray[i];
		}
	}

	free(lowerMedianArray);
	free(upperMedianArray);
	
	//for(i = arraySize - 1; i >= 0; i--)
	//	//printf("Array %d: flux %d: array sorted %d\n" , i, fluctuationArray[i], parasiteSizeArray[i]);
	//	printf("%d\n" , parasiteSizeArray[i]); 
	
	printf("lower: %d \n", lowerMedian);
	printf("upper: %d \n", upperMedian);

	free(fluctuationArray);

	return fluctuations;
}

/// <sumary>
/// Calculates the number of worm fluxuations based on the compactness
/// If the variation between successive values is greater than the absolute compactness threshold value then output the fluctuation
/// </sumary>
int findParasiteFluxAlt(double* parasiteCompactnessArray, int arraySize, FILE* output, double compactnessThreshold)
{
	int fluctuations = 0;

	for (int i = 0; i < arraySize-1; i++)
	{
		if ((parasiteCompactnessArray[i+1] - parasiteCompactnessArray[i] >= compactnessThreshold) || \
			(parasiteCompactnessArray[i+1] - parasiteCompactnessArray[i] <= -compactnessThreshold))
		{
			#ifndef TXT
			#ifndef CSV
				printf("fluctuation between %d and %d\n", i, i+1);
			#endif
			#endif

			#ifdef TXT
				fprintf(output, "Fluctuation between %d and %d\n", i, i+1);
			#endif
			#ifdef CSV
				fprintf(output, "Fluctuation between %d and %d\n", i, i+1);
			#endif
			fluctuations++;
		}
	}

	//printf("number of fluctuations: %d \n", fluctuations);
	return fluctuations;
}

/*
	Modification to original method, creates three quartiles.
*/
int findParasiteFlux3(int arraySize, short* parasiteSizeArray)
{
	printf("--------------- Calculate Flux ---------------\n");

	short* fluctuationArray = (short*)malloc(sizeof(short) * arraySize);
	short* lowerRegionArray = (short*)malloc(sizeof(short) * arraySize/3);
	short* middleRegionArray = (short*)malloc(sizeof(short) * arraySize - ((arraySize/3)*2));
	short* upperRegionArray = (short*)malloc(sizeof(short) * arraySize/3);
	int lowerMedian, middleMedian, upperMedian, i, j, currentSize, fluctuations;
	fluctuations = 0;

	//for (i = 0; i < arraySize; i++)
	//	fluctuationArray[i] = parasiteSizeArray[i];
	memcpy(fluctuationArray, parasiteSizeArray, sizeof(short) * arraySize);

	// Sort the parasite size array into order
	sortArray(parasiteSizeArray, arraySize);
	for(int i = 0; i < arraySize; i++)
		printf("size = %d\n" , parasiteSizeArray[i]);

	// Find the lower median
	for (i = 0; i < (arraySize/3); i++)
		lowerRegionArray[i] = parasiteSizeArray[i];

	//sortArray(lowerRegionArray, arraySize / 3);
	lowerMedian = lowerRegionArray[(arraySize / 3) - 1];

	// -------------------------------------------------- //

	for (i = (arraySize/3), j=0; i < arraySize - (arraySize / 3); i++, j++)
		middleRegionArray[j] = parasiteSizeArray[i];

	//sortArray(middleRegionArray, arraySize - ((arraySize/3)*2));
	middleMedian = middleRegionArray[(arraySize / 3) - 1];

	// -------------------------------------------------- //

	for (i = arraySize - (arraySize / 3), j=0; i < arraySize; i++, j++)
		upperRegionArray[j] = parasiteSizeArray[i];

	//sortArray(upperRegionArray, arraySize - (arraySize/3) - (arraySize - ((arraySize/3)*2)));
	upperMedian = upperRegionArray[(arraySize-(arraySize / 2)) / 2];

	printf("lowerMedian: %d\n", lowerMedian);
	printf("middleMedian: %d\n", middleMedian);
	printf("upperMedian: %d\n", upperMedian);


	//// Set Array to size indicators
	//for (i = 0; i < arraySize; i++)
	//{
	//	if (fluctuationArray[i] <= lowerMedian + 3)
	//		fluctuationArray[i] = 0;
	//	else if (fluctuationArray[i] >= upperMedian - 3)
	//		fluctuationArray[i] = 2;
	//	else
	//		fluctuationArray[i] = 1;
	//}

	// Set Array to size indicators
	for (i = 0; i < arraySize; i++)
	{
		if (fluctuationArray[i] < lowerMedian + 3)
			fluctuationArray[i] = 0;
		else if (fluctuationArray[i] > upperMedian - 3)
			fluctuationArray[i] = 2;
		else
			fluctuationArray[i] = 1;
	}


	//Remove impossible transitions i.e. 202
	//for(i = 0; i < arraySize - 2; i++)
	//{
	//	if(fluctuationArray[i] == 2 && fluctuationArray[i+1] == 0 && fluctuationArray[i+2] == 2)
	//	{
	//		fluctuationArray[i+1] = 2;
	//		i = i + 2;
	//	}
	//	else
	//		i++;
	//}

	//while (currentSize == 1)
	//{
	//	i++;
	//	currentSize = fluctuationArray[i];
	//}
	//// count number of changes from big to small and vice versa
	//for (i; i < arraySize; i++)
	//{
	//	if (fluctuationArray[i] == 0 && currentSize == 2)
	//	{
	//		fluctuations++;
	//		currentSize = 0;
	//		printf("fluctuation at %d\n", i);
	//	}
	//	if (fluctuationArray[i] == 2 && currentSize == 0)
	//	{
	//		fluctuations++;
	//		currentSize = 2;
	//		printf("fluctuation at %d\n", i);
	//	}
	//}


	//i = 0;
	//currentSize = fluctuationArray[i];
	//for (i = 0; i < arraySize && currentSize != 2; i++)
	//	currentSize = fluctuationArray[i];
	//while (currentSize != 2)
	//{
	//	i++;
	//	currentSize = fluctuationArray[i];
	//}
	//printf("i %d\n", i);
	//while (i < arraySize)
	//{
	//	if (fluctuationArray[i] == 0 && currentSize == 2)
	//	{
	//		fluctuations++;
	//		i++;
	//		currentSize = fluctuationArray[i];
	//		printf("fluctuation at %d\n", i);
	//	}
	//	else if (currentSize == 2)
	//	{
	//		i++;
	//	}
	//	else
	//	{
	//		i++;
	//		currentSize = fluctuationArray[i];
	//	}
	//}

	for (i = 0; i < arraySize-1; i++)
		if(fluctuationArray[i]==0 && fluctuationArray[i+1]==2)
		{
			fluctuations++;
			printf("fluctuation at %d\n", i);
		}

	free(lowerRegionArray);
	free(middleRegionArray);
	free(upperRegionArray);
	
	//for(i = arraySize - 1; i >= 0; i--)
	//	printf("Array %d: flux %d: array sorted %d\n" , i, fluctuationArray[i], parasiteSizeArray[i]); 
	
	printf("lowerMedian: %d\n", lowerMedian);
	printf("middleMedian: %d\n", middleMedian);
	printf("upperMedian: %d\n", upperMedian);

	free(fluctuationArray);

	return fluctuations;
}

int findParasiteFluxCompactness(int arraySize, double* parasiteCompactnessArray)
{
	printf("--------------- Calculate Flux ---------------\n");

	double* fluctuationArray = (double*)malloc(sizeof(double) * arraySize);
	double* lowerRegionArray = (double*)malloc(sizeof(double) * arraySize/3);
	double* middleRegionArray = (double*)malloc(sizeof(double) * arraySize - ((arraySize/3)*2));
	double* upperRegionArray = (double*)malloc(sizeof(double) * arraySize/3);
	double lowerMedian, middleMedian, upperMedian;

	int i, j, currentSize, fluctuations;
	fluctuations = 0;

	//for (i = 0; i < arraySize; i++)
	//	fluctuationArray[i] = parasiteSizeArray[i];
	memcpy(fluctuationArray, parasiteCompactnessArray, sizeof(double) * arraySize);

	//printf("-----------------------\n");
	selectSort(parasiteCompactnessArray, arraySize);
	//for(int i = 0; i < arraySize; i++)
	//	printf("compactness = %f\n" , parasiteSizeArray[i]);

	// Find the lower median
	for (i = 0; i < (arraySize/3); i++)
		lowerRegionArray[i] = parasiteCompactnessArray[i];

	//sortArray(lowerRegionArray, arraySize / 3);
	lowerMedian = lowerRegionArray[(arraySize / 3) - 1];

	// -------------------------------------------------- //

	for (i = (arraySize/3), j=0; i < arraySize - (arraySize / 3); i++, j++)
		middleRegionArray[j] = parasiteCompactnessArray[i];

	//sortArray(middleRegionArray, arraySize - ((arraySize/3)*2));
	middleMedian = middleRegionArray[(arraySize / 3) - 1];

	// -------------------------------------------------- //

	for (i = arraySize - (arraySize / 3), j=0; i < arraySize; i++, j++)
		upperRegionArray[j] = parasiteCompactnessArray[i];

	//sortArray(upperRegionArray, arraySize - (arraySize/3) - (arraySize - ((arraySize/3)*2)));
	upperMedian = upperRegionArray[(arraySize-(arraySize / 2)) / 2];

	printf("lowerMedian: %f\n", lowerMedian);
	printf("middleMedian: %f\n", middleMedian);
	printf("upperMedian: %f\n", upperMedian);

	// Set Array to size indicators
	for (i = 0; i < arraySize; i++)
	{
		if (fluctuationArray[i] < lowerMedian + 3)
			fluctuationArray[i] = 0;
		else if (fluctuationArray[i] > upperMedian - 3)
			fluctuationArray[i] = 2;
		else
			fluctuationArray[i] = 1;
	}

	for (i = 0; i < arraySize-1; i++)
	if(fluctuationArray[i]==0 && fluctuationArray[i+1]==2)
	{
		fluctuations++;
		printf("fluctuation at %d\n", i);
	}

	free(lowerRegionArray);
	free(middleRegionArray);
	free(upperRegionArray);

	printf("lowerMedian: %f\n", lowerMedian);
	printf("middleMedian: %f\n", middleMedian);
	printf("upperMedian: %f\n", upperMedian);

	free(fluctuationArray);

	return fluctuations;
}