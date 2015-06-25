#pragma once

////////////////////////////////////////////////////////////////////////////////
/* Switches                                                                   */
////////////////////////////////////////////////////////////////////////////////

//#define RELEASE_BUILD

#define TXT					//depends on RUNS not to be defined
//#define CSV				//depends on RUNS not to be defined
//#define CAL_SIZE
#define COMPACTNESS			// calculates the number of fluctuations using the compactness
//#define TITLES

//#define OUTPUT_PROCESSED	// outputs the processed images to the hard drive

//#define RUNS
//#define NUM_RUNS 40

////////////////////////////////////////////////////////////////////////////////
/* End Switches                                                               */
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#define WIDTH  640
#define HEIGHT 480
#define MAX_DATA_SIZE		WIDTH*HEIGHT		// 307200 elements. 
#define THREADS_PER_BLOCK	512			// OR 1<<6
#define BLOCKS_PER_GRID_ROW (MAX_DATA_SIZE/THREADS_PER_BLOCK)


////////////////////////////////////////////////////////////////////////////////
/* ImageIO                                                                    */
////////////////////////////////////////////////////////////////////////////////
typedef struct		s_rgb
{
	unsigned char	r;
	unsigned char	g;
	unsigned char	b;
	unsigned char	moy;

}					t_rgb;


typedef	struct	s_bmp_fh
{
	short		sType;
	int			iSize;
	short		sReserved1;
	short		sReserved2;
	int			iOffBits;

}				t_bmp_fh;


typedef	struct	s_bmp_sh
{
	int			iSize;
	int			iWidth;
	int			iHeight;
	short		sPlanes;
	short		sBitCount;
	int			iCompression;
	int			iSizeImage;
	int			iXpelsPerMeter;
	int			iYpelsPerMeter;
	int			iClrUsed;
	int			iClrImportant;

}				t_bmp_sh;


typedef	struct	s_bmp_header
{
	t_bmp_fh	first_header;
	t_bmp_sh	second_header;

}				t_bmp_header;


typedef struct		s_bmp
{
	t_bmp_header	header;
	int				width;
	int				width_useless;
	int				height;
	t_rgb			**data;

}					t_bmp;

int	libbmp_copyAndCreateImg(t_bmp *bmp, t_bmp *bmpCopy);
int	libbmp_copyImgData(t_bmp *bmp, t_bmp *bmpCopy);
int	libbmp_load(char *filename, t_bmp *bmp);
int libbmp_write(char *filename, t_bmp *bmp);
void createImageFomArray(t_bmp *img, unsigned char *anArray);
void createBinaryImage(t_bmp *img, unsigned char *anArray);
void binaryImageToArray(t_bmp *img, unsigned char *anArray);
void imageToArray(t_bmp *img, unsigned char *anArray);


////////////////////////////////////////////////////////////////////////////////
/* HelperMethods                                                              */
////////////////////////////////////////////////////////////////////////////////
int getNumberOfImages(char* directory);
void sortArray(short *anArray, int arrayLength);

void selectSort(double array[], int n);

short largestSearch(short* anArray, int arraySize);
short smallestSearch(short* anArray, int arraySize);
float calculateMean(t_bmp *img);
float calculateStandardDeviation(t_bmp *img, float mean);
void compensateForSmallSize(short *anArray, int length);
void copyArray(unsigned char *orgArray, unsigned char *duplicateArray);
void outputArrays(unsigned char *array1, char *arrayName1, unsigned char *array2, char *arrayName2);
void outputArray(unsigned char *array1, char *arrayName);
void outputCSV(unsigned char *imageArray, char *arrayName);


////////////////////////////////////////////////////////////////////////////////
/* Post-processing                                                            */
////////////////////////////////////////////////////////////////////////////////
void erodeImage(int height, int width, unsigned char *anArray);

////////////////////////////////////////////////////////////////////////////////
/* Size Finding Methods                                                       */
////////////////////////////////////////////////////////////////////////////////
double calculateCompactness(double perimiter, double area);
double calculateArea(unsigned char *anArray);
double calculatePerimeter( t_bmp *img, unsigned char *anArray);

double calculatePerimeterOptimised( t_bmp *img, unsigned char *anArray, double initialArea);

int findAndReplace(unsigned char *anArray, int i, int j, int findColour, int replaceColour);

int checkPixels(unsigned char *anArray, int i, int j, int findColour, int replaceColour, int count, int pixelCount);

int* findPixels(int height, int width, unsigned char *anArray);
void findBoundingBox(int heightP, int widthP, unsigned char *anArray);
void getLineSizeWidth(int height, int width, unsigned char *anArray);
void getLineSizeWidthMod(int height, int width, unsigned char *anArray);
void getLineSizeHeight(int height, int width, unsigned char *anArray);
void getLineSizeHeightMod(int height, int width, unsigned char *anArray);
void drawCircle(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour);
int testMaxDilation(int height, int width, unsigned char *anArray, int unsigned charValue);
int getSizeOfBlob(int height, int width, unsigned char *anArray);
int testCircle(int width, int arrayPosition, unsigned char *anArray);
void drawLargestCircle(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour);
int drawCircleAndTest(int height, int width, int startX, int startY, int radius, unsigned char *anArray, unsigned char circleColour);
int getSizeOfBlobCircleTest(int height, int width, unsigned char *anArray);
int findParasiteFlux(int arraySize, short* parasiteSizeArray);

int findParasiteFlux3(int arraySize, short* parasiteSizeArray);

int findParasiteFluxAlt(double* parasiteSizeArray, int arraySize, FILE* csvOutput, double compactnessThreshold);
int findParasiteFluxCompactness(int arraySize, double* parasiteSizeArray);