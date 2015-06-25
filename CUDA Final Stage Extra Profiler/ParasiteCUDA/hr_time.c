#include <windows.h>
#include <stdio.h>

#include "hr_time.h"

//// Dont need.
//#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

LARGE_INTEGER initialTime;
LARGE_INTEGER mFrequency;

DWORD initialTicks;
LONGLONG endTime;
HANDLE timerThread;

// Initialise the timer
// Checks that QPC and QPF is supported
int initialiseTimer( stopWatch *timer ) {
	if(!QueryPerformanceCounter(&initialTime)) {
		printf("HiRes timer counter not supported %d\n", GetLastError());
		return 1;
	}
	if(!QueryPerformanceFrequency(&mFrequency)) {
		printf("HiRes timer freq. not supported %d\n", GetLastError());
		return 1;
	}

	initialTicks = GetTickCount();
	timerThread = GetCurrentThread();
	endTime = 0;

	// Return 0 indicating successful initialisation.
	return 0;
}

void startTimer( stopWatch *timer) {
	LONGLONG newTime;
	double newTicks;
	unsigned long check;
	double milliSecsOff;
	LONGLONG adjustment;

	LARGE_INTEGER currentTime;
	// Set affinity to the first core
	/*
	 * On a multiprocessor machine, it should not matter which processor is called.
     * However, you can get different results on different processors due to bugs in the BIOS or the HAL.
	 */
	DWORD_PTR threadAffMask = SetThreadAffinityMask(timerThread, 1);
	// Query the timer
	QueryPerformanceCounter(&currentTime);
	// Reset affinity
	SetThreadAffinityMask(timerThread, threadAffMask);
	// Resample the frequency
	QueryPerformanceFrequency(&mFrequency);

	newTime = currentTime.QuadPart - initialTime.QuadPart;

	// scale by 1000 for milliseconds
	newTicks = (double) (1000 * newTime)/(double) mFrequency.QuadPart;

	// detect and compensate for performance counter leaps
	// (surprisingly common, see Microsoft KB: Q274323)
	check = GetTickCount() - initialTicks;
	milliSecsOff = (double)(newTicks - check);
	if (milliSecsOff < -100 || milliSecsOff > 100) {
		// We must allow the timer continue
		adjustment = min(milliSecsOff * mFrequency.QuadPart / 1000, newTime - endTime);
		initialTime.QuadPart += adjustment;
		newTime -= adjustment;
	}
	// Record last time for adjustment
	endTime = newTime;
	
	newTicks = (double)(newTime)/(double)mFrequency.QuadPart;
	timer->start = newTicks;
	//printf("Start Time: %f\n", newTicks);
}

void stopTimer( stopWatch *timer) {
	LONGLONG newTime;
	double newTicks;
	unsigned long check;
	double milliSecsOff;
	LONGLONG adjustment;

	LARGE_INTEGER currentTime;
	// Set affinity to the first core
	DWORD_PTR threadAffMask = SetThreadAffinityMask(timerThread, 1);
	// Query the timer
	QueryPerformanceCounter(&currentTime);
	// Reset affinity
	SetThreadAffinityMask(timerThread, threadAffMask);

	// Resample the frequency
	QueryPerformanceFrequency(&mFrequency);

	newTime = currentTime.QuadPart - initialTime.QuadPart;

	// scale by 1000 for milliseconds
	newTicks = (double)(1000 * newTime)/(double)mFrequency.QuadPart;

	// detect and compensate for performance counter leaps
	// (surprisingly common, see Microsoft KB: Q274323)
	check = GetTickCount() - initialTicks;
  	milliSecsOff = (double)(newTicks - check);
	if (milliSecsOff < -100 || milliSecsOff > 100) {
		// We must allow the timer continue
		adjustment = min(milliSecsOff * mFrequency.QuadPart / 1000, newTime - endTime);
		initialTime.QuadPart += adjustment;
		newTime -= adjustment;
	}
	// Record last time for adjustment
	endTime = newTime;

	newTicks = (double)(newTime)/(double)mFrequency.QuadPart;
	timer->stop = newTicks;
	//printf("Stop Time : %f\n", newTicks);
}

// This is the elapsed time in seconds
double getElapsedTime( stopWatch *timer) {
	double duration = (timer->stop) - (timer->start);
	return duration;
}

// This is the elapsed time in milliseconds
double getElapsedTimeInMilli( stopWatch *timer) {
	double duration = ((timer->stop) - (timer->start)) * 1000;
	return duration;
}

// This is the elapsed time in microseconds
double getElapsedTimeInMicro( stopWatch *timer) {
	double duration = ((timer->stop) - (timer->start)) * 1000000;
	return duration;
}

// This is the elapsed time in nanoseconds
double getElapsedTimeInNano( stopWatch *timer) {
	double duration = ((timer->stop) - (timer->start)) * 1000000000;
	return duration;
}