////////////////////////////////////////////////////////////////////////////////
/* For measuring CPU time                                                     */
////////////////////////////////////////////////////////////////////////////////
#define VISUALSTUDIO
////////////////////////////////////////////////////////////////////////////////
/* End for measuring CPU time                                                 */
////////////////////////////////////////////////////////////////////////////////

#ifdef VISUALSTUDIO
#include <time.h>
#include <windows.h>
#include <math.h>
#include <stdio.h>
#else
#include <unistd.h>
#include <sys/times.h>
#endif

#include "cputime.h"
//#include "dimacs.h"


#ifdef VISUALSTUDIO

time_t startCPUTime;

double secondsFromFILETIME (FILETIME *f)
{
	return (f->dwLowDateTime   + pow(2.0,32)*f->dwHighDateTime  )/1.0E7;
}

#else

static double startCPUTime;
static double ticksPerSecond;
static struct tms tmsBuffer;
#endif

double currentCPUTime()
{
	double result;

#ifdef VISUALSTUDIO
	HANDLE hHandle;

	FILETIME creation_time, exit_time, kernel_time, user_time;

	hHandle = GetCurrentProcess();

	GetProcessTimes (hHandle, &creation_time, &exit_time, &kernel_time, &user_time);

	result = secondsFromFILETIME (&kernel_time) + secondsFromFILETIME (&user_time);

#else
	{
		times (&tmsBuffer);
		result = (tmsBuffer.tms_utime + tmsBuffer.tms_stime)/((double) ticksPerSecond);
	}
#endif
	return result*1000;
}


static double CPUTimeLimit;
static int limitSet = 0, nextDisplay;

void initCPUTime()
{
#ifdef VISUALSTUDIO

#else
	ticksPerSecond = (double) sysconf(_SC_CLK_TCK);
	nextDisplay = 1;
#endif

	startCPUTime = currentCPUTime();

}

double getCPUTimeSinceStart()
{

	double result;
	result = currentCPUTime() - startCPUTime;
	return result;

}


void setCPUTimeLimit(double limit)
{
	char message[100];
	CPUTimeLimit = limit;
	limitSet = 1;
	sprintf (message, "CPU limit set to %f", limit);
	//dimacsComment (message);
}

int CPUTimeExpired()
{
	if (getCPUTimeSinceStart() > nextDisplay)
	{
		printf ("\r%d(%d)", nextDisplay, (int)CPUTimeLimit);
		fflush (stdout);
		nextDisplay = getCPUTimeSinceStart()+1;
	}
	return limitSet && (getCPUTimeSinceStart() > CPUTimeLimit);
}
