#include <windows.h>

#ifndef hr_timer
#define hr_timer

typedef struct {
    double start;
    double stop;
} stopWatch;

int initialiseTimer( stopWatch *timer );
void startTimer( stopWatch *timer);
void stopTimer( stopWatch *timer);
double getElapsedTime( stopWatch *timer);
double getElapsedTimeInMilli( stopWatch *timer);
double getElapsedTimeInMicro( stopWatch *timer);
double getElapsedTimeInNano( stopWatch *timer);

#endif