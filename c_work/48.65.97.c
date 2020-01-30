#include<stdio.h>

int main(void)
{

	return 0;
}

#include<stdlib.h>

#pragma warning(disable:4996)

// #include<conio.h>

#define ABS(a,b) (((a)>(b))?((a)-(b)):((b)-(a)))
#define EQUAL(a,b) (ABS(a,b)<0.0001)
#define SWAP(a,b) ({(a) = (a)^(b);(b) = (a)^(b);(a) = (a)^(b);})
#define SWAP(a,b) ({typeof(a) temp__;temp__=(a);(a)=(b);(b)=temp__;})
#define MAX(x,y) (((x)>(y))?(x):(y))

int coun__t = 0;
//printf("\ndebug %d",coun__t++);

#define CTRL(c) (*#c)