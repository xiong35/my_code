#pragma once
/*
	Name:myQS
	Copyright:xiong35
	Author:xiong35
	Date: 23/11/19 11:44
	Description: a qick sort model that fits all types of input.
*/

#include<stdbool.h>

void myQS(void* arr, int num, int size, bool (*compare)(void* pa, void* pb));
//compare(a,b) : (a is in the front of b) => ture

int mySwap(void* lhs, void* rhs, int sz);

