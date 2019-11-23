#include"myQS.h"
//compare(a,b) : (a is in the front of b) => ture
void myQS(void* arr, int num, int size, bool (*compare)(void* pa, void* pb))
{
	//num == 1; return;
	if (num == 1)
	{
		return ;
	}
	//take the first as model
	//while 2 pts havent met
	void* pleft, * pright;
	pleft = arr;
	pright = (void*)((char*)arr + (num - 1) * size);
	while (pleft != pright)
	{
		//r2l find one
		while (compare(arr, pright) && pleft != pright)
			//compare(a,b) : (a is in the front of b) => ture
			//keep finding, if *pright is latter than model
		{
			pright = (void*)((char*)pright - size);
		}
		//l2r find  another
		while (compare(pleft,arr) && pleft != pright)
			//compare(a,b) : (a is in the front of b) => ture
			//keep finding, if *pleft is former than model
		{
			pleft = (void*)((char*)pleft + size);
		}
		//swap
		mySwap(pleft, pright,size);
	}
	//swap with the first one
	mySwap(pleft, arr, size);
	//calculate the num of elements in lhs, rhs.
	int n_lhs = ((char*)pleft - (char*)arr)/size;
	int n_rhs = num - n_lhs - 1;
	//QS lhs
	if (n_lhs)
	{
		myQS(arr, n_lhs, size, compare);
	}
	//QS rhs
	if (n_rhs)
	{
		myQS((void*)((char*)pleft + size), n_rhs, size, compare);
	}
	return;
}
int mySwap(void* lhs, void* rhs, int sz)
{
	void* temp = malloc(sz);
	if (!temp)
		return -1;

	memcpy(temp, lhs, sz);
	memcpy(lhs, rhs, sz);
	memcpy(rhs, temp, sz);

	free(temp);

	return 0;
}
