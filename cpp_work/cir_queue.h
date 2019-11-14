/*
	Name:cir_queue.h
	Copyright:xiong35
	Author:xiong35
	Date: 14/11/19 20:03
	Description:创建TYPE类型的循环队列
*/

#ifndef CIR_QUEUE_H_
#define CIR_QUEUE_H_

#include<vector>
using std::vector;
#define TYPE int

class MyCircularQueue {
private:
	vector<TYPE> queue;
	int head;
	int rear;
	int length;
	int element;
public:
	/** Initialize your data structure here. Set the size of the queue to be k. */
	MyCircularQueue(int k)
	{
		queue.resize(k);
		head = 0;
		rear = 0;
		length = k;
		element = 0;
	}

	/** Insert an element into the circular queue. Return true if the operation is successful. */
	bool enQueue(TYPE value)
	{
		if (isFull())
		{
			return false;
		}
		else
		{
			element++;
			queue[rear] = value;
			rear++;
			rear %= length;
			return true;
		}
	}

	/** Delete an element from the circular queue. Return true if the operation is successful. */
	bool deQueue()
	{
		if (isEmpty())
		{
			return false;
		}
		else
		{
			head++;
			head %= length;
			element--;
			return true;
		}
	}

	/** Get the front item from the queue. */
	TYPE Front()
	{
		if (isEmpty())
			return 0;
		else
			return queue[head];
	}

	/** Get the last item from the queue. */
	TYPE Rear()
	{
		if (isEmpty())
			return 0;
		if (rear == 0)
			return queue[length - 1];
		else
			return queue[rear - 1];
	}

	/** Checks whether the circular queue is empty or not. */
	bool isEmpty()
	{
		return !element;
	}

	/** Checks whether the circular queue is full or not. */
	bool isFull()
	{
		return (element == length);
	}
};
#endif
