#include"stack.h"
#include <iostream>
using std::cin;
using std::cout;
struct RPNstack sorted;
struct RPNstack reversed;
struct RPNstack sorted_rhs;
extern int x_flag;

int main()
{
	char input_[100];
	cin.getline(input_, 100);
	int current;
	current = buildRPN(input_, sorted);
	if (x_flag)
	{
		buildRPN(input_, sorted_rhs, current);
		float x;
		x = solve(sorted,sorted_rhs);
		cout << "x = " << x << std::endl;
	}
	else
	{
		RPN(sorted, reversed);
		cout << "\nthe answer is " << solveRPN(reversed) << std::endl;
	}

	return 0;
}
