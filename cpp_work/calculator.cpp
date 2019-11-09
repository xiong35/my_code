#include"stack.h"
#include <iostream>
using std::cin;
using std::cout;
struct RPNstack sorted;
struct RPNstack reversed;

int main()
{
	char input_[100];
	cin.getline(input_, 100);
	buildRPN(input_, sorted);
	RPN(sorted, reversed);

	cout << "\nthe answer is " << solveRPN(reversed) << std::endl;
	return 0;
}
