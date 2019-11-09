#include "stack.h"
#include <stdlib.h>
#include <iostream>

int x_flag;

//NumStack methods
float NumStack::pop()//need to check pt value
{
	return numlist[pt--];
}

void NumStack::push(float n)
{
	numlist[++pt] = n;
}

//ChStack methods
char ChStack::pop()//need to check pt value
{
	return chlist[pt--];
}

void ChStack::push(char n)
{
	chlist[++pt] = n;
}

int buildRPN(char (&input)[100],RPNstack &lhs,int current)
{
	int i = current;
	float temp = 0;
	int num_flag = 0;
	int dot_flag = 0;
	float dev = 0.1;

	for(; input[i] != '\0';)//=='\n' may cause bug(cin)
	{
		if (IS_NUM(input[i]))//ok
		{
			if (!dot_flag)
			{
				num_flag = 1;
				temp *= 10;
				temp += input[i++] - '0';
			}
			else
			{
				temp += dev * (input[i++] - '0');
				dev /= 10;
			}
		}
		else if (IS_L_BRACKET(input[i]) || IS_X(input[i]))//ok
		{
			if (IS_X(input[i]))
			{
				x_flag = 1;
			}
			if (num_flag)
			{
				lhs.ns.push(temp);//push num in stack
				lhs.cs.push(cPH);
				lhs.count++;

				lhs.cs.push('*');//push the omitted *
				lhs.ns.push(nPH);
				lhs.count++;

				lhs.cs.push(input[i++]);//push bracket
				lhs.ns.push(nPH);
				lhs.count++;

				num_flag = 0;//reset
				temp = 0;
				dev = 0.1;
				dot_flag = 0;
			}
			else
			{
				lhs.cs.push(input[i++]);
				lhs.ns.push(nPH);
				lhs.count++;
			}
		}
		else if(IS_SPACE(input[i]))//ok
		{
			i++;
			continue;
		}
		else if(IS_DOT(input[i]))//ok
		{
			if(dot_flag)
			{
				std::cout<<"invalid input:using double .\n";
				exit(-1);
			}
			dot_flag = 1;
			i++;
		}
		else if(IS_OPERATOR(input[i])||IS_R_BRACKET(input[i]))//ok
		{
			if (lhs.cs.chlist[lhs.count - 1] == ')')
			{
				lhs.cs.push(input[i++]);//push operator
				lhs.ns.push(nPH);
				lhs.count++;
			}
			else
			{
				if (!num_flag )//bug : can't input(-5+6)
				{
					if (lhs.cs.chlist[lhs.count-1] != 'x')
					{
						std::cout << "invalid input: ecpect a number befor" << input[i] << "\n";
						exit(-2);
					}
				}
				if (lhs.cs.chlist[lhs.count - 1] != 'x')
				{
					lhs.ns.push(temp);//push num in stack
					lhs.cs.push(cPH);
					lhs.count++;
				}

				lhs.cs.push(input[i++]);//push operator
				lhs.ns.push(nPH);
				lhs.count++;

				num_flag = 0;//reset
				temp = 0;
				dev = 0.1;
				dot_flag = 0;
			}

		}
		else if(input[i] == '=')
		{
			if (num_flag)
			{
				lhs.ns.push(temp);//push num in stack
				lhs.cs.push(cPH);
				lhs.count++;
			}
			current = i + 1;
			return current;
		}
		else
		{
			std::cout << "invalid input\n";
			exit(-100);
		}
	}
	if(num_flag)
	{
		lhs.ns.push(temp);//push num in stack
		lhs.cs.push(cPH);
		lhs.count++;
	}
	return current;
}

void RPN(RPNstack& sorted, RPNstack& reversed)
{
	ChStack temp;
	int i = 0;
	while (i < sorted.count)
	{
		if (sorted.cs.chlist[i] == cPH)//num
		{
			reversed.cs.push(cPH);
			reversed.ns.push(sorted.ns.numlist[i]);
			reversed.count++;		}
		else//char
		{
			if (temp.pt == -1 && IS_OPERATOR(sorted.cs.chlist[i]))
			{
				temp.push(sorted.cs.chlist[i]);
			}
			else if (IS_L_BRACKET(sorted.cs.chlist[i]))
			{
				temp.push(sorted.cs.chlist[i]);
			}
			else if (IS_R_BRACKET(sorted.cs.chlist[i]))
			{
				while (temp.chlist[temp.pt]!='(')
				{
					reversed.cs.push(temp.pop());
					reversed.ns.push(nPH);
					reversed.count++;
				}
				temp.pop();
			}
			else if (IS_OPERATOR(sorted.cs.chlist[i]))
			{
				if (sorted.cs.chlist[i] == '/' || sorted.cs.chlist[i] == '*')
				{
					while (temp.chlist[temp.pt] != '+'&&
						temp.chlist[temp.pt] != '-'&& temp.chlist[temp.pt] != '(' && temp.pt>-1)
					{
						reversed.cs.push(temp.pop());
						reversed.ns.push(nPH);
						reversed.count++;
					}
					temp.push(sorted.cs.chlist[i]);
				}
				else if (sorted.cs.chlist[i] == '-' || sorted.cs.chlist[i] == '+')
				{
					while (temp.chlist[temp.pt] != '(' && temp.pt>-1)
					{
						reversed.cs.push(temp.pop());
						reversed.ns.push(nPH);
						reversed.count++;
					}
					temp.push(sorted.cs.chlist[i]);
				}
			}
		}
		i++;
	}
	while (temp.pt != -1)
	{
		reversed.cs.push(temp.pop());
		reversed.ns.push(nPH);
		reversed.count++;
	}
	return;
}

float solveRPN(RPNstack & RS)
{
	int i = 0;
	float first, last;
	NumStack temp;
	while (i < RS.count)
	{
		if (RS.cs.chlist[i] == cPH)//num
		{
			temp.push(RS.ns.numlist[i]);
		}
		else//char
		{
			switch (RS.cs.chlist[i])
			{
			case '+':
				temp.push(temp.pop() + temp.pop());
				break;
			case '-':
				last = temp.pop();
				first = temp.pop();
				temp.push(first - last);
				break;
			case '*':
				temp.push(temp.pop() * temp.pop());
				break;
			case '/':
				last = temp.pop();
				first = temp.pop();
				temp.push(first / last);
				break;
			}
		}
		i++;
	}
	return temp.numlist[0];
}

float Subs(RPNstack RS, int n)
{
	int i = 0;
	while (i < RS.count)
	{
		if (RS.cs.chlist[i] == 'x')
		{
			RS.cs.chlist[i] = cPH;
			RS.ns.numlist[i] = n;
		}
		i++;
	}
	RPNstack temp;
	temp.count = 0;
	RPN(RS, temp);
	return solveRPN(temp);
}

float solve(RPNstack& lhs, RPNstack& rhs)
{
	float y1, y2;
	y1 = Subs(lhs, 0) - Subs(rhs, 0);
	y2 = Subs(lhs, 20) - Subs(rhs, 20);
	if (EQUAL(y1,y2))
	{
		return nPH;
	}
	y1 = (y2 - y1) / 20;
	y2 = y2 / y1;
	return (20 - y2);
}
