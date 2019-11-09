/*
	Name:stack.h
	Copyright:xiong35
	Author:xiong35
	Date: 08/11/19 23:18
	Description:define stack(pop,push)(num,char)
*/

#ifndef STACK_H
#define STACK_H

#define ABS(a,b) ((a>b)?(a-b):(b-a))
#define EQUAL(a,b) (ABS(a,b)<0.0001)

#define nPH 3.15151//place holder in numlist
#define cPH '@'//place holder in charlist

#define IS_NUM(x) (x <= '9' && x >= '0')
#define IS_SPACE(x) (x == ' ')
#define IS_L_BRACKET(x) ( x == '(' )
#define IS_R_BRACKET(x) ( x == ')' )
#define IS_X(a) ( a == 'x'||a == 'X' )
#define IS_DOT(x) ( x == '.' )
#define IS_OPERATOR(x) ( x == '+'||x == '-'||x == '*'||x == '/' )


class NumStack
{
public:
	NumStack(){ pt = -1; }
	float pop();
	void push(float n);
	float numlist[100];
	int pt;
};

class ChStack
{
public:
	ChStack(){ pt = -1; }
	char pop();
	void push(char n);
	char chlist[100];
	int pt;
};

struct RPNstack
{
	NumStack ns;
	ChStack cs;
	int count;
};

int buildRPN(char (&input)[100],RPNstack &lhs,int current = 0);
void RPN(RPNstack& sorted, RPNstack& reversed);
float solveRPN(RPNstack & RS);
float Subs(RPNstack RS, int n);
float solve(RPNstack& lhs, RPNstack& rhs);

#endif

