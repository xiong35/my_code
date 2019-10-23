#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<windows.h>

typedef struct body{
	int x;
	int y;
	struct body * pnext; 
}BODY,*PBODY;

int begin(void);
int nxstep(PBODY phead);
void food(void);
void get_dir(void);
PBODY build_body(void);
void move(PBODY phead,int n);
int if_die(int x,int y);
void over(PBODY phead);

char direction = 'd';
int xnext,ynext,score = 0;
char map[40][40]={
"# # # # # # # # # # # # # # # #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"#                             #",
"# # # # # # # # # # # # # # # #",
};

int main(void)
{
	int speed;
	PBODY phead =NULL,pfirst=NULL;
	speed = begin();
	phead = build_body();
	food();
	while (1)
	{
		pfirst = phead->pnext;
		int n = nxstep(pfirst);
		if(n == 1) break;
		if(n == 0||n==7) move(phead,n);
		int o,m;
		system("cls");
		for(m=0 ;m<=17;m++)
		{
			for(o=0;o<=31;o++)
			printf("%c",map[m][o]);
			printf("\n");
		}
		Sleep(720/speed);
		get_dir();
	}
	over(phead);
	return 0;
}
PBODY build_body(void)
{
	PBODY phead_ = NULL;
	PBODY phead = (PBODY)malloc(sizeof(BODY));
	phead_ = phead;
	phead->pnext = (PBODY)malloc(sizeof(BODY));
	phead = phead->pnext ;
	phead->x = 8;
	phead->y = 9;
	phead->pnext = (PBODY)malloc(sizeof(BODY));
	phead = phead->pnext ;
	phead->x = 10;
	phead->y = 9;
	phead->pnext = NULL;
	return phead_;
}

int if_die(int x,int y)
{
	if(map[y][x] == '#') return 1;
	if(map[y][x] == 'o') return 1;
	if(map[y][x] == ' ') return 0;
	if(map[y][x] == '$') return 7;
}

int begin(void)
{
	int n;
	printf("****************************\n\n");
	printf("       ��ӭ����̰����\n\n");
	printf("     ������a,b,cѡ���Ѷ�\n\n");
	printf("  a.����   b.����   c.����\n\n");
	printf("      #ʹ��wasd������λ#\n\n");
	printf("****************************\n");
	char ch = getch();
	if(ch == 'a')
		n = 12;
	else if(ch == 'b')
		n = 8;
	else if(ch == 'c')
		n = 3;
	else
	{
		system("cls");
		printf("��������abc�����˸�ɶ����\n");
		Sleep(1000);
		printf("���ˣ�����Ϸ���ʺ���\n");
		Sleep(1000);
		printf("����Ⱥ��"); 
		Sleep(1000);
		exit(-1) ;
	}
	system("cls");
	printf("\n\n\n      �����������ʼ");
	getch();
	return n;
}

void food(void)
{
	int x,y;
	srand((unsigned)time(NULL));
	x = (rand()*7%31)/2;
	x *= 2;
	srand((unsigned)time(NULL));
	y = rand()*4%15+1;
	while(if_die(x,y) == 1)
	{
		x = (rand()*4%31)/2;
		x *= 2;
		srand((unsigned)time(NULL));
		y = rand()*4%15+1;
	}
	map[y][x] = '$';
	return;
}

void get_dir(void)
{
	extern char direction;
	char n;
	if(kbhit())
	{
		n = getch();
		if(n == 'w' && direction != 's') direction = 'w';
		else if(n == 'a'&& direction != 'd') direction = 'a';
		else if(n == 's'&& direction != 'w') direction = 's';
		else if(n == 'd'&& direction != 'a') direction = 'd';
		else return;
	}
	return;
}

void move(PBODY phead,int n)
{
	PBODY phr,pArr;
	phr = (PBODY)malloc(sizeof(BODY));
	phr->pnext = phead->pnext ;
	pArr = phead->pnext ;
	phead->pnext = phr;
	phr->x = xnext;
	phr->y = ynext;
	map[ynext][xnext]='@';
	while((pArr->pnext)->pnext != NULL)
	{
		map[pArr->y][pArr->x] = 'o';
		pArr = pArr->pnext;
	}
	if(n==0)
	{
	map[(pArr->pnext)->y][(pArr->pnext)->x] = ' ';
	free(pArr->pnext);
	pArr->pnext = NULL;
	}
	if(n==7)
	{
		pArr=pArr->pnext;
		map[pArr->y][pArr->x] = 'o';
		score++;
		food();
	}
	return;
}

int nxstep(PBODY pfirst)
{
	extern char direction;
	int x = pfirst->x ;
	int y = pfirst->y ;
	switch(direction)
	{
		case'w':
		{
			xnext = x;
			ynext = y-1;
		} break;
		case's':
		{
			xnext = x;
			ynext = y+1;
		} break;
		case'a':
		{
			xnext = x-2;
			ynext = y;
		} break;
		case'd':
		{
			xnext = x+2;
			ynext = y;
		} break;
	}
	int n;
	n = if_die(xnext,ynext);
	return n;
}

void over(PBODY phead)
{
	printf("\n*****��Ϸ���������ĵ÷�Ϊ%d*****",score);
	printf("\n            ��q�˳�"); 
	while(1)
	{
		char n = getch();
		if (n == 'q')
			exit(0);
	}
}
