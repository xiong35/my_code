#include<stdio.h>

void get_loc(char n,char pt[][20],char chess_);
void ai(char print[][20],char chess);
int check(char prt[][20]);

int main(void)
{
	printf("******��ӭ����������******\n");
	printf("      ��ѡ����Ϸģʽ\n");
	printf("    a. ����    b. ˫��\n");
	char mod,chess,choose;
	int x,y,boo = 1;
	while(1)  //ѡ��ģʽ 
	{
		mod = getch();
		system("cls");
		if(mod == 'a'){
			chess = 'O';
			printf("��ѡ����ģʽ���㽫ʹ��OΪ����\n");
			printf("���»س���ȷ��");
			getch();
			break;}
		else if(mod == 'b')
		{	
			while(1){
			printf("      ��ѡ��˫��ģʽ\n");
			printf("******������ѡ��ִ��******\n");
 			printf("    a.ѡ��X   b.ѡ��O");
    		choose = getch();
    		if((choose == 'a') || (choose == 'b')){
    			chess = (choose == 'a')? 'X':'O';
				break; }
    		else
    			{	system("cls");
					printf("    ������a��b��Сд��\n");
				}
			}
			break;
		}
		else {
				printf("   ����������a��b��ѡ��\n");
				printf("      ��ѡ����Ϸģʽ\n");
				printf("    a. ����    b. ˫��\n");
			}
	}
	system("cls");
	printf("������1~9��ȷ������λ��(��ͼ)\n");
	printf("     7|8|9\n");
	printf("     -----\n");
	printf("     4|5|6\n");
	printf("     -----\n");
	printf("     1|2|3\n");
	printf("�����������ʼ��Ϸ");
	getch();
	system("cls"); 
	
	char map[20][20]={
	" | | ",
	"-----",
	" | | ",
	"-----",
	" | | ",
	};
	int step = 0;	
	char loc;
	if(mod == 'a')
	{
	while(1)
	{
		int m,n;
		system("cls");
		for(m=0;m<=4;m++)  //��ӡ���� 
		{
			for(n=0;n<=4;n++)
			printf("%c",map[m][n]);
			printf("\n");
		}
		loc = getch();
		get_loc(loc,map,chess);
		if (check(map))
		{
			check(map);
			break;
		}
		chess = (chess == 'O')? 'X':'O';
		boo = 1;
		for(x=0;x<=4;x++)
		{
			for(y=0;y<=4;y++)
			if(map[x][y]==' ') boo = 0;
		}
		if(boo)
		{
			system("cls");
			printf("\n\n    ƽ�֣���\n\n");
			break;
		}	
		ai(map,chess);
		chess = (chess == 'O')? 'X':'O';
		if (check(map))
		{
			check(map);
			break;
		}
		system("cls");
		for(m=0;m<=4;m++)  //��ӡ���� 
		{
			for(n=0;n<=4;n++)
			printf("%c",map[m][n]);
			printf("\n");
		}
	}
	}
	if(mod =='b')   
	{
		while(1)
		{	int m,n;
			system("cls");
			for(m=0;m<=4;m++)  //��ӡ���� 
			{
				for(n=0;n<=4;n++)
				printf("%c",map[m][n]);
			printf("\n");
			}
			loc = getch();
			get_loc(loc,map,chess);
			chess = (chess == 'O')? 'X':'O';
			if (check(map))
			{
				check(map);
				break;
			}
			boo = 1;
			for(x=0;x<=4;x++)
			{
				for(y=0;y<=4;y++)
				if(map[x][y]==' ') boo = 0;
			}
			if(boo)
			{
				system("cls");
				printf("\n\n    ƽ�֣���\n\n");
				break;
			}
			
		}
	}
	return 0;
}
void get_loc(char n,char pt[][20],char chess_)  //�������� 
{	
	switch(n)
	{
		case'1':
			{
				if (pt[4][0] == ' ') 
				pt[4][0] = chess_;
				break;
			}
		case'2':
			{
				if (pt[4][2] == ' ') 
				pt[4][2] = chess_;
				break;
			}
			
		case'3':
			{
				if (pt[4][4] == ' ') 
				pt[4][4] = chess_;
				break;
			}	
			
		case'4':
			{
				if (pt[2][0] == ' ') 
				pt[2][0] = chess_;
				break;
			}						
			
		case'5':
			{
				if (pt[2][2] == ' ') 
				pt[2][2] = chess_;
				break;
			}					
			
		case'6':
			{
				if (pt[2][4] == ' ') 
				pt[2][4] = chess_;
				break;
			}		
			
		case'7':
			{
				if (pt[0][0] == ' ') 
				pt[0][0] = chess_;
				break;
				
			}
		case'8':
			{
				if (pt[0][2] == ' ') 
				pt[0][2] = chess_;
				break;
			}
			
		case'9':
			{
				if (pt[0][4] == ' ') 
				pt[0][4] = chess_;
				break;
			}	
	}
}
int check(char prt[][20])
{
	int m=0;
	for(;m<=4;m+=2)
	{
		if(prt[m][0]==prt[m][2]&&prt[m][0]==prt[m][4]&&prt[m][0]!=' ')
		{	
			system("cls");
			printf("%c һ����ʤ��",prt[m][0]);
			return 1;
		}
	}
	for(m=0;m<=4;m+=2)
	{
		if(prt[0][m]==prt[2][m]&&prt[0][m]==prt[4][m]&&prt[0][m]!=' ')
		{	
			system("cls");
			printf("%c һ����ʤ��",prt[0][m]);
			return 1;
		}	}
	if(prt[0][0]==prt[2][2]&&prt[0][0]==prt[4][4]&&prt[2][2]!=' ')
	{	
		system("cls");
		printf("%c һ����ʤ��",prt[0][0]);
		return 1;
	}	
	if(prt[4][0]==prt[2][2]&&prt[0][4]==prt[4][0]&&prt[2][2]!=' ')
	{	
		system("cls");
		printf("%c һ����ʤ��",prt[4][0]);
		return 1;
	}	
	
	return 0 ;
}
void ai(char print[][20],char chess)
{
	if(print[2][2] == ' ')
	{
		print[2][2] = chess;
		return;
	}
	else
	{
		int m,n;
		for(m=0;m<=4;m+=2)
		{
			if(print[m][0]==print[m][2]&&print[m][0]!=' '&&print[m][4]==' ') {print[m][4]=chess;return;}
			if(print[m][0]==print[m][4]&&print[m][0]!=' '&&print[m][2]==' ') {print[m][2]=chess;return;}
			if(print[m][4]==print[m][2]&&print[m][4]!=' '&&print[m][0]==' ') {print[m][0]=chess;return;}
		}
		for(m = 0;m<=4;m+=2)
		{
			if(print[0][m]==print[2][m]&&print[0][m]!=' '&&print[4][m]==' ') {print[4][m]=chess;return;}
			if(print[0][m]==print[4][m]&&print[0][m]!=' '&&print[2][m]==' ') {print[2][m]=chess;return;}
			if(print[4][m]==print[2][m]&&print[4][m]!=' '&&print[0][m]==' ') {print[0][m]=chess;return;}
		}
		if(print[0][0]==print[2][2]&&print[0][0]!=' '&&print[4][4]==' '){print[4][4]=chess;return;}
		if(print[4][4]==print[2][2]&&print[4][4]!=' '&&print[0][0]==' '){print[0][0]=chess;return;}		
		if(print[0][0]==print[4][4]&&print[0][0]!=' '&&print[2][2]==' '){print[2][2]=chess;return;}
		if(print[0][4]==print[2][2]&&print[0][4]!=' '&&print[4][0]==' '){print[4][0]=chess;return;}
		if(print[4][0]==print[2][2]&&print[4][0]!=' '&&print[0][4]==' '){print[0][4]=chess;return;}
		if(print[0][4]==print[4][0]&&print[0][4]!=' '&&print[2][2]==' '){print[2][2]=chess;return;}	
		
		if(print[0][2]==' '){print[0][2]=chess;return;}
		if(print[2][0]==' '){print[2][0]=chess;return;}
		if(print[2][4]==' '){print[2][4]=chess;return;}
		if(print[4][2]==' '){print[4][2]=chess;return;}
		
		else{
		for(m=0;m<=4;m+=2)
		{
			for(n=0;n<=4;n+=2)
			{
				if(print[m][n]==' '){print[m][n]=chess;return;}
			}
		}
		}
	}
}



