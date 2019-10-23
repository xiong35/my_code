def move(n,ori,mid,des):
    if n == 1:
        print('move 1 from '+ori+' to '+des)
    else:
        move(n-1,ori,des,mid)
        print('move '+str(n)+' from '+ori+' to '+des)
        move(n-1,mid,ori,des)

n = int(input())
move(n,'a','b','c')
