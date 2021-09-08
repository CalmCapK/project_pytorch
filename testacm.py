while True:
    try:
        #m = map(int, input().strip().split())
        #m = list(map(lambda x:int(x), input().split()))
        a, b, x, y = list(map(lambda x:int(x), input().split()))
        cnt = (a+b)//(x+y)
        ans = 0
        for i in range(cnt+1):
            if(i*x<=a and i*y<=b):
                j = min((a-i*x)//y, (b-i*y)//x)
                ans = max(ans, i+j)
        print(ans)
    except EOFError:
        break