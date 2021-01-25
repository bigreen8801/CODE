import math

# 점1 x,y 와 점2 x,y의 거리 구하는 공식
def point_dist(x1,y1,x2,y2):
    
    a = x2 - x1   # 선 a의 길이
    b = y2 - y1    # 선 b의 길이
    
    dist = math.sqrt((a * a) + (b * b))    # (a * a) + (b * b)의 제곱근을 구함
    return dist

# input tree points
x1 = 1811
y1 = 1572

x2 = 1397
y2 = 274

x3 = 2628
y3 = 285

def get_trilateration(x1,y1,x2,y2,x3,y3,target_x,traget_y) : 
    # make the points in a 2d tuple if you want to use static points later
    R1 = (x1,y1)
    R2 = (x2,y2)
    R3 = (x3,y3)

    # you have to introduce the distances 
    d1 = point_dist(x1,y1, target_x, traget_y)
    d2 = point_dist(x2,y2, target_x, traget_y)
    d3 = point_dist(x3,y3, target_x, traget_y)

    # if d1 ,d2 and d3 in known
    # calculate A ,B and C coifficents
    A = R1[0]**2 + R1[1]**2 - d1**2
    B = R2[0]**2 + R2[1]**2 - d2**2
    C = R3[0]**2 + R3[1]**2 - d3**2

    X32 = R3[0] - R2[0]
    X13 = R1[0] - R3[0]
    X21 = R2[0] - R1[0]

    Y32 = R3[1] - R2[1]
    Y13 = R1[1] - R3[1]
    Y21 = R2[1] - R1[1]

    x = (A * Y32 + B * Y13 + C * Y21)/(2.0*(R1[0]*Y32 + R2[0]*Y13 + R3[0]*Y21))
    y = (A * X32 + B * X13 + C * X21)/(2.0*(R1[1]*X32 + R2[1]*X13 + R3[1]*X21))

    # 반올림 수행후 인트로 변환
    x = int(round(x))
    y = int(round(y))

    # # prompt the result
    # print('x, y : ')
    # print(x, y)
    return x, y