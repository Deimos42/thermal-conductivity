import matplotlib.pyplot as plt
import threading
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import math
import os

# параметры по умолчанию
C1 = 8
C2 = 6
a = 7
b = 4
alf = 0.5
shema = np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]])
cell_count = 15 # число разбиений 
layer_count = 10
N = cell_count * 3
num = 0
resul = np.zeros((layer_count, N, N))
lock = threading.Lock()

def u(x,y,t):
    res = math.sin(math.pi*x)*math.sin(math.pi*y)*(math.exp(((-2*(math.pi)**2)*t))) + (x+y)*t
    return res

def u0(x,y): 
    res = 0
    return res

def f(x,y,t):
    res = C1 * t * math.exp((-((x - 7 * a / 8)**2 + (y - 3 * b / 4)**2)) * alf)
    return res

def ugr(x,y,t):
    res = 0
    return res

def ugr3(x,y,t): # g3
    res = C2 * t * math.sin(math.pi * x / a)
    return res

def MakeMatr():
    matr = np.zeros((N, N))
    if b > a:
        ia = a / b
        for i in range(N):
            matr[i][int(N * ia) - 1] = 1
            matr[i][0] = 1
        for j in range(int(N * ia)):
            matr[N - 1][j] = 1
            matr[0][j] = 3
        for i in range(1, int(N * ia) - 1):
            for j in range(1, N - 1):
                matr[j][i] = 2
    else:
        ib = b / a
        for i in range(N):
            matr[int(N * ib) - 1][i] = 1
            matr[0][i] = 3
        for j in range(int(N * ib)):
            matr[j][N - 1] = 1
            matr[j][0] = 1
        for i in range(1, int(N * ib) - 1):
            for j in range(1, N - 1):
                matr[i][j] = 2
    return matr

def Lx(resul, x, y, k):
    hx = 1 / (N - 1)
    return (resul[k][x-1][y]-2*resul[k][x][y]+resul[k][x+1][y])/(hx**2)

def Ly(resul, x, y, k):
    hy = 1/(N -1)
    return (resul[k][x][y-1]-2*resul[k][x][y]+resul[k][x][y+1])/(hy**2)

def Progonka(a, b, c, f):
    nn = len(f)
    y = np.zeros(nn)
    alfa = np.zeros(nn)
    betta = np.zeros(nn)
    y[0] = b[0]
    alfa[0] = - c[0] / y[0]
    betta[0] = f[0] / y[0]
    for i in range(1, nn - 1):
        y[i] = b[i] + a[i] * alfa[i - 1]
        alfa[i] = -c[i] / y[i]
        betta[i] = (f[i] - a[i] * betta[i - 1]) / y[i]
    y[nn - 1] = b[nn - 1] + a[nn - 1] * alfa[nn - 2]
    betta[nn - 1] = (f[nn - 1] - a[nn - 1] * betta[nn - 2]) / y[nn - 1]
    rez = np.zeros(nn)
    rez[nn - 1] = betta[nn - 1]
    for i in range(1, nn):
        rez[nn - 1 - i] = alfa[nn - i - 1] * rez[nn - i] + betta[nn - i - 1]
    return rez

def MakeLayer():
    global num
    global resul
    matr = MakeMatr()
    layer = np.zeros((N, N))
    layer_prom = np.zeros((1,N, N))
    for i in range(N):
        k = 0.5 / (layer_count-1)
        hx = 1 / (N -1)
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        for ig in range(N):
            if matr[ig][i] == 2:
                b[ig]=1 + 2 * k / hx**2
            else:
                b[ig] = 1

        for iv in range(0, N - 1):
            if matr[iv][i] == 2:
                c[iv] = - k / hx**2

        for inn in range(1,N):
            if matr[inn][i] == 2:
                a[inn] = - k / hx**2

        fr=np.zeros(N)
        for ib in range(N):
            if matr[ib][i] == 2:
                fr[ib] = k * f((num - 0.5)/(layer_count-1), ib / (N),i / (N)) + k * Ly(resul, ib, i, num - 1) + resul[num - 1][ib][i]
            elif matr[ib][i] == 1:
                fr[ib] = ugr(ib / N,i / N,(num - 0.5) / (layer_count - 1))
            elif matr[ib][i] == 3:
                fr[ib] = ugr3(ib / N,i / N,(num - 0.5) / (layer_count - 1))
        layer_prom[0][i] = Progonka(a, b, c, fr)

    layer_prom[0] = layer_prom[0].transpose()
    for i in range(N):
        k = 0.5 / (layer_count - 1)
        hy = 1 / (N - 1)
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        for ig in range(N):
            if matr[i][ig] == 2:
                b[ig] = 1 + 2 * k / hy ** 2
            else:
                b[ig] = 1
        for iv in range(0, N - 1):
            if matr[i][iv] == 2:
                c[iv] = - k / hy ** 2
        for inn in range(1, N):
            if matr[i][inn] == 2:
                a[inn] = - k / hy ** 2
        fr = np.zeros(N)
        for ib in range(N):
            if matr[i][ib] == 2:
                fr[ib] = k * f((num) / (layer_count - 1), i / N, ib / N) + k * Lx(layer_prom, i, ib, 0) + layer_prom[0][i][ib]
            elif matr[i][ib] == 1:
                fr[ib] = ugr(i / N, ib / N, num / (layer_count - 1))
            elif matr[i][ib] == 3:
                fr[ib] = ugr3(i / N, ib / N, num / (layer_count - 1))
        layer[i] = Progonka(a, b, c, fr)

    return layer

def Resh():
    global num
    global resul
    lock.acquire()
    matr = MakeMatr()
    layer = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if matr[i][j] == 1:
                layer[i][j] = ugr(i / N, j / N, 0)
            elif matr[i][j] == 2:
                layer[i][j] = u0(i / N, j / N)
            elif matr[i][j] == 3:
                layer[i][j] = ugr3(i / N, j / N, 0)
    resul[0] = layer
    lock.release()
    time.sleep(0.1)
    num = 1
    while num < layer_count:
        lock.acquire()
        resul[num] = MakeLayer()
        lock.release()
        time.sleep(0.1)
        num += 1

def DrawLayer(fig, lnum):
    global num
    i = int(math.sqrt(layer_count * 2))
    j = int(layer_count * 2 / i)
    i = i+(layer_count * 2 % i)
    ax = fig.add_subplot(1, 1, 1, projection = '3d')
    xval = np.linspace(0, 1, N)
    yval = np.linspace(0, 1, N)
    x, y = np.meshgrid(xval, yval)
    z = resul[lnum]
    surf = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = 'inferno')

def Draw(resul):
    lnum = 0
    fig = plt.figure(figsize=(6,6))
    while True:
        lock.acquire()
        global num
        if (lnum <= num):
            print("Слой: ", lnum)
            DrawLayer(fig, lnum)
            plt.draw()
            plt.pause(0.001)
            lnum = lnum + 1
        if lnum == layer_count:
            break
        lock.release()
        time.sleep(0.1)
    plt.draw()
    plt.show()

def SaveInFile(filename):
    with open(filename, 'w') as file:
        file.write(C1.__str__()+'\n')
        file.write(C2.__str__()+'\n')
        file.write(a.__str__()+'\n')
        file.write(b.__str__()+'\n')
        file.write(alf.__str__()+'\n')

def LoadFromFile(filename):
    with open(filename) as file:
        C1 = file.readline().strip('\n')
        C2 = file.readline().strip('\n')
        a = file.readline().strip('\n')
        b = file.readline().strip('\n')
        alf = file.readline().strip('\n')


# начальные параметры заданы в начале файла, мы можем как запустить программу без изменений
# либо изменить некоторые или все параметры
vvod = - 1
while vvod != 5:
    print("0 - Запуск")
    print("1 - Сохранение в файл")
    print("2 - Загрузка из файла")
    print("3 - Ввод параметров задачи")
    print("4 - Ввод количества слоев по времени")
    print("5 - Закрыть")

    vvod = int(input().strip('\n'))

    if vvod == 0:
        th1 = threading.Thread(target=Resh)
        th2 = threading.Thread(target=Draw, args=(resul,))
        th1.start()
        th2.start()
        th1.join()
        th2.join()
        break
    elif vvod == 1:
        print("Введите имя файла: ")
        filename = input().strip('\n')
        SaveInFile(filename)
        print("Сохранено")
    elif vvod == 2:
        print("Введите имя файла: ")
        filename = input().strip('\n')
        LoadFromFile(filename)
        print("Загружено")
    elif vvod == 3:
        С1 = float(input("С1 = ").strip('\n'))
        С2 = float(input("С2 = ").strip('\n'))
        a = float(input("a = ").strip('\n'))
        b = float(input("b = ").strip('\n'))
        alf = float(input("alfa = ").strip('\n'))
    elif vvod == 4:
        layer_count = float(input("Количество слоев по времени = ").strip('\n'))
    elif vvod == 5:
        pass
    else:
        print('Неправильно введённая команда')
        os.system("cls")













