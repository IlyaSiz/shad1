# -*- coding: utf8 -*-
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Параметры модели
xmax = 1
ymax = 1
N = 20
h1 = xmax/N
h2 = ymax/N


# Функция прибыли первой фирмы
def L1(x, y):
    return 10*x*np.e**(-0.5*(x+y)**2)-np.sqrt(x)


# Функция прибыли второй фирмы
def L2(x, y):
    return 10*y*np.e**(-0.5*(x+y)**2)-np.sqrt(y)


# Функция определения индекса списка
def fi(a, b):
    for w in range(0, len(a)+1):
        if a[w] == b:
            return w

# Создания матрицы прибылей
L = np.zeros([N+1, N+1])
for i in range(0, N+1):
    for j in range(0, N+1):
        L[i, j] = L1(i*h1, j*h2)

A = []
e = []
rows, cols = L.shape

for i in range(rows):
    e = []
    for j in range(cols):
        e.append(L[i, j])
    A.append(min(e))

# Итоги работы первой фирмы
a = max(A)
ia = fi(A, a)
xa = ia*h1

B = []
e = []
rows, cols = L.shape
for j in range(cols):
    e = []
    for i in range(rows):
        e.append(L[i, j])
    B.append(max(e))

# Итоги работы второй фирмы
b = min(B)
ib = fi(B, b)
yb = ib*h2

p1 = round(L1(xa, yb), 3)
print("Прибыль первой фирмы -" + str(p1))
p2 = round(L2(xa, yb), 3)
print("Прибыль второй фирмы -" + str(p2))


# Поверхность прибыль- z, объёмы закупок - x,y для первой фирмы
def makeData_L1 ():
    x = [h1*i for i in np.arange(0, N+1)]
    y = [h2*i for i in np.arange(0, N+1)]
    x, y = np.meshgrid(x, y)
    z = []
    for i in range(0, N+1):
        z.append(L1(x[i], y[i]))
    return x, y, z


fig = pylab.figure()
axes = Axes3D(fig)
x, y, z = makeData_L1()
axes.plot_surface(x, y, z)


# Поверхность прибыль- z объёмы закупок- x,y для второй фирмы
def makeData_L2 ():
    x = [h1*i for i in np.arange(0, N+1)]
    y = [h2*i for i in np.arange(0, N+1)]
    x, y = np.meshgrid(x, y)
    z =[]
    for i in range(0, N+1):
        z.append(L2(x[i], y[i]))
    return x, y, z


fig = pylab.figure()
axes = Axes3D(fig)
x, y, z = makeData_L2()
axes.plot_surface(x, y, z)
pylab.show()