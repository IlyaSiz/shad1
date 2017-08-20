# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot as plt

a = 10


# Функция цены от объёма товара
def f(q):
    return a*np.e**(-0.5*q**2)


 #Функция издержек от объёма товара
def h(q):
    return np.sqrt(q)
plt.figure()
q= np.arange(0, 2.01, 0.1)#Массив значений аргумента
plt.title(r'$y=f(q)$') #Заголовок в формате TeX
plt.ylabel(r'$f(q)$') #Метка по оси y в формате TeX
plt.xlabel(r'$q$') #Метка по оси x в формате TeX
plt.grid(True) #Сетка
plt.plot(q,f(q)) #Построение графика
plt.figure()
plt.title(r'$y=h(q)$') #Заголовок в формате TeX
plt.ylabel(r'$h(q)$') #Метка по оси y в формате TeX
plt.xlabel(r'$q$') #Метка по оси x в формате TeX
plt.grid(True) #Сетка
plt.plot(q,h(q)) #Построение графика
plt.show() #Показать график