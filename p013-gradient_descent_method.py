#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# y = x^4 - x^3
def f1(x):
    return x**4 - x**3

# y = x^4 - x^3の導関数。yは3/4で最小値を取る
def f1_prime(x):
    return 4 * x**3 - 3 * x**2

# 最急降下法
def gradient_descent_method(f_prime, init_x):
    learning_rate = 1.0
    eps = 1e-10

    x = init_x
    x_history = [init_x]
    iteration_max = 1000

    # 収束するか最大試行回数に達するまで
    for i in range(iteration_max):

        print(i+1, ":", x)
        # 最急上昇法の場合は-を+にする
        x_new = x - learning_rate * f_prime(x)

        # 収束条件を満たせば終了
        if abs(x - x_new) < eps:
            break

        x = x_new
        x_history.append(x)

    return (x, np.array(x_history))

def main():        
    ans, x_history = gradient_descent_method(f1_prime, 0.1) # 左側から
    
    # グラフを描く
    x = np.arange(-0.3, 1.0, 0.01)
    y = f1(x)
    plt.plot(x, y)
    
    # 移動した点を表示
    plt.plot(x_history, f1(x_history), 'o')
    
    # 点同士を線で結ぶ
    for i in range(len(x_history)-1):
        x1 = x_history[i]
        y1 = f1(x_history[i])
        x2 = x_history[i+1]
        y2 = f1(x_history[i+1])
        plt.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)
    plt.show()    

main()


