#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# y = x^4 - x^3
def f1(x):
    return x**4 - x**3

# y = x^4 - x^3の導関数。yは3/4で最小値を取る
def f1_prime(x):
    return 4 * x**3 - 3 * x**2

# 最急降下法
def gradient_descent_method(f_prime, init_x):
    learning_rate = 0.1
    eps = 1e-10

    i = 1
    x = init_x
    while True:
        print(i, ":", x)
        # 最急上昇法の場合は-を+にする
        x_new = x - learning_rate * f_prime(x)

        # 収束条件
        if abs(x - x_new) < eps:
            return x

        x = x_new
        i += 1

print(gradient_descent_method(f1_prime, 0.1)) # 左側から


    
