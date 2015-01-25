#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 参考：http://matplotlib.org/examples/mplot3d/surface3d_demo.html


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

# f1(x,y) = x^4 + y^4 - 2* x^2 * y
def f1(x, y):
    return x**4 + y**3 - 2*(x**2)*(y)

# ∇f = [∂f/∂x, ∂f/∂y]^T
def gradient_f1(xy):
    x = xy[0]
    y = xy[1]
    
    # (xの偏微分, yの偏微分)
    return np.array([4 * x**3 - 4*x*y, 4 * y**3 - 2 * x**2]);

# 最急降下法
# init_pos = 初期位置. e.g. (x, y)
def gradient_descent_method(gradient_f, init_pos, learning_rate):
    eps = 1e-10

    # 計算しやすいようnumpyのarrayとする
    init_pos = np.array(init_pos)
    pos = init_pos
    pos_history = [init_pos]
    iteration_max = 1000


    # 収束するか最大試行回数に達するまで
    for i in range(iteration_max):

        print(i+1, ":", pos)

        # 最急上昇法の場合は-を+にする
        pos_new = pos - learning_rate * gradient_f(pos)

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

    return (pos, np.array(pos_history))


def show_3d_graph(f):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # X, Yを一定範囲にグリッド状に取り、Z = f(X,Y)を求める
    X = np.arange(-1.0, 1.0, 0.01)
    Y = np.arange(-1.0, 1.0, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = f1(X, Y)

    # 滑らかな3次元プロットを行う
    # rstride, cstride: X, Yを何個飛ばしに見るか。両方を10にすると99%のデータを捨てることになる(多分)
    # cmap: カラーマップ。配色方法の指定
    surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Z軸に数値を表示する範囲を指定
    #ax.set_zlim(-1.01, 1.01)
    # Z軸に数値をn個等間隔に表示
    ax.zaxis.set_major_locator(LinearLocator(10))
    # Z軸に表示する数値のフォーマット
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # グラフ右端に凡例を表示
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    
def main():

    show_3d_graph(f1)
    a = 1/0
    
    
    learning_rates = [ 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 ]

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate)

        # subplotの場所を指定
        plt.subplot(3, 2, (i+1)) # 3行2列の意味

        # グラフのタイトル
        plt.title("learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history)))
        
        # # グラフを描く
        # x = np.arange(-0.3, 1.0, 0.01)
        # y = f1(x)
        # plt.plot(x, y)
        
        # # 移動した点を表示
        # plt.plot(x_history, f1(x_history), 'o')
        
        # # 点同士を線で結ぶ
        # for i in range(len(x_history)-1):
        #     x1 = x_history[i]
        #     y1 = f1(x_history[i])
        #     x2 = x_history[i+1]
        #     y2 = f1(x_history[i+1])
        #     plt.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

    # # タイトルが重ならないようにする
    # plt.tight_layout()

    # # 画像保存用にfigを取り出す
    # fig = plt.gcf()
    # fig.set_size_inches(50.0, 60.0)

    # # 画像を表示
    # plt.show()

    # # 画像を保存
    # fig.savefig('gradient_descent_method.png')

main()


