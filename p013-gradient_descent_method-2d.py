#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 参考：http://matplotlib.org/examples/mplot3d/surface3d_demo.html
#       http://stackoverflow.com/questions/7744697/how-to-show-two-figures-using-matplotlib


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

        # print(i+1, ":", pos)

        # 最急上昇法の場合は-を+にする
        pos_new = pos - learning_rate * gradient_f(pos)

        # 収束条件を満たせば終了
        # np.linalg.norm(): ユークリッド距離を計算する関数
        if abs(np.linalg.norm(pos - pos_new)) < eps:
            break

        pos = pos_new
        pos_history.append(pos)

    return (pos, np.array(pos_history))

# 値の高低を色で表現
def draw_color_map(fig):

    ax1 = fig.add_subplot(111)
    n = 256
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    X,Y = np.meshgrid(x, y)
    Z = f1(X, Y)

    pc = ax1.pcolor(X, Y, Z, cmap='RdBu')

    # グラフに凡例を表示
    # http://stackoverflow.com/questions/18874135/how-to-plot-pcolor-colorbar-in-a-different-subplot-matplotlib
    # http://stackoverflow.com/questions/13784201/matplotlib-2-subplots-1-colorbar
    fig.subplots_adjust(right=0.8) # 右端を空ける？
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7]) # カラーバー用のaxesを用意
    fig.colorbar(pc, cax=cbar_ax) # カラーバーを描画

def draw_contour(fig):
    # 等高線を描く
    ax1 = fig.add_subplot(111)
    n = 256
    x = np.linspace(-1.5, 1.5, n)
    y = np.linspace(-1.5, 1.5, n)
    X,Y = np.meshgrid(x, y)

    level_num = 20
    # 等高線で同じ高さとなるエリアを色分け
    ax1.contourf(X, Y, f1(X, Y), level_num, alpha=.75, cmap=plt.cm.hot)
    # 等高線を引く
    C = ax1.contour(X, Y, f1(X, Y), level_num, colors='black', linewidth=.5)
    ax1.clabel(C, inline=1, fontsize=10)
    ax1.set_title('contour')


def main():
    # カラーマップを表示
    fig1 = plt.figure(1)
    draw_color_map(fig1)
    
    # 等高線を表示
    fig2 = plt.figure(2)
    draw_contour(fig2)

    plt.show()
    
    
def aaa():
    
    input()

    learning_rates = [ 0.1, 0.2, 0.4, 0.6 ]

    # 収束する様子を表示するためのグラフ
    fig2 = plt.figure(2)

    for i, learning_rate in enumerate(learning_rates):
        ans, pos_history = gradient_descent_method(gradient_f1, (0.1, 0.1), learning_rate)

        # subplotの場所を指定
        plt.subplot(2, 2, (i+1)) # 2行2列の意味

        # グラフのタイトル
        plt.title("learning rate: " + str(learning_rate) + ", iteration: " + str(len(pos_history)))

        # 移動した点を表示
        for pos in pos_history:
            plt.plot(pos[0], pos[1], 'o')
        
        # 点同士を線で結ぶ
        for i in range(len(pos_history)-1):
            x1 = pos_history[i][0]
            y1 = pos_history[i][1]
            x2 = pos_history[i+1][0]
            y2 = pos_history[i+1][1]
            plt.plot([x1, x2], [y1, y2], linestyle='-', linewidth=2)

        # 等高線を表示
        n = 64
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X,Y = np.meshgrid(x, y)
        
        # plt.axes([0.025, 0.025, 0.95, 0.95])
        
        plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)
        C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
        plt.clabel(C, inline=1, fontsize=10)

        plt.xticks(())
        plt.yticks(())
        
        
    # タイトルが重ならないようにする
    plt.tight_layout()

    # 画像保存用にfigを取り出す
    # fig = plt.gcf()
    # fig.set_size_inches(50.0, 60.0)

    # 画像を表示
    fig2.show()
    
    # 画像を保存
    # fig.savefig('gradient_descent_method.png')

    # 終了を防ぐ
    input()

main()


