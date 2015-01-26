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
def draw_pcolor(fig):


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

    # # グラフ右端に凡例を表示
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 表示
    fig.show()
    

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
    # 等高線を表示
    fig1 = plt.figure(1)
    draw_contour(fig1)

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


