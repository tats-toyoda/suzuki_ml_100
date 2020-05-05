import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import japanize_matplotlib
import scipy
from scipy import stats
from numpy.random import randn
import copy

def min_sq(x, y): # 最小二乗法の切片と傾きを求める関数
    x_bar, y_bar = np.mean(x), np.mean(y)
    beta_1 = np.dot(x - x_bar, y - y_bar) / np.linalg.norm(x - x_bar) ** 2
    beta_0 = y_bar - beta_1 * x_bar
    return [beta_1, beta_0]

N = 100
a = np.random.normal(loc=2, scale=1, size=N) # 平均・標準偏差・サイズ
b = randn(1) # 係数
x = randn(N)
y = a * x + b + randn(N) # ここまで人工データの生成
a1, b1 = min_sq(x, y) # 回帰係数・切片
xx = x - np.mean(x) # ここで中心化する
yy = y - np.mean(y) # ここで中心化する
a2, b2 = min_sq(xx, yy) # 中心化後の回帰係数・切片

x_seq = np.arange(-5, 5, 0.1)
y_pre = x_seq * a1 + b1
yy_pre = x_seq * a2 + b2
plt.scatter(x, y, c="black")
plt.axhline(y=0, c="black", linewidth=0.5)
plt.axvline(x=0, c="black", linewidth=0.5)
plt.plot(x_seq, y_pre, c="blue", label="中心化前")
plt.plot(x_seq, yy_pre, c="orange", label="中心化後")
plt.legend(loc="upper left")