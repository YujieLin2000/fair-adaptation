import numpy as np
import matplotlib.pyplot as plt

# 绘图参数设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
# 计算点point到点line_point1, line_point2组成向量的距离
# 对于曲线1上的点，找到曲线2上离其最近的两个点，求曲线1上的点到这两个最近点之间的距离
def point_distance_line(x1, y1, x2, y2, label):
    # x1, y1 曲线2
    # x2, y2 曲线1
    val = dict(zip(x1, y1))
    dis_list = []
    for point in np.array(tuple(zip(x2, y2))):
        xx = point[0]
        zzy = point[1]
        val_list = list(x1)
        val_list.append(xx)  # 将曲线1上的点添加到曲线点中，找到距离该点最近的两个点
        sort_val = sorted(val_list)
        # 如果当前点是最后一个点，取其前两个点
        if sort_val.index(xx) == len(val_list) - 1:
            lind = sort_val.index(xx) - 1
            rind = sort_val.index(xx) - 2
        # 如果当前点是第一个点，取其后两个点
        elif sort_val.index(xx) == 0:
            lind = sort_val.index(xx) + 1
            rind = sort_val.index(xx) + 2
        # 否则，取其前一个点和后一个点
        else:
            lind = sort_val.index(xx) - 1
            rind = sort_val.index(xx) + 1

        plx = sort_val[lind]
        prx = sort_val[rind]
        ply = val[plx]
        pry = val[prx]
        line_point1 = np.array([plx, ply])
        line_point2 = np.array([prx, pry])

        mfy = ply + ((xx - plx) * (pry - ply)) / (prx - plx)   # 曲线1横坐标代入曲线2对应直线后得到的纵坐标

        # 计算向量
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        # 求距离
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        # label等于0 ,曲线2纵坐标小于曲线1纵坐标
        if label == 0:
            if zzy >= mfy:
                dis_list.append(distance)
            else:
                dis_list.append(-distance)
        # label等于1  曲线2纵坐标大于曲线1纵坐标
        elif label == 1:
            if zzy <= mfy:
                dis_list.append(distance)
            else:
                dis_list.append(-distance)
    return dis_list


x1 = np.arange(1, 10)
y1 = 2 * np.log(x1)

x2 = np.arange(1, 10)
y2 = 2 * np.log(x1) + np.random.random()
plt.plot(x1, y1, 'r-o', label = '曲线2')
plt.plot(x2, y2, 'b-*', label = '曲线1')
plt.legend()
plt.grid(alpha = 1)
plt.show()
print(point_distance_line(x1, y1, x2, y2, label=0))
print(point_distance_line(x1, y1, x2, y2, label=1))
