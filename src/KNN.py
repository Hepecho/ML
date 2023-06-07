from collections import Counter
import numpy as np
from datetime import datetime
import pandas as pd
import BallTree as bt
import data_preprocess
import joblib


class KNN:
    def __init__(self, k=5, leaf_size=40):
        """
        一个依靠球树进行高效计算的`k`近邻（kNN）模型。

        Parameters
        ----------
        k : int
            预测过程中的邻居数，缺省值5
        leaf_size : int
            球树中每片叶子上的最大数据点数量，缺省值40
        """
        self._ball_tree = bt.BallTree(leaf_size=leaf_size)
        self.hyperparameters = {
            "k": k,
            "leaf_size": leaf_size,
        }

    def fit(self, X_train, y_train):
        """
        Notes
        -----
        训练模型
        Parameters
        ----------
        X_train : csr_matrix
          训练数据集(tf-idf) --csr_matrix(N，M)
        y_train : ndarray
          训练标记集 --ndarray(N,*)
        """
        X = X_train.toarray()  # 转换密集矩阵ndarray
        X = X.astype(np.float16)  # 转换为低精度数据类型
        if X.ndim != 2:  # 维度不为2
            raise Exception("X must be two-dimensional")
        self._ball_tree.fit(X, y_train)

    def predict(self, X):
        """
        Notes
        -----
        用训练好的模型预测样本X的类别
        Parameters
        ----------
        X : ndarray
          测试数据集--(N，M)非稀疏矩阵
        Returns
        -------
        labels : int
          标签labels
        """
        predictions = []
        H = self.hyperparameters
        for x in X:
            pred = None
            nearest = self._ball_tree.nearest_neighbors(H["k"], x)
            targets = [n.val for n in nearest]
            # 为了与sklearn / scipy.stats.mode保持一致，在出现平局的情况下返回最小的类ID。
            counts = Counter(targets).most_common()
            pred, _ = sorted(counts, key=lambda xr: (-x[1], x[0]))[0]
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X_test, y_test):
        """
        对测试集进行测试
        Parameters
        ----------
        X_test : csr_matrix
          测试集数据 --(N，M)稀疏矩阵
        y_test :
          测试集标记 --(N,)数组
        Returns
        -------
        right_rate : float
          right_rate准确率
        """
        X = X_test.toarray()  # 转换一般矩阵
        N = X_test.shape[0]
        # 错误值计数
        error_num = 0
        # 获取预测值
        presicts = self.predict(X)
        # 循环遍历测试集中的每一个样本
        for i in range(N):
            # 与答案进行比较
            if presicts[i] != y_test[i]:
                # 若错误  错误值计数加1
                error_num += 1
        # 返回准确率
        return 1 - (error_num / N)


if __name__ == '__main__':

    """# 使用自己写的KNN训练
    knn = KNN()
    time1 = datetime.now()
    knn.fit(data_preprocess.X_train_tfidf, data_preprocess.twenty_train.target)
    time2 = datetime.now()
    print("训练共用时：", (time2 - time1).seconds, "秒")

    # 打印参数
    print(knn.hyperparameters)

    # 保存模型
    joblib.dump(knn, './model/dump_knn.pkl')  # 将模型保存到本地
    knn2 = joblib.load('./model/dump_knn.pkl')  # 调入本地模型

    # 测试模型准确度
    time3 = datetime.now()
    score_test = knn2.score(data_preprocess.X_test_tfidf, data_preprocess.twenty_test.target)
    time4 = datetime.now()
    print("测试共用时：", (time4 - time3).seconds, "秒")
    print(score_test)"""

    # 调参
    from sklearn.neighbors import KNeighborsClassifier  # KNN
    from sklearn.neighbors import BallTree  # ball-tree
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    Xtd = data_preprocess.X_train_tfidf.toarray()  # 转换一般矩阵
    # Xtd = Xtd.astype(np.float32)  # 转换为低精度数据类型

    knn = KNeighborsClassifier(algorithm='ball_tree')
    knn.fit(Xtd, data_preprocess.twenty_train.target)
    param_list = {
        'n_neighbors': range(3, 15, 1)
    }
    # GridSearchCV参数说明
    # param_grid字典类型，放入参数搜索范围
    # scoring = None 使用分类器的score方法
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    # cv = 3 交叉验证参数，默认None，使用三折交叉验证。指定fold数量=3
    grid = GridSearchCV(estimator=knn, param_grid=param_list, cv=3,
                        scoring=None, n_jobs=-1)
    # 在验证集上训练
    Xnd = data_preprocess.X_vali_tfidf.toarray()  # 转换一般矩阵
    grid.fit(Xnd, data_preprocess.y_valit_target)

    means = grid.cv_results_['mean_test_score']
    params = grid.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))
    # 输出最优训练器的精度
    print(grid.best_params_)
    print(grid.best_score_)

    # 画出迭代图
    import matplotlib.pyplot as plt
    # 更新配置
    plt_params = {
        'font.family': 'SimHei',  # 设置字体,使图形中的中文正常编码显示
        'axes.unicode_minus': False  # 使坐标轴刻度表签正常显示正负号
    }
    length = len(means)
    params_value = []
    for param in params:
        params_value.append(param['n_neighbors'])
    plt.rcParams.update(plt_params)
    plt.plot(params_value, means, '-b', label='n_neighbors')
    plt.xlabel('n_neighbors')
    plt.ylabel('score')
    plt.legend(loc=0)
    plt.title('KNN')
    plt.grid()
    plt.show()
