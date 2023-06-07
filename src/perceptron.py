import pandas as pd
import numpy as np
import data_preprocess
from datetime import datetime
import joblib


# 训练感知机模型
class MyPerceptron:
    def __init__(self, l_rate=0.9):
        self.w = None  # 权重向量
        self.b = 0  # 偏置
        self.hyperparameters = {"l_rate": l_rate}  # 学习效率

    def get_params(self, deep=True):
        """
        获取模型参数

        Parameters
        ----------
        deep : boolean, optional
               如果是True，将返回这个模型和包含的子对象的参数，这些子对象是模型。
        Returns
        -------
        params : mapping of string to any
                参数名称映射到它们的值
        """
        out = dict()
        for key in ['l_rate']:
            value = getattr(self.hyperparameters, key, None)  # 返回超参数对象属性值
            if deep and hasattr(value, 'get_params'):  # 判断对象是否包含对应的属性
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def fit(self, X_train, y_train):
        """
        Notes
        -----
        通过训练集计算权重w和偏置b
        Parameters
        ----------
        X_train : csr_matrix
          训练数据集(tf-idf) --csr_matrix(N，M)
        y_train : ndarray
          训练标记集 --ndarray(N, *)
        """
        # 用样本点的特征数更新初始w
        self.w = np.zeros(X_train.shape[1])
        i = 0
        while i < X_train.shape[0]:
            X = X_train[i].toarray()  # 将第i篇文章的tf-idf稀疏矩阵转换为ndarray数组
            X = np.squeeze(X, 0)  # 删除ndarray冗余的维数
            y = 1 if y_train[i] == 1 else -1
            # 如果y*(wx+b)≤0 说明是误判点，更新w,b
            if y * (np.dot(self.w, X) + self.b) <= 0:
                self.w += self.hyperparameters['l_rate'] * np.dot(y, X)
                self.b += self.hyperparameters['l_rate'] * y
                i = 0  # 有误分类点，从头开始
            else:
                i += 1

    def score(self, X_test, y_test):
        """
        Notes
        -----
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
        # 用样本点的特征数更新初始w，如x1=(3,3)T，有两个特征，则self.w=[0,0]
        i = 0
        error_num = 0
        while i < X_test.shape[0]:
            X = X_test[i].todense()  # 将第i篇文章的tf-idf稀疏矩阵转换为一般矩阵便于计算
            X = X.getA()  # 将矩阵matrix转化为ndarray
            X = np.squeeze(X, 0)  # 删除ndarray冗余的维数
            y = 1 if y_test[i] == 1 else -1
            # 如果y*(wx+b)≤0 说明是误判点，更新w,b
            if y * (np.dot(self.w, X) + self.b) <= 0:
                error_num += 1
            i += 1
        return 1 - error_num/i


if __name__ == '__main__':
    """# 使用自己写的MyPerceptron类训练
    perceptron = MyPerceptron()
    time1 = datetime.now()
    perceptron.fit(data_preprocess.X_train_tfidf, data_preprocess.twenty_train.target)
    time2 = datetime.now()
    print("训练共用时：", (time2 - time1).seconds, "秒")
    print(perceptron.w)
    print(perceptron.b)
    print(perceptron.hyperparameters)

    # 保存模型
    joblib.dump(perceptron, './model/dump_perception.pkl')  # 将模型保存到本地
    perceptron2 = joblib.load('./model/dump_perception.pkl')  # 调入本地模型

    # 测试模型准确率
    time3 = datetime.now()
    score_test = perceptron2.score(data_preprocess.X_test_tfidf,
                                   data_preprocess.twenty_test.target)
    time4 = datetime.now()
    print("测试共用时：", (time4 - time3).seconds, "秒")
    print(score_test)"""

    # 调参
    from sklearn.linear_model import Perceptron  # 感知机
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    Xtd = data_preprocess.X_train_tfidf.toarray()  # 转换一般矩阵
    Xtd = Xtd.astype(np.float32)  # 转换为低精度数据类型

    clf = Perceptron(fit_intercept=True, max_iter=1000, shuffle=True)
    clf.fit(Xtd, data_preprocess.twenty_train.target)
    param_list = {
        'eta0': np.arange(0.05, 1.05, 0.05)
    }
    # GridSearchCV参数说明
    # param_grid字典类型，放入参数搜索范围
    # scoring = None 使用分类器的score方法
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    # cv = 3 交叉验证参数，默认None，使用三折交叉验证。指定fold数量=3
    grid = GridSearchCV(estimator=clf, param_grid=param_list, cv=3,
                        scoring=None, n_jobs=-1)
    # 在验证集上训练
    Xnd = data_preprocess.X_vali_tfidf.toarray()  # 转换一般矩阵
    Xnd = Xnd.astype(np.float32)  # 降低精度，加速拟合
    grid.fit(Xnd, data_preprocess.y_valit_target)

    means = grid.cv_results_['mean_test_score']
    params = grid.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   {'l_rate': %.2f}" % (mean, param['eta0']))
    # 输出最优训练器的精度
    print("{'l_rate': ", grid.best_params_['eta0'], "}")
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
        params_value.append(param['eta0'])
    plt.rcParams.update(plt_params)
    plt.plot(params_value, means, '-b', label='l_rate')
    plt.xlabel('l_rate')
    plt.ylabel('score')
    plt.legend(loc=0)
    plt.title('Perceptron')
    plt.grid()
    plt.show()
