import pandas as pd
import numpy as np
import data_preprocess
from datetime import datetime
import joblib


# 训练NB模型
class MyNaiveBayes:
    def __init__(self, eps=1e-9):
        self.hyperparameters = {"eps": eps}  # var_smoothing
        self.labels = None  # 类标签
        self.cmean = None  # 每个类的每个特征的均值(K, M)
        self.cstd = None  # 每个类的每个特征的方差(K, M)
        self.Py = None  # 先验概率分布(K,)

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
        for key in ['eps']:
            value = getattr(self.hyperparameters, key, None)  # 返回超参数对象属性值
            if deep and hasattr(value, 'get_params'):  # 判断对象是否包含对应的属性
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        设置参数
        该方法既适用于简单的estimator，也适用于嵌套对象（如管道）。
        后者的参数形式为``<组件>__<参数>``，这样就有可能更新嵌套对象的每个组件。
        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value
        return self

    def fit(self, X_train, y_train):
        """
        Notes
        -----
        获取训练集每一类中每个特征的均值和方差、先验概率以及类标签的取值集合
        Parameters
        ----------
        X_train : csr_matrix
          训练数据集(tf-idf) --csr_matrix(N，M)
        y_train : ndarray
          训练标记集 --ndarray(N,*)
        """
        N, M = X_train.shape  # N--文档数 M--特征维度
        self.labels = np.unique(y_train)
        K = len(self.labels)  # K--类标签总数
        self.cmean = np.zeros((K, M), dtype='float32')
        self.cstd = np.zeros((K, M), dtype='float32')
        self.Py = np.zeros((K,), dtype='float32')
        X = X_train.toarray()  # 转换一般矩阵
        X = X.astype(np.float32)  # 转换为低精度数据类型
        # 按类别计算各个特征的均值和方差
        for i, c in enumerate(self.labels):
            X_c = X[y_train == c, :]
            self.cmean[i, :] = np.mean(X_c, axis=0)  # [i, :]表示每列第i个数据 axis=0表示按列求平均
            self.cstd[i, :] = np.var(X_c, axis=0) + self.hyperparameters["eps"]  # axis=0表示按列求方差
            self.Py[i] = X_c.shape[0] / N

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
        return self.labels[self._log_posterior(X).argmax(axis=1)]  # 搜索每行最大值对应索引

    def _log_posterior(self, X):
        """
        Notes
        -----
        计算每个类别的（未归一化的）对数后验
        Parameters
        ----------
        X : ndarray
          测试数据集--(N，M)非稀疏矩阵
        Returns
        -------
        log_posterior : ndarray
          log_posterior--(N, K)对于X中每个样本每个类别的未归一化的对数后验概率
        """
        K = len(self.labels)
        log_posterior = np.zeros((X.shape[0], K))  # 行表示样本，列表示类别
        for i in range(K):
            log_posterior[:, i] = self._log_class_posterior(X, i)  # [:,i]所有行的第i个数据
        return log_posterior

    def _log_class_posterior(self, X, class_idx):
        """
        Notes
        -----
        计算第class_idx个类别的对数后验
        Parameters
        ----------
        X : ndarray
          测试数据集--(N，M)非稀疏矩阵
        class_idx : int
          类别下标
        Returns
        -------
        log_class_posterior : ndarray
          log_class_posterior--(N,)对于X每个样本在第class_idx个类别下的每个特征，未归一化的对数后验概率
        """
        mean = self.cmean[class_idx]  # 均值 (M,)一维数组
        prior = self.Py[class_idx]  # 先验概率
        std = self.cstd[class_idx]  # 方差  (M,)一维数组
        # log likelihood = 代入X的正态分布取对数
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * std))
        log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / std, axis=1)
        # 虽然X(N,M) mean(M,)数组维度不一样 ，但sum可以从右往左匹配维度，进行四则运算，axis=1<=1（最大维数）表示按行求和
        return log_likelihood + np.log(prior)

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

    """# 使用自己写的NB类训练
    naive_bayes = MyNaiveBayes()
    time1 = datetime.now()
    naive_bayes.fit(data_preprocess.X_train_tfidf, data_preprocess.twenty_train.target)
    time2 = datetime.now()
    print("训练共用时：", (time2 - time1).seconds, "秒")
    # 打印参数
    print(naive_bayes.labels)
    print(naive_bayes.Py)
    print(naive_bayes.cmean)
    print(naive_bayes.cstd)

    # 保存模型
    joblib.dump(naive_bayes, './model/dump_naive_bayes.pkl')  # 将模型保存到本地
    naive_bayes2 = joblib.load('./model/dump_naive_bayes.pkl')  # 调入本地模型

    # 测试模型准确度
    time3 = datetime.now()
    score_test = naive_bayes2.score(data_preprocess.X_test_tfidf, data_preprocess.twenty_test.target)
    time4 = datetime.now()
    print("测试共用时：", (time4 - time3).seconds, "秒")
    print(score_test)"""

    # 调参
    from sklearn.naive_bayes import GaussianNB  # 高斯朴素贝叶斯
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics

    Xtd = data_preprocess.X_train_tfidf.toarray()  # 转换一般矩阵
    # Xtd = Xtd.astype(np.float32)  # 转换为低精度数据类型

    clf = GaussianNB()
    clf.fit(Xtd, data_preprocess.twenty_train.target)
    param_list = {
        'var_smoothing': [0.10, 0.05, 1e-2, 0.005, 1e-4, 1e-5]
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
        params_value.append(param['var_smoothing'])
    plt.rcParams.update(plt_params)
    plt.plot(params_value, means, '-b', label='var_smoothing')
    plt.xlabel('var_smoothing')
    plt.ylabel('score')
    plt.legend(loc=0)
    plt.title('GaussianNB')
    plt.grid()
    plt.show()
