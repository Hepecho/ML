import pandas as pd
import numpy as np
import data_preprocess
from datetime import datetime
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


if __name__ == '__main__':
    Xtd = data_preprocess.X_train_tfidf.toarray()  # 转换一般矩阵
    Xnd = data_preprocess.X_test_tfidf.toarray()  # 转换一般矩阵
    # Xtd = Xtd.astype(np.float32)  # 转换为低精度数据类型

    time1 = datetime.now()
    svc = SVC()
    svc.fit(Xtd, data_preprocess.twenty_train.target)
    time2 = datetime.now()
    print("训练共用时：", (time2 - time1).seconds, "秒")

    # 测试模型准确率
    time3 = datetime.now()
    score_test = svc.score(Xnd, data_preprocess.twenty_test.target)
    time4 = datetime.now()
    print("测试共用时：", (time4 - time3).seconds, "秒")
    print(score_test)

    param_list = {
        "C": [0.1, 1, 10], "gamma": [1, 0.1, 0.01]
    }
    # GridSearchCV参数说明
    # param_grid字典类型，放入参数搜索范围
    # scoring = None 使用分类器的score方法
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    # cv = 3 交叉验证参数，默认None，使用三折交叉验证。指定fold数量=3
    grid = GridSearchCV(estimator=svc, param_grid=param_list, cv=3,
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
    C_value = []
    gamma_value = []
    for param in params:
        C_value.append(param['C'])
        gamma_value.append(param['gamma'])
    C_value = list(set(C_value))
    plt.rcParams.update(plt_params)
    i = -1
    # C_value = list(set(C_value))
    for c in C_value:
        i += 1
        plt.figure()
        plt.plot(gamma_value[3*i:(3*i+3):1], means[3*i:(3*i+3):1], '-r', label='gamma')
        plt.xlabel('gamma')
        plt.ylabel('score')
        plt.legend(loc=0)
        plt.title('SVM C={}'.format(c))
        plt.grid()
    plt.show()
