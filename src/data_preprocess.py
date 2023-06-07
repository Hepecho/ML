# from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
categories = [
    'alt.atheism',
    'soc.religion.christian',
    'comp.graphics',
    'sci.med']
"""categories = [
    'alt.atheism',
    'comp.graphics',
    'misc.forsale',
    'rec.autos',
    'sci.crypt',
    'sci.med',
    'soc.religion.christian',
    'talk.politics.guns'
]"""
# categories = None  # 选择所有类
# 加载训练集、测试集
"""twenty_train = datasets.load_files("./20news-bydate/20news-bydate-train")
twenty_test = datasets.load_files("./20news-bydate/20news-bydate-test")"""
twenty_train = fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=42)
twenty_test = fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=42)
print("原始训练集大小：", len(twenty_train.data))
print("测试集大小: ", len(twenty_train.data))

# voc_file = open("./src/voc.txt", "w")

# 划分验证集
X_train_data, X_vali_data, y_train_target, y_valit_target = train_test_split(
    twenty_train.data, twenty_train.target, test_size=0.33, random_state=42)
# print(len(X_train_data))
print("验证集大小：", len(X_vali_data))
# print(y_train_target)
print("验证集标签：{}".format(y_valit_target))

# 将文本文件变成数字的特征表示(词袋模型)
# 1)使用CountVectorizer构建词频向量
# CountVectorizer支持单词或者连续字符的N-gram模型的计数,利用scipy.sparse矩阵只在内存中保存特征向量中非0元素位置以节省内存
# 创建词频转换器
count_vect = CountVectorizer(stop_words='english')  # 实例化
# 转换训练集 将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i个文本下的词频
# fit_transform是fit和transform两种方法的简写
# fit方法用于构建特征空间（也就是构建词典）
# transform方法使用该空间将文本数据转化为特征矩阵
X_train_counts = count_vect.fit_transform(twenty_train.data)
print("词频矩阵，稀疏编码 done\n")
# print(count_vect.vocabulary_, file=voc_file)  # 输出词典
# voc_file.close()
print("训练集词频矩阵大小：{}".format(X_train_counts.shape))  # (i,j),x 第i个列表元素，**词典中索引为j的元素**， 词频x

# 2)转化为TF-IDF特征向量
# 用TfidfTransformer将词频向量转为Tfidf形式
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

"""
csr_matrix，全称Compressed Sparse Row matrix，即按行压缩的稀疏矩阵存储方式，由三个一维数组indptr, indices, data组成。
这种格式要求矩阵元按行顺序存储，每一行中的元素可以乱序存储。那么对于每一行就只需要用一个指针表示该行元素的起始位置即可。
一维数组data（数值）:有序地存储了所有的非零值，它具有与非零元素同样多数量的元素，通常由变量nnz表示；
一维数组indptr（行偏移量）：表示某一行的第一个元素在data里面的起始偏移位置，在行偏移的最后补上矩阵总的元素个数
一维数组Indices（列号:）: 其使用如下方式包含列索引信息:indices[indptr[i]:indptr[i+1]]是一个具有行i中非零元素的列索引的整数数组。
  Len(indice)==len(data)==nnz;
"""

print("TF-IDF矩阵 done\n")
print(X_train_tfidf)
print("训练集TF-IDF稀疏矩阵shape: {}".format(X_train_tfidf.shape))

X_test_counts = count_vect.transform(twenty_test.data)  # 注意：不能加fit_，否则会产生新的特征
X_test_tfidf = tfidf_transformer.transform(X_test_counts)  # 注意：不能加fit_，否则会产生新的特征

X_vali_counts = count_vect.transform(X_vali_data)  # 注意：不能加fit_，否则会产生新的特征
X_vali_tfidf = tfidf_transformer.transform(X_vali_counts)  # 注意：不能加fit_，否则会产生新的特征
