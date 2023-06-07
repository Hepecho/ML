import numpy as np
import heapq
from copy import copy
from collections import Hashable


def euclidean(x, y):
    """
    计算两个向量间的欧式距离
    """
    return np.sqrt(np.sum((x - y) ** 2))


#                           Priority Queue                            #
class PQNode(object):
    def __init__(self, key, val, priority, entry_id):
        """一个通用的节点对象，用于保存 :class:`PriorityQueue`中的条目"""
        self.key = key
        self.val = val
        self.entry_id = entry_id
        self.priority = priority

    def __repr__(self):
        fstr = "PQNode(key={}, val={}, priority={}, entry_id={})"
        return fstr.format(self.key, self.val, self.priority, self.entry_id)

    def to_dict(self):
        """返回该节点内容的字典表示"""
        d = self.__dict__
        d["id"] = "PQNode"
        return d

    def __gt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id > other.entry_id
        return self.priority > other.priority

    def __ge__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority >= other.priority

    def __lt__(self, other):
        if not isinstance(other, PQNode):
            return -1
        if self.priority == other.priority:
            return self.entry_id < other.entry_id
        return self.priority < other.priority

    def __le__(self, other):
        if not isinstance(other, PQNode):
            return -1
        return self.priority <= other.priority


class PriorityQueue:
    def __init__(self, capacity, heap_order="max"):
        """
        一个使用二进制堆的优先级队列实现
        Notes
        -----
        优先级队列是一个数据结构，用于存储一个数值集合中最大或最小的元素的最高"capacity"。
        由于使用二进制堆，``PriorityQueue``提供`O(log N)` :meth:`push`和:meth:`pop`操作。
        Parameters
        ----------
        capacity: int
            队列中可容纳的最大项目数。
        heap_order: {"max", "min"}
            优先级队列是否应该保留`capacity`最小（`heap_order` = 'min'）
            或`capacity`最大（`heap_order` = 'max'）优先级的项目。
        """
        assert heap_order in ["max", "min"], "heap_order must be either 'max' or 'min'"
        self.capacity = capacity
        self.heap_order = heap_order

        self._pq = []
        self._count = 0
        self._entry_counter = 0

    def __repr__(self):
        fstr = "PriorityQueue(capacity={}, heap_order={}) with {} items"
        return fstr.format(self.capacity, self.heap_order, self._count)

    def __len__(self):
        return self._count

    def __iter__(self):
        return iter(self._pq)

    def push(self, key, priority, val=None):
        """
        向队列中添加一个新的（键，值）对，优先级为`priority`
        Notes
        -----
        如果队列处于饱和状态，并且`priority`超过当前队列中最大/最小优先级的项目的优先级，
        用（`key`, `val`）替换当前队列项目。
        Parameters
        ----------
        key : hashable object
            要插入队列的键
        priority : comparable
            键值对的优先级
        val : object
            与`key`相关的值。默认为None
        """
        if self.heap_order == "max":
            priority = -1 * priority

        item = PQNode(key=key, val=val, priority=priority, entry_id=self._entry_counter)
        heapq.heappush(self._pq, item)

        self._count += 1
        self._entry_counter += 1

        while self._count > self.capacity:
            self.pop()

    def pop(self):
        """
         从队列中移除具有最大/最小（取决于`self.heap_order`）优先级的项目并返回。
        Notes
        -----
        与:meth:`peek`相比，这个操作是`O(log N)`。
        Returns
        -------
        item : :class:`PQNode` instance or None
            具有最大/最小优先级的项目，取决于``self.heap_order'`。
        """
        item = heapq.heappop(self._pq).to_dict()
        if self.heap_order == "max":
            item["priority"] = -1 * item["priority"]
        self._count -= 1
        return item

    def peek(self):
        """
        返回具有最大/最小（取决于`self.heap_order`）优先级的项目，*不*从队列中移除该项目。
        Notes
        -----
        与:meth:`pop`相比，这个操作是O(1)
        Returns
        -------
        item : :class:`PQNode` instance or None
            具有最大/最小优先级的项目，取决于``self.heap_order'`。
        """
        item = None
        if self._count > 0:
            item = copy(self._pq[0].to_dict())
            if self.heap_order == "max":
                item["priority"] = -1 * item["priority"]
        return item


# ======================BallTree======================= #
class BallTreeNode:
    def __init__(self, centroid=None, X=None, y=None):
        self.left = None
        self.right = None
        self.radius = None
        self.is_leaf = False

        self.data = X
        self.targets = y
        self.centroid = centroid

    def __repr__(self):
        fstr = "BallTreeNode(centroid={}, is_leaf={})"
        return fstr.format(self.centroid, self.is_leaf)

    def to_dict(self):
        d = self.__dict__
        d["id"] = "BallTreeNode"
        return d


class BallTree:
    def __init__(self, leaf_size=40):
        """
        Notes
        -----
        球树是一棵二叉树，其中每个节点都定义了一个`D`维的超球（"球"），包含了要搜索的点的一个子集。
        树的每个内部节点将数据点分成两个不相干的集合，与不同的球相关联。
        虽然球本身可能相交，但每个点根据其与球中心的距离被分配到分区中的一个或另一个球。
        树上的每个叶子节点都定义了一个球，并列举了该球内的所有数据点。
        Parameters
        ----------
        leaf_size : int
            每个叶子上的最大数据点数量。默认值是40。
        """
        self.root = None
        self.leaf_size = leaf_size
        self.metric = euclidean  # 设置距离为欧式距离

    def fit(self, X, y=None):
        """
        使用O(MlogN) `k`-d构建算法，递归地构建一个球树

        Notes
        -----
        递归地将数据分为由中心点`C`和半径`r`定义的节点，从而使节点下面的每个点都位于由`C`和`r`定义的超球内。
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
          训练数据集，N个文档，M个特征
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, *)` or None
          训练标记集
        """
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)  # 找到根节点划分数据集
        self.root = BallTreeNode(centroid=centroid)
        self.root.radius = np.max([self.metric(centroid, x) for x in X])
        self.root.left = self._build_tree(left_X, left_y)
        self.root.right = self._build_tree(right_X, right_y)

    def _build_tree(self, X, y):
        centroid, left_X, left_y, right_X, right_y = self._split(X, y)

        if X.shape[0] <= self.leaf_size:
            leaf = BallTreeNode(centroid=centroid, X=X, y=y)
            leaf.radius = np.max([self.metric(centroid, x) for x in X])
            leaf.is_leaf = True
            return leaf

        node = BallTreeNode(centroid=centroid)
        node.radius = np.max([self.metric(centroid, x) for x in X])
        node.left = self._build_tree(left_X, left_y)
        node.right = self._build_tree(right_X, right_y)
        return node

    @staticmethod
    def _split(X, y=None):
        # 找到方差最大的维度
        split_dim = np.argmax(np.var(X, axis=0))

        # 在split_dim维度上为X和y重新排序
        sort_ixs = np.argsort(X[:, split_dim])
        X, y = X[sort_ixs], y[sort_ixs] if y is not None else None

        # 按split_dim的中位数值划分
        med_ix = X.shape[0] // 2
        centroid = X[med_ix]  # , split_dim

        # 在中心点将数据分成两半（中位数总是出现在右边的分割上）
        left_X, left_y = X[:med_ix], y[:med_ix] if y is not None else None
        right_X, right_y = X[med_ix:], y[med_ix:] if y is not None else None
        return centroid, left_X, left_y, right_X, right_y

    def nearest_neighbors(self, k, x):
        """
        Notes
        -----
        使用KNS1算法找到球树中与查询向量`x`最接近的`k`个邻居。
        Parameters
        ----------
        k : int
            最近临的邻居数
        x : :py:class:`ndarray <numpy.ndarray>` of shape `(1, M)`
            查询向量
        Returns
        -------
        nearest : :class:`PQNode`s的列表，长度为`k`。
            `X`中最接近查询矢量的`k`点的列表。每个:class:`PQNode`的`key`属性包含点本身，
            `val`属性包含其目标，`distance`属性包含其与查询向量的距离。
        """
        # 保持一个最大优先级的队列，优先级 = 与x的距离
        PQ = PriorityQueue(capacity=k, heap_order="max")
        nearest = self._knn(k, x, PQ, self.root)
        for n in nearest:
            n.distance = self.metric(x, n.key)
        return nearest

    def _knn(self, k, x, PQ, root):
        dist = self.metric
        dist_to_ball = dist(x, root.centroid) - root.radius
        dist_to_farthest_neighbor = dist(x, PQ.peek()["key"]) if len(PQ) > 0 else np.inf

        if dist_to_ball >= dist_to_farthest_neighbor and len(PQ) == k:
            return PQ
        if root.is_leaf:
            targets = [None] * len(root.data) if root.targets is None else root.targets
            for point, target in zip(root.data, targets):
                dist_to_x = dist(x, point)
                if len(PQ) == k and dist_to_x < dist_to_farthest_neighbor:
                    PQ.push(key=point, val=target, priority=dist_to_x)
                else:
                    PQ.push(key=point, val=target, priority=dist_to_x)
        else:
            l_closest = dist(x, root.left.centroid) < dist(x, root.right.centroid)
            PQ = self._knn(k, x, PQ, root.left if l_closest else root.right)
            PQ = self._knn(k, x, PQ, root.right if l_closest else root.left)
        return PQ
