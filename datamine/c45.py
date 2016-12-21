#coding=utf-8
#author=godpgf
import math
import numpy as np
from .statistics import Statistics

class DecisionTreeNode(object):

    def __init__(self, max_gain_ratio, row, start, lens):
        self.max_gain_ratio = max_gain_ratio
        self.row = row
        self.start = start
        self.lens = lens
        self.children = {}

class DecisionTree(object):

    def __init__(self, x, y, root):
        self.x = x
        self.y = y
        self.root = root

    def predict(self, x):
        p = self.root
        while p.max_gain_ratio != 0:
            if x[p.row] not in p.children:
                return self.predict_node(p)
            p = p.children[x[p.row]]
        res = None
        max_count = 0
        all_count = 0
        for key, value in p.children.items():
            if value[1] > max_count:
                max_count = value[1]
                res = key
            all_count += value[1]

        return res, float(max_count)/all_count

    def predict_node(self, node):
        dic = Statistics.get_element_count(self.y, node.start, node.lens)
        res = None
        max_count = 0
        all_count = 0
        for key, value in dic.items():
            if value > max_count:
                max_count = value
                res = key
            all_count += value

        return res, float(max_count)/all_count


class C45(object):

    @classmethod
    def create_tree(cls, x, y, min_element = 0):
        visable = [False for i in range(len(x[0]))]
        max_gain_ratio, row, dic = cls.get_max_gain_ratio_row(x, y, visable)
        root = DecisionTreeNode(max_gain_ratio,row,0,len(x))
        cls.insert_tree(x, y, visable, root, dic, min_element)
        return DecisionTree(x,y,root)

    @classmethod
    def pruning_tree(cls, root, x, y):
        #剪枝
        pass

    @classmethod
    def insert_tree(cls, x, y, visable, root, dic, min_element):

        if root.row != -1:
            visable[root.row] = True

        #将样本中的数据放置在不同区域,以构成新子树
        # 样本取值:(开始位置,结束位置)
        sample_trans = {}
        curIndex = root.start
        for key, value in dic.items():
            sample_trans[key] = [curIndex,curIndex]
            curIndex += len(value)

        #重新排列样本,不同的区间分给不同的子树
        curIndex = 0
        while curIndex < root.lens:
            offset = root.start + curIndex
            #area = None
            if root.max_gain_ratio != 0:
                area = sample_trans[x[offset][root.row]]
            else:
                #有可能所有特征都用过了，或者没有用过的特征自己的取值都相等，或者没有用过的特征对应的标签都相等
                area = sample_trans[y[offset]]
            if offset == area[1]:
                #样本正好落在合适区间的结束位置
                curIndex += 1
                area[1] += 1
            elif offset >= area[0] and offset < area[1]:
                #样本已经在区间中
                curIndex = area[1] - root.start
            elif offset < area[0]:
                #样本不在自己应该在的区间
                x[[offset,area[1]],:] = x[[area[1],offset],:]
                y[[offset,area[1]]] = y[[area[1],offset]]
                area[1] += 1
            else:
                raise Exception("样本不可能大于自己的区间，因为遍历这个样本时前面的元素都是放置好的")

        #插入子树
        for key, value in sample_trans.items():
            if root.max_gain_ratio == 0:
                root.children[key] = (value[0], value[1]-value[0])
            else:
                if value[1]-value[0] > min_element:
                    max_gain_ratio, row, dic = cls.get_max_gain_ratio_row(x, y, visable, value[0],value[1]-value[0])
                else:
                    max_gain_ratio, row, dic = 0, -1, Statistics.get_res_split(y, y, value[0],value[1]-value[0])
                child = DecisionTreeNode(max_gain_ratio, row, value[0], value[1]-value[0])
                root.children[key] = child
                cls.insert_tree(x,y,visable,child,dic,min_element)

        if root.row != -1:
            visable[root.row] = False



    """
    选择还没有使用的事件（特征）中的其中一个，使得结果熵减少量最大
    如果所有事件对熵的减少量都是0，所有表示事件对结果毫无影响。此时直接附带上结果。
    返回熵的增益率和行号(附带详细的划分情况提高程序效率)
    """

    @classmethod
    def get_max_gain_ratio_row(cls, x, y, visable, start = 0, lens = None):
        if lens is None:
            lens = len(x)

        max_gain_ratio = 0
        row = -1
        dic = None
        for i in range(0, len(visable)):
            if visable[i] is False:
                gr,d = C45.gain_ratio(x[:,i], y, start, lens)
                if gr > max_gain_ratio:
                    max_gain_ratio = gr
                    row = i
                    dic = d

        if max_gain_ratio == 0:
            dic = Statistics.get_res_split(y, y, start, lens)

        return max_gain_ratio, row, dic

    #########################################################################3

    """
    信息增益率 gainRatio = Gain(A)/splitE(A) 用来属性选择度量(附带详细的划分情况提高程序效率)
    """
    @classmethod
    def gain_ratio(cls, x, y, start = 0, lens = None):
        #计算当前列的信息增益Gain(A)
        gain,dic = cls.gain(x, y, start, lens)
        #C4.5对ID3的改进
        splitInfo = cls.infoDj(x, start, lens)
        return gain * splitInfo if splitInfo != 0 else 0, dic

    """
    某条数据的信息增益，即"y的熵-有x的参与下y的熵"(附带详细的划分情况提高程序效率)
    """
    @classmethod
    def gain(cls, x, y, start = 0, lens = None):
        sv_total,dic = cls.entropyA_S(x, y, start, lens)
        return cls.infoDj(y,start,lens) - sv_total, dic

    """
    得到在x的参与下y的熵(附带详细的划分情况提高程序效率)
    """
    @classmethod
    def entropyA_S(cls, x, y, start = 0, lens = None):

        if lens is None:
            lens = len(x)

        dic = Statistics.get_res_split(x, y, start, lens)

        sv_total = 0
        for key, value in dic.items():
            sv_total += cls.getPi(len(value),lens) * cls.infoDj(value)
        return sv_total, dic


    """
    得到某个划分后的熵
    """
    @classmethod
    def infoDj(cls, x, start = 0, lens = None):
        if lens is None:
            lens = len(x)
        counts, allCount = cls.get_counts(x, start, lens)
        return cls.entropy_S(counts,allCount)



    @classmethod
    def get_counts(cls, x, start, lens):

        dic = Statistics.get_element_count(x, start, lens)
        counts = []
        allCount = 0
        for key, value in dic.items():
            allCount += value
            counts.append(value)
        return counts, allCount

    """
    计算某个事件的熵，比如：打球（YES、NO） infoD = -E(pi*log2 pi)
    counts表示事件每一条分类的数量，比如：YES数量、NO数量
    allCount表示事件的样本数
    """
    @classmethod
    def entropy_S(cls, counts, allCount):
        infoD = 0;
        for count in counts:
            infoD += cls.info(count, allCount)# 当前计算的分裂信息
        return -infoD

    """
    注意：因为x_pi < 1，所以返回值是负的，用的时候要转化成正的
    负信息熵 entropy: info(T) = (i = 1...k)pi * log（2）pi
    x:当Si前值
    total:
    S总值
    返回当前值的 信息分裂 当前属性列的集合的信息分裂的和即为
    """
    @classmethod
    def info(cls, x, total) :
        if x == 0 :
            return 0;

        x_pi = cls.getPi(x, total)
        return x_pi * cls.logYBase2(x_pi)

    #log2y
    @classmethod
    def logYBase2(cls, y) :
        return math.log(y) / math.log(2);

    """
    pi=|C(i,d)|/|D|
    x:当Si前值
    total:S总值
    返回当前值所占总数的比例
    """
    @classmethod
    def getPi(cls, x, total):
		return float(x) / total