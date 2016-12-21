#coding=utf-8
#author=godpgf

class Statistics(object):


    # 得到x中元素的数量
    @classmethod
    def get_element_count(cls, x, start, lens):
        dic = {}
        for i in range(start,start+lens):
            if x[i] in dic:
                dic[x[i]] += 1
            else:
                dic[x[i]] = 1
        return dic

    # 得到某个x[i]的分类下y的划分
    @classmethod
    def get_res_split(cls, x, y, start, lens):
        dic = {}
        for i in range(start, start+lens):
            if x[i] in dic:
                dic[x[i]].append(y[i])
            else:
                dic[x[i]] = [y[i]]
        return dic