#coding=utf-8
#author=godpgf
from .statistics import Statistics

class NaiveBayes(object):

    def __init__(self):
        pass

    def train(self, x, y):
        self.target_dic = Statistics.get_element_count(y,0,len(y))
        self.target_count = len(y)
        self.x_dic = []
        self.x_count = []
        for i in range(len(x[0])):
            dic = Statistics.get_res_split(y,x[:,i],0,self.target_count)
            element_dic = {}
            count_dic = {}
            for key, value in dic.items():
                element_dic[key] = Statistics.get_element_count(value,0,len(value))
                count_dic[key] = len(value)
            self.x_dic.append(element_dic)
            self.x_count.append(count_dic)

    def predict(self, x):
        res_dic = {}
        is_use = []
        for i in range(len(x)):
            use = True
            for key, value in self.target_dic.items():
                if x[i] not in self.x_dic[i][key]:
                    use = False
                    break
            is_use.append(use)
        for key, value in self.target_dic.items():
            percent = float(value) / self.target_count
            for i in range(len(x)):
                if is_use[i]:
                    percent *= float(self.x_dic[i][key][x[i]])/self.x_count[i][key]
            res_dic[key] = percent
        return res_dic
