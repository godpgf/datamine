#coding=utf-8
#author=godpgf

import sys
import datetime
import random
import xlrd
import xlwt
import numpy as np
from datamine import C45, DataProcessTool, Statistics, NaiveBayes

#统计准确率
def normal_percent(y):
    count = 0
    for d in y:
        if d == u'结项回款':
            count+=1
    return float(count)/len(y)

#从所有数据中选择需要使用的
def filt_x(x, title):
    return np.array([[x[i][title[u'区域']],
                      x[i][title[u'城市']],
                      x[i][title[u'来源']],
                      x[i][title[u'行业']],
                      x[i][title[u'公司规模']],
                      x[i][title[u'产品需求']]
                      ] for i in range(len(x))])

#处理城市数据
def city_normalize(x):
    for i in range(len(x)):
        is_other_city = True
        for c in [u"北京",
                  u"深圳",
                  u"上海",
                  u"广州",
                  u"杭州",
                  u"成都",
                  u"南京",
                  u"武汉",
                  u"郑州",
                  u"重庆",
                  u"西安",
                  u"厦门",
                  u"东莞",
                  u"苏州",
                  u"长沙"
                  ]:
            if c in x[i]:
                x[i] = c
                if c in [u"北京",u"深圳",u"上海",u"广州"]:
                    x[i] = u'大城市'
                else:
                    x[i] = u'中城市'
                is_other_city = False
                break
        if is_other_city:
            x[i] = u'小城市'

def need_normal(x):
    for i in range(len(x)):
        if u'、' not in x[i]:
            if x[i] not in [u'未知',u'收款',u'无法获取',u'微信wap',u'代付',u'多级商户',u'账户系统']:
                x[i] = u'小众单一需求'
            elif x[i] in [u'代付',u'多级商户',u'账户系统']:
                x[i] = u'特殊单一需求'
        else:
            num = len(x[i].split(u'、'))
            if num >= 4 :
                x[i] = u'三个以上需求'
            elif x[i] not in [u'收款、代付',u'收款、多级商户',u'收款、账户系统',u'收款、微信wap']:
                x[i] = u'小众联合需求'

#统计c45准确率
def get_c45_percent(dt, x):
    res,percent = dt.predict(x)
    if res != u'结项回款':
        percent = 1 - percent
    return percent

#统计朴素贝叶斯准确率
def get_nb_percent(nb, x):
    res = nb.predict(x)
    percent = 0
    all_percent = 0
    for key, value in res.items():
        if key == u'结项回款':
            percent = value
        all_percent += value
    return percent / all_percent

def read_table_file(table_file_path):
    data = xlrd.open_workbook(table_file_path)
    table_excel = data.sheets()[0]
    table = []
    title = {}
    for i in range(0,table_excel.nrows):
        if i == 0:
            for j in range(table_excel.ncols):
                title[table_excel.cell(i,j).value] = j
        else:
            table.append([table_excel.cell(i,j).value for j in range(table_excel.ncols)])

    return np.array(table),title

"""
def split_table(table, title):
    predict_table = []
    normal_table = []
    for t in table:
        if t[title[u'PSS阶段']] == u'待处理':
            predict_table.append(t)
        elif t[title[u'PSS阶段']] == u'结项回款':
            normal_table.append(t)
        elif len(t[title[u'PSS阶段']]) > 0:
            t[title[u'PSS阶段']] = u'尚未付款'
            normal_table.append(t)
    random.shuffle(normal_table)
    return np.array(normal_table), np.array(predict_table)
"""

#过滤掉没有用的数据
def filter_table(table, title):
    use_table = []
    for t in table:
        if t[title[u'PSS阶段']] != u'待处理' and len(t[title[u'PSS阶段']]) > 0:
            use_table.append(t.copy())
    random.shuffle(use_table)
    return np.array(use_table)

#得到训练数据
def filter_train_table(table, title):
    use_table = []
    for t in table:
        if t[title[u'PSS阶段']] == u'结项回款':
            use_table.append(t)
        elif t[title[u'PSS阶段']] == u'CLOSE' \
                or t[title[u'PSS阶段']] == u'一周内无人接听' \
                or (datetime.datetime.now() - datetime.datetime.strptime(t[title[u'创建时间']], "%Y-%m-%d %H:%M:%S")).days > 30:
            #print (datetime.datetime.now() - datetime.datetime.strptime(t[title[u'创建时间']], "%Y-%m-%d %H:%M:%S")).days
            #已知肯定不会付款的和长时间都没付款的，当做反例
            t[title[u'PSS阶段']] = u'尚未付款'
            use_table.append(t)
    return np.array(use_table)

def cal_accurate_rate(x, y):
    dic = Statistics.get_res_split(x, y, 0, len(y))
    name = []
    lens = []
    acc = []
    for key, value in dic.items():
        name.append(key)
        count = 0
        for t in value:
            if t == u'结项回款':
                count+=1
        lens.append(len(value))
        acc.append(float(count) / len(value))
    return name, lens, acc

#在不做任何特征处理的情况下,统计表格各个维度,输出统计结果
def statistic_table(table, title, statistics_file_path):
    #遍历所有维度,得到某个维度下数据和准确率的关系
    name = []
    value = []
    lens = []
    acc = []
    #for i in [u'区域',u'城市',u'来源',u'行业',u'公司规模',u'产品需求']:
    for i in [u'产品需求']:
        v,l,a = cal_accurate_rate(table[:,title[i]], table[:,title[u'PSS阶段']])
        for j in range(len(v)):
            name.append(i)
            value.append(v[j])
            lens.append(l[j])
            acc.append(a[j])
    index = np.array(acc).argsort()

    xls = xlwt.Workbook(encoding = 'utf-8')
    sheet = xls.add_sheet("Worksheet")
    sheet_title = [u'维度',u'取值',u'包含线索数量',u'结项回款率']
    for i in range(len(sheet_title)):
        sheet.write(0,i,sheet_title[i])

    row = 1
    for i in range(len(index)-1, -1, -1):
        id = index[i]
        sheet.write(row,0,name[id])
        sheet.write(row,1,value[id])
        sheet.write(row,2,"%d"%lens[id])
        sheet.write(row,3,"%.2f%%"%(acc[id]*100))
        row+=1
        #print(u"%s,%s,%d,%.2f%%"%(name[id], value[id], lens[id], acc[id]*100))
    xls.save(statistics_file_path)

def process_table(table, title):
    #去掉区域的缺失值
    DataProcessTool.replace_data(table[:,title[u'区域']],{u"其他":''})
    DataProcessTool.fast_replace_missing_data(table[:,title[u'区域']],set(['']))

    #去掉城市的缺失值,并重新归类
    DataProcessTool.fast_replace_missing_data(table[:,title[u'城市']],set(['']))
    city_normalize(table[:,title[u'城市']])

    #去掉来源缺失值
    DataProcessTool.fast_replace_missing_data(table[:,title[u'来源']],set(['']))

    #去掉行业缺失值
    DataProcessTool.replace_data(table[:,title[u'行业']],{u"待填写":''})
    DataProcessTool.fast_replace_missing_data(table[:,title[u'行业']],set(['']))

    #去掉公司规模缺失值
    DataProcessTool.replace_data(table[:,title[u'公司规模']],{u"待填写":u''})
    DataProcessTool.fast_replace_missing_data(table[:,title[u'公司规模']],set(['']))

    #需求中缺失数据太多，专门做一个分类存放
    DataProcessTool.replace_data(table[:, title[u'产品需求']], {u"待填写": u'未知',u"":u"未知",u"其他":u"未知"})
    need_normal(table[:, title[u'产品需求']])


def process_predict_table(table, title):
    #去掉区域的缺失值
    DataProcessTool.replace_data(table[:,title[u'区域']],{u"其他":''})

    #去掉城市的缺失值,并重新归类
    DataProcessTool.fast_replace_missing_data(table[:,title[u'城市']],set(['']))
    city_normalize(table[:,title[u'城市']])

    #去掉行业缺失值
    DataProcessTool.replace_data(table[:,title[u'行业']],{u"待填写":''})

    #去掉公司规模缺失值
    DataProcessTool.replace_data(table[:,title[u'公司规模']],{u"待填写":u''})

    # 需求中缺失数据太多，专门做一个分类存放
    DataProcessTool.replace_data(table[:, title[u'产品需求']], {u"待填写": u'未知', u"": u"未知", u"其他": u"未知"})
    need_normal(table[:, title[u'产品需求']])

def create_c45_model(x, y):
    return C45.create_tree(x, y, 4)

def create_naive_bayes_model(x, y):
    nb = NaiveBayes()
    nb.train(x, y)
    return nb

def predict(normal_table, predict_table, title, model_name):
    model_dic = {"naive_bayes":(create_naive_bayes_model,get_nb_percent),"c45":(create_c45_model,get_c45_percent)}
    fun_pair = model_dic[model_name]
    creater = fun_pair[0]
    predicter = fun_pair[1]
    train_x = filt_x(normal_table, title)
    train_y = normal_table[:,title[u'PSS阶段']]
    model = creater(train_x, train_y)

    predict_x = filt_x(predict_table, title)
    res = {}
    for i in range(len(predict_x)):
        if len(predict_table[i][title[u'企业全称']]) > 0:
            res[predict_table[i][title[u'企业全称']]] = predicter(model,predict_x[i])
    return res, normal_percent(normal_table[:,title[u'PSS阶段']])
    """
    percent = []
    des = []
    for i in range(len(predict_x)):
        des.append([predict_table[i][title[u'企业全称']],
                    predict_table[i][title[u'区域']],
                    predict_table[i][title[u'城市']],
                    predict_table[i][title[u'来源']],
                    predict_table[i][title[u'行业']],
                    predict_table[i][title[u'公司规模']],
                    predict_table[i][title[u'产品需求']],])
        percent.append(predicter(model,predict_x[i]))

    index = np.array(percent).argsort()
    xls = xlwt.Workbook(encoding = 'utf-8')
    sheet = xls.add_sheet("Worksheet")
    sheet_title = [u'企业全称',u'区域',u'城市',u'来源',u'行业',u'公司规模',u'产品需求',u'股价结项回款率']
    for i in range(len(sheet_title)):
        sheet.write(0,i,sheet_title[i])

    row = 1
    for i in range(len(index)-1, -1, -1):
        id = index[i]
        for j in range(7):
            sheet.write(row,j,des[id][j])
        sheet.write(row,7,"%.2f%%"%(percent[id]*100))
        row+=1
    xls.save(predict_file_path)
    """

def legitimate_city(x):
    city_list = []
    for d in x:
        if len(d) > 1:
            city_list.append(d)
    city_list.sort()
    city = {}
    for c in city_list:
        if c not in city:
            for key, value in city.items():
                if key in c:
                    city[c] = value
                    break
        if c not in city:
            city[c] = c
    DataProcessTool.replace_data(x,city)
    DataProcessTool.replace_data(x,{u'待确认':u'',u'SH':u'',u'50 人以下':u'',u'无':u'',u'不详':u''})
    DataProcessTool.fast_replace_missing_data(x,set([u'']))

def save_res(res, mean_percent, predict_file_path, table, title):
    inv_title = {}
    for key, value in title.items():
        inv_title[value] = key
    xls = xlwt.Workbook(encoding='utf-8')
    sheet = xls.add_sheet("Worksheet")
    for i in range(len(table[0])):
        sheet.write(0, i, inv_title[i])
    sheet.write(0, len(table[0]), u"结项回款概率(超过%.0f%%就不错)"%(mean_percent*100))
    row = 1
    for i in range(len(table)):
        for j in range(len(table[0])):
            sheet.write(row,j,table[i][j])
        name = table[i][title[u'企业全称']]
        if name in res:
            sheet.write(row,len(table[0]),"%.0f"%(res[name]*100))
        row+=1
    xls.save(predict_file_path)

def main(table_file_path, statistics_file_path, predict_file_path, model_name):
    try:
        table,title = read_table_file(table_file_path)
        legitimate_city(table[:,title[u'城市']])

        #normal_table, predict_table = split_table(table, title)
        use_table = filter_table(table, title)
        statistic_table(use_table, title, statistics_file_path)
        process_table(use_table, title)

        train_table = filter_train_table(use_table,title)
        test_model(train_table, title)

        predict_table = table.copy()
        process_predict_table(predict_table, title)
        res, mean_percent = predict(train_table, predict_table, title, model_name)
        save_res(res, mean_percent, predict_file_path, table, title)
    except Exception, e:
        print Exception,":",e
        return




#测试模型-------------------------------------------------------------------------------------
def test_model(table, title):
    train_table = np.array(table[0:int(len(table)*0.75)])
    test_table = np.array(table[int(len(table)*0.75):-1])

    #测试数据的准确率
    train_acc = normal_percent(train_table[:,title[u'PSS阶段']])
    test_acc = normal_percent(test_table[:,title[u'PSS阶段']])
    print("训练数据结项回款率%.2f%%"%(train_acc*100))
    print("测试数据结项回款率%.2f%%"%(test_acc*100))

    train_x = filt_x(train_table, title)
    train_y = train_table[:,title[u'PSS阶段']]
    dt = C45.create_tree(train_x, train_y, 4)

    right_count, all_count, lens = percent(dt, train_x, train_y, train_acc, get_c45_percent)
    print("C4.5算法训练数据数量:%d,预测数量:%d,回款率:%.2f%%"%(lens,all_count,float(right_count)/all_count * 100))

    test_x = filt_x(test_table, title)
    test_y = test_table[:,title[u'PSS阶段']]

    right_count, all_count, lens = percent(dt, test_x, test_y, test_acc, get_c45_percent)
    print("C4.5算法测试数据数量:%d,预测数量:%d,回款率:%.2f%%"%(lens,all_count,float(right_count)/all_count * 100))

    nb = NaiveBayes()
    nb.train(train_x, train_y)
    right_count, all_count, lens = percent(nb, train_x, train_y, train_acc, get_nb_percent)
    print("朴素贝叶斯算法训练数据数量:%d,预测数量:%d,回款率:%.2f%%"%(lens,all_count,float(right_count)/all_count * 100))

    right_count, all_count, lens = percent(nb, test_x, test_y, test_acc, get_nb_percent)
    print("朴素贝叶斯算法测试数据数量:%d,预测数量:%d,回款率:%.2f%%"%(lens,all_count,float(right_count)/all_count * 100))

def percent(model, x, y, decide_percent, cal_fun):
    right_count = 0
    all_count = 0
    for i in range(len(y)):
        percent = cal_fun(model, x[i])
        if percent > decide_percent:
            if y[i] == u'结项回款':
                right_count += 1
            all_count += 1
    return right_count, all_count, len(y)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("需要输入4个参数:数据源文件地址,生成的统计文件地址,生成的预测文件地址,算法模型名字")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])