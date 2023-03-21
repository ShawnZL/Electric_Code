# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import csv
import json
import os
import shutil

info_list = ['表无封印', '封印缺失', '夸接线', '表箱锁坏', '表箱无资产编号', '表箱门坏', '门叶子坏', '表箱无箱盖', '电表下盖有损坏','表未固定','表无电']
info_dic = {'表无封印': '1.txt', '封印缺失': '2.txt', '夸接线': '3.txt', '表箱锁坏': '4.txt', '表箱无资产编号': '5.txt', '表箱门坏': '6.txt', '门叶子坏': '7.txt', '表箱无箱盖': '8.txt', '电表下盖有损坏': '9.txt','表未固定':'10.txt', '表无电':'11.txt'}
info_list2 = ['现场正常', '电表无封印', '无危险标志', '表盖倾斜', '表无封印', '表箱门锁坏', '表箱无箱盖', '表无封', '表箱无锁', '表箱门坏', '表失电', '表箱无箱']
info_dic2 = {'现场正常': '1', '电表无封印': '2', '无危险标志': '3', '表盖倾斜': '4', '表无封印': '5', '表箱门锁坏': '6',
             '表箱无箱盖': '7', '表无封': '8', '表箱无锁': '9', '表箱门坏': '10', '表失电': '11', '表箱无箱': '12'}
def txtTocsv():
    """
    :param name: txtTocsv
    :return: .csv
    """
    out = open('test.csv', 'w', newline='')  # 要转成的.csv文件，先创建一个LF1big.csv文件
    csv_writer = csv.writer(out, dialect='excel')

    f = open("test.txt", "r")
    for line in f.readlines():
        # line = line.replace(',', '\t')  # 将每行的逗号替换成空格
        list = line.split()  # 将字符串转为列表，从而可以按单元格写入csv
        csv_writer.writerow(list)

def doTXT():
    """
    去除""
    :return: 返回一个txt文件
    """
    ss = "elec.txt"
    s2 = "test1.txt"
    with open(ss, "r", encoding='UTF-8-sig') as fr, open(s2, "w", encoding='UTF-8-sig') as fw:
        file_data = fr.readlines()
        for row in file_data:  # 读取每一行
            tmp = str(row).strip("\n\r").split(',')  # 以","为分界符，分成数组
            print(tmp)
            b = eval(tmp[9])  # eval为python自带函数，可以去掉数组值两边引号，具体可查
            l = row.replace(tmp[9], b)
            fw.write(l)
    os.remove(ss)
    os.rename(s2, ss)

def delete_blank():
    """
    删除file2和info无内容的行
    :return:test1.csv
    """
    df = pd.read_csv("test1.csv")
    for i in range(len(df)):
        if df['file2'][i] == '"[]"' and df['info'][i] == '""':
            print(df.loc[i])
            # df.drop(i,inplace=True)
    # df.to_csv('test1.csv',index=False,encoding="utf-8")

def select_info():
    """
    选择file2和info空的分开处理
    :return: file.csv和info.csv
    """
    df = pd.read_csv("info.csv")
    for i in range(len(df)):
        # if df['file2'][i] == '"[]"' and df['info'][i] == '""':
        if df['file2'][i] != '"[]"':
            print(df.loc[i])
            df.drop(i,inplace=True)
    df.to_csv('info.csv',index=False,encoding="utf-8")

def select_file():
    """
    处理file.csv，目的就是将其中的file2分解开来。
    :return:
    """
    df = pd.read_csv('file.csv')
    # print(df.to_string())
    print(type(df['file2']))
    str1 = r'\"path\":\"'
    str2 = r'\",\"info\":'
    code_end = r',\"code\":\"\"}'
    strend = r',\"code\":\"\"}]'
    for row in range(10,len(df)):
        print(df['file2'][row])
        tempstr = df['file2'][row]
        len4 = tempstr.find(code_end)
        len5 = tempstr.find(strend)
        tempend = len4 + 17
        len1 = int(tempstr.find(str1)) # +11照片名开始
        len2 = int(tempstr.find(str2)) # 照片名结束位置
        len3 = int(tempstr.find(strend))
        # 照片名
        # print(tempstr[len1 + 11:len2])
        name = tempstr[len1 + 11: len2]

        # info
        temp_info = tempstr[len2 + 12: len4]
        info_deal(temp_info, name)
        print(temp_info)
        # print(len(tempstr))
        print('begin')
        while (len4 != len5):
            tempstr = tempstr[len4 + 17:]
            len4 = tempstr.find(code_end)
            len5 = tempstr.find(strend)
            tempend = len4 + 17
            len1 = int(tempstr.find(str1))  # +11照片名开始
            len2 = int(tempstr.find(str2))  # 照片名结束位置
            len3 = int(tempstr.find(strend))
            # 照片名
            print(tempstr[len1 + 11:len2])
            name = tempstr[len1 + 11: len2]

            # info
            temp_info = tempstr[len2 + 12: len4]
            info_deal(temp_info, name)
            print(temp_info)
            print(len(tempstr))
            print('end')

def info_deal(str, name):
    """
    处理info信息
    :param str: info
    :return: 返回进入list中
    """
    len1 = str.find(r'\"')
    while(len1 + 3 != len(str)):
        print(str)
        str = str[len1 + 2:]
        len1 = str.find(r'\"')
        info = str[0:len1]
        print(info)
        # if (info != ',' and (info not in temp_list)):
        #     temp_list.append(info)
        if (info != ','):
            path = info_dic[info]
            with open(path,'a+') as fw:
                fw.write(name + '\n')

def read_txt():
    """
    读取txt文件将并且替换其中的字符
    :return:
    """
    with open('elec.txt',"r", encoding="UTF-8-sig") as fr, open('test.txt', 'w', encoding='UTF-8-sig') as fw:
        file_data = fr.readlines()
        d_file = 'temp%d'
        for row in file_data:
            row = row.replace(']","[', ']" "[')
            row = row.replace('","', '" "')
            fw.write(row)


def select_info():
    """
    处理info.csv信息文件
    :return:
    """
    df = pd.read_csv('info.csv')
    """
    写信息进入info_list2
    for i in range(len(df)):
        temp_str = df['info'][i]
        if temp_str[1:-1] not in info_list2:
            info_list2.append(temp_str[1:-1])
    print(info_list2)
    """
    for i in range(len(df)):
        str1 = df['file1'][i]
        info1 = df['info'][i]
        info1 = info1[1:-1]
        print(info1)
        print(str1)
        str_path = r'{\"path\":\"'
        str_end = r'\",\"info\":'
        for inum in range(3):
            start_path = str1.find(str_path)
            end_path = str1.find(str_end)
            print(start_path, end_path)
            # name
            path_name = str1[start_path+12: end_path]
            write_info(info_dic2[info1], inum, path_name)
            print(str1[start_path+12: end_path])
            str1 = str1[end_path+10:]
            print(str1)

def write_info(str1, num, path_name):
    path = str1 + '.' + str(num) + '.txt'
    with open(path, 'a+') as fw:
        fw.write(path_name + '\n')

def read_picture():
    """
    根据txt读取picture
    :return: file
    """

    with open('info/12.2.txt', 'r') as fr:
        for line in fr.readlines():
            line = line.strip('\n')
            srcfile = os.path.join(r'/Users/shawnzhao/Downloads/sample', line) # 选定文件
            dstfile = os.path.join(r'/Users/shawnzhao/Downloads/pic/12.2/',line) # 指定文件夹
            try:
                shutil.copyfile(srcfile, dstfile) # 复制文件
                os.remove(srcfile)
            except:
                print(line)

if __name__ == '__main__':
    # select_file()
    # info_deal(r'[\"表箱锁坏\",\"表箱门坏\",\"表箱无箱盖\"]','1.jpg')
    read_picture()