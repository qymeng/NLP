#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import math
import numpy


class HMM:
    def __init__(self, file):
        key = file.split("\\")[2]
        key = key.split("_")[0]
        self.trainFile = file  # 训练集文件
        self.modelFile = ".\\model\\" + key + "_model"  # 模型文件
        self.testFile = ".\\test\\" + key + "_test"  # 测试文件
        self.outputFile = ".\\result\\" + key + "_result"  # 输出文件
        self.MIN_VALUE = -1.0e+100
        # 隐马尔可夫模型(HMM)
        # self.listStatus
        # self.listSymbol
        self.pi = dict()
        self.matrixA = dict()
        self.matrixB = dict()

    # 1.训练模型
    def train(self):
        # (1)----------------统计训练集信息----------------
        fin = open(self.trainFile, mode="r", encoding="utf-8")
        # 定义临时变量
        setStatus = set()
        setSymbol = set()
        statusCount = dict()
        preStatusCount = dict()
        countPi = dict()
        countA = dict()
        countB = dict()
        BEGIN = True  # 句首标记
        # 循环统计
        for line in fin.readlines():
            line = line.strip()  # 可能是空行
            if not line:
                BEGIN = True
            elif BEGIN:  # 句首
                segments = line.split()
                symbol = segments[0]
                status = segments[1]
                setSymbol.add(symbol)  # 加入到集合中
                setStatus.add(status)
                statusCount.setdefault(status, 0)  # 计数状态频度
                statusCount[status] += 1
                countPi.setdefault(status, 0)  # 计数句首状态频度
                countPi[status] += 1
                countB.setdefault(status, {})  # 计数输出映射频度
                countB[status].setdefault(symbol, 0)
                countB[status][symbol] += 1
                preStatus = status
                BEGIN = False
            else:  # 非句首
                segments = line.split()
                symbol = segments[0]
                status = segments[1]
                setSymbol.add(symbol)  # 加入到集合中
                setStatus.add(status)
                preStatusCount.setdefault(preStatus, 0)  # 计数前一个状态的频度
                preStatusCount[preStatus] += 1
                statusCount.setdefault(status, 0)  # 计数状态频度
                statusCount[status] += 1
                countA.setdefault(preStatus, {})  # 计数状态转移频度
                countA[preStatus].setdefault(status, 0)
                countA[preStatus][status] += 1
                countB.setdefault(status, {})  # 计数输出映射频度
                countB[status].setdefault(symbol, 0)
                countB[status][symbol] += 1
                preStatus = status
                BEGIN = False
        fin.close()
        # (2)----------------获取并导出模型信息-----------
        fout = open(self.modelFile, mode="w", encoding="utf-8")
        self.listStatus = sorted(list(setStatus))
        self.listSymbol = sorted(list(setSymbol))
        # Status集合
        fout.write("Status\n" + str(len(setStatus)) + "\n\n")
        # Symbol集合
        fout.write("Symbol\n" + str(len(setSymbol)) + "\n\n")
        # Pi
        for status in self.listStatus:
            self.pi.setdefault(status, self.MIN_VALUE)
        total = 0
        for status, value in countPi.items():
            total += value
        for status in self.listStatus:
            if status in countPi.keys():
                self.pi[status] = math.log(countPi[status]/total)

        fout.write("Pi\n")
        for status in self.listStatus:
            fout.write(status+" "+str(self.pi[status])+"\n")
        fout.write("\n")
        # 矩阵A
        for status1 in self.listStatus:
            self.matrixA.setdefault(status1, {})
            for status2 in self.listStatus:
                self.matrixA[status1].setdefault(status2, self.MIN_VALUE)
        for status1, item in countA.items():
            for status2 in item.keys():
                self.matrixA[status1][status2] = math.log(
                    item[status2] / preStatusCount[status1])

        fout.write("A\n")
        for status1 in self.listStatus:
            for status2 in self.listStatus:
                fout.write(status1+":"+status2+" " +
                           str(self.matrixA[status1][status2])+"    ")
            fout.write("\n")
        fout.write("\n")
        # 矩阵B
        for status in self.listStatus:
            self.matrixB.setdefault(status, {})
            for symbol in self.listSymbol:
                self.matrixB[status].setdefault(
                    symbol, self.MIN_VALUE)
        for status, item in countB.items():
            for symbol in item.keys():
                self.matrixB[status][symbol] = math.log(
                    float(item[symbol])/float(statusCount[status]))

        fout.write("B\n")
        for status in self.listStatus:
            for symbol in self.listSymbol:
                fout.write(status+":"+symbol+" " +
                           str(self.matrixB[status][symbol])+"\n")

        fout.close()

    # 2.Viterbi算法
    def getBValue(self, status, symbol, num):
        if status not in self.matrixB.keys():
            return math.log(1/num)
        elif symbol not in self.matrixB[status].keys():
            return math.log(1/num)
        else:
            return self.matrixB[status][symbol]

    def viterbi(self, buffer):
        length = len(buffer)  # 字符串长度
        num = len(self.listStatus)  # 状态数
        delta = numpy.zeros([length, num])
        path = numpy.zeros([length, num], dtype="int")
        #(1)初始化
        for j in range(num):
            delta[0][j] = self.pi[self.listStatus[j]] + \
                self.getBValue(self.listStatus[j], buffer[0], num)
            path[0][j] = j
        #(2)归纳推导
        for i in range(1, length):
            for j in range(num):
                maxIndex = 0
                max = delta[i-1][0] + \
                    self.matrixA[self.listStatus[0]][self.listStatus[j]]
                for k in range(1, num):
                    temp = delta[i-1][k] + \
                        self.matrixA[self.listStatus[k]][self.listStatus[j]]
                    if temp > max:
                        maxIndex = k
                        max = temp
                delta[i][j] = max + \
                    self.getBValue(self.listStatus[j], buffer[i], num)
                path[i][j] = maxIndex
        #(3)终止和路径读出
        result = list()
        max = delta[length-1][0]
        maxIndex = 0
        for i in range(1, num):
            if delta[length-1][i] > max:
                max = delta[length-1][i]
                maxIndex = i
        result.append(self.listStatus[maxIndex])
        i = length-1
        while i > 0:
            result.append(self.listStatus[path[i][maxIndex]])
            maxIndex = path[i][maxIndex]
            i -= 1

        result.reverse()
        return result

    # 3.测试模型
    def test(self):
        fin = open(self.testFile, mode="r", encoding="utf-8")
        fout = open(self.outputFile, mode="w", encoding="utf-8")
        #循环处理每一句话
        lines = fin.readlines()
        numOfLines = len(lines)
        n = 0
        buffer = list()

        for line in lines:
            n += 1
            line = line.strip()
            if not line and len(buffer) == 0:  # 连续空行
                fout.write("\n")
            elif not line or n == numOfLines:   # 句末或测试文本末
                result = self.viterbi(buffer)
                length = len(result)
                for i in range(length):
                    fout.write(buffer[i]+" "+result[i]+"\n")
                fout.write("\n")
                buffer.clear()
            elif line is not None:
                buffer.append(line)

        fin.close()
        fout.close()
