#!/usr/bin/python3
# -*- coding: UTF-8 -*-


class DataProcess:
    """处理语料库文件的类"""

    def __init__(self, file):
        self.rawFile = file
        self.trainFile1 = ".\\data\\wordseg_train.utf8"
        self.trainFile2 = ".\\data\\postag_train.utf8"
        self.testFile1 = ".\\test\\wordseg_test"
        self.testFile2 = ".\\test\\postag_test"
        self.resultFile1 = ".\\result\\wordseg_result.txt"
        self.resultFile2 = ".\\result\\postag_result.txt"

    #将输入文件转换成用于分词和用于词性标注的训练文件
    def toTrainFile(self):
        fin = open(self.rawFile, mode="r", encoding="gbk")
        fout1 = open(self.trainFile1, mode="w", encoding="utf-8")
        fout2 = open(self.trainFile2, mode="w", encoding="utf-8")

        for line in fin.readlines():
            line = line.strip()
            if line is None:  # 训练文件，不需要留出空行
                continue
            segmentList = line.split("  ")
            for segment in segmentList[1:]:
                segment = segment.strip()
                word = segment.split("/")[0]
                word = word.lstrip("[")
                tag = segment.split("/")[1]
                tag = tag.split("]")[0]
                #写入用于分词的训练文件
                length = len(word)
                if length == 1:
                    fout1.write(word + " S\n")
                elif length == 2:
                    fout1.write(word[0] + " B\n")
                    fout1.write(word[1] + " E\n")
                else:
                    fout1.write(word[0] + " B\n")
                    for i in range(1, length - 1):
                        fout1.write(word[i] + " M\n")
                    fout1.write(word[-1] + " E\n")
                #写入用于词性标注的训练文件
                fout2.write(word + " " + tag + "\n")
            fout1.write("\n")
            fout2.write("\n")

        fin.close()
        fout1.close()
        fout2.close()

    #将输入文件转换成分词测试文件
    def toTestFile1(self):
        fin = open(self.rawFile, mode="r", encoding="gbk")
        fout = open(self.testFile1, mode="w", encoding="utf-8")

        for line in fin.readlines():
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            length = len(line)
            for i in range(0, length):
                fout.write(line[i]+"\n")
            fout.write("\n")

        fin.close()
        fout.close()

    #将输入文件转换成结果文件
    def toResultFile1(self):
        fin = open(self.rawFile, mode="r", encoding="utf-8")
        fout = open(self.resultFile1, mode="w", encoding="utf-8")

        buffer = ""
        for line in fin.readlines():
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            symbol = line.split()[0]
            status = line.split()[1]
            buffer += symbol
            if status == "S" or status == "E":
                fout.write(buffer+" ")
                buffer = ""

        fin.close()
        fout.close()

    #将输入文件转换成词性标注测试文件：
    def toTestFile2(self):
        fin = open(self.rawFile, mode="r", encoding="utf-8")
        fout = open(self.testFile2, mode="w", encoding="utf-8")

        buffer = ""
        for line in fin.readlines():
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            symbol = line.split()[0]
            status = line.split()[1]
            buffer += symbol
            if status == "S" or status == "E":
                fout.write(buffer+"\n")
                buffer = ""

        fin.close()
        fout.close()

    #将输入文件转换成结果文件
    def toResultFile2(self):
        fin = open(self.rawFile, mode="r", encoding="utf-8")
        fout = open(self.resultFile2, mode="w", encoding="utf-8")

        for line in fin.readlines():
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            symbol = line.split()[0]
            status = line.split()[1]
            fout.write(symbol+"/"+status+" ")

        fin.close()
        fout.close()
