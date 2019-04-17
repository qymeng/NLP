#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import sys
import DataProcess
import HMM

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc < 3:
        print("程序结束")
        print("用法：")
        print("\tpython Main.py [语料库文件] [测试文件]")
        sys.exit(1)
    # 1：处理语料库文件
    print("[处理语料库文件]")
    dataProcessInst = DataProcess.DataProcess(sys.argv[1])
    print("输入文件：", sys.argv[1])
    dataProcessInst.toTrainFile()
    print("输出文件：", dataProcessInst.trainFile1, "(用于分词训练)")
    print("输出文件：", dataProcessInst.trainFile2, "(用于词性标注训练)")
    print()

    # 2：训练分词模型
    print("[训练分词模型]")
    hmmInst1 = HMM.HMM(dataProcessInst.trainFile1)
    print("训练文件：", hmmInst1.trainFile)
    hmmInst1.train()
    print("模型文件：", hmmInst1.modelFile)
    print()

    # 3：将测试文件转成指定格式
    print("[处理分词测试文件]")
    dataProcessInst = DataProcess.DataProcess(sys.argv[2])
    print("输入文件：", sys.argv[2])
    dataProcessInst.toTestFile1()
    print("输出文件：", dataProcessInst.testFile1, "(用于分词测试)")
    print()

    # 4：自动分词
    print("[自动分词]")
    print("测试文件：", hmmInst1.testFile)
    hmmInst1.test()
    print("输出文件：", hmmInst1.outputFile)
    print()

    # 5：将输出文件转换成普通文本和标注测试文件
    print("[处理分词输出文件]")
    dataProcessInst = DataProcess.DataProcess(hmmInst1.outputFile)
    print("输入文件：", hmmInst1.outputFile)
    dataProcessInst.toResultFile1()
    print("输出文件：", dataProcessInst.resultFile1, "(用于观察分词结果)")
    dataProcessInst.toTestFile2()
    print("输出文件：", dataProcessInst.testFile2, "(用于词性标注测试)")
    print()

    # 6：训练词性标注模型
    print("[训练词性标注模型]")
    hmmInst2 = HMM.HMM(dataProcessInst.trainFile2)
    print("训练文件：", hmmInst2.trainFile)
    hmmInst2.train()
    print("模型文件：", hmmInst2.modelFile)
    print()

    # 7：词性标注
    print("[词性标注]")
    print("测试文件：", hmmInst2.testFile)
    hmmInst2.test()
    print("输出文件：", hmmInst2.outputFile)
    print()

    # 8：将输出文件转换成普通文本
    print("[处理词性标注输出文件]")
    dataProcessInst = DataProcess.DataProcess(hmmInst2.outputFile)
    print("输入文件：", hmmInst2.outputFile)
    dataProcessInst.toResultFile2()
    print("输出文件：", dataProcessInst.resultFile2, "(用于观察词性标注结果)")
    print()
