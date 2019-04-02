#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import sys


class HMMSegmentation:
	"""隐马尔可夫模型分词类"""

	def __init__(self, file1, file2, file3):
		self.trainingFile = file1  # 训练集文件
		self.testingFile = file2  # 测试集文件
		self.outputFile = file3  # 分词结果

		self.tagList = ['S', 'B', 'M', 'E']  # tag种类
		self.endFlagList = ["。", "？", "！"]  # 句末符号种类，用于计算pi

		self.wordList = list()  # 存放所有种类汉字
		self.tagDict = dict()  # 计数各tag频度
		self.tagWordDict = dict()  # 计数各tag代表各汉字的频度
		self.headCount = dict()  # 计数句首汉字的tag频度(S、B)

		self.pi = {'S': 0.5, 'B': 0.5, 'M': 0, 'E': 0}
		self.matrixA = dict()  # 隐含状态转移矩阵
		self.matrixB = dict()  # 观测状态转移矩阵

	def getTag(self, i, len):
		if len == 1:  # 单字
			return 'S'
		if i == 0:  # 词首
			return 'B'
		if i == len - 1:  # 词尾
			return 'E'
		return 'M'  # 词尾

	def train(self):
		print("[训练开始]")
		#----------------读取文件----------------
		fin = open(self.trainingFile, mode="r", encoding="utf-8")

		separator = "  "
		tagPatternStream = ""

		for line in fin:
			headFlag = True  # 每一行都置句首flag为true
			result = line.split(separator)
			for segment in result:
				segment = segment.strip()
				lenSegment = len(segment)
				for i in range(lenSegment):
					word = segment[i]
					#存这个汉字
					if word not in self.wordList:
						self.wordList.append(word)
					#获取汉字的tag，并计数tag频度
					tag = self.getTag(i, lenSegment)
					self.tagDict.setdefault(tag, 0)
					self.tagDict[tag] += 1
					#加入到tag字符串中
					tagPatternStream += tag
					#记录tag与汉字的映射频度
					self.tagWordDict.setdefault(tag, {})
					self.tagWordDict[tag].setdefault(word, 0)
					self.tagWordDict[tag][word] += 1
					#记录句首tag频度
					if headFlag:
						self.headCount.setdefault(tag, 0)
						self.headCount[tag] += 1
						headFlag = False
					if word in self.endFlagList:
						headFlag = True
			tagPatternStream += ","
		fin.close()
		#---------------tag转移矩阵----------------------
		tagTransDict = dict()
		lenTagPatternStream = len(tagPatternStream)
		for i in range(lenTagPatternStream-1):
			tag1 = tagPatternStream[i]
			tag2 = tagPatternStream[i+1]
			if tag1 == ',' or tag2 == ',':
				continue
			tagTransDict.setdefault(tag1, {})
			tagTransDict[tag1].setdefault(tag2, 0)
			tagTransDict[tag1][tag2] += 1
		#----------------------------------------
		#计算分布情况pi
		self.pi['S'] = float(self.headCount['S']) / \
                    float(self.headCount['S']+self.headCount['B'])
		self.pi['B'] = 1-self.pi['S']
		print("初始tag分布(π)：")
		for temp in self.pi.items():
			print(temp)
		#计算matrixA(4x4)
		for tag1 in self.tagList:
			self.matrixA.setdefault(tag1, {})
			for tag2 in self.tagList:
				self.matrixA[tag1].setdefault(tag2, 0)
		# A[tag1][tag2]=P(tag2|tag1)=P(tag1·tag2)/P(tag2)
		for tag1, item in tagTransDict.items():
			for tag2 in item.keys():
				self.matrixA[tag1][tag2] = float(item[tag2])/float(self.tagDict[tag1])
		# 初始化matrixB
		for tag in self.tagList:
			self.matrixB.setdefault(tag, {})
			for word in self.wordList:
				self.matrixB[tag].setdefault(word, 1.0/float(self.tagDict[tag]))
				#self.matrixB[tag].setdefault(word, 0)
		# B[tag][word]=P(word|tag)=P(word·tag)/P(tag)
		for tag, item in self.tagWordDict.items():
			for word in item.keys():
				self.matrixB[tag][word] = float(item[word])/float(self.tagDict[tag])
		#----------------------------------------
		print("[训练结束]")

	def test(self):
		print("[测试开始]")
		#----------------Viterbi算法------------------------
		fin = open(self.testingFile, mode="r", encoding="utf-8")
		fout = open(self.outputFile, mode="w", encoding="utf-8")

		#计算weight(状态tag下，前一个字是word的可能性)
		weight = dict()
		count = 0
		for line in fin:
			count += 1
			line = line.strip()
			lenLine = len(line)
			if lenLine <= 0:  # 会去掉空行
				continue
			word = line[0]
			for tag in self.tagList:
				weight.setdefault(1, {})
				if word not in self.matrixB[tag].keys():
					self.matrixB[tag].setdefault(word, 1.0/float(self.tagDict[tag]))
				weight[1].setdefault(tag, self.pi[tag]*self.matrixB[tag][word])
			for i in range(1, lenLine):
				word = line[i]
				weight.setdefault(i+1, {})
				for tag1 in self.tagList:
					weight[i+1].setdefault(tag1, 0)
					maxProb = 0
					for tag2 in self.tagList:
						maxProb = max(maxProb, weight[i][tag2]*self.matrixA[tag2][tag1])
					if word not in self.matrixB[tag1].keys():
						self.matrixB[tag1][word] = 1.0 / float(self.tagDict[tag1])
					weight[i+1][tag1] = maxProb * self.matrixB[tag1][word]
			#计算path后回溯
			tagPath = list()
			buffer = list()
			for tag in self.tagList:
				buffer.append([tag, weight[lenLine][tag]])
			tag1, _ = max(buffer, key=lambda x: x[1])
			tagPath.append(tag1)

			for i in range(lenLine, 1, -1):
				buffer = list()
				for tag in self.tagList:
					buffer.append([tag, weight[i-1][tag]*self.matrixA[tag][tag1]])
				tag1, _ = max(buffer, key=lambda x: x[1])
				tagPath.append(tag1)

			tagPath.reverse()
			result = ""

			lenPath = len(tagPath)
			for i in range(lenPath):
				tag = tagPath[i]
				if tag == 'S':
					result += (line[i]+"  ")
					continue
				if tag == 'B' or tag == 'M':
					result += line[i]
					continue
				if tag == 'E':
					result += (line[i]+"  ")
			result += '\n'
			fout.writelines(result)

		fout.close()
		fin.close()
		print("共计：", count, "行")
		#----------------------------------------
		print("[测试结束]")
		print("输出文件：", self.outputFile)


if __name__ == "__main__":
	argc = len(sys.argv)
	if(argc < 4):
		print("程序结束")
		print("用法：")
		print("\tpython main.py [训练集文件] [测试集文件] [输出文件]")
		sys.exit(1)
	print("[程序运行]")
	print("训练集：", sys.argv[1])
	print("测试集：", sys.argv[2])

	segInst = HMMSegmentation(sys.argv[1], sys.argv[2], sys.argv[3])
	segInst.train()
	segInst.test()
