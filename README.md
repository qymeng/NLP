# NLP
## (1)HMMSegmentation

## (2)HMMPOSTagger
> HMMPOSTagger
>> __pycache__              (python生成的缓存文件)
>> data                     (输入数据目录)
>>> 199801.txt              (语料库文件)
>>> postag_train.utf8       (处理语料库文件得到的用于词性标注的训练文件)
>>> wordseg_train.utf8      (处理语料库文件得到的用于自动分词的训练文件)
>> model                    (模型文件目录)
>>> postag_model            (训练得到的词性标注HMM数据)
>>> wordseg_model           (训练得到的自动分词HMM数据)
>> result                   (输出文件目录)
>>> postag_result           (词性标注测试得到的输出文件)
>>> postag_result.txt       (转换得到的直观的输出文件)
>>> wordseg_result          (自动分词得到的输出文件)
>>> wordseg_result.txt      (转换得到的直观的输出文件)
>> test                     (测试文件目录)
>>> postag_test             (将自动分词得到的结果文件处理成词性标注的输入文件)
>>> test1.txt               (测试文件1)
>>> test2.txt               (测试文件2)
>>> wordseg_test            (将某个输入文件处理成自动分词测试的输入文件)
>> DataProcess.py           
>> HMM.py       
>> Main.py
>> test.bat                 (用于测试的批处理文件)