我们可以使用一个kenlm的python包去训练一个语言模型，并对每个句子进行打分。


安装kenlm:
```
pip install https://github.com/kpu/kenlm/archive/master.zip
```

## 训练语言模型
首先下载语言数据，我们可以下载Bible数据：
```
wget https://github.com/vchahun/notes/raw/data/bible/bible.en.txt.bz2
```
然后创建一个process.py文件，对数据进行分词等预处理：
```
import sys
import nltk

for line in sys.stdin:
    for sentence in nltk.sent_tokenize(line):
        print(' '.join(nltk.word_tokenize(sentence)).lower())
```

接下来就是训练一个语言模型：
```
bzcat bible.en.txt.bz2 |\
python process.py |\
./kenlm/bin/lmplz -o 3 > bible.arpa
```

然后可以将训练好的语言模型转换为二进制格式，便于导入模型等：
```
./kenlm/bin/build_binary bible.arpa bible.klm
```

最后，我们就可以使用这个语言模型去对每个句子进行打分：
```
import kenlm
model = kenlm.LanguageModel('bible.klm')
model.score('in the beginning was the word')
```
