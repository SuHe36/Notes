#使用Moses的训练语言模型与翻译模型的记录

参考网址：https://github.com/txAnnie/Moses/blob/master/moses%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83%E4%B8%8E%E6%B5%8B%E8%AF%95


语料准备：
为了训练一个翻译系统，我们需要一个在句子上对齐的平行语料
cd Moses
mkdir corpus
cd corpus
wget http://www.statmt.org/wmt13/training-parallel-nc-v8.tgz
tar zxvf training-parallel-nc-v8.tgz

tokenlsation:在语料的单词和单词之间或者单词和标点之间插入空白，然后进行后续操作。
/home/tx/Moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < /home/tx/Moses/corpus/training/news-commentary-v8.fr-en.en > /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.en

/home/tx/Moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < /home/tx/Moses/corpus/training/news-commentary-v8.fr-en.fr > /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.fr

Truecaser：提取一些关于文本的统计信息
/home/tx/Moses/mosesdecoder/scripts/recaser/train-truecaser.perl --model /home/tx/Moses/corpus/truecase-model.en --corpus /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.en

/home/tx/Moses/mosesdecoder/scripts/recaser/train-truecaser.perl --model /home/tx/Moses/corpus/truecase-model.fr --corpus /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.fr

truecasing：将语料中每句话的字和词组都转换为没有格式的形式，减少数据稀疏性问题
/home/tx/Moses/mosesdecoder/scripts/recaser/truecase.perl --model /home/tx/Moses/corpus/truecase-model.en < /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.en > /home/tx/Moses/corpus/news-commentary-v8.fr-en.true.en

/home/tx/Moses/mosesdecoder/scripts/recaser/truecase.perl --model /home/tx/Moses/corpus/truecase-model.fr < /home/tx/Moses/corpus/news-commentary-v8.fr-en.tok.fr > /home/tx/Moses/corpus/news-commentary-v8.fr-en.true.fr

cleaning： 将长语句和空语句删除，并且将不对齐语句进行处理。
/home/tx/Moses/mosesdecoder/scripts/training/clean-corpus-n.perl /home/tx/Moses/corpus/news-commentary-v8.fr-en.true fr en /home/tx/Moses/corpus/news-commentary-v8.fr-en.clean 1 80
~、

语言模型训练（法语到英语）:
使用Kenlｍ生成一个合适的３元语言模型
/home/tx/Moses/mosesdecoder/bin/lmplz -o 3 < /home/tx/Moses/corpus/news-commentary-v8.fr-en.true.en > news-commentary-v8.fr-en.arpa.en

将*.arpa.en文件二值化：

/home/tx/Moses/mosesdecoder/bin/build_binary news-commentary-v8.fr-en.arpa.en news-commentary-v8.fr-en.bl

训练翻译系统：
cd corpus
nohup nice ~/Moses/mosesdecoder/scripts/training/train-model.perl -root-dir train -corpus ../news-commentary-v8.fr-en.clean -f fr -e en -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:$HOME/lm/news-commentary-v8.fr-en.blm.en:8-external-bin-dir ~／Moses/mosesdecoder/tools >& training.out &
(******nohup 指后台不挂断地运行命令*******
nohup命令：如果你正在运行一个进程，而且你觉得在退出帐户时该进程还不会结束，那么可以使用nohup命令。该命令可以在你退出帐户/关闭终端之后继续运行相应的进程。nohup就是不挂断的意思( no hang up)。
使用 jobs 查看任务。
使用 fg %n　关闭。)

优化（调整）：
cd ~/corpus
wget http://www.statmt.org/wmt12/dev.tgz
tar zxvf dev.tgz

tokenise and truecase the corpus we need:
cd corpus/
/home/tx/Moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en < dev/news-test2008.en > news-test2008.tok.en
/home/tx/Moses/mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr < dev/news-test2008.fr > news-test2008.tok.fr
/home/tx/Moses/mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.en < news-test2008.tok.en > news-test2008.true.en
/home/tx/Moses/mosesdecoder/scripts/recaser/truecase.perl --model truecase-model.fr < news-test2008.tok.fr > news-test2008.true.fr

优化过程：
cd working
nohup nice ／home/tx/Mose/mosesdecoder/scripts/training/mert-moses.pl /home/tx/Mose/corpus/news-test2008.true.fr /home/tx/Mose/corpus/news-test2008.true.en /home/tx/Mose/mosesdecoder/bin/moses train/model/moses.ini --mertdir /home/tx/Mose/mosesdecoder/bin/ &> mert.out &	

测试：
/home/tx/Moses/mosesdecoder/bin/moses -f /home/tx/Moses/corpus/working/mert-work/moses.ini 
然后输入一句英文。

生成n-best list:
/home/tx/Moses/mosesdecoder/bin/moses -f moses.ini -n-best-list listfile 5 < in
(in为输入文件.　　listfile为输出文件)
	


注：服务器下working 里为法语到英语模型，working2　里为中文错字到正字模型。