

把数据文件输入到datareader.py里面的_read()函数里，然后读取出包含[tokens,metadata, labels, d_tags]等key的dict，
这个dict又被封装成一个instance

然后这些封装的instance在trainer.py的315,316行
` raw_train_generator = self.iterator(self.train_data, num_epochs=1, shuffle=self.shuffle)`
`  train_generator = lazy_groups_of(raw_train_generator, num_gpus)`
来再去调用allennlp的data_iterator.py的__call__()函数结构，实现由token转换为词表里面的下标。
这个词典的嵌入是在train.py里面的`iterator.index_with(vocab)`来嵌入的。

然后最后是在trainer.py里面的第339行`batch_group`可以认为是最后生成的一个batch的数据，
这里的batch_group就是一个dict，是调用前面的allenNLP的接口data_iterator.py的接口，__call__()函数，生成的一个batch的训练数据。
里面存储了4个key，分别是[tokens, metadata, labels, d_tags]，对应的value是他们对应的vocab.txt里面的下标值。


现在的问题是，参数能够传递到seq2labels_model里面，但是在调用self.text_field_embedder的时候，只传入了tokens，没有传入insert_word的信息


可以考虑对allennlp中的basic_text_field_embedder.py进行重写，重写里面的forward函数