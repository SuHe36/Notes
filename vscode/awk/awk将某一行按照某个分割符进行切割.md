参考网址：https://blog.csdn.net/Cengineering/article/details/78529292

在实现用awk将某一行按照某个分割符进行切割之前，我们需要先了解awk的一些内建变量。
## awk的内建变量FS,OFS,RS,ORS,RT,NF,NR,FNR的作用与区别

- FS与OFS的作用与区别
  - FS(Font Space):指定字段列的分隔符,默认是空格
  ![](../figure/18.png)
  - OFS(Output Font Space):指定输出字段的列分隔符,默认是空格
  ![](../figure/19.png)
- RS、ORS、RT的作用与区别
  - RS(Row Space)：指定行分割符，默认是\n
  ![](../figure/20.png)
  - ORS(Output Row Space)：指定输出行分割符。默认是\n
  

## awk实现按照":"符号分割每一行，并且取最后一个元素大于20的行。

现在有一个文件，存储的是dict的(key:value)的形式的数据，文件名为data.txt,比如：
```
a:20
b:10
c:d:30
```
现在我们希望按照":"字符进行分割每一行，因为有的key中包含":"字符，所以我们只能用分割后的最后一个元素与20进行比较。

所以最后使用的命令如下：
```
cat data.txt | awk -F":" '{if($NF>=20) print $0}'
```

其中$NF$表示最后一列