python的字典的键是不可变的，对应的就是一个对象能不能作为字典的key，就取决于他有没有__hash__方法，
所以所有python自带的对象中，除了list,set,dict以及内部包含上面这三种类型之一的tuple不能作为python字典的key以外，
其他的对象都能作为key.


而为什么可以通过__hash__方法来判断数据是否是可变的呢。
可以查看参考资料：https://blog.csdn.net/lnotime/article/details/81192207


这里有个很好的尝试，就是我们之所以说list,set和dict没有__hash__方法是因为，他们的数据结构内部__hash__函数反回了None。

这里作者将list的__hash__返回了```hash(self[0])```，也就是返回了list的第一个元素的hash值。

此时list就可以作为dict的key了。

所以通过这个例子我们可以知道，判定一个数据类型能够作为key，就是这个数据是否是不可变的数据类型，也就是__hash__能够返回一个值。

真正的python里面判定键值是否是唯一的，是通过__hash__函数和__eq__函数一起来确定的。
_hash__返回值相等，且eq判断也相等，才会被认为是同一个键。


