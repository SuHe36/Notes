有时我们想将两个文件按行合并。比如机器翻译中常用的source文件和target文件。

这时我们可以使用`paste -d`命令进行合并，但是`paste -d`命令只能采取单个字符作为间隔符，
如果我们想采用多个字符，如` ### `做分隔符，我们可以使用`/dev/null`作为空文件，来实现这一目标。

比如有file1和file2，那么如果我们想使用`abc`作为分隔符，那么我们可以使用如下的命令：

`paste -d abc file1 /dev/null /dev/null file2`

这个命令也等价于：

`paste -d abc file1 - - file2 < /dev/null`

那么我们想要实现使用`###`做分隔符，和上面的方法一样，只是把`abc`换成`###`

