有时在`git push`之前忘记了`git pull`，那么我们应该怎么解决呢：

只需要执行：
`git pull --rebase origin master`

--rebase的作用就是取消掉本地库中刚刚的commit，并把他们接到更新后的新的版本库中。

![](../figure/99.png)