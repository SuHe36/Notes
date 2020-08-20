在使用git时，我们有时commit后，发现提交了一些.pyc文件。

那么我们可以使用`git rm --cached *.pyc`命令来删除他们；然后在执行`git commit --amend`就可以了。

当然可以修改.gitignore文件，默认在提交时删除那些.]