# git分支

如果想新建一个分支并同时切换到那个分支，可以使用一个带有-b参数的git checkout命令：

    $ git checkout -b mybranck
这样就切换到了mybranck分支。这条命令其实是下面两条命令的简写：

    $ git branck mybranch
    $ git checkout mybranch

那么我们在mybranch上进行了一些修改后，想和原master分支合并的话，我们首先需要切换到master分支，然后再合并mybranch分支：

    $ git checkout master
    $ git merge mybranck

如果我们想要删除mybranck分支，使用如下的命令：

    $ git branch -d mybranch

如果合并两个分支时，存在冲突的话，可以使用git status命令来查看那些因包含冲突而未能合并的文件。

# git版本回退
可以使用git log查看commit ID，然后使用git reset cid来进行回退。
但是有时我们回退了版本，发现回退错误，想撤销回退，这是可以使用git reflog，查看历史所有版本的cid，然后使用git reset cid来切换到那个版本。

git reset --hard cid 可以强制会退到以前的版本，包括把本地的修改复原到以前

**版本回滚之后，如何再回滚到以前的版本呢：使用git reset --hard cid1就回滚到了cid1版本，那么我们如何回滚到cid1以后的版本呢，这是就需要我们使用到git reflog查看以前版本的cid，然后可以继续使用git reset --hard cid2来进行回滚**

