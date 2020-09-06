## 1
在使用java创建`List<List<String>>`类型的数据时，下面这种方法是错误的：

`List<List<String>> res = new ArrayList<ArrayList<String>>()`

正确的方法应该是：
`List<List<String>> res = new ArrayList<>(); `


## 2
在向一个List<List<String>>()对象中添加一个List<String>()数据时，正确的做法是：
```
List<List<String>> res = new ArrayList<>();
List<String> temp = new ArrayList<>();
temp.add("a");
temp.add("b");
res.add(new ArrayList<>(temp));
// 不能直接使用res.add(temp);
```
