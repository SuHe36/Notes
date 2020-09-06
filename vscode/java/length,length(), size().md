主要记载下length, length(), size()方法的区别以及使用情况。

首先区别一下length和length();

**length不是方法，是属性，数组的属性；**
```
public static void main(String[] args) {
	int[] intArray = {1,2,3};
	System.out.println("这个数组的长度为：" + intArray.length);
}
```

**length()是字符串string的一个方法；**
```
public static void main(String[] args) {
	String str = "HelloWorld";
	System.out.println("这个字符串的长度为：" + str.length());
}
```

进入length()方法看一下具体实现：
```
private final char value[];
 
public int length() {
        return value.length;
    }
```
所以length()方法，归根结底还是用到length这个底层的属性。

**size()方法，是List集合的一个方法；**

```
public static void main(String[] args) {
	List<String> list = new ArrayList<String>();
	list.add("a");
	list.add("b");
	list.add("c");
	System.out.println("这个list的长度为：" + list.size());
}
```
在List的方法中，是没有length()方法的；

看一段ArrayList的源码：
```
private final E[] a;
 
ArrayList(E[] array) {
       if (array==null)
             throw new NullPointerException();
       a = array;
}
 
public int size() {
       return a.length;
}
```
由这段可以看出list的底层实现其实就是数组，size()方法最后要找的其实还是数组的length属性。
**另外，除了list，set和map也有size()方法，所以准确的说,size()方法是针对集合而言。**

总结：
- length：数组的属性
- length()：String的方法
- size()：集合的方法
