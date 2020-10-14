可以使用java中的`PriorityQueue<>()`接口去创建优先队列，优先队列其实就是一个最大堆或者一个最小堆。


下面以`ListNode`为例，说明建立最小堆与最大堆的区别：

```
        Queue<ListNode> priorityQueue = new PriorityQueue<ListNode>(n, new Comparator<ListNode>(){
            @Override
            public int compare(ListNode l1, ListNode l2){
//           l1表示排在前面的数，l2表示排在后面的数
//          如果l1.val < l2.val，值为负，false,表示不需要调整顺序，也就是升序，
//          如果la.val > l2.val，值为正,true,表示需要调整顺序，也是升序
//            如果想实现降序，返回l2.val -l1.val就可以
                return l1.val - l2.val;
            }
        });
```



同时，我们可以将比较器写在外面,作为一个新的变量：
```
public static Comparator<ListNode> cmp = new Comparator<ListNode>(){
    @Override
    public int compare(ListNode l1, ListNode l2){
        return l1.val - l2.val;
    }
};
Queue<ListNode> prioprityQueue = new PriorityQueue<>(100, cmp);
```


如果传入的是一个具有三个属性值`<age, tall, length>`的`person`类，如果我们想先按照age的从小到大进行排序，如果age相等再按照tall的从小到高进行排序，如果age和tall都相等，则按照length从小到高排序。

```
public static Comparator<person> cperson = new Comparator<person>() {
        @Override
        public int compare(person o1, person o2) {
            if (o1.getAge() != o2.getAge()){
                return o1.getAge() - o2.getAge();
            }
            else if(o1.getAge() == o2.getAge() && o1.getTall()!=o2.getTall()){
                return o1.getTall() - o2.getTall();
            }
            else {
                return o1.getName().length() - o2.getName().length();
            }

        }
    };

Queue<person> priorityQueue = new PriorityQueue<>(100,cperson);
```



还有这种Java8的方法：
`Queue<ListNode> queue = new PriorityQueue<>((v1,v2)->v1.val-v2.val)`