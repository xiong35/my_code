
# 《精通python设计模式》读书笔记

# SOLID原则

|     |                                     |              |
| --- | ----------------------------------- | ------------ |
| SRP | The Single Responsibility Principle | 单一责任原则 |
| OCP | The Open Closed Principle           | 开放封闭原则 |
| LSP | The Liskov Substitution Principle   | 里氏替换原则 |
| ISP | The Interface Segregation Principle | 接口分离原则 |
| DIP | The Dependency Inversion Principle  | 依赖倒置原则 |

---

## 1 工厂模式（Factory）

客户端请求一个对象（而不是自己实例化对象），而无需知道这个对象是谁生成的

### 1.1 工厂方法

执行单个函数，他接受一个参数来指明我们需要什么，由工厂方法来处理所有细节  

适用条件：创建对象的代码散落在不同的地方，难以跟踪他们的情形  
一般会根据对象种类分组，创建多个工厂方法  

### 1.2 抽象工厂

是一组工厂方法，每个方法产生不同种的对象，但是这一组方法具有相似性，会放在一起用  

---

## 2 建造者模式（Builder）

当一个对象需要多个步骤来创造，并且要求可以自定义构造过程时就可以用建造者模式  
如：创建html时，body, head 等部分都要一步一步单独创造  

组成成分：

- Builder：内置一系列统一接口的方法，用来个性化的创造一个对象
- Director：接受参数来指派builder创造对象

> 对比工厂模式：建造者模式适合创造复杂对象

## 3 原型模式（prototype）

当我们已有一个对象，并想创建他的一个副本的时候使用原型模式
一般用```copy.deepcopy()```方法复制，再用```\_\_dict\_\_.update(**kwargs)微调  

## 4 适配器模式（Adapter）

一个额外的代码层，能让两个接口通信  
常用字典储存{通用接口：特殊方法}来实现  

## 5 修饰器模式（Decorator）

能在不影响原有功能的情况下添加功能  

修饰器中定义一个修饰函数  
修饰器接受一个函数作为参数，返回包装过的**函数**  

e.g. ：定义一个包装递推数列，增加储存功能的修饰器

    def memoize(fn):
        known = dict()

        @functools.wraps(fn)
        def memoizer(*args):
            if args not in known:
                known[args] = fn(*args)
            return known[args]
            
        return memoizer

其中```@functools.wraps(fn)```的作用是保证包装后函数的名字等信息不改变，一般包装时都会加上这一句，不加也无太大妨碍  
