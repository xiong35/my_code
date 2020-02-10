
# Python OS 模块探究

## 常用函数

### os.access(path,mode)

测试一个文件是否（以怎样的形式）存在，返回值为Bool型  

path为要测试的文件目录  

mode:  

- os.F_OK: 作为access()的mode参数，测试path是否存在
- os.R_OK: 包含在access()的mode参数中 ， 测试path是否可读
- os.W_OK 包含在access()的mode参数中 ， 测试path是否可写
- os.X_OK 包含在access()的mode参数中 ，测试path是否可执行

e.g.

    ret = os.access("/tmp/foo.txt", os.R_OK)
    print ("R_OK - 返回值 %s"% ret)

    True

### os.getcwd()

返回当前目录

### os.listdir(path)

返回path下的所有文件的列表(包括隐藏文件)

### os.chdir(path)

更改目录，返回None

e.g.

    print(os.getcwd())
    print(os.chdir('..'))
    print(os.getcwd())

    /home/ylxiong/Documents/my_code2242787668
    None
    /home/ylxiong/Documents

### os.mkdir(path[, mode])

创建目录

e.g.

    os.mkdir( path, 0755 )

### os.remove(path)

删除**文件**

### os.removedirs(path)

**递归地**删除**目录**  

从叶目录开始删除（若不是叶目录则不能删），遇到非空目录为止

### os.rmdir(path)

只删除一个**空目录**

### os.rename(src, dst)

对文件/目录改名，从src改为det

---

## os.path模块

| 内容                                | 含义                                           |
| ----------------------------------- | ---------------------------------------------- |
| os.path.abspath(path)               | 返回绝对路径                                   |
| os.path.basename(path)              | ～文件名                                       |
| os.path.commonprefix(list)          | 返回list(多个路径)中，所有path共有的最长的路径 |
| os.path.dirname(path)               | 返回(相对当前根目录的)文件路径                 |
| os.path.exists(path)                | 路径存在则返回True,路径损坏返回False           |
| os.path.expanduser(path)            | 把path中包含的"~"和"~user"转换成用户目录       |
| os.path.getctime(path)              | 返回文件 path 创建时间                         |
| os.path.getsize(path)               | 返回文件大小，如果文件不存在就返回错误         |
| os.path.isabs/file/dir(path)        | 判断是否为绝对路径/文件/目录                   |
| os.path.join(path1[, path2[, ...]]) | 把目录和文件名合成一个路径                     |
| os.path.relpath(path[, start])      | 从start开始计算相对路径                        |
| os.path.samefile(path1, path2)      | 判断目录或文件是否相同                         |
| os.path.split(path)                 | 把路径分割成 dirname 和 basename，返回一个元组 |

---

## 不常见函数

### os.chflags(path, flags)，os.chmod(path, mode)， os.chown(path, uid, gid)

改变文件属性

### os.chroot(path)

改变当前进程的根目录
