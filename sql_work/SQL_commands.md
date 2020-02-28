
# SQL语言常用命令

## 基本命令

| 命令                    | 作用                        |
| ----------------------- | --------------------------- |
| SHOW DATABASES;         | 查看当前所有数据库          |
| USE BaseName;           | 打开指定库                  |
| SHOW TABLES\[FROM Name\]; | 查看当前库/\[Name\]库里所有表 |

## 1 查询

### 1.1 基础查询

    SELECT 查询列表
    FROM 表名

查询列表可以是表里的字段/常量/表达式/函数  
如果列名和关键字冲突, 用``框起来表示这不是关键字

e.g.

```
USE myemployees;
SELECT
  last_name AS 姓,
  salary AS '薪水',
  email 邮箱,
  `name` "名字",
  DISTINCT department_id
FROM employees;
```

notice:

- 加不加AS都行
- 可以把别名引起来避免歧义
- DISTINCT:去重

SELECT还可以当print用:

    SELECT '666'+111;   #out: 777
    SELECT 7%3;
    SELECT version();

### 1.2 条件查询

    SELECT List
    FROM Name
    WHERE condition

e.g.

    SELECT *
    FROM employees
    WHERE salary>12000 or department_id<>90;    #也可以!=

    SELECT *
    FROM employees
    WHERE email LIKE '%@qq____';

    SELECT *
    FROM employees
    WHERE salary BETWEEN 10000 AND 12000;
    
    SELECT *
    FROM employees
    WHERE id IN ('foo','bar');

    SELECT *
    FROM employees
    WHERE bonus <=> 100;

其中 \[ % \] == \[ .* \], \[ _ \] == \[ . \]  
BETWEEN a AND b == <=b and >= a  
\[<=>\] == \[ = OR IS \]  
