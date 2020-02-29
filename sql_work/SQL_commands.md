
# SQL语言之查询

## 基本命令

| 命令                      | 作用                          |
| ------------------------- | ----------------------------- |
| SHOW DATABASES;           | 查看当前所有数据库            |
| USE BaseName;             | 打开指定库                    |
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
    WHERE bonus <=> 100;

\[<=>\] == \[ = OR IS \]  

### 1.3 模糊查询

    SELECT *
    FROM employees
    WHERE email LIKE '%@qq____';

    SELECT *
    FROM employees
    WHERE salary BETWEEN 10000 AND 12000;
    
    SELECT *
    FROM employees
    WHERE id IN ('foo','bar');

其中 \[ % \] == \[ .* \], \[ _ \] == \[ . \]  
BETWEEN a AND b == <=b and >= a  

### 1.4 排序查询

    SELECT (salary*12*IFNULL(bonus,0)) AS 年薪
    FROM employees
    WHERE department_id>=90;
    ORDER BY 年薪 [ASC] DESC

    SELECT *
    FROM employees
    WHERE id IN ('foo','bar')
    ORDER BY salary ASC, employee_id DEC;

### 1.5 多表查询

    SELECT `name`, boyName
    FROM boys, beauty
    WHERE beauty.boyfriend_id=boys.id;

    SELECT e.last_name, e.job_id, j.job_title
    FROM employees AS e, jobs AS j
    WHERE e.job_id=j.job_id;

自连接

    SELECT 
        e.employee_id, e.last_name,
        m.employee_id, m.last_name
    FROM employees e, employees m;

### 1.6 分页查询

    SELECT *
    FROM employees
    WHERE id IN ('foo','bar')
    ORDER BY salary ASC, employee_id DEC
    LIMIT <Begin>, <Num>;

Begin: 开始索引【从0开始！！！】  
Num: 一共要看多少条  
**Begin == (Page - 1) \* Num**

---

## 2 函数

### 2.1 常见函数

#### 2.1.1 字符函数

    CONCAT(string, string[, string])

    LENGTH(string)  # 字节数, utf8下一个汉字3个字节

    IFNULL(obj, rt_val)

    UPPER(string), LOWER(string)

    SBUSTR(string, begin, length)
    # == string[begin-1:begin-1+len]
    # 索引从1开始

    INSTR(string, sub)
    # rt: index of sub [if none: 0]

    TRIM([str FROM ]string)
    # trim beginning/ending spaces[strs] from string

    LPAD(string, len, str)
    # use str to left pad string
    # so that the whole length == len
    # if len(string)>len, the right part will be trimmed

    REPLACE(string, target, dest)

#### 2.1.2 数学函数

    ROUND(num[, acc])
    # 离0四舍五入, 保留acc位小数

    CEIL(num), FLOOR(num)
    # 向上/下取整, 负数也向上/下

    TRUNCATE(num[, acc])
    # 在acc位小数截断

    MOD(a, b)   # a%b, == a-a/b*b

#### 2.1.3 日期函数

    NOW()   # YY-MM-DD HH-MM-SS

    CURDATE()   # YY-MM-DD
    CURTIME()   # HH-MM-SS

    YEAR(date)      # 2019
    MONTH(date)     # 9
    MONTHNAME(date) # September
    HOUR(date)
    # ...

    STR_TO_DATE('9-13@@@1999', '%m-%d@@@%Y')
    DATE_FORMAT('2018/7/6', '%Y年%m月%d日')
    # 1999-09-13
    # Y:4位年份, y:2位年份
    # H:24小时制, h:12小时
    # c:月份1,2...  m:月份01,02...
    # i:分钟01,02...

#### 2.1.4 其他

    SELECT VERSION()
    SELECT DATABASE()   # show current DB
    SELECT USER()

    MD5(string)

    EXISTS(SELECT ...)  # rt 1/0

### 2.2 流程控制函数

    IF(condition, true, false)

    SELECT salary, department_id,
    CASE department_id
    WHEN 30 THEN salary*1.1
    WHEN 40 THEN salary*1.2
    # ...
    ELSE salary
    END AS New_Salary
    FROM employees;

    SELECT salary,
    CASE 
    WHEN salary>20000 THEN 'A'
    WHEN salary>15000 THEN 'B'
    # ...
    ELSE 'D'
    END AS Salary_Rank
    FROM employees;

### 2.3 分组函数

#### 2.3.1 基本函数

    SUM([DSTINCT] index)
    MIN, MAX, AVG, COUNT 
    # MIN/MAX 支持字符串, 日期, 数字
    # 什么都支持
    # 都忽略Null值

#### 2.3.2 分组查询

    SELECT
        ROUND(MAX(salary), 2) AS 'max',
        department_id,
        job_id
    FROM employees
    WHERE salary > 12000 
    GROUP BY department_id,job_id
    HAVING max > 13000
    ORDER BY max ASC;

    SELECT COUNT(*), department_id
    FROM employees
    GROUP BY department_id;
    # 计算每组id包含的人数

GROUP BY后面接多个字段的话, 完全相同的才会分到一个组

WHERE 分组前筛选  
HAVING 分组后筛选  
能WHERE就WHERE  

MySQL支持HAVING后接别名  

ORDER BY在最后

---

## 3 连接

    SELECT List
    FROM Table1
    <连接类型> JOIN Table2
    ON <连接条件>
    # ...

分类

- 内连接: inner
- 外连接
  - 左外: left
  - 右外: right
  - 全外: full
- 交叉连接: cross

### 3.1 内连接

    SELECT city, COUNT(*) AS num
    FROM departments AS d
    INNER JOIN locations AS l
    ON d.location_id = l.location_id
    GROUP BY city
    HAVING num > 3;

### 3.2 外连接

用于查询一个表里有, 另一个表没有的信息  
结果==内连接结果+从表没有的结果(**所有信息**都用Null填充)  

确定主表:

- 左外连接: left 左边为主
- 右外连接: right 右边为主

显然左右连接可以在小幅度修改后替换

    SELECT d.*, e.employee_id
    FROM departments AS d
    LEFT JOIN employees AS e
    ON d.department_id = e.department_id
    WHERE e.employee_id IS NULL;

**全外**  
效果:左外U右外  
MySQL不支持  

### 3.3  交叉连接

笛卡尔乘积

---

## 4 子查询

### 4.1 标量子查询

和一个值比较

    SELECT *
    FROM employees
    WHERE salary > (
        SELECT salary
        FROM employees
        WHERE last_name = 'Abel'
    );

### 4.2 列子查询

和一个竖着的列表比较, 常用IN/NOT IN

### 4.3 行子查询

    SELECT *
    FROM employees
    WHERE (employee_id,salary)=(
        SELECT MIN(employee_id), MAX(salary)
        FROM employees
    );

### 4.4 SELECT子查询

一般可用连接查询代替

    SELECT d.*, (
        SELECT COUNT(*)
        FROM employees AS e
        WHERE e.department_id = d.department_id
    ) AS num
    FROM departments AS d;

### 4.5 表[和谐]子查询

    SELECT avg_dep.*, g.grade_level
    FROM(
        SELECT AVG(salary) AS `avg`, department_id
        FROM employees
        GROUP BY department_id
    ) AS avg_dep
    INNER JOIN job_grade AS g
    ON avg_dep.avg BETWEEN g.lowest_sal AND highest_sal;

## 5 联合查询

用于从多个表里提取共同信息

    SELECT * FROM employees WHERE email LIKE '%a%' OR department_id > 90;
    # ==
    SELECT * FROM employees WHERE email LIKE '%a%'
    UNION
    SELECT * FROM employees WHERE department_id > 90;

UNION: 取并集, 可叠加  
如果列名上下不一致, **以第一个为准**, 但是**数量一定要一样**  
**会自动去除几个表里一样的信息**, 如果不想去, 改成```UNION ALL```
