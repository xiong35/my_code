
# SQL的操控

## 1 插入

表里有个标志: NULLable, 默认是True  
如果NULLable, 要么都不写, 要么插入NULL  
可省略列名, 但是所有值得和原表一样

    INSERT INTO employees(first_name, last_name)
    VALUE('Tom', NULL);
    # ↑可以批量插入
    # 另一种插入方式(用的少)
    INSERT INTO employees
    SET id=999, first_name='Jobs';

    INSERT INTO employees(id, last_name)
    SELECT 999, 'Tom';

## 2 修改

### 2.1 修改单表

    UPDATE employees
    SET salary = 10000
    WHERE id = 10;

### 2.2 修改多表

    UPDATE boys AS bo
    INNER JOIN beauty AS b
    ON bo.id=b.boyfriend_id
    SET b.phone='119'
    WHERE bo.boy_name='XXX';