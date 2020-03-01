def add(num1, num2):
    sum = 0
    carry = 0
    while (num2 != 0):
        sum = num1 ^ num2
        carry = (num1 & num2) << 1
        num1 = sum
        num2 = carry
    return num1

input = (233,455)
print(add(*input))
