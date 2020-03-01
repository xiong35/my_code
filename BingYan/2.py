# python
def find_zero(nums) -> int:
    num_len = len(nums)
    # 找左边最近的0
    l_zero = []
    for i in range(num_len):
        l_zero.append(float('inf'))
    for index, num in enumerate(nums):
        if num == 0:
            l_zero[index] = 0
        else:
            if index == 0:
                continue
            else:
                l_zero[index] = l_zero[index-1]+1
    # 找右边最近的0
    r_zero = []
    for i in range(num_len):
        r_zero.append(float('inf'))
    for index, num in enumerate(nums[::-1]):
        if num == 0:
            r_zero[index] = 0
        else:
            if index == 0:
                continue
            else:
                r_zero[index] = r_zero[index-1]+1
    # 取两者较小值
    r_zero = r_zero[::-1]
    for i in range(num_len):
        nums[i] = min(l_zero[i],r_zero[i])
    return nums

nums = [1,0,1]
print(find_zero(nums))