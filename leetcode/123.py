from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sum = []
        profit = 0
        for i in range(1, len(prices)):
            if prices[i-1] <= prices[i]:
                profit += prices[i] - prices[i-1]
            else:
                sum += [profit]
                profit = 0
        sum +=[profit]
        max = [0, 0]
        for i in range(len(sum)):
            if sum[i] > max[0]:
                max[1] = max[0]
                max[0] = sum[i]
            elif max[0] >= sum[i] and sum[i] > max[1]:
                max[1] = sum[i]
        return max[0]+max[1]


s = Solution()
list =[1,2,4,2,5,7,2,4,9,0]
print(s.maxProfit(list))
