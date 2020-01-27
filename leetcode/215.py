class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        list = []
        for i in range(nums.size()):
            if list.size() < k:
                