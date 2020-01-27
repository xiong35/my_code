from typing import List


class Solution:
    move = [
        [[0, 0], [-1, 1]],
        [[0, 0], [1, -1]],
        [[1, 0], [1, 0]],
        [[-1, 0], [-1, 0]],
        [[0, 1], [0, 1]],
        [[0, -1], [0, -1]]
    ]
    positions = [[0, 0], [1, 0]]
    count = 0
    min = 99999

    def minimumMoves(self, grid: List[List[int]]) -> int:
        x = len(grid[0])
        y = len(grid)
        if self.positions[0][1] == y-1 and self.positions[1][0] == x-1:
            self.min = min(self.count, self.min)
            return 0
        for i in range(6):
            temp = self.move[i]+self.positions
            x1 = temp[0][0]
            y1 = temp[0][1]
            x2 = temp[1][0]
            y2 = temp[1][1]
            if not(x1 >= 0 and y1 >= 0 and x2 < x and y2 < y):
                continue
            if (grid[y1][x1] or grid[y2][x2]):
                continue
            if not(x1 == x2 or y1 == y2):
                continue
            positions = temp
            self.count += 1
            self.minimumMoves(grid)
            self.count -= 1
            positions = temp - self.move[i]
        return min


s = Solution()
grid = [[0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0, 0]]
print(s.minimumMoves(grid))
