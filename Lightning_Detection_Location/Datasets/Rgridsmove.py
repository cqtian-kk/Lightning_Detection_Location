import random
import numpy as np

def generate_grid(width, height, depth):
    """生成三维网格"""
    grid = [[[' ' for _ in range(depth)] for _ in range(height)] for _ in range(width)]
    return grid

def get_neighbours(x, y, z, width, height, depth):
    """获取一个点的所有邻居点"""
    neighbours = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0]:  # 只允许在 z 轴向下移动
                if dx != 0 or dy != 0 or dz != 0:
                    new_x = x + dx
                    new_y = y + dy
                    new_z = z + dz
                    if 0 <= new_x < width and 0 <= new_y < height and 0 <= new_z < depth:
                        neighbours.append((new_x, new_y, new_z))
    return neighbours

def random_walk_dfs(width, height, depth, step_length):
    """随机深度优先搜索"""
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    z = random.randint(depth//8, depth-1)
    path = [(x, y, z)]  # 记录路径
    grid = generate_grid(width, height, depth)
    grid[x][y][z] = 'X'  # 起点
    stack = [(x, y, z)]  # 深度优先搜索使用的栈
    steps = 1  # 记录步数
    while stack:
        if steps >= step_length:
            break
        x, y, z = stack.pop()
        neighbours = get_neighbours(x, y, z, width, height, depth)
        unvisited_neighbours = [(nx, ny, nz) for nx, ny, nz in neighbours if grid[nx][ny][nz] == ' ']
        if unvisited_neighbours:
            new_x, new_y, new_z = random.choice(unvisited_neighbours)
            grid[new_x][new_y][new_z] = 'X'  # 标记新的点为已访问
            stack.append((x, y, z))  # 将当前点入栈
            stack.append((new_x, new_y, new_z))  # 将新的点入栈
            path.append((new_x, new_y, new_z))  # 记录新的点到路径
            steps += 1  # 步数加一

    return path


def random_grids(lats, lons, highs, step_length):
    width, height, depth = len(lats),len(lons),len(highs)
    path = random_walk_dfs(width, height, depth, step_length)
    result = []
    for x, y, z in path:
        result.append([lats[x],lons[y],highs[z]])
    return result







