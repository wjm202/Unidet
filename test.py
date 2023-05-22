import sys
import collections
n,m=3,3
# for line in sys.stdin:
#     i+=1
#     if i<=n:
#         matrix_number.append(list(map(int,line.split(" "))))
#     elif i<=2*n:
#         matrix_weight.append(list(map(int,line.split(" "))))
#     elif i==2*n+1:
#         q=int(line)
#     else:
#         coordinate.append(list(map(int,line.split(" "))))
matrix_number=[[1, 2, 3],[4,5,6],[7,8,9]]
matrix_weight=[[1,1,1],[1,1,1],[1,1,1]]
q=3
coordinate=[[1,2],[2,2],[3,3]]
def bfs(x,y,x_final,y_final):
    global matrix_number,matrix_weight,m,n,visited,anns
    if x==x_final and y==y_final:
        anns+=1
        return
    visited[x][y]=True
    queue=collections.deque([])
    queue.append((x,y))
    while queue:
        x,y=queue.popleft()
        for new_x, new_y in [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]:
            if new_x>=0 and new_x<n and new_y>=0 and new_y<m and \
                matrix_number[new_x][new_y]>=matrix_weight[x][y]+matrix_number[x][y]\
                and visited[new_x][new_y]==False:
                queue.append((new_x,new_y))
    return


for i in range(q):
    x_final,y_final=coordinate[i][0]-1,coordinate[i][1]-1
    anns=0
    for x in range(n):
        for y in range(m):
            visited=[[False]*n for _ in range(m)]
            bfs(x,y,x_final,y_final)
    print(anns)
