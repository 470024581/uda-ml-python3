from random import shuffle

a = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
b = [1,2,3,4,5,6]
c = list(zip(a, b))  
shuffle(c)
a, b = zip(*c)
print(a)
print(b)
