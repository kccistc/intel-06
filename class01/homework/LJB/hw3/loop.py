#for loop
items = [1, 2, 3,4, 'Hello', 6.24]

for k in range(0, len(items)):
    print(items[k])
print('=====================')
for item in items:
    print(item)
print('=====================')
items = [[1,2], [3,4], [5,6]]
for item in items:
    print(item[0], item[1])
print('=====================')
for item1, item2 in items:
    print(item1, item2)
print('=====================')
info = {'A' : 1, 'B': 2, 'C': 3}
for key in info:
    print(key, info[key])
print('=====================')
for key, value in info.items():
    print(key, value)