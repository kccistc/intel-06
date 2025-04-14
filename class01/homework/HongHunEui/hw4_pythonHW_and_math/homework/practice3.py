n = input()
n = int(n)

numbers = list(range(1, n * n + 1))

for i in range(n):
    row = numbers[i * n:(i + 1) * n]
    print(' '.join(map(str, row)))