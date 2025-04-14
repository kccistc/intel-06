def print_number_square(n):
    matrix = np.arange(1, n*n+1).reshape(n, n)
    for row in matrix:
        print(" ".join(map(str, row)))

n = int(input("Enter the size of the square: "))
print_number_square(n)