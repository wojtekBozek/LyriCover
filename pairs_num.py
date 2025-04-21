def min_elements_for_pairs(x):
    n = 2  # Start with 2 elements, as pairs require at least 2 elements
    while (n * (n - 1)) // 2 < x:
        n += 1
    return n

# Example usage:
x = 1000  # The number of distinct pairs
result = min_elements_for_pairs(x)
print(f"Minimum number of elements required to form {x} distinct pairs: {result}")