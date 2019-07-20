import numpy as np

arr = np.array([1, 2, 3, 4], dtype=np.int)

print(arr.dtype)
print(arr.shape)
print(arr.data)
print(arr.strides)


print(arr.size)
print(arr.ndim)

re = arr.reshape(1, 4)

print(re.shape)
print(re)
print(re.itemsize)

range = np.arange(10)
print(range)
