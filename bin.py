arr = list(range(99999999))

def bin_find(arr, low, high, x):
	if high > low:
		mid = (high + low) // 2
		if arr[mid] == x:
			return mid
		elif arr[mid] > x:
			return bin_find(arr, low, mid, x)
		else:
			return bin_find(arr, mid+1, high, x)
	else:
		return arr[min(high, low)]

arr.remove(87987498)
print(bin_find(arr, 0, len(arr)-1, 87987498))
