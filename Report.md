# CSCE 435 Group project

## 1. Group members:
1. Jessica Williams
2. Stephanie Lam
3. Steve Wang
4. Paola Gallegos Tamez

---

## 2. _due 10/25_ Project topic
Sorting algorithms

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)

We will compare each of the four algorithms (Bucket Sort, QuickSort, Sample Sort, Bubble Sort) by implementing in MPI as well as Cuda. We plan to use reverse sorted, random, and 10% noisy data and compare each of the implementations across those as well.

### Bucket Sort
function bucketsort(data,n_buckets)
	create n_buckets empty arrays
	loop over data:
		add data[i] to correct bucket

	for each bucket:
		bitonicsort(bucket)


	for each bucket:
		copy bucket contents to answer array


### QuickSort
create function that takes in an array, two integers, low and high
	in function
		set integer as pivot, equal to element at position high in array
		set integer as k, equal to low - 1
		for each element from low to high
			if element at array is less than pivot
				k increases
				swap the numbers at position element and k
			
		end for
		swap element at k+1 and element at high
		return k+1

create second function takes in array , two integers, low and hig
	in function
		if low is less than high
			set integer as p, call first function with values given
			recursive call with array, low and p-1
			recursive call with array, p+1, and high
		end if

in main function call second function with array, 0 and n-1, n being the size of the array.

### Bubble Sort
create function that takes in an array of items
loop = list.count
for i = 0 to loop-1
	boolean swapped = false
	for j = 0 to loop-1
	    if list at index j > list at index j+1
	        swap(list at index j, list at index j+1)
	        swapped = true
	    if (not swapped) then break
	return list

### Sample Sort
With unsorted array of size n
Create m buckets (m likely # of threads)

for element in array
    bucket index = element index / m
    add element to bucket[ bucket index]

for each bucket in buckets
    sort bucket with quicksort

need m-1 pivot elements

piv_num = m-1

create sample_selection array
for each bucket in buckets
    mod_number = ceil(number of elements in bucket /(piv_num+1))
    for i in range of 1 and piv_num
        append bucket[imod_number] element to sample_selection


Global splitters = []

sort sample_selection

mod_number= ceil (number of elements in sample_selection/ (piv_num +1))
for i in range of 1 and piv_num
    append sample_selection[ i mod_number] to global splitters

we should have m-1 global splitters, now use these splitters for bucket sort
perform bucket sort given m buckets and the bucket partitions being the global splitters

### 3. Team communication
Our team will mainly be using discord as our means of communication due to the fact that it is easy to use and if we ever need to voice call or meet up remotely, we do not have to set up a zoom meeting.
