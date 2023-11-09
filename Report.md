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
```
function bucketsort(data,n_buckets)
	create n_buckets empty arrays
	loop over data:
		add data[i] to correct bucket

	for each bucket:
		bitonicsort(bucket)


	for each bucket:
		copy bucket contents to answer array
```

### QuickSort
create function that takes in an array, two integers, low and high
```
	function
		set integer as pivot, equal to element at position high in array
		set integer as k, equal to low - 1
		for each element from low to high
			if element at array is less than pivot
				k increases
				swap the numbers at position element and k
			
		end for
		swap element at k+1 and element at high
		return k+1
```

create second function takes in array , two integers, low and high
```
	in function
		if low is less than high
			set integer as p, call first function with values given
			recursive call with array, low and p-1
			recursive call with array, p+1, and high
		end if
```

in main function call second function with array, 0 and n-1, n being the size of the array.

### Bubble Sort
create function that takes in an array of items
```
function bubbleSort(list: array of items)
	loop = list.count
	for i = 0 to loop-1
		boolean swapped = false
		for j = 0 to loop-1
			if list at index j > list at index j+1
				swap(list at index j, list at index j+1)
				swapped = true
			if (not swapped)
				break
	return list
```

### Sample Sort
With unsorted array of size n
Create m buckets (m likely # of threads)
```
for element in array
    bucket index = element index / m
    add element to bucket[ bucket index]

for each bucket in buckets
    sort bucket with quicksort

need m-1 pivot elements

piv_num = m-1
```
create sample_selection array
```
for each bucket in buckets
    mod_number = ceil(number of elements in bucket /(piv_num+1))
    for i in range of 1 and piv_num
        append bucket[imod_number] element to sample_selection


Global splitters = []

sort sample_selection

mod_number= ceil (number of elements in sample_selection/ (piv_num +1))
for i in range of 1 and piv_num
    append sample_selection[ i mod_number] to global splitters
```
we should have m-1 global splitters, now use these splitters for bucket sort
perform bucket sort given m buckets and the bucket partitions being the global splitters

### 3. Project implementation

All CUDA implementations were tested with 256 threads and 4096 NUM_VALS

### Bucket Sort
Required Code Regions: data_init, comm, comp, comm_large, comp_large, comm_small
comp_small, correctness_check

CUDA Implementation (Pseudocode):
```
function bucketsort(data,sizes,n_buckets)
        compute thread index
        add data of index to correct bucket
    increment bucket size
    
function main()
    generateData(data) //on host
    
    copy data to device
    bucketSort(data,sizes,n_buckets) //on kernel
    for each bucket:
            bitonicsort(bucket) //on kernel
    for each bucket:
            copy bucket contents to answer array on host
```

MPI Implementation (Pseudocode):
```
function main()
    generateData(data)
    
    loop over local data
        add data to correct bucket
        increment bucket size
    
    add then broadcast bucket sizes
    
    sort local bucket
    
    gather all buckets together
```

### Quicksort

### Bubble Sort
Required Code Regions: data_init, comm, comp, comm_large, comp_large, comm_small
comp_small, correctness_check

Cuda Implementation (Psuedocode):
```
// Function to generate a random float
function random_float() -> float
    return random float value between 0 and 1

// Function to fill an array with random float values
function array_fill(arr: float[], length: int)
    for i from 0 to length - 1
        arr[i] = random_float()

// Function to verify the correctness of the sorted array
function verify(values: float[]) -> int
    for i from 0 to length of values - 2
        if values[i] > values[i+1]
            return -1
    return 1

// CUDA kernel for a single step of bubble sort
function bubble_sort_step(dev_values: float[])
    i = threadIdx.x + blockDim.x * blockIdx.x
    next = i + 1
    if next < d_NUM_VALS
        if dev_values[i] > dev_values[next]
            swap dev_values[i] and dev_values[next]

// Function to perform bubble sort on the GPU
function bubble_sort(values: float[])
    allocate device memory for dev_values
    copy values from host to dev_values
    set d_NUM_VALS to NUM_VALS
    define grid and block dimensions
    for i from 0 to NUM_VALS - 2
        call bubble_sort_step kernel with dev_values as argument
    synchronize GPU
    copy dev_values from device to host
    free device memory

// Main function
function main(argc: int, argv: string[]) -> int
    THREADS = convert argv[1] to int
    NUM_VALS = convert argv[2] to int
    BLOCKS = NUM_VALS / THREADS
    start = current time
    allocate memory for values
    call array_fill with values and NUM_VALS as arguments
    call bubble_sort with values as argument
    stop = current time
    size = NUM_VALS * size of float
    data_size_gb = kernel_call * size * 4 * (1e-9) // Size in GB
    kernel_execution_time_s = (stop - start) / CLOCKS_PER_SEC // Kernel execution time in seconds
    effective_bandwidth_gb_s = data_size_gb / kernel_execution_time_s
    if verify(values)
        print "Sort successful"
    else
        print "Sort unsuccessful"
    free memory for values
    return 0

```

MPI Implementation (Pseudocode):

```
function bubbleSort(data: array of double, dataSize: int)
    swapped = true
    while swapped
        swapped = false
        for i from 0 to dataSize - 2
            if data[i] > data[i+1]
                swap data[i] and data[i+1]
            end if
        end for
    end while
end function

function verify(data: array of double, dataSize: int) -> int
    for i from 0 to dataSize - 2
        if data[i] > data[i+1]
            return -1
        end if
    end for
    return 1
end function

function main(argc: int, argv: array of string) -> int
    taskid, numprocs, mtype, source, destination, numworkers, rc: int
    status: MPI_Status
    workcom: MPI_Comm
    comp, master_initialization, comm_large, comm_small, comp_small: string
    test: array of double
    chunkSize: int
    MPI_Init(argc, argv)
    MPI_Comm_rank(MPI_COMM_WORLD, taskid)
    MPI_Comm_size(MPI_COMM_WORLD, numprocs)
    MPI_Comm_split(MPI_COMM_WORLD, taskid != 0, taskid, workcom)
    if numprocs < 2
        MPI_Abort(MPI_COMM_WORLD, rc)
        exit(1)
    end if
    numworkers = numprocs - 1
    test = new double[1000]
    chunkSize = 1000 / (numprocs - 1)
    if taskid == MASTER
        for i from 0 to 1000 - 1
            test[i] = rand() % 1000
        end for
        mtype = FROM_MASTER
        destination = 1
        for dest from 1 to numprocs - 1
            MPI_Send(chunkSize, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD)
            MPI_Send(test[(dest-1) * chunkSize], chunkSize, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD)
        end for
        mtype = FROM_WORKER
        for source from 1 to numprocs - 1
            MPI_Recv(test[(source-1) * chunkSize], chunkSize, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, status)
        end for
    end if

    if taskid > MASTER
        mtype = FROM_MASTER
        MPI_Recv(chunkSize, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, status)
        localData = new double[chunkSize]
        MPI_Recv(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, status)
        bubbleSort(localData, chunkSize)
        mtype = FROM_WORKER
        MPI_Send(localData, chunkSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD)
        delete[] localData
    end if
    if taskid == MASTER
        if verify(test, 1000)
            print("Sort successful")
        else
            print("Sort unsuccessful")
        end if
    end if

    return 0
end function
```

### Sample Sort
Required Code Regions: data_init, comm, comp, comm_large, comp_large, comm_small
comp_small, correctness_check

MPI Implementation:

For sample sort, the first MPI gather gathers all the splitters from the processors and combines them. Then, MPI all to all will send data from all processes to the communicators, the buckets are sorted locally, and then eventually are gathered together to form a sorted array

CUDA Implementation:

In the cuda kernel, the computation for sorting each block/bucket is performed. After each bucket is sorted, we obtain the splitters from them with a cuda memcpy, and then each kernel places elements into their respective buckets. Another memcpy is used to copy the sorted values from the GPU to the main process into one array.


### 4. Team communication
Our team will mainly be using discord as our means of communication due to the fact that it is easy to use and if we ever need to voice call or meet up remotely, we do not have to set up a zoom meeting.
