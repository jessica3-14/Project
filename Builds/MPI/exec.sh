for si in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
  for (( n=2; n<=1024 ; n*=2)); do
    for mode in 0 1 2 3; do
      #sbatch mpi.grace_job $si $n
      echo "$n" "$si" "$mode"
    done
  done
done
