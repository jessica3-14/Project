for si in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
  for (( n=64; n<=4096 ; n*=2)); do
    for mode in 0 1 2 3; do
      #sbatch bitonic.grace_job $si $n $mode
      echo "$n" "$si" "$mode"
    done
  done
done
