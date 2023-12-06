for si in 65536 262144 1048576 4194304 16777216; do
  for (( n=64; n<=1024 ; n*=2)); do
    #for mode in 0 1 2 3; do
      sbatch bitonic.grace_job $n $si 0
      #echo "$n" "$si" 0
    #done
  done
done
