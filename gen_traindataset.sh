for i in `seq 0 9`
do
    python src/gen_my_beta_dist.py $i 200 &
done
wait