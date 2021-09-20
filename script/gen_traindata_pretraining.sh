#!/usr/bin/bash
echo "生成训练集"
for i in `seq 0 9`
do
    python src/gen_my_beta_dist.py $i 200 0 &
done
wait

echo "生成训练集"
for i in `seq 10 19`
do
    python src/gen_my_beta_dist.py $i 200 0 &
done
wait

echo "生成验证集"
for i in `seq 0 4`
do
    python src/gen_my_beta_dist.py $i 100 1 &
done
wait