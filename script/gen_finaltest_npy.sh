
for((i=0,j=5000;i<=50000 && j<=50000;i+=5000,j+=5000));do
    python src/gen_finaltest_npy.py $i $j &
done


