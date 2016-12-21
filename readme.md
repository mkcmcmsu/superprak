
scp -r  edu-cmc-stud16-618-11@bluegene1.hpc.cs.msu.ru:~/
ssh edu-cmc-stud16-618-11@bluegene1.hpc.cs.msu.ru
make
cp main /gpfs/data/edu-cmc-stud16-618-08
mpisubmit.bg -n 128 -w 00:05:00 -m smp ./main 1000 1000
llmap (check task execution)
compile and run on bluegene (mpi + openmp)

scp -r  edu-cmc-stud16-618-11@bluegene1.hpc.cs.msu.ru:~/
ssh edu-cmc-stud16-618-11@bluegene1.hpc.cs.msu.ru
cd poisson-equation
make bluegene-openmp
cp main /gpfs/data/edu-cmc-stud16-618-08
mpisubmit.bg -n 128 -w 00:05:00 -m smp -env OMP_NUM_THREADS=3 ./main 1000 1000
llmap (check task execution)
run locally
