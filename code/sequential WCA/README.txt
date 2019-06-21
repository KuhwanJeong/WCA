compile		make
run
./wca0.exe $1 $2 $3 $4 $5
$1	data	simul/mnist
$2	alpha	5/10/20 correspond to alpha=0.5/1.0/2.0
$3	b0	1(simul)/20(mnist) correspond to b0=0.1/2.0
$4	eps	4(simul)/5(mnist) correspond to eps=0.0001/0.00001
$5	permutation	1/.../10 correspond to train1.txt/.../train10.txt
