h2s=$1
i=$2
base=$3
dir=$h2s/base_$base/sim_$i
echo $dir
mkdir -p $dir

make sim chr=22 d=1000 n=600 h2s=$1 dc=10 dir=$dir base=$base > $dir/log.txt
make move dir=$dir h2s=$1