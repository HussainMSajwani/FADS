d=$4
h2s=$3
i=$2
genvar=$1
dir=d_$d/genVar_$genvar/h2s_$h2s/sim_$i
mkdir -p $dir

make sim chr=22 d=$d n=1200 h2s=$h2s dc=10 dir=$dir genvar=$genvar > $dir/log.txt
make move dir=$dir h2s=$1

#
#			--nonlinear exp \
#			--expbase $(base) \
#			--proportionNonlinear 0
