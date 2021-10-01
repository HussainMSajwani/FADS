d=$4
h2s=$3
i=$2
genvar=$1
dir=d_$d/genVar_$genvar/h2s_$h2s/sim_$i
mkdir -p $dir

make threshold d=3000 n=1200 h2s=0.5 dc=10 genvar=0.5 chr=22 i=1 method=p outdir=/home/shussain/FADS/experiments/genvar/test name="30092021"
#make move dir=$dir h2s=$1

#
#			--nonlinear exp \
#			--expbase $(base) \
#			--proportionNonlinear 0
