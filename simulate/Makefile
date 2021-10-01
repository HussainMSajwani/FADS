SHELL:=/bin/bash
#CEU hapmap3: https://mathgen.stats.ox.ac.uk/impute/data_download_1000G_pilot_plus_hapmap3.html
#ref_panel_dir=/home/shussain/hapgen2/1000GP_Phase3/
ref_panel_dir=/home/shussain/hapgen2/CEU_impute/
n_pheno=1
genVar=0.95
name=`date +'%d%m%Y'`
dir=d_$(d)/genVar_$(genvar)/h2s_$(h2s)/sim_$(i)

threshold: $(dir)/PS/output
	mkdir -p $(outdir)/$(dir)
	python train.py $(genvar) $(i) $(method) $(h2s) $(d) /home/shussain/Simulated_data/$(name)/$(dir) $(outdir)/$(dir)


sim: $(dir)/PS/output
#generate genotype in oxford format. chromosome $chr
$(dir)/prelim/prelim_$(chr):
	d_max=`wc -l < $(ref_panel_dir)data/chr$(chr).legend`; \
	if [ "$$d_max" -lt "$d" ]; then echo "Not enough SNPs in reference panel"; exit 64; fi; \
	rand=`awk -v d_max=$$d_max -v d=$$d 'BEGIN{srand();print int(rand()*(d_max-d))}'`; \
	first=`cat $(ref_panel_dir)/data/chr$(chr).legend | head -$$(($$rand+1)) | tail -1 | cut -d' ' -f2`; \
	dth=`cat $(ref_panel_dir)/data/chr$(chr).legend | head -$$(($$d+$$rand)) | tail -1 | cut -d' ' -f2`; \
	echo $$dth; \
	dummyDL=`sed -n '2'p $(ref_panel_dir)data/chr$(chr).legend | cut -d ' ' -f 2`; \
	~/hapgen2/hapgen2 -m $(ref_panel_dir)genetic_maps/genetic_map_chr$(chr)_combined_b*.txt\
					  -l $(ref_panel_dir)data/chr$(chr).legend\
					  -h $(ref_panel_dir)data/chr$(chr).haps\
					  -o $(dir)/prelim/prelim_$(chr) \
					  -n $(n) 0 \
					  -int $$first $$dth \
					  -dl $$first 0 0 0 \
					  -no_haps_output
	

#transform from oxford format into plink format
$(dir)/prelim/prelim_plink_chr$(chr): $(dir)/prelim/prelim_$(chr)
	~/plink/plink-1.9/plink --data $(dir)/prelim/prelim_$(chr).controls \
							--oxford-single-chr $(chr) \
							--make-bed \
							--out $(dir)/prelim/prelim_plink_chr$(chr)
	touch $@


$(dir)/PS/output: $(dir)/prelim/prelim_$(chr)
	mkdir -p $(dir)/PS
	
	Rscript commandlineFunctions.R \
			--NrSamples=$(n) --NrPhenotypes=$(n_pheno) \
			--tNrSNP=$(d) --format="oxgen"\
			--genotypefile=$(dir)/prelim/prelim_$(chr).controls \
			--cNrSNP=$(dc) \
			--genVar=$(genVar) --h2s=$(h2s) --phi=1 \
			--directory=$(dir)/PS \
			--subdirectory=output \
			--showProgress \
			--saveTable \
			--savePLINK \

	rm -rf temp
	mkdir -p ~/Simulated_data/$(name)/$(dir)
	mv $(dir)/* ~/Simulated_data/$(name)/$(dir)
	echo !!



clean:
	rm -rf prelim
	rm -rf temp
	rm -rf final
	rm -rf PS