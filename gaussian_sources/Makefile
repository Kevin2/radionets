-include .makerc
dataset: simulate_bundles\
		 fft_samp\
		 calc_normalization

simulate_bundles: ${data}/gaussian_sources_train0.h5

fft_samp: ${data}/fft_samp_train0.h5

calc_normalization: ${data}/normalization_factors.csv
cnn_training: ./training_done
find_lr: ./lr_find

######################################################################

# ${data}/gaussian_sources_train0.h5:
# 		mkdir -p ${data}/
# 		python simulate_bundles.py \
# 		${num_bundles_train} ${num_bundles_valid} ${num_bundles_test} \
# 		${num_img} ${size_img} ${data}

# ${data}/fft_samp_train0.h5:
# 		python create_fft_pairs.py \
# 		${data} ${data}/fft_samp_ ../simulations/layouts/vlba.txt \
# 		-fourier ${fourier} \
# 		-amp_phase ${amp_phase} \
# 		-noise ${noise} \
# 		-preview ${preview} \
# 		-specific_mask False \
# 		-compressed False\
# 		-lon -80 \
# 		-lat 50 \
# 		-steps 50

# ${data}/normalization_factors.csv:
# 		python calculate_normalization.py\
# 		${data}/ \
# 		${data}/${norm_path} \

./training_done:
		mkdir -p ${path}
		python train_cnn.py ${data}\
		${path}/${path_model}\
		${arch} ${data}/${norm_path}\
		${epochs} ${lr} ${lr_type} ${bs} -fourier ${fourier} -amp_phase ${amp_phase}\
		-pretrained ${pretrained} ${pre_path}\
		-inspection ${inspection}
		# touch ./training_done

./lr_find:
		mkdir -p ${path}
		python find_lr.py ${data}\
		${arch} ${path} ${lr_type} ${bs} ${data}/${norm_path} -max_iter ${max_iter}\
		-min_lr ${min_lr} -max_lr ${max_lr} -fourier ${fourier} -amp_phase ${amp_phase}\
		-pretrained ${pretrained} ${pre_path}\
		-save ${save}

./plot_loss:
		mkdir -p ${path}
		python plot_loss.py ${data}\
		${path}/${path_model}\
		${arch} -fourier ${fourier}\

clean_norm:
	rm -r ${norm_path}

clean_data:
	rm -r ${data}/
