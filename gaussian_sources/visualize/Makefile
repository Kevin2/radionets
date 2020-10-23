-include .makerc
all: save_predictions visualize_predictions

visualize_predictions: build/visualize_predictions_done
save_predictions: ./pred_save
msssim: ./mean_msssim

build/visualize_predictions_done:
	mkdir -p ${out_path}
	mkdir -p ${out_path}diff/
	mkdir -p ${out_path}dynamic_range/
	mkdir -p ${out_path}blob/
	python visualize_predictions.py \
	${out_path} \
	-fourier ${fourier} -amp_phase ${amp_phase} -diff ${diff} \
	-dr ${dr} -sensitivity ${sensitivity} -blob ${blob} ${mode}

./pred_save:
	mkdir -p ${out_path}
	python save_predictions.py ${data} ${data}/${norm_path} \
	${arch} ${path_model} ${out_path} -fourier ${fourier} -separate ${separate} ${mode}

./mean_msssim:
	mkdir -p ${out_path}
	python mean_msssim.py ${data} ${data}/${norm_path} \
	${arch} ${path_model} ${out_path} -fourier ${fourier}

clean:
	rm -rf ${out_path}

