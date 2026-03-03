# compile:
# 	g++ ${FILE}.cpp src/*.cpp -o ${FILE} \
# 	-I include \
# 	-I include/libtorch/include \
# 	-I include/libtorch/include/torch/csrc/api/include \
# 	-L include/libtorch/lib -ltorch -ltorch_cpu -lc10

compile:
	cd build \
	&& cmake .. \
		-DCMAKE_PREFIX_PATH="$(pwd)/../include/libtorch" \
	&& make -j$(nproc) \
	&& mv database_engine_bin ../database_engine_bin

run:
	./database_engine_bin ../configs/database.yaml ${ACTION}
