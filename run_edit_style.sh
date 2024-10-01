cuda_device=$1
folders=$2
image_path=$3

IFS=' ' read -r -a array <<< "$folders"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:/usr/local/cuda/lib64:/usr/local/cuda/cudnn/lib:$LD_LIBRARY_PATH
data_path="./preprocessed"


for name in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=$cuda_device python train_edit_style.py --source_dir $data_path/$name/gaussian \
                                --style_image_path $image_path\
                                --train_skip 16
done
