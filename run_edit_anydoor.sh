cuda_device=$1
folders=$2
image_path=$3

IFS=' ' read -r -a array <<< "$folders"

data_path="./testdataset"

for name in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=$cuda_device python train_edit_anydoor.py --source_dir $data_path/$name/gaussian \
                                --style_image_path $image_path \
                                --train_skip 64
done