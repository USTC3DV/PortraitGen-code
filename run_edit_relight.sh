cuda_device=$1
folders=$2
bg_setting=$3
prompt=$4

IFS=' ' read -r -a array <<< "$folders"
data_path="./preprocessed"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:/usr/local/cuda/lib64:/usr/local/cuda/cudnn/lib:$LD_LIBRARY_PATH
for name in "${array[@]}"
do

CUDA_VISIBLE_DEVICES=$cuda_device python train_edit_relight.py --source_dir $data_path/$name/gaussian \
                                    --prompt "$prompt" \
                                    --train_skip 16\
                                    --bg_setting $bg_setting

done