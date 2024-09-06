cuda_device=$1
folders=$2
prompt=$3
exp_name=$4
IFS=' ' read -r -a array <<< "$folders"
data_path="./testdataset"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:/usr/local/cuda/lib64:/usr/local/cuda/cudnn/lib:$LD_LIBRARY_PATH
for name in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=$cuda_device python train_edit_ip2p.py --source_dir $data_path/$name/gaussian \
                                       --prompt "$prompt" \
                                       --train_skip 8\
                                       --exp_name "$exp_name"
done