

cuda_device=$1
folders=$2
exp_name=$3
IFS=' ' read -r -a array <<< "$folders"
data_path="./preprocessed"

export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:/usr/local/cuda/lib64:/usr/local/cuda/cudnn/lib:$LD_LIBRARY_PATH
for name in "${array[@]}"
do
    CUDA_VISIBLE_DEVICES=$cuda_device python train_recon.py --source_dir $data_path/$name/gaussian --exp_name "$exp_name"
    touch $data_path/$name/.fearecon_done
done