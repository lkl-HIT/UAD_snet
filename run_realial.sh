datapath=/home/lkl/dataset/Real-IAD
datasets=('audiojack' 'bottle_cap' 'button_battery' 'end_cap' 'eraser' 'fire_hood' 'mint' 'mounts' 'pcb' 'phone_battery' 'plastic_nut' 'plastic_plug' 'porcelain_doll' 'regulator' 'rolled_strip_base'
           'rolled_strip_base' 'sim_card_set' 'switch' 'tape' 'terminalblock' 'toothbrush' 'toy' 'toy_brick' 'transistor1' 'usb' 'usb_adaptor' 'u_block' 'vcpill' 'wooden_beads' 'woodstick' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu 0 \
--seed 0 \
--log_group simplenet_realiad_ema \
--log_project RealIAD_Results \
--results_path results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 40 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 4 \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" realiad_1 $datapath
