num=1

for lr in 1e-3
do
for bs in 64
do
for wd in 1e-4
do
for i in $(seq 1 $num)
do

# python -m torch.distributed.run --nproc_per_node=2 main.py \
# --distributed \
python main.py \
--hdf5_path '/data/sd0809/TianTanData/data_align_3mod.hdf5' \
--mask_path '/data/sd0809/TianTanData/annotations/WT' \
--task 'tumor_classification' \
--modality T1_Ax T1_E_Ax T2_Ax \
--model_dim 2d \
--model metaformer \
--drop_path_rate 0.1 \
--n_seg_classes 1 \
--crop_H 224 \
--crop_W 224 \
--crop_D 1 \
--seg_loss_fn 'DiceCE' \
--optim 'adamw' \
--sched 'warmup_cosine' \
--warmup_epochs 30 \
--learning_rate $lr \
--weight_decay $wd \
--batch_size $bs \
--num_workers 8 \
--epochs 200 \
--val_every 5 \
--save_folder '/output' \
--manual_seed 4294967295 \
--test_seed $i \
--gpu_id 0 \
--start_epoch 1 \

done
done
done
done