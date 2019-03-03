/home/disk1/vis/suying02/anaconda2/bin/python train.py --train-list /home/ssd1/info/reweight_shuffle_above_2000.txt \
 --val-list  /home/ssd1/info/val_filelist.txt --schedule 2 4 6 8 10 12 14 15 16 17 18 19 20 21 22 23 \
 --gamma 0.5 -c checkpoints/imagenet/resnext101-reweight -a resnext101 --num-classes 5000 \
 --root /home/ssd1/ --gpu-id 2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
 --root-val /home/ssd1/val_images_resized \
 --train-epoch 700 --test-epoch 300 --lr 0.0025 \
 --resume checkpoints/imagenet/resnext101/checkpoint-23-1.pth.tar 
 #--weight-loss --weight-file /home/disk1/vis/suying02/checkpoints/imagenet/resnext101/ratio.json
