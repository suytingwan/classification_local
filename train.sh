/home/disk1/vis/suying02/anaconda2/bin/python train.py --train-list /home/ssd1/webvision_2018/info/train_filelist_all.txt \
 --val-list  /home/ssd1/webvision_2018/info/val_filelist.txt --schedule 2 4 6 8 10 12 14 \
 --gamma 0.5 -c checkpoints/imagenet/resattention92 --a resattention92 --num-classes 5000 \
 --root /home/ssd1/webvision_2018 --gpu-id 0,1,2,3,4,5,6,7,8 \
 --root-val /home/ssd1/webvision_2018/val_images_resized \
 --train-epoch 400 --test-epoch 256 --lr 0.10 --data-augmentation 
 #--resume /home/vis/suying02/noisy-webdata/checkpoints/imagenet/resnet101/checkpoint-2.pth.tar
