/home/disk1/vis/suying02/anaconda2/bin/python test.py --train-list /home/ssd1/webvision_2018/info/train_filelist_all.txt \
 --val-list /home/ssd1/webvision_2018/info/train_filelist_all.txt --schedule 4 7 9 11 12 13 14 15 16 17 18 19 20 \
 --gamma 0.5 -c ensemble_checkpoints/imagenet/resnext101-zjf-rmnoise -a resnext101 --num-classes 5000 \
 --root /home/ssd1/webvision_2018 --gpu-id 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
 --root-val /home/ssd1/webvision_2018 \
 --train-epoch 100 --test-epoch 7000 --lr 0.000391 --output-feature \
 --save-dir ensemble_checkpoints/imagenet/resnext101-zjf-rmnoise \
 --resume ensemble_checkpoints/imagenet/resnext101-zjf-rmnoise/checkpoint-57.pth.tar
