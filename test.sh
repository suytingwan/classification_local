/home/disk1/vis/suying02/anaconda2/bin/python test.py --train-list /home/ssd1/info/train_filelist_all.txt \
 --val-list  /home/ssd1/info/train_filelist_all.txt --schedule 2 4 6 8 10 12 14 \
 --gamma 0.5 -c checkpoints/imagenet/resnext101 -a resnext101 --num-classes 5000 \
 --root /home/ssd1 --gpu-id 0,1,2,3,4,5,6,7 \
 --root-val /home/ssd1 \
 --train-epoch 400 --test-epoch 4000 --lr 0.10 --save-dir checkpoints/imagenet/resnext101 \
 --resume /home/disk1/vis/suying02/checkpoints/imagenet/resnext101/model_best.pth.tar
