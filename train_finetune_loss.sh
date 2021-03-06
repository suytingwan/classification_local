/home/disk1/vis/suying02/anaconda2/bin/python train.py --train-list /home/ssd1/webvision_2018/info/train_filelist_all_from_loss.txt \
 --val-list  /home/ssd1/webvision_2018/info/val_filelist.txt --schedule 2 4 6 8 10 12 14 16 18 20 \
 --gamma 0.5 -c checkpoints/imagenet/resnext101 -a resnext101 --num-classes 5000 \
 --root /home/ssd1/webvision_2018 --gpu-id 0,1,2,3,4,5,6 \
 --root-val /home/ssd1/webvision_2018/val_images_resized \
 --train-epoch 360 --test-epoch 300 --lr 0.005 \
 --resume /home/disk1/vis/suying02/webvision/checkpoints/imagenet/resnext101/model_best.pth.tar \
 --weight-loss --weight-file checkpoints/imagenet/resnext101/ratio_from_loss.json
