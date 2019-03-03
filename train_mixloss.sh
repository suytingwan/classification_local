/home/vis/suying02/anaconda2/bin/python train.py --train-list /home/ssd3/train_filelist_google_flickr_binary_lower_reweight5.txt \
 --val-list  /home/ssd1/vis/info/val_filelist.txt --schedule 67 68 69 70 71 72 \
 --gamma 0.5 -c checkpoints/imagenet/resnext101-mixloss -a resnext101 --num-classes 5000 \
 --root /home/ssd1/vis --gpu-id 0 \
 --root-val /home/ssd1/vis/val_images_resized \
 --train-epoch 60 --test-epoch 40 --lr 0.00025 \
 --mix-loss --output-feature \
 --resume checkpoints/imagenet/resnext101-mixloss/checkpoint-66.pth.tar 
  #--weight-loss --weight-file checkpoints/imagenet/resnext101/ratio.json 
