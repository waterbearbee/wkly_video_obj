python test_net.py \
        --dataset=charades \
        --net=vgg16 \
        --model_path=panet_obj_norm_action_loss_kpsel_vgg16_lre-5_pretrain \
        --bs=1 \
        --vis=0 \
        --checksession 1\
        --checkepoch 16 \
        --checkpoint 7482 \
        --cuda