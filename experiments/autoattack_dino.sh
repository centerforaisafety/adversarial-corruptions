# 2/255, 4/255, 8/255
for eps in 0.0078431373 0.0156862745 0.031372549; do
    # resnet50 on ImageNet
    python experiments/eval_autoattack.py --data_dir ./datasets/imagenet \
        --epsilon $eps --weights weights/imagenet/resnet50_linf_eps8.0.ckpt \
        --num_classes 1000 --model resnet50
    # dinov2 on ImageNet
    for weights in dinov2_vitb14 dinov2_vitl14; do
        python experiments/eval_autoattack.py --data_dir ./datasets/imagenet \
            --epsilon $eps --weights weights/imagenet/$weights \
            --num_classes 1000 --model dinov2
    done

    # resnet50 on ImageNet100
    python experiments/eval_autoattack.py --data_dir ./datasets/imagenet100 \
        --epsilon $eps --weights weights/imagenet100/pgd-linf-8.pth \
        --num_classes 100 --model resnet50
    # dinov2 on ImageNet100
    for weights in dinov2_vitb14 dinov2_vitl14; do
        python experiments/eval_autoattack.py --data_dir ./datasets/imagenet100 \
            --epsilon $eps --weights weights/imagenet100/$weights \
            --num_classes 100 --model dinov2
    done
done

