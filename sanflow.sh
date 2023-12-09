CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
CATEGORIES="bottle"

for category in $CATEGORIES
do
    echo $category
    python main.py --gpu 0 --dataset mvtec --inp 256 --action-type norm-test --class-name $category --checkpoint './weights_WRN50/'$category'.pt' --pro --viz 
    # python main.py --gpu 0 --dataset mvtec --inp 320 --action-type norm-test --class-name $category --checkpoint './weights_WRN101/'$category'.pt' -enc 'wide_resnet101_2' --pro --viz 
    # python main.py --gpu 0 --inp 256 --lr 5e-4 --meta-epochs 20 --sub-epoch 4 --class-name $category -bs 4 -pl 3 --pro
    # python main.py --gpu 0 --inp 320 --lr 5e-4 --meta-epochs 20 --sub-epoch 4 --dataset mvtec --class-name $category -bs 4 -pl 3 --pro -enc 'wide_resnet101_2'
done
