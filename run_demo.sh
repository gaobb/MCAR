# setting
dataset='coco2014'               # 'voc2007', 'voc2012', 'coco2014'
imgsize=448                      #  256, 448
basemodel='resnet101'            # 'mobilenetv2', 'resnet50', 'resnet101'
poolingstyle='gwp'               # 'avg', 'gwp'
topN=4
threshold=0.5
datapath='./images/coco14val/'   #'./images/voc12test/'  
resumemodel=../MCAR-models/$dataset-$basemodel-$poolingstyle-$imgsize-$topN-$threshold.pth.tar  #model_best.pth.tar
savepath='./visualization/'$basemodel-$poolingstyle-$imgsize-$topN-$threshold

# runing
echo $resumemodel
CUDA_VISIBLE_DEVICES=0  python -u ./src/mcar_demo.py \
  --data-path $datapath  \
  --dataset-name $dataset \
  --image-size $imgsize \
  --bm $basemodel \
  --ps $poolingstyle  \
  --topN $topN \
  --threshold $threshold \
  --sp $savepath \
  --resume $resumemodel
