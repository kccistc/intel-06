# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/: ???
./splitted_dataset/train: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/val: ???
./splitted_dataset/train/<class#>: ???​
./splitted_dataset/train/<class#>: ???​
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| |
|EfficientNet-B0|0.986|406.34|02:18|1||4.0M|
|DeiT-Tiny|0.980|228.02|04:37|1||5.5M|
|MobileNet-V3-large-1x|1|935.15||1||3.0M|


## FPS 측정 방법
