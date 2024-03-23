# MFNet
Source code for "**Spatial Continuity and Non-equal Importance in Weakly Supervised Salient Object Detection**".

## Prerequisites

### training data
link: https://pan.baidu.com/s/1omTCChQFWwNFhQ79AVD8rg.    code: oipw

### testing datasets
link: https://pan.baidu.com/s/1PBzDP1Hnf3RIvpARmxn2yA.    code: oipw

## Training
### 1st training stage
Case1: Please refer to [this repository](https://github.com/DUTyimmy/generatePGT).

Case2: We also upload ready-made pseudo labels in **Training data** (the link above), you can directly use our offered two kinds of pseudo labels for convenience. CAMs are also presented if you needed.

### 2nd training stage

#### 1, setting the training data to the proper root as follows:

```
MF_code -- data -- DUTS-Train -- image -- 10553 samples

                -- ECSSD (not necessary) 
                
                -- pseudo labels -- label0_0 -- 10553 pseudo labels
                
                                 -- label1_0 -- 10553 pseudo labels
```
#### 2, training
```Run main.py```

Here you can set ECCSD dataset as validation set for optimal results by setting ```--val``` to ```True```, of course it is not necessary in our work.

## Testing
```Run test_code.py```

You need to configure your desired testset in ```--test_root```.  Here you can also perform PAMR and CRF on saliency maps for a furthur refinements if you want, by setting ```--pamr``` and ```--crf``` to True. **Noting** that the results in our paper do not adopt these post-process for a fair comparison.

## Generating Robustness Benchmark

```Run corrupt.py```


## Acknowledge
Thanks to pioneering helpful works:

  - [MFnet](https://github.com/DUTyimmy/MFNet)


