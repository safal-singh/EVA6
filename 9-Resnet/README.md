# TARGET
* Custom ResNet architecture for CIFAR10 with following architecture
1. Prep Layer - 3x3 Conv (stride=padding=1) > BatchNorm > ReLU (64 channels)
2. Layer 1
      1. X = 3x3 Conv (stride=padding=1) > MaxPool (2x2, stride=2) > BatchNorm > ReLU (128 channels)
      2. R1 = ResBlock((3x3 Conv-BN-ReLU-3x3 Conv-BN-ReLU))(X) (128 channels)
      3. Add(X, R1)
3. Layer2 - 3x3 Conv (stride=padding=1) > MaxPool (2x2, stride=2) > BatchNorm > ReLU (256 channels)
4. Layer3
      1. X = 3x3 Conv (stride=padding=1) > MaxPool (2x2, stride=2) > BatchNorm > ReLU (512 channels)
      2. R2 = ResBlock((3x3 Conv-BN-ReLU-3x3 Conv-BN-ReLU))(X) (512 channels)
      3. Add(X, R1)
5. MaxPool (4x4, stride=4)
6. Fully connected layer (512 > 10)
7. Log Softmax activation

* One Cycle Learning Rate Scheduler
1. Total Epochs-24
2. Max LR at 5th epoch
3. LR min = 0
4. LR max - Find using LR test
5. No Annihilation/Tempering

* Augmentation sequence - RandomCrop 32x32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8). Implemented using Albumentations lib.
* Batch size - 512
* Target Accuracy - 93%

# RESULTS
## Model summary
![Model summary] (S9ModelSummary.jpg)

