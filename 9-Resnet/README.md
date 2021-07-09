Custom ResNet architecture for CIFAR10 with following architecture
1. Prep Layer - 3x3 Conv (stride=padding=1) > BatchNorm > ReLU (64 channels)
2. Layer 1 -
      a. X = 3x3 Conv (stride=padding=1) > MaxPool (2x2, stride=2) > BatchNorm > ReLU (128 channels)
      b. 
