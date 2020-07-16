PyTorch Stanford Cars Baselines (90.0% with ResNet-50)
======================================================

Baselines (currently just ResNets and MobileNets) on Stanford Cars Using a minimally modified version of the canonical PyTorch ImageNet example.

| Model       | Accuracy |
|-------------|----------|
|ResNet-18    |   86.0   |
|ResNet-50    |   90.0   |
|ResNet-101   |   90.1   |
|ResNet-152   |  *90.1*  |
|MobileNetv2  |   87.1   |
|ResNet-18\*  |   61.5   |
|ResNet-50\*  |    7.9   |
|ResNet-101\* |          |
|ResNet-152\* |          |
|MobileNetv2\*|          |

* denotes no pretraining.
Each model just uses the default learning rate schedule (decay by 10 every 30 epochs), and 90 epochs of training.

Dependencies:
+ PyTorch
+ SciPy (for parsing original `.mat` metadata files)

To reproduce a result:
run `download_and_prepare.sh` to download the original images and organize them into an `ImageFolder` dataset.
Run `main.py` (exact same options as the official PyTorch ImageNet example).
e.g., `python3 main.py --pretrained --arch resnet50`

Please submit a PR if you have an improvement to a baseline that is established in the literature (e.g., new data augmentation strategy, regularization, additional image resolution, etc.) and requires minimal hyperparameter tuning.
