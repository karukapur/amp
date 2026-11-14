# Local model/data assets

`imagenet_class_index.json` is required runtime metadata: the original 1,000
ImageNet class indices, WordNet IDs and names. It is included with the source.

Place the torchvision ResNet-50 V1 `resnet50-0676ba61.pth` checkpoint here for the
default checkpoint path, or pass `--checkpoint /path/to/checkpoint.pth`.
Checkpoints, subset images and dataset archives are ignored by Git. The runner
does not download them automatically. Full ImageNet can live elsewhere and be
selected with `--imagenet-root`.

The original local pilot uses `resnet_results/manifest.json` with retained images
in this directory. Those local assets are optional when using your own dataset.
