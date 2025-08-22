# Why forking this repo?
In order to pursue my master thesis in AIGD task, I decided to implement the NN approach proposed in the paper to analyze the results mentioned.
For everything else that is not the new NN files, refer to the original [project page](https://utkarshojha.github.io/universal-fake-detection/) of the [paper](https://arxiv.org/abs/2302.10174).

# How to use the NN approach

1. Create a feature bank
First of all adjust the paths in the create_feature_bank.py file:
```bash
[...]
fake_dataset = BankDataset(image_dir="dataset/1_fake")
real_dataset = BankDataset(image_dir="dataset/0_real")
[...]
```
Then run:
```bash
python create_feature_bank.py
```

2. Run NN approach file with the generated feature bank
```bash
python neighbouring_approach.py --fake_path=path/to/fake --real_path=path/to/real --feature_bank=path/to/feature_bank --arch=CLIP:ViT-L/14 --max_sample={default:1000} --k={default:1}
```

