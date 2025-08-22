import torch
import argparse
import os
import torch.utils.data
import clip
import numpy as np
from sklearn.metrics import average_precision_score
from create_feature_bank import extract_features, BankDataset
from dataset_paths import DATASET_PATHS
import random
from validate import find_best_threshold, calculate_acc

###############################################

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def load_paths(image_dir):
    image_paths = []
    for fname in os.listdir(image_dir):            
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):  
            image_paths.append(os.path.join(image_dir, fname))
    return image_paths


def predict_if_fake(input_features, real_feats, fake_feats, img_name, k):
    """
    Paper-style per-class k-NN:
      - find k nearest neighbors within REAL bank
      - find k nearest neighbors within FAKE bank
      - compare their average distances
    """

    input_features = input_features.float()
    real_feats = real_feats.float()
    fake_feats = fake_feats.float()

    # Normalize input extracted features
    input_features = input_features / input_features.norm(dim=-1, keepdim=True)

    real_dists = 1 - torch.cosine_similarity(input_features, real_feats, dim=1)
    fake_dists = 1 - torch.cosine_similarity(input_features, fake_feats, dim=1)

    # Take top-k nearest within each class
    topk_real = torch.topk(real_dists, k, largest=False).values.mean().item()
    topk_fake = torch.topk(fake_dists, k, largest=False).values.mean().item()

    # Score: lower distance means closer → more likely that class
    decision = 1 if topk_fake < topk_real else 0  # 1=fake, 0=real
    score = topk_real - topk_fake  # positive if closer to fake

    print(f"Img: {img_name} | decision={decision} | d_real={topk_real:.3f} | d_fake={topk_fake:.3f} | score={score:.3f}")

    return decision, score

def build_prediction_array(real_feats, fake_feats, k, images, paths, max_sample):
    scores = []
    labels = []
    max_iter = min(len(images), max_sample)
    for i in range(max_iter):
        img = images[i].unsqueeze(0).to(device)
        input_features = extract_features(img, model, device)
        label, score = predict_if_fake(input_features, real_feats, fake_feats, paths[i], k)
        scores.append(score)
        labels.append(label)
    return labels,scores


###############################################

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fake_path', type=str, default=None, help='Fake path of generated images')
    parser.add_argument('--real_path', type=str, default=None, help='Real path for real images')
    parser.add_argument('--feature_bank', type=str, default=None, help='Feature bank path to use')

    parser.add_argument('--arch', type=str, default='res50',help="CLIP:ViT-L/14 is suggested to extract features")
    parser.add_argument('--max_sample',type=int,default=1000)
    parser.add_argument('--k', type=int, default=1)

    opt = parser.parse_args()

    data = torch.load(opt.feature_bank)
    fake_feats = data["fake"]
    real_feats = data["real"]

    k = opt.k

    print(f"Loading model {opt.arch}")
    model_name = ""
    if(opt.arch == "CLIP:ViT-L/14"): model_name = "ViT-L/14"
    model, _ = clip.load(model_name,device)
    print ("Model loaded..")
    model.to(device)
    model.eval()

    fake_images = BankDataset(opt.fake_path)
    real_images = BankDataset(opt.real_path)

    fake_paths = load_paths(opt.fake_path)
    real_paths = load_paths(opt.real_path)

    max_sample = opt.max_sample
    
    fake_predicted_labels, fake_predicted_scores = build_prediction_array(real_feats,fake_feats,k,fake_images,fake_paths,max_sample)
    fake_labels = np.ones(len(fake_predicted_labels))
    
    real_predicted_labels, real_predicted_scores = build_prediction_array(real_feats,fake_feats,k,real_images,real_paths,max_sample)
    real_labels = np.zeros(len(real_predicted_labels))

    all_labels = np.concatenate([real_labels,fake_labels])
    all_scores = np.concatenate([np.array(real_predicted_scores),np.array(fake_predicted_scores)])
    all_pred = np.concatenate([np.array(real_predicted_labels),np.array(fake_predicted_labels)])
    
    mAP = average_precision_score(all_labels,all_scores)

    r_acc0, f_acc0, acc0 = calculate_acc(all_labels, all_scores, 0.0)

    best_thres = find_best_threshold(all_labels, all_scores)
    r_acc1, f_acc1, acc1 = calculate_acc(all_labels, all_scores, best_thres)


    print("\n")
    print(f"mAP: {mAP}")
    print("----------------------------\n")
    print("Accuracy with 0.0 threshold:")
    print(f"real:{r_acc0}|fake:{f_acc0}|total:{acc0}\n")
    print("Accuracy with best threshold:")
    print(f"real:{r_acc1}|fake:{f_acc1}|total:{acc1}\n")