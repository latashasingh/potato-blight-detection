"""Grad-CAM: shows which leaf regions the classifier looked at.

Run:  python src/gradcam.py --data_dir data --model efficientnet_b0 --n 6
(uses outputs/best_model.pt and outputs/test_split.json from train.py)
"""
import argparse, json, os
import numpy as np, torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import build_model, MEAN, STD


def gradcam(model, layer, x, cls=None):
    acts, grads = [], []
    h1 = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    out = model(x); cls = int(out.argmax(1)) if cls is None else cls
    model.zero_grad(); out[0, cls].backward()
    h1.remove(); h2.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1))[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, cls, out.softmax(1)[0, cls].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data"); ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--out", default="outputs"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.ImageFolder(a.data_dir, transform=tf)
    model = build_model(a.model, len(ds.classes))
    model.load_state_dict(torch.load(os.path.join(a.out, "best_model.pt"), map_location="cpu")); model.eval()
    layer = model.features[-1] if a.model == "efficientnet_b0" else model.layer4[-1]
    te = json.load(open(os.path.join(a.out, "test_split.json")))["test_indices"]
    pick = np.random.RandomState(0).choice(te, size=min(a.n, len(te)), replace=False)

    fig, axes = plt.subplots(2, len(pick), figsize=(3 * len(pick), 6))
    for k, i in enumerate(pick):
        x, y = ds[i]; cam, c, p = gradcam(model, layer, x.unsqueeze(0))
        img = x.permute(1, 2, 0).numpy() * STD + MEAN
        cam_big = np.array(transforms.functional.resize(
            transforms.functional.to_pil_image(torch.tensor(cam).unsqueeze(0).float()), (224, 224)))
        axes[0, k].imshow(np.clip(img, 0, 1)); axes[0, k].axis("off")
        axes[0, k].set_title(f"true: {ds.classes[y].split('___')[-1]}", fontsize=9)
        axes[1, k].imshow(np.clip(img, 0, 1)); axes[1, k].imshow(cam_big, cmap="jet", alpha=0.45); axes[1, k].axis("off")
        axes[1, k].set_title(f"pred: {ds.classes[c].split('___')[-1]} ({p:.2f})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "gradcam_examples.png"), dpi=150)
    print("saved", os.path.join(a.out, "gradcam_examples.png"))


if __name__ == "__main__":
    main()
