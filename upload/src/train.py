
import argparse, json, os
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(name, n_classes):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for p in m.features[:4].parameters():       # freeze early layers
            p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in list(m.conv1.parameters()) + list(m.layer1.parameters()) + list(m.layer2.parameters()):
            p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError("model must be efficientnet_b0 or resnet18")
    return m


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)      # NEW (1/3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = transforms.Compose([                       # augmentation
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    full_train = datasets.ImageFolder(a.data_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(a.data_dir, transform=eval_tf)
    classes = full_train.classes
    labels = np.array(full_train.targets)

    # 70 / 15 / 15 stratified split
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=labels, random_state=a.seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=labels[rest], random_state=a.seed)
    train_dl = DataLoader(Subset(full_train, tr), batch_size=a.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(Subset(full_eval, va), batch_size=a.batch, num_workers=2)
    test_dl = DataLoader(Subset(full_eval, te), batch_size=a.batch, num_workers=2)
    print(f"classes={classes}  train={len(tr)} val={len(va)} test={len(te)}  device={device}")

    # class-imbalance handling: weighted loss
    counts = np.bincount(labels[tr], minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = build_model(a.model, len(classes)).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    best_recall, best_path = -1, os.path.join(a.out, "best_model.pt")
    bad = 0                                                   # NEW (2/3)
    for ep in range(a.epochs):
        model.train(); total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            total += loss.item() * len(x)
        yv, pv = evaluate(model, val_dl, device)
        rec = recall_score(yv, pv, average="macro")       # select on validation recall
        print(f"epoch {ep+1}/{a.epochs}  loss={total/len(tr):.4f}  val_acc={(yv==pv).mean():.4f}  val_macro_recall={rec:.4f}")
        if rec > best_recall:                                 # NEW (3/3)
            best_recall = rec; bad = 0; torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping at epoch {ep+1} (no improvement for {a.patience} epochs)")
                break

    # final report ONCE on the untouched test set
    model.load_state_dict(torch.load(best_path, map_location=device))
    yt, pt = evaluate(model, test_dl, device)
    report = classification_report(yt, pt, target_names=classes, output_dict=True)
    print(classification_report(yt, pt, target_names=classes))
    json.dump({"classes": classes, "model": a.model, "report": report}, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    cm = confusion_matrix(yt, pt)
    fig, ax = plt.subplots(figsize=(5, 4.5)); ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([c.split("___")[-1] for c in classes], rotation=30, ha="right")
    ax.set_yticklabels([c.split("___")[-1] for c in classes])
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=150)
    json.dump({"test_indices": [int(i) for i in te]}, open(os.path.join(a.out, "test_split.json"), "w"))


if __name__ == "__main__":
    main()
