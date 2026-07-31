"""Supervised feasibility test: GRU on proprio windows -> injured leg / splint length / friction.

Offline (no sim). Input: npz from collect_student_obs.py. Windows are kept only
where the gt is constant (no reset inside the window) — the per-class window
counts printed below therefore double as an episode-survival report per injury
condition. Train/test split is BY ENV to avoid leakage.

Usage:
  python3 train_injury_estimator.py --data biomech/student_obs_v0p3.npz \
      [--window 50 --stride 10 --hidden 128 --epochs 12]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--window", type=int, default=50)
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--epochs", type=int, default=12)
args = parser.parse_args()

torch.manual_seed(0)
rng = np.random.RandomState(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(args.data)
obs, idx, spl, fric = d["obs"], d["idx"], d["splint"], d["fric"]
S, E, D = obs.shape
W, STRIDE = args.window, args.stride
print(f"data: steps={S} envs={E} dim={D}  window={W} stride={STRIDE}")

Xs, y_cls, y_spl, y_fric, env_ids = [], [], [], [], []
for e in range(E):
    for t0 in range(0, S - W, STRIDE):
        sl = slice(t0, t0 + W)
        if (idx[sl, e] != idx[t0, e]).any() or (spl[sl, e] != spl[t0, e]).any():
            continue  # a reset crossed the window
        Xs.append(obs[sl, e])
        y_cls.append(idx[t0, e] + 1)  # 0=healthy, 1..4=FL..RR
        y_spl.append(spl[t0, e])
        y_fric.append(fric[t0, e])
        env_ids.append(e)

X = torch.tensor(np.array(Xs, dtype=np.float32))
y_cls = torch.tensor(np.array(y_cls, dtype=np.int64))
y_spl = torch.tensor(np.array(y_spl, dtype=np.float32))
y_fric = torch.tensor(np.array(y_fric, dtype=np.float32))
env_ids = np.array(env_ids)
counts = np.bincount(y_cls.numpy(), minlength=5)
print(f"windows: {len(X)}  per-class [Normal,FL,FR,RL,RR]: {counts.tolist()}")
print("  (a near-zero class count = that injury condition rarely survives a full window)")

perm = rng.permutation(E)
test_envs = set(perm[: int(0.3 * E)].tolist())
te_mask = np.array([e in test_envs for e in env_ids])
tr, te = np.where(~te_mask)[0], np.where(te_mask)[0]

mu = X[tr].reshape(-1, D).mean(0)
sd = X[tr].reshape(-1, D).std(0).clamp_min(1e-3)


class Est(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.gru = nn.GRU(D, h, batch_first=True)
        self.cls = nn.Linear(h, 5)
        self.reg = nn.Linear(h, 2)  # [length, friction]

    def forward(self, x):
        _, hn = self.gru(x)
        h = hn[-1]
        return self.cls(h), self.reg(h)


model = Est(args.hidden).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
mu_d, sd_d = mu.to(dev), sd.to(dev)

inj_tr = tr[(y_cls[tr] > 0).numpy()]
spl_m, spl_s = y_spl[inj_tr].mean(), y_spl[inj_tr].std().clamp_min(1e-6)
fric_m, fric_s = y_fric[inj_tr].mean(), y_fric[inj_tr].std().clamp_min(1e-6)
print(f"train injured: spl {spl_m:.3f}±{spl_s:.3f}  fric {fric_m:.3f}±{fric_s:.3f}")

BS = 512
for epoch in range(args.epochs):
    model.train()
    order = rng.permutation(len(tr))
    tot, nb = 0.0, 0
    for i in range(0, len(tr), BS):
        b = tr[order[i:i + BS]]
        xb = (X[b].to(dev) - mu_d) / sd_d
        logits, reg = model(xb)
        cls_b = y_cls[b].to(dev)
        loss = nn.functional.cross_entropy(logits, cls_b)
        inj = cls_b > 0
        if inj.any():
            t_spl = ((y_spl[b].to(dev) - spl_m.to(dev)) / spl_s.to(dev))[inj]
            t_frc = ((y_fric[b].to(dev) - fric_m.to(dev)) / fric_s.to(dev))[inj]
            loss = loss + nn.functional.mse_loss(reg[inj, 0], t_spl)
            loss = loss + nn.functional.mse_loss(reg[inj, 1], t_frc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
        nb += 1
    print(f"epoch {epoch}: loss {tot / nb:.4f}", flush=True)

model.eval()
preds_c, preds_r = [], []
with torch.no_grad():
    for i in range(0, len(te), BS):
        b = te[i:i + BS]
        logits, reg = model((X[b].to(dev) - mu_d) / sd_d)
        preds_c.append(logits.argmax(1).cpu())
        preds_r.append(reg.cpu())
pc = torch.cat(preds_c)
pr = torch.cat(preds_r)
tc = y_cls[te]
acc = (pc == tc).float().mean().item() * 100
print(f"\n[classification] acc {acc:.1f}%  (5-way, test windows n={len(te)})")

inj = tc > 0
p_spl = pr[inj, 0] * spl_s + spl_m
p_frc = pr[inj, 1] * fric_s + fric_m
t_spl, t_frc = y_spl[te][inj], y_fric[te][inj]


def report(name, p, t, unit):
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    mae = (p - t).abs().mean()
    base = (t - t.mean()).abs().mean()
    print(
        f"[{name}] R2 {r2:.3f}  MAE {mae:.4f}{unit}  "
        f"(predict-mean baseline MAE {base:.4f}{unit}, n={len(t)})"
    )


report("splint length", p_spl, t_spl, "m")
report("foot friction", p_frc, t_frc, "")
