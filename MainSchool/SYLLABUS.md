# Sets, Graphs & Symmetry for High-Energy Physics
### Deep learning on point clouds — from DeepSets to equivariant & hypergraph networks
**TIFR ML School 2026 · Main School** · *(redesign of the IISER-K 2025 set/graph lectures)*

---

## The one idea behind the whole course

> **Everything in this course is _constrained message passing_.**
> A model is just a choice of three things:
> 1. **What relations you allow** — none (a set) → pairwise edges (a graph) → soft all-to-all edges (attention) → groups of nodes (a hypergraph).
> 2. **How you aggregate** — the permutation-invariant pool (sum / mean / max / attention) that turns neighbours into one vector.
> 3. **Which symmetry you bake in** — permutation only, or also rotations / translations / the full Lorentz group.
>
> **DeepSets is the atom.** Graphs add edges to it; attention makes those edges soft; equivariance constrains the functions; hypergraphs lift edges to groups — and we will see that a hypergraph layer is *literally two nested DeepSets*. The course closes the loop on the idea it opens with.

This is why the modules are ordered the way they are: each one adds exactly **one** new ingredient to the previous.

| # | Module | Relations | Aggregation | Symmetry baked in | Flagship HEP model |
|---|--------|-----------|-------------|-------------------|--------------------|
| 0 | Primer *(optional/pre-school)* | — | — | — | data formats & ML refresher |
| 1 | **Sets — DeepSets, PFN, EFN** | none | sum / mean | permutation | Particle/Energy Flow Networks |
| 2 | **Graphs — message passing** | kNN / dynamic | max / mean over neighbours | permutation | ParticleNet, GravNet |
| 3 | **Attention — Transformers** | soft all-to-all | attention-weighted sum | permutation (+ physics prior) | Particle Transformer (ParT) |
| 4 | **Equivariant graph nets** | radius / all-to-all | over invariants | E(n), Lorentz SO(3,1) | EGNN, LorentzNet, PELICAN |
| 5 | **Hypergraph nets** | hyperedges (groups) | two-stage DeepSets | permutation | AllSet / HGNN |

Every module ships **both** a *theory* part (worked derivations in the notebook markdown / slides) **and** an *explicit training* part (a real training loop on real or realistic data, with HEP-style evaluation).

---

## Audience, prerequisites, logistics

- **Audience:** physics graduate students / postdocs comfortable with Python and the basics of neural networks (an MLP, backprop, an optimizer). The PreSchool notebooks (`PyTorch_101`, `Optimization_and_Initialization`, `Activation_Functions`) are the assumed background.
- **Format (assumed, adjustable):** ~5 sessions of ≈2 h, each = short theory block + guided hands-on. Re-scope freely to your real slot count.
- **Compute:** notebooks run on a laptop (CPU / Apple-MPS) with small subsets; full-size training (ParticleNet/ParT) wants a GPU. Each notebook exposes `N_PER_CLS`, `MAX_PART`, `epochs` knobs to trade accuracy for speed.
- **Stack:** modern PyTorch + PyTorch-Geometric (native scatter — **we drop `torch_scatter`/`torch_cluster`**), plus `uproot` / `awkward` / `vector` for reading HEP ROOT files, and `scikit-learn` for metrics. Module 1 needs **no** PyG.

---

## HEP evaluation conventions (used throughout)

We do **not** stop at accuracy. For tagging we report:
- **ROC / AUC**, and especially
- **background rejection `1/εB` at a fixed signal efficiency `εS`** (e.g. εS = 0.3 / 0.5) — the number experimentalists actually care about.

Standard benchmark datasets:
- **Top Tagging Reference dataset** (Kasieczka et al., 1902.09914) — binary top vs QCD; *the* common yardstick across ParticleNet / ParT / LorentzNet / PELICAN → lets us compare every model on one task.
- **JetClass** (Qu et al., 2202.03772) — 10-class, large; used for the multiclass capstone. A 100k example file is used in Module 1.

---

## Module 0 — Primer *(optional / pre-school)*
**Goal:** remove all data-wrangling friction before the real lectures.
- **Concepts:** why HEP data resists CNNs/RNNs — jets & events are *variable-length, unordered* sets of 4-vectors; the symmetries we will exploit (permutation, rotation/translation, Lorentz, infrared-&-collinear).
- **Hands-on:** read a ROOT file with `uproot`/`awkward`/`vector`; build a clean `Dataset`; visualize a jet in the η–φ plane; understand **padding & masking** for variable-length sets (the single most common source of bugs in this field).

## Module 1 — From particles to sets: DeepSets, PFN & EFN  ✅ *(built — `Module1_DeepSets_PFN_EFN.ipynb`)*
**Theory**
- Permutation **invariance vs equivariance** — formal definitions via permutation matrices.
- The **Deep Sets theorem** (Zaheer 2017): any permutation-invariant function = `ρ(Σ_i φ(x_i))`; why **sum** is canonical and what mean/max cost in expressiveness.
- Two ways to build it: the canonical invariant form (= **PFN**) and the permutation-**equivariant linear layer** `x' = Γx + Λ(x − mean)` (derivation included).
- HEP instantiation: **Particle Flow Networks** and **Energy Flow Networks**; **IRC safety** as an architectural constraint (EFN's energy-weighted sum).
**Hands-on (train):**
1. Warm-up: images → point clouds; train a DeepSets classifier; **numerically demonstrate permutation invariance**.
2. Real: **train a PFN and an EFN** for top-vs-QCD tagging on JetClass; evaluate with ROC/AUC + background rejection.
3. **IRC-safety demo:** collinear-split and soft-add particles and watch the EFN output stay put while the PFN moves.
- **Refs:** Zaheer 1703.06114; Qi *PointNet* 1612.00593; Komiske–Metodiev–Thaler *EFN/PFN* 1810.05165.

## Module 2 — From sets to graphs: message passing, ParticleNet & GravNet  ✅ *(built & executed — `Module2_Graphs_MessagePassing_ParticleNet.ipynb`; ParticleNet test AUC ≈ 0.992)*
**Theory**
- Relational inductive bias (Battaglia 1806.01261); the **MPNN** message→aggregate→update formalism (Gilmer 1704.01212) — show DeepSets = MPNN with no edges.
- **Graph construction in HEP:** kNN in (η,φ), radius graphs, fully-connected, **dynamic / learned** graphs.
- **EdgeConv / DGCNN** → **ParticleNet**; **GravNet / GarNet** (learned latent-space graphs for irregular calorimetry).
**Hands-on (train):** implement **EdgeConv from scratch** in PyG; **train ParticleNet**; **visualize the dynamic adjacency** per layer.
- **Refs:** Wang *DGCNN* 1801.07829; Qu–Gouskos *ParticleNet* 1902.08570; Qasim *GravNet* 1902.07987.

## Module 3 — Attention is message passing: Transformers for particles  ✅ *(built & executed — `Module3_Attention_Transformers_ParT.ipynb`; ParT test AUC ≈ 0.991)*
**Theory**
- Scaled dot-product attention; **self-attention = message passing on a soft complete graph** (`softmax(QKᵀ/√d)V ≡ φ(v_i, ⊕_j ψ(q_i,k_j))`).
- Sets need **no positional encoding**; permutation-equivariance of attention.
- **Set Transformer** (induced/pooled attention, O(N)) → **Particle Transformer (ParT)** and its physics-informed **pairwise interaction bias**.
**Hands-on (train):** multi-head attention from scratch; **train a mini Particle Transformer**; benchmark vs ParticleNet on the same data.
- **Refs:** Vaswani 1706.03762; Lee *Set Transformer* 1810.00825; Qu *ParT* 2202.03772.

## Module 4 — Physics-informed symmetry: equivariant graph networks  ✅ *(built & executed — `Module4_Equivariant_EGNN_LorentzNet.ipynb`; verified E(3)/Lorentz equivariance + data-efficiency experiment)*
**Theory**
- Groups & representations; the equivariance condition `f(ρ_g x) = ρ'_g f(x)`.
- **Two design philosophies:**
  - *Invariant-feature / scalarization* — build messages from invariants (distances, Minkowski dot products): **EGNN** (E(n)), **LorentzNet**, **PELICAN** (permutation-equivariant + Lorentz). Simple, robust.
  - *Equivariant-feature / steerable* — carry features in group representations, combine with tensor products / Clebsch–Gordan: **LGN**, Tensor-Field Networks. More expressive, heavier (covered conceptually).
- Why bake in Lorentz symmetry: **data efficiency** and generalization.
**Hands-on (train):** implement **EGNN** and verify equivariance (rotate input → output rotates); **complete the Lorentz-invariant message-passing net** (Minkowski dot products) with a full training loop on top tagging; headline experiment: **equivariant vs non-equivariant accuracy at small training-set size**.
- **Refs:** Satorras *EGNN* 2102.09844; Gong *LorentzNet* 2201.08187; Bogatskiy *LGN* 2006.04780, *PELICAN* 2211.00454.

## Module 5 — Beyond pairwise: hypergraph networks (+ outlook)  ✅ *(built & executed — `Module5_Hypergraph_AllSet.ipynb`; AllSet = two nested DeepSets, hypergraph tagger AUC ≈ 0.991, loop-closer to Module 1)*
**Theory**
- Why higher-order relations in HEP: multi-particle correlations / **higher-order energy correlators**; **sets-of-sets** (event = set of jets = sets of particles); **combinatorial assignment** (which particles came from which parent).
- Hypergraphs & the **incidence matrix**; a hyperedge = a *group* of nodes.
- **Hypergraph message passing:** HGNN, and **AllSet** (Chien 2022) — node→hyperedge then hyperedge→node updates, *each a multiset (DeepSets / Set-Transformer) function*. **The payoff:** a hypergraph layer = two stacked DeepSets aggregations. The atom from Module 1 reappears at the top.
- Relation to **SPANet**-style attention for assignment.
**Hands-on (train):** implement an **AllSet-style hypergraph layer by reusing the Module-1 DeepSets aggregator twice**; apply to a HEP combinatorial / event-level task.
- **Capstone / outlook:** the unifying table above; a brief look at **foundation models & self-supervised pretraining for particle physics** (masked particle modelling, OmniJet-style).
- **Refs:** Feng *HGNN* 1809.09401; Chien *AllSet* 2106.13264; Fenton *SPANet* 2010.09206.

---

## Cross-cutting threads (woven through every module)
- **Training mechanics:** ragged-batch handling (padding+masking, or PyG `Batch`), input normalization of physics features, class imbalance, reproducibility.
- **Inductive bias ↔ data trade-off:** revisited each module — how much symmetry to bake in vs. learn from data.
- **Interpretability:** attention maps (M3), layer-wise relevance propagation on ParticleNet (M2).

## Repository layout (this folder)
```
MainSchool/
├── SYLLABUS.md                                  ← this file
├── Module1_DeepSets_PFN_EFN.ipynb               ← ✅ built, end-to-end
├── Module2_Graphs_MessagePassing_ParticleNet.ipynb   ← ✅ built & executed
├── Module3_Attention_Transformers_ParT.ipynb         ← ✅ built & executed
├── Module4_Equivariant_EGNN_LorentzNet.ipynb         ← ✅ built & executed
└── Module5_Hypergraph_AllSet.ipynb                   ← ✅ built & executed

../data/    shared data (FashionMNIST present; JetClass_example_100k.root auto-discovered)
```

## Datasets & setup notes
- **Images** (Module 1 warm-up): `torchvision` MNIST (auto-download), falling back to the FashionMNIST already in `../data`.
- **JetClass example** (`JetClass_example_100k.root`, ~136 MB): the notebook searches `../data/`, the legacy 2023 ICTS path, and `./`. Source: <https://hqu.web.cern.ch/datasets/JetClass/example/>. Copy it into `../data/` for portability.
- **Environment:** `numpy torch torchvision matplotlib scikit-learn uproot awkward vector` for Modules 1–3 plus `torch_geometric` (Modules 2+). Avoid the old `torch.set_default_dtype(float64)` and the deprecated `np.Inf` / `pytorch_lightning` usages — all fixed in the redesign.

## Key references (by module)
- **M1:** Zaheer 1703.06114 · Qi 1612.00593 · Komiske–Metodiev–Thaler 1810.05165
- **M2:** Gilmer 1704.01212 · Battaglia 1806.01261 · Wang 1801.07829 · Qu–Gouskos 1902.08570 · Qasim 1902.07987
- **M3:** Vaswani 1706.03762 · Lee 1810.00825 · Qu 2202.03772
- **M4:** Satorras 2102.09844 · Gong 2201.08187 · Bogatskiy 2006.04780 / 2211.00454
- **M5:** Feng 1809.09401 · Chien 2106.13264 · Fenton 2010.09206
