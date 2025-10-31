# SRB – Phase II Falsification

**Goal:** Triangulate SRB effects via multi-signal metrics (per Sam):
1) Meaning Transfer — Transfer Entropy (prompt → response)  
2) Human Coherence — NSM primitives (+ SimpleBERT sanity check)  
3) Creative Surprise — Cross-Entropy (NLL baseline)  
4) Geometry/TDA — UMAP trajectories + Persistent Homology (Betti)  
5) External Verification — separate low-vocab BERT checker

## Run
```bash
python experiments/phase2/runners/run_phase2.py --config experiments/phase2/config/defaults.yaml