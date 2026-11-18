# GaitKeeper

Disentangling identity from disease in 3D Parkinsonian gait (CARE-PD). See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

## Setup

```bash
conda create -n gaitkeeper python=3.9.21 -y
conda activate gaitkeeper
pip install chumpy==0.70 --no-build-isolation   # old setup.py imports pip, breaks in isolated builds
pip install -r requirements.txt

git clone https://github.com/TaatiTeam/CARE-PD external/CARE-PD
```

Check it worked (should print `2.6.0+cu118 True`):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data

Not in the repo (too big). Put the CARE-PD pickles in `Datasets/CAREPD/`.
