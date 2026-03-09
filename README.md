# Federated Learning IDS — Privacy Attack Analysis

An empirical study of **gradient inversion attacks** and **privacy defence mechanisms** on a Federated Learning-based Intrusion Detection System, built using the [Flower](https://github.com/adap/flower) framework and [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset.

This project extends the base FL-IDS implementation with a full privacy vulnerability analysis — demonstrating that gradient updates leaked during federated training can be exploited to reconstruct private network traffic data, and evaluating three defence strategies against this threat.

---

## Key Results

| Defence | Parameter | Reconstruction MSE | Privacy Level |
|---|---|---|---|
| No Defence (Baseline) | — | 0.000056 | ❌ Critical Risk |
| DP Noise | σ = 0.1 | 0.670612 | ✅ Strong |
| DP Noise | σ = 2.0 | 2.930902 | ✅✅ Strongest |
| Gradient Clipping | Norm = 0.5 | 1.219867 | ✅✅ Strong |
| Sparse Gradients | Top 25% | 1.004138 | ✅✅ Strong |

> Higher MSE = harder to reconstruct = better privacy protection.
> DP Noise at σ=2.0 provides **52,000x improvement** in privacy over the undefended baseline.

---

## Repository Structure

```
fl-ids-privacy-attack/
├── data/                    # Place UNSW-NB15 CSV files here
├── utils/                   # Data loading and plot utilities
├── client.py                # FL client (Flower)
├── server.py                # FL server with FedAvg aggregation
├── simulation.py            # Single-command FL simulation
├── gradient_attack.py       # Gradient inversion attack
├── defence_dp.py            # Differential Privacy noise defence
├── defence_clipping.py      # Gradient clipping defence
├── defence_sparse.py        # Sparse gradient communication defence
├── docker-compose.yaml      # Docker simulation setup
├── Dockerfile.client
├── Dockerfile.server
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/fl-ids-privacy-attack.git
cd fl-ids-privacy-attack
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
pip install -U flwr["simulation"]
```

**4. Download the dataset**

Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place both files in the `data/` folder.

---

## Running the FL Simulation

There are 3 options:

**Option 1 — Single command (recommended)**
```bash
python simulation.py
```

**Option 2 — Manual (multiple terminals)**
```bash
# Terminal 1
python server.py

# Terminal 2, 3, 4 (run each separately)
python client.py
```
> Note: At least 3 clients are needed to satisfy `min_fit_clients`, `min_evaluate_clients`, and `min_available_clients`. Default server port is 8080.

**Option 3 — Docker**
```bash
docker-compose up --build
```

**Visualize model architecture**
```bash
python utils/plot.py
```

---

## Running the Privacy Attack & Defences

**Gradient Inversion Attack**
```bash
python gradient_attack.py > attack_output.txt 2>&1
```
Reconstructs private training data from intercepted gradients using the DLG method.

**Differential Privacy Defence**
```bash
python defence_dp.py > dp_output.txt 2>&1
```
Tests Gaussian noise injection at σ ∈ {0.0, 0.1, 0.5, 1.0, 2.0}.

**Gradient Clipping Defence**
```bash
python defence_clipping.py > clipping_output.txt 2>&1
```
Tests gradient norm clipping at values ∈ {10.0, 1.0, 0.5, 0.1}.

**Sparse Gradient Defence**
```bash
python defence_sparse.py > sparse_output.txt 2>&1
```
Tests top-k% gradient sparsification at k ∈ {50%, 25%, 10%, 1%}.

---

## Tech Stack

- **Python 3.10**
- **TensorFlow / Keras** — Neural network training
- **Flower (flwr)** — Federated learning framework
- **scikit-learn** — Preprocessing
- **pandas / numpy** — Data handling

---

## Concepts Demonstrated

- Federated Learning (FedAvg aggregation)
- Gradient Inversion Attacks (Adversarial ML)
- Differential Privacy (Gaussian noise injection)
- Gradient Clipping
- Sparse Gradient Communication
- Privacy-Utility Tradeoff Analysis
- Network Intrusion Detection (UNSW-NB15)

---

## Future Enhancements

- Personalize datasets for each client instead of using a common sampled dataset
- Evaluate attacks on larger batch sizes to reflect realistic FL deployments
- Combine multiple defences (DP noise + clipping) for stronger guarantees
- Extend to multi-class attack classification

---

## Base Implementation Credit

The FL-IDS base code is adapted from [oqadiSAK/fl-ids](https://github.com/oqadiSAK/fl-ids). The gradient inversion attack and all defence scripts are original contributions.

---

## License

MIT
