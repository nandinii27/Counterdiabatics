# DAQC with Counter-Diabatic Driving for Ising Ground State Preparation

## Overview

This notebook implements a **Digital-Analog Quantum Computing (DAQC)** protocol augmented with a first-order **Counter-Diabatic (CD)** correction term to suppress diabatic transitions during quantum annealing of a 4-qubit all-to-all coupled Ising model.

The annealing Hamiltonian interpolates between a transverse-field driver $H_0 = \sum_i \sigma_i^z$ and a problem Hamiltonian $H_p = \sum_{i<j} J_{ij} \sigma_i^x \sigma_j^x$:

$$H(\lambda) = (1 - \lambda) H_0 + \lambda H_p, \quad \lambda \in [0, 1]$$ 

---

## Architecture

### Circuit Structure

The DAQC circuit decomposes the time-evolution operator into three interleaved blocks per Trotter step at each $\lambda_m = m/M$:

| Block | Gate | Description |
|-------|------|-------------|
| `H_p` | `MSGate(n, θ)` | Mølmer–Sørensen all-to-all XX coupling |
| `H_0` | `Phase(φ)` | Local transverse-field evolution |
| `CD`  | `S · MS_row · S†`, `S · MS_col · S†` | Row/column decomposition of the CD operator |

The CD block applies the gauge potential $\mathcal{A}_\lambda^{(1)} \propto \sum_{i<j} J_{ij} (\sigma_i^y \sigma_j^x + \sigma_i^x \sigma_j^y)$ via conjugation of MS gates with S/S† to rotate $\sigma^x \to \sigma^y$ on the target qubit.

### Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| `n` | 4 | Number of qubits |
| `M` | 3 | Trotter steps |
| `T` | 1.0 | Total annealing time |
| `N` | 1 | Sublayers per step |
| `J` | upper-triangular, all-ones | Coupling matrix (fully connected) |

---

## Noise Model

Depolarizing noise is applied to all MS gate variants with error probability $p = 10^{-4}$:

```
NoiseModel: depolarizing_error(p=1e-4, n_qubits=4)
Applied to: MS, MS_row, MS_col
Simulator: AerSimulator (density_matrix method)
Shots: 2000
```

---

## Dependencies

```
qiskit >= 1.0
qiskit-aer
numpy
matplotlib
```

---

## Usage

Run all cells sequentially in `solution.ipynb`. The circuit is drawn to `circuit-mpl.jpeg` and measurement histograms over all $2^4 = 16$ computational basis states are displayed inline.

---

