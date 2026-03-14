from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import qiskit.quantum_info as qi
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import MSGate
from qiskit.circuit.exceptions import CircuitError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


Edge = Tuple[int, int, float]


def weights_to_edges(J: Sequence[Sequence[float]]) -> List[Edge]:
    """Convert an upper-triangular weight matrix into a weighted edge list."""
    edges: List[Edge] = []
    n = len(J)
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(J[i][j])
            if weight != 0.0:
                edges.append((i, j, weight))
    return edges


def bitstring_to_spins(bitstring: Sequence[int] | str) -> List[int]:
    """Map computational-basis bits to Ising spins."""
    return [1 if int(bit) == 0 else -1 for bit in bitstring]


def cut_value(bitstring: Sequence[int] | str, edges: Iterable[Edge]) -> float:
    """Classical MaxCut objective."""
    value = 0.0
    for i, j, weight in edges:
        if int(bitstring[i]) != int(bitstring[j]):
            value += weight
    return value


def maxcut_energy(bitstring: Sequence[int] | str, edges: Iterable[Edge]) -> float:
    r"""Energy of the cost Hamiltonian

    .. math::
        H_C = - \sum_{(i,j)\in E} \frac{J_{ij}}{2}\left(1 - Z_i Z_j\right).
    """
    spins = bitstring_to_spins(bitstring)
    energy = 0.0
    for i, j, weight in edges:
        energy += -0.5 * weight * (1 - spins[i] * spins[j])
    return energy


def maxcut_bruteforce(n: int, edges: Iterable[Edge]) -> Tuple[float, List[str]]:
    """Return the exact MaxCut value and all optimal bitstrings."""
    best_value = None
    best_states: List[str] = []

    for bits in product([0, 1], repeat=n):
        bitstring = "".join(map(str, bits))
        value = cut_value(bitstring, edges)
        if best_value is None or value > best_value:
            best_value = value
            best_states = [bitstring]
        elif value == best_value:
            best_states.append(bitstring)

    return float(best_value), best_states


def solve_maxcut_from_matrix(J: Sequence[Sequence[float]]) -> Dict[str, object]:
    """Solve MaxCut exactly for the graph encoded by ``J``."""
    edges = weights_to_edges(J)
    best_cut, solutions = maxcut_bruteforce(len(J), edges)
    return {
        "n": len(J),
        "edges": edges,
        "best_cut": best_cut,
        "solutions": solutions,
        "ground_energy": maxcut_energy(solutions[0], edges),
    }


def build_lambda_schedule(T: float, dt: float) -> np.ndarray:
    """Use the same schedule convention as the original notebook."""
    return np.arange(int(T / dt) + 1) * dt / T


def cd_denominator_terms(J: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """Precompute the scalar sums entering the CD coefficient."""
    n = len(J)
    sum_J_squared = sum(J[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
    sum_J_fourth = sum(J[i][j] ** 4 for i in range(n) for j in range(i + 1, n))
    pairwise_sum_terms = sum(
        J[i][j] ** 2 * J[i][k] ** 2
        + J[i][j] ** 2 * J[j][k] ** 2
        + J[i][k] ** 2 * J[j][k] ** 2
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    )
    return float(sum_J_squared), float(sum_J_fourth), float(pairwise_sum_terms)


def calc_alpha_1(lam: float, J: Sequence[Sequence[float]]) -> float:
    """Counterdiabatic prefactor used in ``solution_v2.ipynb``."""
    sum_J_squared, sum_J_fourth, pairwise_sum_terms = cd_denominator_terms(J)
    denominator = (1 - lam) ** 2 * 4 * sum_J_squared + lam**2 * (
        sum_J_fourth + 6 * pairwise_sum_terms
    )
    if np.isclose(denominator, 0.0):
        return 0.0
    return float(-0.25 * sum_J_squared / denominator)


def _xx_label(num_qubits: int, i: int, j: int) -> str:
    label = ["I"] * num_qubits
    label[num_qubits - 1 - i] = "X"
    label[num_qubits - 1 - j] = "X"
    return "".join(label)


def _matrix_msgate_operator(theta_matrix: Sequence[Sequence[float]]) -> qi.Operator:
    """Return an operator equivalent to a matrix-parameter MSGate.

    Newer Qiskit versions accept ``MSGate(n, theta_matrix)`` directly. Older versions,
    including the local project venv, reject nested-list parameters, so we synthesize
    the same XX interaction unitary explicitly.
    """
    n = len(theta_matrix)
    try:
        return qi.Operator(MSGate(n, theta_matrix))
    except (CircuitError, TypeError):
        labels = []
        coeffs = []
        for i in range(n):
            for j in range(i + 1, n):
                angle = float(theta_matrix[i][j])
                if angle != 0.0:
                    labels.append(_xx_label(n, i, j))
                    coeffs.append(angle / 2.0)

        if not labels:
            return qi.Operator(np.eye(2**n, dtype=complex))

        h_ms = qi.SparsePauliOp(labels, coeffs=coeffs).to_matrix()
        return qi.Operator(expm(-1j * h_ms))


def add_H_p_block(
    qc: QuantumCircuit,
    lam: float,
    J: Sequence[Sequence[float]],
    dt: float,
    N: int,
    label: str = "MS",
) -> None:
    """Add the analog interaction block."""
    n = len(J)
    theta_layer = (2 * dt * lam / N * np.array(J, dtype=float)).tolist()
    ms_op = _matrix_msgate_operator(theta_layer)
    qc.unitary(ms_op, list(range(n)), label=label)


def add_H_0_block(qc: QuantumCircuit, lam: float, n: int, dt: float, N: int) -> None:
    """Add the local-field block."""
    local_angle = 2 * dt * (1 - lam) / N
    for qubit in range(n):
        qc.p(local_angle, qubit)


def add_CD_block(
    qc: QuantumCircuit,
    lam: float,
    J: Sequence[Sequence[float]],
    T: float,
    dt: float,
    N: int,
    row_label: str = "MS_row",
    col_label: str = "MS_col",
) -> None:
    """Add the CD correction as sparse row and column MSGate blocks."""
    n = len(J)
    alpha_1 = calc_alpha_1(lam, J)
    J_prime = (-4 * dt / T * alpha_1 * np.array(J, dtype=float)).tolist()

    for row in range(n):
        theta_row = [[0.0 for _ in range(n)] for _ in range(n)]
        for col in range(row + 1, n):
            theta_row[row][col] = J_prime[row][col]

        if any(theta_row[row][col] != 0.0 for col in range(row + 1, n)):
            ms_row = _matrix_msgate_operator(theta_row)
            qc.s(row)
            qc.unitary(ms_row, list(range(n)), label=row_label)
            qc.sdg(row)

    for col in range(1, n):
        theta_col = [[0.0 for _ in range(n)] for _ in range(n)]
        for row in range(col):
            theta_col[row][col] = J_prime[row][col]

        if any(theta_col[row][col] != 0.0 for row in range(col)):
            ms_col = _matrix_msgate_operator(theta_col)
            qc.s(col)
            qc.unitary(ms_col, list(range(n)), label=col_label)
            qc.sdg(col)


def build_daqc_circuit(
    J: Sequence[Sequence[float]],
    T: float,
    dt: float = 1.0,
    N: int = 1,
    with_cd: bool = False,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the DAQC circuit with optional CD correction."""
    n = len(J)
    lambda_schedule = build_lambda_schedule(T, dt)
    qc = QuantumCircuit(n, n if measure else 0)

    for qubit in range(n):
        qc.x(qubit)

    for lam in lambda_schedule:
        for _ in range(N):
            add_H_p_block(qc, lam, J, dt=dt, N=N)
            add_H_0_block(qc, lam, n=n, dt=dt, N=N)
            if with_cd:
                add_CD_block(qc, lam, J, T=T, dt=dt, N=N)
        qc.barrier()

    for qubit in range(n):
        qc.h(qubit)

    if measure:
        qc.measure(range(n), range(n))
    return qc


def circuit_style() -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Color scheme for the DAQC circuit drawings."""
    return {
        "displaycolor": {
            "MS": ("#6a4c93", "#ffffff"),
            "MS_row": ("#ff6b6b", "#000000"),
            "MS_col": ("#ffd54f", "#000000"),
        }
    }


def build_noise_model(p_ms: float, n: int) -> NoiseModel:
    """Depolarizing noise applied only to the MS-type blocks."""
    error_ms = depolarizing_error(p_ms, n)
    noise_model = NoiseModel()
    for label in ("MS", "MS_row", "MS_col"):
        noise_model.add_all_qubit_quantum_error(error_ms, label)
    noise_model.add_basis_gates(["unitary"])
    return noise_model


def run_counts(
    circuit: QuantumCircuit,
    noise_model: NoiseModel | None = None,
    shots: int = 2000,
    method: str = "density_matrix",
) -> Dict[str, int]:
    """Run a noisy or noiseless circuit and return measurement counts."""
    simulator = AerSimulator(method=method, noise_model=noise_model)
    return simulator.run(circuit, shots=shots).result().get_counts()


def counts_to_probabilities(counts: Dict[str, int], shots: int | None = None) -> Dict[str, float]:
    """Normalize counts into a probability distribution."""
    total = shots if shots is not None else sum(counts.values())
    return {bitstring: count / total for bitstring, count in counts.items()}


def success_probability(
    counts: Dict[str, int],
    solutions: Sequence[str],
    shots: int | None = None,
) -> float:
    """Probability mass assigned to the exact classical MaxCut solutions."""
    total = shots if shots is not None else sum(counts.values())
    return sum(counts.get(bitstring, 0) for bitstring in solutions) / total


def benchmark_success_probability(
    J: Sequence[Sequence[float]],
    T_values: Sequence[float],
    dt: float = 1.0,
    N: int = 1,
    p_ms: float = 1e-6,
    shots: int = 2000,
) -> Dict[str, object]:
    """Compare DAQC and DAQC+CD success probabilities across multiple times."""
    exact = solve_maxcut_from_matrix(J)
    noise_model = build_noise_model(p_ms=p_ms, n=len(J))

    probabilities = []
    probabilities_cd = []

    for T in T_values:
        qc = build_daqc_circuit(J, T=T, dt=dt, N=N, with_cd=False, measure=True)
        qc_cd = build_daqc_circuit(J, T=T, dt=dt, N=N, with_cd=True, measure=True)

        counts = run_counts(qc, noise_model=noise_model, shots=shots)
        counts_cd = run_counts(qc_cd, noise_model=noise_model, shots=shots)

        probabilities.append(success_probability(counts, exact["solutions"], shots=shots))
        probabilities_cd.append(success_probability(counts_cd, exact["solutions"], shots=shots))

    return {
        "T_values": list(T_values),
        "probabilities": probabilities,
        "probabilities_cd": probabilities_cd,
        "solutions": exact["solutions"],
        "best_cut": exact["best_cut"],
        "edges": exact["edges"],
    }
