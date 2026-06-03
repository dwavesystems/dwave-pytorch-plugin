import numpy as np
from dimod import ExactSolver
from scipy.linalg import expm
from tqdm.auto import tqdm

from dwave.system import DWaveSampler

_I = np.eye(2, dtype=float)

PAULI_X = np.array([[0.0, 1.0],
                    [1.0, 0.0]])

PAULI_Z = np.array([[1.0, 0.0],
                    [0.0, -1.0]])


def kronecker_reduce(ops: list[np.ndarray]):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


print("ZII\n", kronecker_reduce([PAULI_Z, _I, _I]).astype(int))
print("")
print("IIZ\n", kronecker_reduce([_I, _I, PAULI_Z]).astype(int))
print("")


def single_site(pauli: np.ndarray, i: int, n: int):
    chain = [_I] * n
    chain[i] = pauli
    return kronecker_reduce(chain)


def two_site(pauli_a: np.ndarray, pauli_b: np.ndarray,
             i: int, j: int, n: int) -> np.ndarray:
    assert i != j
    chain = [_I] * n
    chain[i] = pauli_a
    chain[j] = pauli_b
    return kronecker_reduce(chain)


def linear_A(s: float) -> float:
    return 1.0 - s


def linear_B(s: float) -> float:
    return s


VISUALIZE = False
if VISUALIZE:
    import matplotlib.pyplot as plt
    _s = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(_s, linear_B(_s), label="LinearB", c="blue")
    ax.plot(_s, linear_A(_s), label="LinearA", c="red")
    ax.legend()
    ax.legend_.set_title("Schedules")
    plt.tight_layout()
    plt.savefig("schedule")


class QuantumAnnealer:

    def __init__(self, h, J, n_qubits=None, schedule_A=linear_A, schedule_B=linear_B):
        self.n_qubits = int(n_qubits)
        self.h = h
        self.J = J
        self.schedule_A = schedule_A
        self.schedule_B = schedule_B
        self.H_driver = self._build_driver()
        self.H_problem = self._build_problem()

    def _build_driver(self):
        n = self.n_qubits
        H = np.zeros((2 ** n, 2 ** n), dtype=float)
        for i in range(n):
            H -= single_site(PAULI_X, i, n)
        return H

    def _build_problem(self):
        n = self.n_qubits
        H = np.zeros((2 ** n, 2 ** n), dtype=float)
        terms = [("h", k, v) for k, v in self.h.items() if v != 0.0] + \
                [("J", k, v) for k, v in self.J.items() if v != 0.0]
        for kind, key, bias in terms:
            if kind == "h":
                H += bias * single_site(PAULI_Z, key, n)
            else:
                i, j = key
                H += bias * two_site(PAULI_Z, PAULI_Z, i, j, n)
        return H

    def hamiltonian(self, s: float) -> np.ndarray:
        return self.schedule_A(s) * self.H_driver + self.schedule_B(s) * self.H_problem

    def anneal(self, t_anneal: float, num_steps: int):
        dt = t_anneal / num_steps
        dim = 2 ** self.n_qubits
        # Initialize to |+> state
        psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)

        step_iter = tqdm(range(num_steps), desc="annealing")
        for step in step_iter:
            s_mid = (step + 1) / num_steps
            H = self.hamiltonian(s_mid)
            U = expm(-1j * dt * H)
            psi = U @ psi

        return psi

    def ground_state(self, s: float = 1.0):
        """Lowest eigenpair of H(s)."""
        vals, vecs = np.linalg.eigh(self.hamiltonian(s))
        return float(vals[0]), vecs[:, 0]


def _demo(n: int) -> None:
    rng = np.random.default_rng(0)
    h = {i: rng.uniform(-0.5, 0.5) for i in range(n)}
    J = {(i, j): rng.choice([-1.0, 1.0])
         for i in range(n) for j in range(i + 1, n)}

    annealer = QuantumAnnealer(h=h, J=J, n_qubits=n)
    ss = ExactSolver().sample_ising(h=h, J=J)
    assert np.isclose(np.sort(ss.record.energy), np.sort(annealer.H_problem.diagonal())).all()

    print("Problem Hamiltonian matched:", (annealer.H_problem == annealer.hamiltonian(1.0)).all())
    print("Driver Hamiltonian matched:", (annealer.H_driver == annealer.hamiltonian(0.0)).all())

    state = annealer.anneal(t_anneal=1000.0, num_steps=100)

    e0, gs = annealer.ground_state(s=1.0)
    final_overlap = float(np.abs(np.vdot(gs, state)) ** 2)

    idx = int(np.argmax(np.abs(gs) ** 2))
    bitstring = format(idx, f"0{n}b")
    gs_spins = np.array([1 - 2 * int(b) for b in bitstring])

    assert np.isclose(e0, ss.lowest().record.energy.min())
    assert np.all(ss.lowest().record.sample == gs_spins)

    print(f"n_qubits           : {n}")
    print(f"exact ground energy: {e0:.6f}")
    print(f"ground bitstring   : {bitstring}  (spins={gs_spins.tolist()})")
    print(f"final |<gs|psi>|^2 : {final_overlap:.6f}")


if __name__ == "__main__":
    _demo(4)
