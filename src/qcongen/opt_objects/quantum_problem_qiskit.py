"""Quantum platform interface using Qiskit."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import Session
from qiskit.primitives import Estimator 

from qcongen.opt_objects.ising import IsingHamiltonian 
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit.providers.backend import BackendV2 as Backend
from qiskit_aer import AerSimulator
# from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
from scipy.optimize import minimize
from qiskit_ibm_runtime import SamplerV2 as Sampler
from scipy.optimize import OptimizeResult as Result
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger('qcongen')


class QuantumProblem:

    def __init__(self, hamiltonian: IsingHamiltonian, sample_size: int, backend: Backend | None = None, init_params: np.ndarray | None = None):
        self.backend: Backend = backend if backend else AerSimulator()
        self.init_params: np.ndarray = init_params if init_params is not None else [np.pi, np.pi/2, np.pi, np.pi/2]
        self.sample_size = sample_size
        self.hamiltonian = hamiltonian
        self.sparse_pauli_hamiltonian = self._build_ising_paulis(hamiltonian)
        self.logical_circuit = self._create_logical_circuit(self.sparse_pauli_hamiltonian)
        self.physical_circuit = self._create_physical_circuit(self.logical_circuit)
        self.optimized_circuit: QuantumCircuit | None = None
        self.result: Result | None = None 
        self.objective_func_vals: list[float] = []
        

    def _build_ising_paulis(self, hamiltonian: IsingHamiltonian) -> SparsePauliOp:
        """Convert the hamiltonian to Pauli list."""

        J_ij = []
        h_i = []

        for i in range(hamiltonian.J.shape[0]):
            for j in range(hamiltonian.J.shape[1]):
                if hamiltonian.J[i, j] != 0:
                    J_ij.append((i, j))
        
        for i in range(hamiltonian.h.shape[0]):
            if hamiltonian.h[i] != 0:
                h_i.append(i)
        
        pauli_list = []
        for (i, j) in J_ij:
            paulis = ["I"] * hamiltonian.J.shape[0]
            paulis[i] = "Z"
            paulis[j] = "Z"

            weight = hamiltonian.J[i, j]
            pauli_str = "".join(paulis)[::-1]
            pauli_list.append((pauli_str, weight))

        for i in h_i:
            paulis = ["I"] * hamiltonian.J.shape[0]
            paulis[i] = "Z"

            weight = hamiltonian.h[i]
            pauli_str = "".join(paulis)[::-1]
            pauli_list.append((pauli_str, weight))

        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
        print(cost_hamiltonian)

        return cost_hamiltonian


    def _create_logical_circuit(self, cost_hamiltonian: SparsePauliOp) -> QuantumCircuit:
        """Create a QAOA circuit for the Ising problem."""

        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
        
        circuit.measure_all()

        return circuit


    def _create_physical_circuit(self,logical_circuit: QuantumCircuit) -> QuantumCircuit:
        """Create a physical circuit from the logical circuit."""

        pm = generate_preset_pass_manager(optimization_level=3,
                                            backend=self.backend)
        candidate_circuit = pm.run(logical_circuit)
        return candidate_circuit
    


    
    def optimize_circuit(self) -> Result:
        with Session(backend=self.backend) as session:
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000

            self.result: Result = minimize(
                self._cost_func_estimator,
                x0=self.init_params,
                args=(self.physical_circuit, self.sparse_pauli_hamiltonian, estimator, self.objective_func_vals),
                method="COBYLA",
            )
            
            self.optimized_circuit = self.physical_circuit.assign_parameters(self.result.x)

            if self.result.success:
                logger.debug("Circuit optimization successful")
            else:
                logger.warning("Circuit optimization did not converge")
                
            return self.result

    def sample_circuit(self) -> dict[int, tuple[np.ndarray, int]]:
        sampler: Sampler = Sampler(mode=self.backend)
        sampler.options.default_shots = self.sample_size


        pub = (self.optimized_circuit, )
        
        job = sampler.run([pub])
        counts_int = job.result()[0].data.meas.get_int_counts()
        samples_dict = self._process_results(counts_int)
        
        self._log_sampling_statistics(samples_dict)

        
        return samples_dict


    def _process_results(self, final_distribution_int: dict[int, int]) -> dict[int, tuple[np.ndarray, int]]:
        """Process the results of the circuit sampling.
        
        Args:
            final_distribution_int: Dictionary mapping integer representations to counts
            
        Returns:
            Dictionary mapping integer representations to tuples of (bitstring array, count)
        """
        results: dict[int, tuple[np.ndarray, int]] = {}
        for bitstring, count in final_distribution_int.items():
            binary = np.binary_repr(bitstring, width=self.hamiltonian.J.shape[0])
            arr = np.array([int(digit) for digit in binary][::-1])
            results[bitstring] = (arr, count)
        return results
    
    def _log_sampling_statistics(self, samples_dict: dict[int, tuple[np.ndarray, int]]) -> None:
        """Log detailed statistics about the sampling results."""
        total_samples = sum(sample_tuple[1] for sample_tuple in samples_dict.values())
        n_unique_samples = len(samples_dict)
        logger.debug(f"Generated {n_unique_samples} unique samples from {total_samples} total measurements")
        
        sorted_by_freq = sorted(samples_dict.items(), key=lambda x: x[1][1], reverse=True)
        
        logger.debug("\nTop 5 most frequent samples:")
        logger.debug("  Rank | Count |   %   ")
        logger.debug("  -----|-------|-------")
        
        for i, (_, (sample, count)) in enumerate(sorted_by_freq[:5], 1):
            percentage = (count / total_samples) * 100
            logger.debug(f"    {i:2d} | {count:5d} | {percentage:5.1f}")
            logger.debug(f"    Sample: {sample}")
    def _cost_func_estimator(self, params: np.ndarray, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: Estimator, objective_func_vals: list[float]) -> float:

        isa_hamiltonian: SparsePauliOp = hamiltonian.apply_layout(ansatz.layout)

        pub = (ansatz, isa_hamiltonian, params)
        job = estimator.run([pub])

        results = job.result()[0]
        cost: float = float(results.data.evs)

        objective_func_vals.append(cost)

        return cost

class SimulatedQuantumProblem:
    """Quantum problem using statevector simulation for exact results."""

    def __init__(self, hamiltonian: IsingHamiltonian, sample_size: int, init_params: np.ndarray | None = None):
        self.init_params: np.ndarray = init_params if init_params is not None else [np.pi, np.pi/2, np.pi, np.pi/2]
        self.sample_size = sample_size
        self.hamiltonian = hamiltonian
        self.sparse_pauli_hamiltonian = self._build_ising_paulis(hamiltonian)
        self.logical_circuit = self._create_logical_circuit(self.sparse_pauli_hamiltonian)
        self.optimized_circuit: QuantumCircuit | None = None
        self.result: Result | None = None 
        self.objective_func_vals: list[float] = []

    def _build_ising_paulis(self, hamiltonian: IsingHamiltonian) -> SparsePauliOp:
        """Convert the hamiltonian to Pauli list."""
        J_ij = []
        h_i = []

        for i in range(hamiltonian.J.shape[0]):
            for j in range(hamiltonian.J.shape[1]):
                if hamiltonian.J[i, j] != 0:  
                    J_ij.append((i, j))
        
        for i in range(hamiltonian.h.shape[0]):
            if hamiltonian.h[i] != 0:
                h_i.append(i)
        
        pauli_list = []
        for (i, j) in J_ij:
            paulis = ["I"] * hamiltonian.J.shape[0]
            paulis[i] = "Z"
            paulis[j] = "Z"
            weight = hamiltonian.J[i, j]
            pauli_str = "".join(paulis)[::-1]
            pauli_list.append((pauli_str, weight))

        for i in h_i:
            paulis = ["I"] * hamiltonian.J.shape[0]
            paulis[i] = "Z"
            weight = hamiltonian.h[i]
            pauli_str = "".join(paulis)[::-1]
            pauli_list.append((pauli_str, weight))

        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)

        return cost_hamiltonian

    def _create_logical_circuit(self, cost_hamiltonian: SparsePauliOp) -> QuantumCircuit:
        """Create a QAOA circuit for the Ising problem."""
        circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
        
        return circuit

    def optimize_circuit(self) -> Result:
        """Optimize the QAOA circuit using statevector simulation."""
        estimator = StatevectorEstimator()
        

        self.result: Result = minimize(
            self._cost_func_estimator,
            x0=self.init_params,
            args=(self.logical_circuit, self.sparse_pauli_hamiltonian, estimator, self.objective_func_vals),
            method="BFGS",
        )
        self.optimized_circuit = self.logical_circuit.assign_parameters(self.result.x)
        
        if self.result.success:
            logger.debug("Circuit optimization successful")
        else:
            logger.warning("Circuit optimization did not converge")
            
        return self.result

    def sample_circuit(self) -> dict[int, tuple[np.ndarray, int]]:
        """Sample from the optimized circuit using statevector simulation."""
        cr = ClassicalRegister(self.optimized_circuit.num_qubits)
        meas_circuit = self.optimized_circuit.copy()
        meas_circuit.add_register(cr)
        meas_circuit.measure_all(inplace=True)

        sampler = StatevectorSampler()

        job = sampler.run([(meas_circuit, None)], shots=self.sample_size)
        result = job.result()[0]
        
        bit_array = result.data.meas
        samples_dict = {}
        
        for num in bit_array.array:
            num = int(num[0])
            binary = np.binary_repr(num, width=self.optimized_circuit.num_qubits)
            arr = np.array([int(bit) for bit in binary])
            
            if num in samples_dict:
                samples_dict[num] = (arr, samples_dict[num][1] + 1)
            else:
                samples_dict[num] = (arr, 1)
        
        self._log_sampling_statistics(samples_dict)
        
        return samples_dict

    def _log_sampling_statistics(self, samples_dict: dict[int, tuple[np.ndarray, int]]) -> None:
        """Log detailed statistics about the sampling results."""
        total_samples = sum(sample_tuple[1] for sample_tuple in samples_dict.values())
        n_unique_samples = len(samples_dict)
        logger.debug(f"Generated {n_unique_samples} unique samples from {total_samples} total measurements")
        
        sorted_by_freq = sorted(samples_dict.items(), key=lambda x: x[1][1], reverse=True)
        
        logger.debug("\nTop 5 most frequent samples:")
        logger.debug("  Rank | Count |   %   ")
        logger.debug("  -----|-------|-------")
        
        for i, (_, (sample, count)) in enumerate(sorted_by_freq[:5], 1):
            percentage = (count / total_samples) * 100
            logger.debug(f"    {i:2d} | {count:5d} | {percentage:5.1f}")
            logger.debug(f"    Sample: {sample}")

    def _cost_func_estimator(self, params: np.ndarray, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp, estimator: StatevectorEstimator, objective_func_vals: list[float]) -> float:
        """Estimate cost function using statevector simulation."""
        bound_circuit = ansatz.assign_parameters(params)
        
        pub = (bound_circuit, [hamiltonian])
        job = estimator.run([pub])
        result = job.result()[0]
        
        cost: float = float(result.data.evs[0])
        objective_func_vals.append(cost)
        
        return cost