import random
import numpy as np

# Define global quantum gates as NumPy arrays with complex data types

# Pauli-X (NOT) gate: Flips the state of a qubit
X = np.array([[0, 1],
              [1, 0]], dtype=complex)

# Pauli-Z gate: Applies a phase flip to the |1⟩ state
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# Hadamard (H) gate: Creates superposition states
H = (1 / np.sqrt(2)) * np.array([[1,  1],
                                 [1, -1]], dtype=complex)

class Qbit(object):
    """
    Class representing a quantum register with n qubits.
    """
    def __init__(self, n):
        """
        Initialize the quantum register.

        Parameters:
        n (int): Number of qubits in the register.
        """
        self.n = n
        # Initialize the quantum state vector in the |0...0⟩ state
        self.qbit = np.zeros(2 ** n, dtype=complex)
        self.qbit[0] = 1
        # Reshape the state vector into an n-dimensional tensor
        self.qbit = np.reshape(self.qbit, [2 for _ in range(n)])

    def SET(self, bitstring):
        """
        Set the quantum register to a specific basis state.

        Parameters:
        bitstring (str): Binary string representing the desired basis state.
        """
        qbit = np.zeros(2 ** self.n, dtype=complex)
        qbit[int(bitstring, 2)] = 1
        self.qbit = np.reshape(qbit, [2 for _ in range(self.n)])

    def print(self):
        """
        Print the quantum state in terms of its basis states and amplitudes.
        """
        self.n = len(self.qbit.shape)
        qbit = np.reshape(self.qbit, 2 ** self.n)
        kstr = ''
        for k in range(2 ** self.n):
            num = len(bin(k)) - 2
            zeros = self.n - num
            r = np.round(qbit[k].real, 4)
            if r > 2. ** (-2 ** (2 * self.n)):
                kstr += f' + {r} |{"0" * zeros}{bin(k)[2:]}>'
            elif r < -2. ** (-2 ** (2 * self.n)):
                kstr += f' - {-r} |{"0" * zeros}{bin(k)[2:]}>'
        if kstr[1] == '-':
            print(kstr[1:])
        else:
            print(kstr[2:])

    def C_SWAP(self, C, I1, I2):
        """
        Apply a controlled SWAP (Fredkin) gate on three qubits.

        Parameters:
        C (int): Control qubit index.
        I1 (int): First target qubit index.
        I2 (int): Second target qubit index.
        """
        # Initialize an 8x8 identity matrix for the CSWAP gate
        M = np.identity(8, dtype=complex)
        # Define the SWAP operation on the target qubits when control qubit is |1⟩
        # Swap positions in the matrix corresponding to swapping I1 and I2 when C is |1⟩
        M[-3, -2] = 1
        M[-3, -3] = 0
        M[-2, -3] = 1
        M[-2, -2] = 0
        # Reshape the matrix into a tensor suitable for n-qubit operations
        CSWAP = np.reshape(M, [2 for _ in range(6)])

        # Swap axes to bring the control and target qubits to the front
        qbit = self.qbit
        qbit = np.swapaxes(qbit, 0, C)
        qbit = np.swapaxes(qbit, 1, I1)
        qbit = np.swapaxes(qbit, 2, I2)
        # Apply the CSWAP operation using tensor contraction
        qbit = np.tensordot(CSWAP, qbit, axes=([0, 1, 2], [0, 1, 2]))
        # Swap the axes back to their original positions
        qbit = np.swapaxes(qbit, 2, I2)
        qbit = np.swapaxes(qbit, 1, I1)
        self.qbit = np.swapaxes(qbit, 0, C)

    def NOT(self, I):
        """
        Apply the Pauli-X (NOT) gate to a specified qubit.

        Parameters:
        I (int): Index of the qubit to apply the NOT gate.
        """
        # Swap the target qubit to the front for gate application
        qbit = np.swapaxes(self.qbit, 0, I)
        # Apply the X gate via tensor contraction
        qbit = np.tensordot(X, qbit, axes=(0, 0))
        # Swap the qubit back to its original position
        self.qbit = np.swapaxes(qbit, 0, I)

    def CNOT(self, C, I):
        """
        Apply the Controlled-NOT (CNOT) gate with a control and target qubit.

        Parameters:
        C (int): Control qubit index.
        I (int): Target qubit index.
        """
        # Initialize a 4x4 identity matrix for the CNOT gate
        M = np.identity(4, dtype=complex)
        # Define the CNOT operation: flip target qubit when control is |1⟩
        M[-1, -2] = 1
        M[-1, -1] = 0
        M[-2, -1] = 1
        M[-2, -2] = 0
        # Reshape the matrix into a tensor suitable for two-qubit operations
        CNOT = np.reshape(M, [2 for _ in range(4)])
        # Swap the control and target qubits to the front
        qbit = self.qbit
        qbit = np.swapaxes(qbit, 0, C)
        qbit = np.swapaxes(qbit, 1, I)
        # Apply the CNOT operation via tensor contraction
        qbit = np.tensordot(CNOT, qbit, axes=([0, 1], [0, 1]))
        # Swap the qubits back to their original positions
        qbit = np.swapaxes(qbit, 1, I)
        self.qbit = np.swapaxes(qbit, 0, C)

    def CN_NOT(self, C, I):
        """
        Apply a multi-controlled NOT (Toffoli-like) gate.

        Parameters:
        C (list of int): List of control qubit indices.
        I (int): Target qubit index.
        """
        n = len(C)
        # Initialize a 2^(n+1) x 2^(n+1) identity matrix for the multi-controlled NOT gate
        Mn = np.identity(2 ** (n + 1), dtype=complex)
        # Define the gate operation to flip the target qubit when all controls are |1⟩
        Mn[-1, -2] = 1
        Mn[-1, -1] = 0
        Mn[-2, -1] = 1
        Mn[-2, -2] = 0
        # Reshape the matrix into a tensor suitable for (n+1)-qubit operations
        CnNOT = np.reshape(Mn, [2 for _ in range(2 * (n + 1))])

        # Swap control qubits to the front
        qbit = self.qbit
        for i in range(n):
            qbit = np.swapaxes(qbit, i, C[i])
        # Swap the target qubit to the (n)th position
        qbit = np.swapaxes(qbit, n, I)
        # Apply the multi-controlled NOT gate via tensor contraction
        qbit = np.tensordot(CnNOT, qbit, axes=(range(n + 1), range(n + 1)))
        # Swap the target qubit back to its original position
        qbit = np.swapaxes(qbit, n, I)
        # Swap control qubits back to their original positions
        for i in reversed(range(n)):
            qbit = np.swapaxes(qbit, i, C[i])
        self.qbit = qbit

    def CNOTin(self):
        """
        Apply a chain of CNOT gates to the qubits, flipping each qubit conditionally.
        """
        # This method seems to be intended for a specific use case and may need further clarification.
        # For the purposes of this code, we'll leave it as is.
        pass  # Placeholder for actual implementation or clarification

    def H(self, I):
        """
        Apply the Hadamard (H) gate to a specified qubit.

        Parameters:
        I (int): Index of the qubit to apply the H gate.
        """
        # Swap the target qubit to the front for gate application
        qbit = np.swapaxes(self.qbit, 0, I)
        # Apply the H gate via tensor contraction
        qbit = np.tensordot(H, qbit, axes=(1, 0))
        # Swap the qubit back to its original position
        self.qbit = np.swapaxes(qbit, 0, I)

    def H_n(self, n):
        """
        Apply the Hadamard gate to the first n qubits.

        Parameters:
        n (int): Number of qubits to apply the H gate to.
        """
        for k in range(n):
            self.H(k)

    def Z(self, I):
        """
        Apply the Pauli-Z gate to a specified qubit.

        Parameters:
        I (int): Index of the qubit to apply the Z gate.
        """
        # Swap the target qubit to the front for gate application
        qbit = np.swapaxes(self.qbit, 0, I)
        # Apply the Z gate via tensor contraction
        qbit = np.tensordot(Z, qbit, axes=(1, 0))
        # Swap the qubit back to its original position
        self.qbit = np.swapaxes(qbit, 0, I)

    def CZin(self):
        """
        Apply a multi-controlled Z gate to the system, flipping the phase of the |11...1⟩ state.
        """
        # Define the index corresponding to the |11...1⟩ state
        index = tuple([1] * self.n)
        # Apply a phase flip by multiplying by -1
        self.qbit[index] *= -1

    def CnZ(self):
        """
        Apply a multi-controlled Z gate to the system, flipping the phase of the |11...1⟩ state.
        """
        # This method seems redundant with CZin; included for compatibility
        self.CZin()

    def CN_Z(self, C):
        """
        Apply a multi-controlled Z gate with specified control qubits.

        Parameters:
        C (list of int): List of control qubit indices.
        """
        n = len(C)
        # Initialize a 2^n x 2^n identity matrix
        Mn = np.identity(2 ** n, dtype=complex)
        # Flip the phase of the |11...1⟩ state
        Mn[-1, -1] = -1
        # Reshape into a tensor suitable for n-qubit operations
        CnZ = np.reshape(Mn, [2 for _ in range(2 * n)])

        # Swap control qubits to the front
        qbit = self.qbit
        for i in range(n):
            qbit = np.swapaxes(qbit, i, C[i])
        # Apply the CnZ gate via tensor contraction
        qbit = np.tensordot(CnZ, qbit, axes=(range(n), range(n)))
        # Swap control qubits back to their original positions
        for i in reversed(range(n)):
            qbit = np.swapaxes(qbit, i, C[i])
        self.qbit = qbit

    def X_n(self, n):
        """
        Apply the Pauli-X (NOT) gate to the first n qubits.

        Parameters:
        n (int): Number of qubits to apply the X gate to.
        """
        for k in range(n):
            self.NOT(k)

    def Clarifies(self):
        """
        Analyze the quantum state to find the probabilities of different outcomes.

        Returns:
        arr (list): List of tuples containing amplitudes and corresponding bitstrings.
        """
        u = 0
        self.n = len(self.qbit.shape)
        # Flatten the state vector
        qbit = np.reshape(self.qbit, 2 ** self.n)
        arr = []
        for k in range(2 ** self.n):
            r = qbit[k].real
            num = len(bin(k)) - 2
            zeros = self.n - num
            binum = '0' * zeros + bin(k)[2:]
            # Include states with significant amplitudes
            #if abs(r) > 2. ** (-self.n):
            arr.append((r, binum[:2 * int(self.n / 5)]))
        # Print the probability of obtaining the first solution
        return arr

def Solutions(n, q, A):
    """
    Generate the solution states for the given problem.

    Parameters:
    n (int): Number of qubits.
    q (int): Parameter related to the problem size.
    A (np.ndarray): Random matrix used in the problem.

    Returns:
    trushbits (list): List of bitstrings representing the solution states.
    """
    # Define indices for different qubit registers
    InputQubits = range(2 * n)
    Q1 = range(2 * n, 2 * n + q)
    Q2 = range(2 * n + q, 2 * n + 2 * q)
    Q3 = range(2 * n + 2 * q, 3 * n + 2 * q)
    OracleBit = 3 * n + 2 * q
    # Initialize the quantum register
    state = Qbit(3 * n + 2 * q + 1)
    # Apply Hadamard gates to create superposition
    state.H_n(2 * n)
    # Apply NOT gate to the first qubit in Q1
    state.NOT(Q1[0])
    for i in range(n):
        for j in range(2 * n):
            for r in range(q):
                state.C_SWAP(InputQubits[j], Q1[r], Q2[r])
            for r in range(q):
                s = (r + A[i][j]) % q
                state.C_SWAP(InputQubits[j], Q1[s], Q2[r])
        state.CNOT(Q1[0], Q3[i])
    # Apply multi-controlled NOT gate as part of the oracle
    state.CN_NOT(Q3, OracleBit)
    # Flatten the state vector
    qbit = np.reshape(state.qbit, 2 ** (3 * n + 2 * q + 1))
    trushbits = []
    for k in range(2 ** (3 * n + 2 * q + 1)):
        num = len(bin(k)) - 2
        zeros = 3 * n + 2 * q + 1 - num
        r = np.round(qbit[k].real, 3 * n + 2 * q)
        if abs(r) > 2. ** (-2 * (n + q)):
            bit = zeros * '0' + bin(k)[2:]
            trushbits.append(bit[2 * n:])
    return trushbits

class Circuit(object):
    """
    Class representing the quantum circuit for Grover's algorithm.
    """
    def __init__(self, n, trush):
        """
        Initialize the quantum circuit.

        Parameters:
        n (int): Number of qubits.
        trush (list): List of bitstrings representing the solution states.
        """
        self.n = n
        self.trush = trush
        # Initialize the quantum register
        self.state = Qbit(2 * n)
        # Apply Hadamard gates to create superposition
        self.state.H_n(2 * n)

    def AddBit(self):
        """
        Add an ancilla qubit to the quantum register based on the solution states.
        """
        n = self.n
        Sol = [int(i[-1]) for i in self.trush]
        qbit1 = np.reshape(self.state.qbit, 2 ** (2 * n))
        qbit2 = np.zeros(2 ** (2 * n + 1), dtype=complex)
        for i in range(2 ** (2 * n)):
            if Sol[i] == 1:
                qbit2[2 * i + 1] = qbit1[i]
            else:
                qbit2[2 * i] = qbit1[i]
        self.state.qbit = np.reshape(qbit2, [2 for _ in range(2 * n + 1)])

    def OmitBit(self):
        """
        Remove the ancilla qubit by summing over its states.
        """
        qbit = self.state.qbit
        # Sum over the last axis (ancilla qubit)
        qbit = np.sum(qbit, axis=-1)
        # Create a new quantum register without the ancilla qubit
        NewState = Qbit(2 * self.n)
        NewState.qbit = qbit
        self.state = NewState

    def OracleFlip(self):
        """
        Apply the oracle operation that flips the phase of the solution states.
        """
        self.AddBit()
        # Apply Z gate to the ancilla qubit
        self.state.Z(-1)
        self.OmitBit()

    def Diffusion(self):
        """
        Apply the diffusion operator (inversion about the mean).
        """
        self.state.H_n(2 * self.n)
        self.state.X_n(2 * self.n)
        self.state.CN_Z(range(2 * self.n))
        self.state.X_n(2 * self.n)
        self.state.H_n(2 * self.n)

    def GroverRotation(self):
        """
        Perform one Grover iteration (oracle followed by diffusion).
        """
        self.OracleFlip()
        self.Diffusion()

    def GroverSearch(self, num):
        """
        Perform multiple Grover iterations.

        Parameters:
        num (int): Number of Grover iterations to perform.
        """
        for _ in range(num):
            self.GroverRotation()

def Collapse(lst, n, A, t):
    """
    Simulate the measurement process to find a solution.

    Parameters:
    lst (list): List of tuples containing amplitudes and bitstrings.
    n (int): Number of qubits.
    A (np.ndarray): Random matrix used in the problem.
    """
    prob = np.array([i[0] ** 2 for i in lst], dtype='f8')

    # Randomly select a state based on the probabilities
    k = 0
    while k == 0:
        t += 1
        k = np.random.choice(np.arange(len(prob)), p=prob)
    num = len(bin(k)) - 2
    zeros = 2 * n - num
    binum = '0' * zeros + bin(k)[2:]
    x = np.array([int(i) for i in binum])
    if np.sum((A @ x) % n) == 0:
        return binum, t
    return None, t

# Main execution starts here
if __name__ == "__main__":
    # Prompt the user to input the number of qubits
    NUMQUBITS = int(input("Enter the number of qubits (NUMQUBITS): "))
    # Generate a random matrix A with elements in the range [0, NUMQUBITS)
    RANDOMMATRIX = np.random.randint(NUMQUBITS, size=(NUMQUBITS, 2 * NUMQUBITS))
    print("Random Matrix A:")
    print(RANDOMMATRIX)

    # Generate the solution states
    TRUSH = Solutions(NUMQUBITS, NUMQUBITS, RANDOMMATRIX)
    print("Number of solution: " + str(sum([int(i[-1]) for i in TRUSH])-1))



    # Initialize the quantum circuit for Grover's algorithm
    c = Circuit(NUMQUBITS, TRUSH)

    # Calculate the optimal number of Grover iterations
    if NUMQUBITS == 2:
        for i in [1,2,3]:
            if i == 1:
                c.GroverSearch(1)
                arr = c.state.Clarifies()
            # Simulate the measurement process to find a solution
            binum, non  = Collapse(arr, NUMQUBITS, RANDOMMATRIX, 0)
            if binum != None:
                break
    else:
        num = 0
        num0 = 0
        tri = 0
        tot = 0
        binum = None
        for i in range(2 * NUMQUBITS + 1):
            tot += 1
            num = int(np.floor(np.pi / 4 * 2. ** (i / 2.)))
            c.GroverSearch(num-num0)
            num0 = num
            arr = c.state.Clarifies()
            # Simulate the measurement process to find a solution
            binum, tri  = Collapse(arr, NUMQUBITS, RANDOMMATRIX, tri)
            if binum != None:
                break

        c.state.print()


    if binum == None:
        print("There is no non-trivial Solution!")
    else:
        print("Solution: " + binum)
