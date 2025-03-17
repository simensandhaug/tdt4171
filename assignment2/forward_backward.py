import numpy as np


transition_matrix = np.array([[0.7, 0.3],
                              [0.3, 0.7]])

observation_matrix = {
    True: np.array([0.9, 0.2]),
    False: np.array([0.1, 0.8])
}

prior = np.array([0.5, 0.5])

def forward_backward(evidence_sequence):
    fwd = prior
    forward_messages = []

    for ev in evidence_sequence:
        fwd = observation_matrix[ev] * (transition_matrix @ fwd)

        # Normalize
        fwd = fwd / np.sum(fwd)

        forward_messages.append(fwd.copy())

    return forward_messages


e_1_2 = [True, True]
e_1_5 = [True, True, False, True, True]

# Compute forward messages
forward_messages_1_2 = forward_backward(e_1_2)
forward_messages_1_5 = forward_backward(e_1_5)

# Extract final
P_X2_given_e1_2 = forward_messages_1_2[-1][0]
P_X5_given_e1_5 = forward_messages_1_5[-1][0]

# Print results
print(f"P(X2 | e1:2) = {P_X2_given_e1_2:.3f} (Expected: 0.883)")
print(f"P(X5 | e1:5) = {P_X5_given_e1_5:.3f}")

# Print all normalized forward messages
for i, fwd in enumerate(forward_messages_1_5, start=1):
    print(f"f{i}: {fwd}")
