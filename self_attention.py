import numpy as np

class SelfAttention:
    def __init__(self, d_model=5):
        self.d_model = d_model
        self.W_Q = np.eye(d_model)
        self.W_K = np.eye(d_model)
        self.W_V = np.eye(d_model)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, tokens, embeddings_dict):
        X = np.array([embeddings_dict[t] for t in tokens])
        
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        
        scores = (Q @ K.T) / np.sqrt(self.d_model)
        weights = self.softmax(scores)
        output = weights @ V
        
        return output, weights

def run_experiment():
    # Feature Map: [Electronics, Weight, Fire, Action, Emotions]
    embeddings = {
        "turn":      [2.5, 0.0, 0.0, 3.0, 0.0], # High value in 'Electronics'
        "on":        [3.0, 0.0, 0.0, 1.0, 0.0],
        "luggage":   [0.0, 5.0, 0.0, 0.0, 0.0],
        "light":     [0.2, 0.2, 0.2, 0.2, 0.2], # Neutral starting point
        "cigarette": [0.0, 0.0, 5.0, 0.0, 0.0],
        "makes":     [0.0, 0.0, 0.0, 2.0, 1.5],
        "of":        [0.0, 0.0, 0.0, 0.0, 0.5],
        "problems":  [0.0, 0.0, 0.0, 0.0, 5.0]  # High value in 'Emotions'
    }

    sentences = [
        ["turn", "on", "light"],
        ["light", "luggage"],
        ["light", "cigarette"],
        ["makes", "light", "of", "problems"]
    ]

    model = SelfAttention(d_model=5)

    print("self-attention: 'light' multi-contextual test")
    print("-"* 20)
    for tokens in sentences:
        output, weights = model.forward(tokens, embeddings)
        light_idx = tokens.index("light")
        
        print(f"\n-> Analyzing Sentence: '{' '.join(tokens)}'")
        
        attention_map = {
            token: float(value) 
            for token, value in zip(tokens, np.round(weights[light_idx], 3))
        }
        print(f"  Attention map for 'light': {attention_map}")

        context_vec = [float(v) for v in np.round(output[light_idx], 2)]
        print(f"  Contextualized vector: {context_vec}")

if __name__ == "__main__":
    run_experiment()