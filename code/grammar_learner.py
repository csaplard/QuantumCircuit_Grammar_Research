
import sys
import os
import numpy as np
import pandas as pd
import time
from signal_processing import sax_encoding

# np.random.seed(42)  # Removed global seed for publication integrity

# Files
BOOT_FILE = "files/noise_deconvolution/expanded_torino_bootstrap_10k.csv"
SYN_FILE = "files/noise_deconvolution/expanded_torino_synthetic_10k.csv"
RESULTS_FILE = "files/noise_deconvolution/grammar_learning_results.txt"

# Params
HIDDEN_DIM = 16
SEQ_LEN = 20
EPOCHS = 50
LR = 0.01

# ==============================================================================
# NUMPY LSTM FOR CLASSIFICATION (Character Prediction)
# ==============================================================================
class CharLSTM:
    def __init__(self, vocab_size, hidden_dim, learning_rate):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Xavier init
        scale = np.sqrt(2.0 / (vocab_size + hidden_dim))
        self.W_ih = np.random.randn(4 * hidden_dim, vocab_size) * scale
        self.W_hh = np.random.randn(4 * hidden_dim, hidden_dim) * scale
        self.b = np.zeros(4 * hidden_dim)
        self.b[0:hidden_dim] = 1.0 # Forget gate bias
        
        # Output layer (hidden -> vocab probabilities)
        self.W_out = np.random.randn(vocab_size, hidden_dim) * scale
        self.b_out = np.zeros(vocab_size)
        
        # Adam
        self._init_adam()

    def _init_adam(self):
        self.t_adam = 0
        self.params = ['W_ih', 'W_hh', 'b', 'W_out', 'b_out']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        
    def sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
    def softmax(self, x): 
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def forward(self, x_indices):
        T = len(x_indices)
        H = self.hidden_dim
        
        h = np.zeros(H)
        c = np.zeros(H)
        cache = []
        
        outputs = []
        
        for t in range(T):
            x_onehot = np.zeros(self.vocab_size)
            x_onehot[x_indices[t]] = 1.0
            
            gates = self.W_ih @ x_onehot + self.W_hh @ h + self.b
            f = self.sigmoid(gates[0:H])
            i = self.sigmoid(gates[H:2*H])
            cc = np.tanh(gates[2*H:3*H])
            o = self.sigmoid(gates[3*H:4*H])
            
            c_new = f * c + i * cc
            h_new = o * np.tanh(c_new)
            
            # Predict next char
            logits = self.W_out @ h_new + self.b_out
            probs = self.softmax(logits)
            outputs.append(probs)
            
            cache.append({'x': x_onehot, 'h_prev': h, 'c_prev': c, 
                          'f': f, 'i': i, 'cc': cc, 'o': o, 
                          'c_new': c_new, 'gates': gates, 'h_new': h_new})
            
            h = h_new
            c = c_new
            
        return outputs, cache

    def backward(self, x_indices, y_indices, cache):
        T = len(x_indices)
        H = self.hidden_dim
        
        dW_ih = np.zeros_like(self.W_ih)
        dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros_like(self.b)
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        
        dh_next = np.zeros(H)
        dc_next = np.zeros(H)
        
        loss = 0
        
        for t in reversed(range(T)):
            probs = self.softmax(self.W_out @ cache[t]['h_new'] + self.b_out) # Recompute for safety
            
            # Cross entropy loss gradient
            # dL/dy = probs - target
            dy = probs.copy()
            dy[y_indices[t]] -= 1.0
            
            loss += -np.log(probs[y_indices[t]] + 1e-10)
            
            dW_out += np.outer(dy, cache[t]['h_new'])
            db_out += dy
            
            dh = self.W_out.T @ dy + dh_next
            
            # LSTM gradients
            s = cache[t]
            do = dh * np.tanh(s['c_new'])
            dc = dh * s['o'] * (1 - np.tanh(s['c_new'])**2) + dc_next
            di = dc * s['cc']
            dcc = dc * s['i']
            df = dc * s['c_prev']
            
            df_raw = df * s['f'] * (1 - s['f'])
            di_raw = di * s['i'] * (1 - s['i'])
            dcc_raw = dcc * (1 - s['cc']**2)
            do_raw = do * s['o'] * (1 - s['o'])
            
            dgates = np.concatenate([df_raw, di_raw, dcc_raw, do_raw])
            
            dW_ih += np.outer(dgates, s['x'])
            dW_hh += np.outer(dgates, s['h_prev'])
            db += dgates
            
            dh_next = self.W_hh.T @ dgates
            dc_next = dc * s['f']
            
        # Clip
        for g in [dW_ih, dW_hh, db, dW_out, db_out]:
            np.clip(g, -5, 5, out=g)
            
        # Update (Adam)
        grads = {'W_ih': dW_ih, 'W_hh': dW_hh, 'b': db, 'W_out': dW_out, 'b_out': db_out}
        self.t_adam += 1
        beta1, beta2 = 0.9, 0.999
        for p in self.params:
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * grads[p]
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * grads[p]**2
            m_hat = self.m[p] / (1 - beta1**self.t_adam)
            v_hat = self.v[p] / (1 - beta2**self.t_adam)
            setattr(self, p, getattr(self, p) - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8))
            
        return loss / T

# ==============================================================================
# MAIN LOGIC
# ==============================================================================
def train_model(file_path_or_data, label, alphabet_size=5, hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN, epochs=EPOCHS, lr=LR, data_is_array=False, seed=None):
    """
    Trains the LSTM model. 
    Args:
        file_path_or_data: Path to CSV or numpy array of signal if data_is_array=True.
        label: Name for logging.
        alphabet_size: Number of SAX symbols (e.g., 5, 7, 9).
        hidden_dim, seq_len, epochs, lr: LSTM parameters.
        data_is_array: If True, file_path_or_data is treated as the signal numpy array.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
        
    print(f"\n[{label}] Training Grammar Learner (Alphabet={alphabet_size}, Seed={seed})...")
    
    if data_is_array:
        sig = file_path_or_data
    else:
        file_path = file_path_or_data
        print(f"  Source: {file_path}")
        if not os.path.exists(file_path):
            print("  Error: File not found.")
            return float('inf'), float('inf'), None, None
        
        # Load and Normalize
        df = pd.read_csv(file_path)
        # Assume 'signal' column exists or take the first numeric one
        if 'signal' in df.columns:
            sig = df['signal'].values
        else:
            # Fallback to first column
            sig = df.iloc[:,0].values
    
    # Normalize
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-10)
    
    # SAX Encoding
    # Using existing tool with Quantile forcing to prevent degeneracy
    sax_str = sax_encoding(sig, alphabet_size=alphabet_size, behavior='quantile')
    print(f"  SAX String length: {len(sax_str)}")
    
    # To indices (0 to alphabet_size-1)
    vocab = "abcdefghijklmnopqrstuvwxyz"[:alphabet_size]
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    data_idx = np.array([char_to_idx[c] for c in sax_str])
    
    # Train/Val Split (80/20)
    split = int(0.8 * len(data_idx))
    train_data = data_idx[:split]
    val_data = data_idx[split:]
    
    model = CharLSTM(vocab_size=alphabet_size, hidden_dim=hidden_dim, learning_rate=lr)
    
    losses = []
    
    print(f"  Training ({epochs} epochs)...")
    t0 = time.time()
    
    for epoch in range(epochs):
        # Train on distinct batches
        total_loss = 0
        n_batches = 0
        
        # Simple strided iterating
        for i in range(0, len(train_data) - seq_len - 1, seq_len):
            x = train_data[i : i+seq_len]
            y = train_data[i+1 : i+seq_len+1]
            
            # Forward + Backward
            outputs, cache = model.forward(x)
            loss = model.backward(x, y, cache)
            total_loss += loss
            n_batches += 1
            
        avg_train_loss = total_loss / n_batches if n_batches > 0 else 0
        losses.append(avg_train_loss)
        
        # Optional: Print every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"    Epoch {epoch+1:02d}: Loss = {avg_train_loss:.4f}", flush=True)
            
    # Validation
    total_val_loss = 0
    n_val = 0
    for i in range(0, len(val_data) - seq_len - 1, seq_len):
        x = val_data[i : i+seq_len]
        y = val_data[i+1 : i+seq_len+1]
        outputs, _ = model.forward(x)
        
        # Calc loss manually
        batch_loss = 0
        for t in range(seq_len):
            batch_loss += -np.log(outputs[t][y[t]] + 1e-10)
        total_val_loss += batch_loss / seq_len
        n_val += 1
        
    final_val_loss = total_val_loss / n_val if n_val > 0 else float('inf')
    dt = time.time() - t0
    
    print(f"  Final Validation Loss: {final_val_loss:.4f} (Time: {dt:.1f}s)")
    
    # Perplexity = exp(loss)
    ppl = np.exp(final_val_loss)
    print(f"  Perplexity: {ppl:.2f}")
    
    return final_val_loss, ppl, model, val_data 

def extract_grammar(model, data_indices, seq_len=SEQ_LEN):
    # print("\n=== EXTRACTING GRAMMAR (Learned Transition Probabilities) ===")
    
    vocab_size = model.vocab_size
    # Init matrix
    trans_matrix = np.zeros((vocab_size, vocab_size)) 
    counts = np.zeros(vocab_size)
    
    # Run through data with model state
    # To save time, just pick random segments
    indices = range(0, len(data_indices) - seq_len - 1, seq_len)
    
    for i in indices:
        x = data_indices[i : i+seq_len]
        # Forward pass to get probabilities at each step
        outputs, _ = model.forward(x)
        
        for t in range(len(outputs) - 1): 
            # current symbol
            curr = x[t] 
            # predicted probs for next
            probs = outputs[t]
            
            trans_matrix[curr] += probs
            counts[curr] += 1
            
    # Normalize rows
    # Divide sum of prob vectors by count to get Average Probability Vector P(next|curr)
    prob_matrix = trans_matrix / (counts[:, None] + 1e-10) 
    
    return prob_matrix

def print_grammar_summary(prob_matrix):
    size = prob_matrix.shape[0]
    vocab = "abcdefghijklmnopqrstuvwxyz"[:size]
    print(f"\n[Grammar Fingerprint {size}x{size}]")
    print("   " + "      ".join(vocab))
    for i in range(size):
        row_str = f"{vocab[i]}: "
        for j in range(size):
            row_str += f"{prob_matrix[i,j]:.4f} "
        print(row_str)

def main():
    print("=== GRAMMAR DISCOVERY EXPERIMENT ===")
    # 1. Bootstrap
    loss_boot, ppl_boot, model_boot, val_boot = train_model(BOOT_FILE, "BOOTSTRAP (Potential Grammar)", alphabet_size=7)
    
    # Extract Grammar
    matrix = extract_grammar(model_boot, val_boot)
    print_grammar_summary(matrix)

if __name__ == "__main__":
    main()
