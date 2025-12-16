import random
import math

# --- CONSTANTES DO CIRCUITO ---
V_PLUS = 9.0
V_MINUS = 0.0
V_REF = V_PLUS / 2  # Terra Virtual
GAIN = 3.2   # Ganho do AmpOp A
DELTA_V = V_PLUS - V_MINUS # 9V

# Tabelas Verdade
AND_TABLE = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
OR_TABLE  = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
NAND_TABLE= {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
NOR_TABLE = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
XOR_TABLE = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0} # Só pra teste, 1 neurônio não resolve XOR
INHIBIT_TABLE = {(0, 0): 0, (0, 1): 0, (1, 0): 1, (1, 1): 0} # A AND NOT B
porta_table = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1} # A OR B

def clip(v: float, vmin: float = 0.0, vmax: float = 1.0) -> float:
    return max(vmin, min(vmax, v))

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

def sigmoid_derivative(x):
    # Derivada da sigmoide: f'(x) = f(x) * (1 - f(x))
    s = sigmoid(x)
    return s * (1 - s)

def frac_to_voltage(frac: float) -> float:
    # Converte 0.0-1.0 para 0V-9V
    return V_MINUS + frac * DELTA_V

def forward_pass(w1: float, w2: float, w_bias: float, x1: int, x2: int):
    """
    Calcula a passagem direta (Forward) e retorna os valores intermediários
    necessários para o gradiente.
    """
    
    # 1. Tensão de Bias (Threshold)
    # O LM324 satura em ~7.5V, então limitamos o bias também (caso venha de um buffer)
    v_bias = clip(frac_to_voltage(w_bias), V_MINUS, 7.5)
    
    # 2. Nó de Entrada (Média Ponderada Dinâmica)
    # R_ref sempre conectado. R1 conecta se x1=1. R2 conecta se x2=1.
    soma_v = V_REF
    n = 1.0 # Divisor (começa com 1 do R_ref)
    
    if x1:
        soma_v += frac_to_voltage(w1)
        n += 1.0
    if x2:
        soma_v += frac_to_voltage(w2)
        n += 1.0
        
    v_in = soma_v / n
    
    # 3. Amplificação (AmpOp A)
    # Formula: V_out = V_ref + Gain * (V_in - V_ref)
    v_a_raw = V_REF + GAIN * (v_in - V_REF)
    v_a = clip(v_a_raw, V_MINUS, 7.5) # Saturação do OpAmp real
    
    # 4. Predição (Comparador / Hinge)
    # Margem de decisão: Va > Vbias
    pred_binaria = 1 if v_a > v_bias else 0
    
    return v_a, v_bias, n, pred_binaria

def train_neuron(target_table: dict, gate_name: str = "Custom", lr: float = 0.001, epochs: int = 500000):
    w1 = random.uniform(0.0, 1.0)
    w2 = random.uniform(0.0, 1.0)
    w_bias = random.uniform(0.0, 1.0)
    
    print(f"--- Treinando {gate_name} (MSE Sigmoide) ---")
    
    for epoch in range(epochs):
        total_error = 0.0
        exemplos = list(target_table.items())
        random.shuffle(exemplos)
        
        for (x1, x2), y_target in exemplos:
            v_a, v_bias, n, _ = forward_pass(w1, w2, w_bias, x1, x2)
            z = v_a - v_bias
            
            # MSE com Sigmoide
            y_pred = sigmoid(z)
            error = y_target - y_pred
            total_error += error ** 2
            
            # Gradiente: dL/dz = -2 * error * sigmoid_derivative(z)
            delta = -2 * error * sigmoid_derivative(z)
            
            if x1: w1 -= lr * delta * GAIN * (1.0/n) * DELTA_V
            if x2: w2 -= lr * delta * GAIN * (1.0/n) * DELTA_V
            
            w_bias -= lr * delta * (-1.0) * DELTA_V
            
            w1 = clip(w1)
            w2 = clip(w2)
            w_bias = clip(w_bias, 0.0, 7.5 / DELTA_V)
    
        if total_error < 1e-5: break
            
    return w1, w2, w_bias

# --- EXECUÇÃO ---
if __name__ == "__main__":
    while True:
        print("Digite a tabela verdade desejada como uma string de 4 bits.")
        print("Ordem de Entrada: (0,0), (1,0), (0,1), (1,1)")
        print("Exemplo AND: 0001")
        print("Exemplo OR:  0111")
        
        s_input = input("Tabela (4 bits): ").strip()
        
        if len(s_input) != 4 or not all(c in '01' for c in s_input):
            print("Erro: A entrada deve ter exatamente 4 caracteres '0' ou '1'.")
            exit()
            
        # Construindo a tabela na ordem (0,0), (1,0), (0,1), (1,1)
        custom_table = {
            (0, 0): int(s_input[0]),
            (1, 0): int(s_input[1]), 
            (0, 1): int(s_input[2]), 
            (1, 1): int(s_input[3])
        }
        
        # Identificação de portas conhecidas (Baseado na ordem 00, 10, 01, 11)
        known_gates = {
            "0001": "AND",
            "0111": "OR",
            "1110": "NAND",
            "1000": "NOR",
            "0110": "XOR",
            "0100": "INHIBIT (A AND NOT B)",
            "0010": "INHIBIT (B AND NOT A)"
        }
        
        gate_name = known_gates.get(s_input, f"Custom: {s_input}")
        
        w1_final, w2_final, w_bias_final = train_neuron(custom_table, gate_name=gate_name)
        
        # Exibir Resultados
        v1 = frac_to_voltage(w1_final)
        v2 = frac_to_voltage(w2_final)
        v_th = frac_to_voltage(w_bias_final)
        
        # Calculo de Resistencia (R = V * 10k / V_PLUS)
        r1 = v1 * 10.0 / V_PLUS
        r2 = v2 * 10.0 / V_PLUS
        r_th = v_th * 10.0 / V_PLUS
        
        print(f"\nRESULTADOS FINAIS ({gate_name}):")
        print(f"  Potenciômetro P1 (w1): {v1:.2f} V")
        print(f"  Potenciômetro P2 (w2): {v2:.2f} V")
        print(f"  Potenciômetro TH (wb): {v_th:.2f} V")
        
        print("-" * 40)
        print("TESTE DE VERIFICAÇÃO:")
        
        erros = 0
        for (x1, x2), target in custom_table.items():
            va, vb, n, pred = forward_pass(w1_final, w2_final, w_bias_final, x1, x2)
            status = "OK" if pred == target else "ERRO"
            if pred != target: erros += 1
            
            print(f"  In({x1}, {x2}) | Divisor n={n} | Va={va:5.2f}V vs Bias={v_th:5.2f}V | LED={pred} (Meta {target}) -> {status}")

        if erros == 0:
            print("\nSUCESSO: A rede aprendeu a porta perfeitamente!")
        else:
            print(f"\nFALHA: A rede errou {erros} casos (talvez precise de mais épocas ou a porta não é linearmente separável).")