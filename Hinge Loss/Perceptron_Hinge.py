import random
import math

# --- CONSTANTES DO CIRCUITO ---
V_PLUS = 9.0
V_MINUS = 0.0
V_REF = V_PLUS / 2  # Terra Virtual
GAIN = 3.2   # Ganho do AmpOp A
DELTA_V = V_PLUS - V_MINUS # 9V

def clip(v: float, vmin: float = 0.0, vmax: float = 1.0) -> float:
    return max(vmin, min(vmax, v))

def frac_to_voltage(frac: float) -> float:
    # Converte 0.0-1.0 para 0V-9V
    return V_MINUS + frac * DELTA_V

def forward_pass(w1: float, w2: float, w_bias: float, x1: int, x2: int):
    """
    Calcula a passagem direta (Forward).
    """
    
    # 1. Tensão de Bias (Threshold)
    v_bias = clip(frac_to_voltage(w_bias), V_MINUS, 7.5)
    
    # 2. Nó de Entrada
    soma_v = V_REF
    n = 1.0 
    
    if x1:
        soma_v += frac_to_voltage(w1)
        n += 1.0
    if x2:
        soma_v += frac_to_voltage(w2)
        n += 1.0
        
    v_in = soma_v / n
    
    # 3. Amplificação
    v_a_raw = V_REF + GAIN * (v_in - V_REF)
    v_a = clip(v_a_raw, V_MINUS, 7.5)
    
    # 4. Predição Binária
    pred_binaria = 1 if v_a > v_bias else 0
    
    return v_a, v_bias, n, pred_binaria

def train_neuron(target_table: dict, gate_name: str = "Custom", lr: float = 0.001, epochs: int = 500000):
    """
    Treina usando Hinge Loss (Perceptron com Margem).
    Objetivo: y * (Va - Vbias) >= margem
    """
    
    # Inicialização Aleatória
    w1 = random.uniform(0.0, 1.0)
    w2 = random.uniform(0.0, 1.0)
    w_bias = random.uniform(0.0, 1.0)
    
    margem = 0.3 # Margem de segurança (Zona Morta)

    print(f"--- Treinando {gate_name} (Hinge Loss - Perceptron Puro) ---")
    
    for epoch in range(epochs):
        errors_count = 0
        
        # Embaralha
        exemplos = list(target_table.items())
        random.shuffle(exemplos)
        
        for (x1, x2), y_target in exemplos:
            # Forward
            v_a, v_bias, n, _ = forward_pass(w1, w2, w_bias, x1, x2)
            
            # Distância
            z = v_a - v_bias
            
            # Alvo Bipolar: 1 ou -1
            y_sign = 1.0 if y_target == 1 else -1.0
            
            # Hinge Loss Condition:
            # Queremos que o sinal alinhado com a distância seja maior que a margem.
            # Validação: y_sign * z >= margem
            
            L = max(0, margem - y_sign * z)
            
            if L > 0:
                # VIOLAÇÃO!
                errors_count += 1
                
                # Gradiente do Hinge Loss:
                # L = margem - y_sign * z
                # dL/dz = -y_sign
                delta = -y_sign
                
                # --- Atualização w1 ---
                # dL/dw1 = dL/dz * dz/dw1 = delta * (GAIN/n * DELTA_V)
                if x1:
                    grad_w1 = delta * (GAIN / n) * DELTA_V
                    w1 -= lr * grad_w1
                
                # --- Atualização w2 ---
                if x2:
                    grad_w2 = delta * (GAIN / n) * DELTA_V
                    w2 -= lr * grad_w2
                    
                # --- Atualização Bias ---
                # dL/dwb = dL/dz * dz/dwb = delta * (-DELTA_V)
                grad_bias = delta * (-DELTA_V)
                w_bias -= lr * grad_bias
                
                # Clip
                w1 = clip(w1)
                w2 = clip(w2)
                w_bias = clip(w_bias, 0.0, 7.5 / DELTA_V)
        
        if errors_count == 0:
            # Se passou por todos os exemplos sem violar a margem, ACABOU.
            # Não tenta "melhorar" o que já está bom.
            # Isso preserva a "personalidade" da solução encontrada.
            break
            
    return w1, w2, w_bias

# --- EXECUÇÃO ---
if __name__ == "__main__":
    while True:
        print("Digite a tabela verdade (4 bits). Ex: 0001 (AND)")
        s_input = input("Tabela: ").strip()
        
        if len(s_input) != 4 or not all(c in '01' for c in s_input):
            print("Erro: 4 bits apenas.")
            continue
            
        custom_table = {
            (0, 0): int(s_input[0]),
            (1, 0): int(s_input[1]), 
            (0, 1): int(s_input[2]), 
            (1, 1): int(s_input[3])
        }
        
        known_gates = {
            "0001": "AND", "0111": "OR", "1110": "NAND", "1000": "NOR",
            "0110": "XOR", "0100": "INHIBIT A", "0010": "INHIBIT B"
        }
        gate_name = known_gates.get(s_input, f"Custom: {s_input}")
        
        w1_final, w2_final, w_bias_final = train_neuron(custom_table, gate_name=gate_name)
        
        v1 = frac_to_voltage(w1_final)
        v2 = frac_to_voltage(w2_final)
        v_th = frac_to_voltage(w_bias_final)
        
        print(f"\nRESULTADOS ({gate_name}):")
        print(f"  P1 (w1): {v1:.2f} V")
        print(f"  P2 (w2): {v2:.2f} V")
        print(f"  TH (wb): {v_th:.2f} V")
        
        print("-" * 30)
        erros = 0
        for (x1, x2), target in custom_table.items():
            va, vb, n, pred = forward_pass(w1_final, w2_final, w_bias_final, x1, x2)
            status = "OK" if pred == target else "ERRO"
            if pred != target: erros += 1
            print(f"  In({x1},{x2}) | Va={va:4.2f}V Bias={vb:4.2f}V | Out={pred} ({target}) -> {status}")
            
        if erros == 0: print("\nSUCESSO!")
        else: print(f"\nFALHA ({erros} erros)")
