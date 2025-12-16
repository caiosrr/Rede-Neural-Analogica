import random
import math

# --- CONSTANTES ELÉTRICAS ---
# CAMADA 1 (Oculta - N1 e N2)
L1_VCC    = 9.0   # Alimentação dos pots de Bias e OpAmps
L1_SIGNAL = 9.0   # Tensão que entra nas chaves (Input)
L1_REF    = 4.5   # Terra Virtual
L1_SAT    = 7.5   # Saída Máxima (Input para a próxima camada)

# CAMADA 2 (Saída - N3)
L2_VCC    = 7.5   # Alimentação dos pots de Bias e OpAmp
L2_SIGNAL = 7.5   # Tensão que entra nos pesos (Vem de N1/N2)
L2_REF    = 3.75  # Terra Virtual
L2_SAT    = 6   # Saída Máxima do N3

GAIN = 3.2 # Ganho dos AmpOps (Igual para todos)

# Tabelas Verdade
AND_TABLE = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
OR_TABLE  = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
NAND_TABLE= {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
NOR_TABLE = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
XOR_TABLE = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

def clip(v: float, vmin: float, vmax: float) -> float:
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

class HardwareNeuron:
    def __init__(self, name, v_signal, v_supply, v_ref, v_sat):
        self.name = name
        
        # Parâmetros Físicos Específicos desta Camada
        self.v_signal = v_signal # Tensão aplicada ao resistor de peso quando chave fecha
        self.v_supply = v_supply # Tensão máxima do potenciômetro de Bias
        self.v_ref    = v_ref    # Referência (Terra Virtual)
        self.v_sat    = v_sat    # Tensão máxima de saída do OpAmp
        
        # Pesos (0.0 a 1.0 - Posição do Potenciômetro)
        self.w1 = random.uniform(0.1, 0.9)
        self.w2 = random.uniform(0.1, 0.9)
        self.w_bias = random.uniform(0.1, 0.9)
        
        # Memória
        self.last_va = 0.0
        self.last_bias_v = 0.0
        self.last_n = 1.0
        self.last_out_logic = 0
        
        #  (Velocidade)
        self.vel_w1 = 0.0
        self.vel_w2 = 0.0
        self.vel_bias = 0.0

    def get_weight_voltage(self, w_frac):
        # Converte fração do pot (0-1) para tensão real baseada no sinal de entrada
        return w_frac * self.v_signal

    def get_bias_voltage(self, w_frac):
        # Bias é alimentado pelo VCC da camada
        return w_frac * self.v_supply

    def forward(self, in1_active, in2_active):
        # 1. Divisor de Tensão Variável
        soma_v = self.v_ref
        n = 1.0 
        
        if in1_active:
            soma_v += self.get_weight_voltage(self.w1)
            n += 1.0
        
        if in2_active:
            soma_v += self.get_weight_voltage(self.w2)
            n += 1.0
            
        v_in = soma_v / n
        self.last_n = n
        
        # 2. Amplificação
        v_a_raw = self.v_ref + GAIN * (v_in - self.v_ref)
        
        # Clipa na saturação específica desta camada
        self.last_va = clip(v_a_raw, 0.0, self.v_sat)
        
        # 3. Comparador
        # O Bias não deve exceder a tensão de saturação do OpAmp anterior/atual
        self.last_bias_v = clip(self.get_bias_voltage(self.w_bias), 0.0, self.v_sat)
        
        self.last_out_logic = 1 if self.last_va > self.last_bias_v else 0
        return self.last_out_logic

def print_res(neuron, layer_name):
    v1 = neuron.get_weight_voltage(neuron.w1)
    v2 = neuron.get_weight_voltage(neuron.w2)
    vb = neuron.get_bias_voltage(neuron.w_bias)
    
    # R = V * 10k / V_supply_of_pot
    r1 = (v1 / neuron.v_signal) * 10.0 if neuron.v_signal > 0 else 0
    r2 = (v2 / neuron.v_signal) * 10.0 if neuron.v_signal > 0 else 0
    rb = (vb / neuron.v_supply) * 10.0 if neuron.v_supply > 0 else 0
    
    print(f"\n[{layer_name}] {neuron.name}:")
    print(f"  P1 (w1): {neuron.w1*100:5.1f}% -> {v1:.2f}V")
    print(f"  P2 (w2): {neuron.w2*100:5.1f}% -> {v2:.2f}V ")
    print(f"  PB (wb): {neuron.w_bias*100:5.1f}% -> {vb:.2f}V")

def train_network(target_table, epochs=500000, lr=0.001):
    n1 = HardwareNeuron("Oculto 1", v_signal=L1_SIGNAL, v_supply=L1_VCC, v_ref=L1_REF, v_sat=L1_SAT)
    n2 = HardwareNeuron("Oculto 2", v_signal=L1_SIGNAL, v_supply=L1_VCC, v_ref=L1_REF, v_sat=L1_SAT)
    n3 = HardwareNeuron("Saída",    v_signal=L2_SIGNAL, v_supply=L2_VCC, v_ref=L2_REF, v_sat=L2_SAT)
    
    margem = 0.3
    decay = 1e-5
    
    for i in range(epochs):
        total_error = 0.0
        exemplos = list(target_table.items())
        random.shuffle(exemplos)
        
        for (x1, x2), y_target in exemplos:
            out_n1 = n1.forward(x1, x2)
            out_n2 = n2.forward(x1, x2)
            n3.forward(out_n1, out_n2)
            
            z_n3 = n3.last_va - n3.last_bias_v
            y_sign = 1.0 if y_target == 1 else -1.0
            
            # MSE com Sigmoide Deslocada (Shifted)
            z_shifted = z_n3 - (y_sign * margem)
            
            # Se já passou da margem, zera o erro
            if (y_target == 1 and z_n3 > margem) or (y_target == 0 and z_n3 < -margem):
                delta_n3 = 0.0
            else:
                y_pred = sigmoid(z_shifted)
                error = y_target - y_pred
                total_error += error ** 2
                delta_n3 = -2 * error * sigmoid_derivative(z_shifted)
            
            if delta_n3 == 0: continue

            # Update N3 (Sem Momentum, Com Decay)
            factor_n3 = lr * delta_n3 * (1.0/n3.last_n) * GAIN * n3.v_signal
            decay_val = lr * decay
            old_w1_n3, old_w2_n3 = n3.w1, n3.w2
            
            if out_n1: n3.w1 -= (factor_n3 + decay_val * n3.w1)
            if out_n2: n3.w2 -= (factor_n3 + decay_val * n3.w2)
            
            n3.w_bias -= lr * delta_n3 * (-1.0) * n3.v_supply
            
            # Backprop N1/N2
            dist_n1 = (n1.last_va - n1.last_bias_v)
            delta_n1 = delta_n3 * old_w1_n3 * sigmoid_derivative(dist_n1)
            
            dist_n2 = (n2.last_va - n2.last_bias_v)
            delta_n2 = delta_n3 * old_w2_n3 * sigmoid_derivative(dist_n2)

            # Update N1
            factor_n1 = lr * delta_n1 * (1.0/n1.last_n) * GAIN * n1.v_signal
            if x1: n1.w1 -= (factor_n1 + decay_val * n1.w1)
            if x2: n1.w2 -= (factor_n1 + decay_val * n1.w2)
            n1.w_bias -= lr * delta_n1 * (-1.0) * n1.v_supply
            
            # Update N2
            factor_n2 = lr * delta_n2 * (1.0/n2.last_n) * GAIN * n2.v_signal
            if x1: n2.w1 -= (factor_n2 + decay_val * n2.w1)
            if x2: n2.w2 -= (factor_n2 + decay_val * n2.w2)
            n2.w_bias -= lr * delta_n2 * (-1.0) * n2.v_supply

            for n in [n1, n2, n3]:
                n.w1 = clip(n.w1, 0.1, 0.9)
                n.w2 = clip(n.w2, 0.1, 0.9)
                # Limita o Bias um pouco abaixo da saturação para garantir margem se o sinal saturar
                # Ex: Se satura em 7.5V, limita bias em 7.0V
                max_bias = (n.v_sat - 0.5) / n.v_supply
                n.w_bias = clip(n.w_bias, 0.1, max_bias)
        
        if total_error < 1e-6: break
            
    return n1, n2, n3

if __name__ == "__main__":
    while True:
        print("Digite a tabela verdade desejada como uma string de 4 bits.")
        print("Ordem de Entrada: (0,0), (1,0), (0,1), (1,1)")
        print("Exemplo XOR: 0110")
        
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
            "1001": "XNOR",
            "0100": "INHIBIT (A AND NOT B)",
            "0010": "INHIBIT (B AND NOT A)"
        }
        
        gate_name = known_gates.get(s_input, f"Custom: {s_input}")
        
        best_n1, best_n2, best_n3 = None, None, None
        min_errors = 999
        
        print(f"--- Treinando {gate_name} (MSE Shifted Sigmoid - Multi-Start) ---")
        
        best_n1, best_n2, best_n3 = None, None, None
        min_errors = 999
        best_margin_min = -1.0
        
        # 20 Tentativas para fugir de mínimos locais (XOR é difícil)
        for attempt in range(20): 
            n1, n2, n3 = train_network(custom_table, epochs=100000)
            
            current_errors = 0
            current_margin_min = 999.0
            
            for (x1, x2), target in custom_table.items():
                y1 = n1.forward(x1, x2)
                y2 = n2.forward(x1, x2)
                y3 = n3.forward(y1, y2)
                
                # Verifica margens
                m1 = abs(n1.last_va - n1.last_bias_v)
                m2 = abs(n2.last_va - n2.last_bias_v)
                m3 = abs(n3.last_va - n3.last_bias_v)
                
                min_m = min(m1, m2, m3)
                if min_m < current_margin_min:
                    current_margin_min = min_m
                
                if y3 != target:
                    current_errors += 1
            
            # Critério: Menos erros > Maior Margem
            if current_errors < min_errors:
                min_errors = current_errors
                best_margin_min = current_margin_min
                best_n1, best_n2, best_n3 = n1, n2, n3
            elif current_errors == min_errors:
                if current_margin_min > best_margin_min:
                    best_margin_min = current_margin_min
                    best_n1, best_n2, best_n3 = n1, n2, n3
            
            # Se achou solução perfeita com boa margem, para
            if min_errors == 0 and best_margin_min > 0.2:
                break

        print(f"\n=== RESULTADOS PARA {gate_name} ===")
        
        if best_n1:
            print_res(best_n1, "CAMADA 1")
            print_res(best_n2, "CAMADA 1")
            print_res(best_n3, "CAMADA 2 (SAÍDA)")
            
            print("\n--- TESTE FINAL ---")
            total_errors = 0
            for (x1, x2), target in custom_table.items():
                y1 = best_n1.forward(x1, x2)
                y2 = best_n2.forward(x1, x2)
                y3 = best_n3.forward(y1, y2)
                
                # Verifica margens
                m1 = abs(best_n1.last_va - best_n1.last_bias_v)
                m2 = abs(best_n2.last_va - best_n2.last_bias_v)
                m3 = abs(best_n3.last_va - best_n3.last_bias_v)
                
                margin_ok = (m1 > 0.1) and (m2 > 0.1) and (m3 > 0.1)
                
                if y3 == target and margin_ok:
                    status = "OK"
                else:
                    status = "ERRO"
                    if not margin_ok: status += " (Margem)"
                    total_errors += 1
                    
                print(f" In({x1},{x2}) -> Oculta[{y1},{y2}] -> Out {y3} (Meta {target}) -> {status}")
                if not margin_ok:
                    print(f"    [DEBUG] Margens: N1={m1:.2f}V, N2={m2:.2f}V, N3={m3:.2f}V (Min 0.10V)")

            if total_errors == 0:
                print("\n>>> SUCESSO: A rede aprendeu a porta perfeitamente! <<<")
            else:
                print(f"\n>>> FALHA: A rede errou {total_errors} casos. <<<")
        else:
            print("Erro crítico no treinamento.")