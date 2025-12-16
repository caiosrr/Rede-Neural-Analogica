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
L2_SAT    = 6.0   # Saída Máxima do N3

GAIN = 3.2 # Ganho dos AmpOps (Igual para todos)

# Tabelas Verdade
AND_TABLE = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}
OR_TABLE  = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
NAND_TABLE= {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
NOR_TABLE = {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}
XOR_TABLE = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

def clip(v: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, v))

def sigmoid_derivative(x):
    # Derivada suave para passar pelo degrau do comparador (Surrogate Gradient)
    try:
        s = 1 / (1 + math.exp(-x))
    except OverflowError:
        s = 0 if x < 0 else 1
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
        self.w1 = random.uniform(0, 1)
        self.w2 = random.uniform(0, 1)
        self.w_bias = random.uniform(0, 1)
        
        # Memória
        self.last_va = 0.0
        self.last_bias_v = 0.0
        self.last_n = 1.0
        self.last_out_logic = 0
        
        # Momentum (Velocidade)
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
    
    print(f"\n[{layer_name}] {neuron.name}:")
    print(f"  P1 (w1): {neuron.w1*100:5.1f}% -> {v1:.2f}V")
    print(f"  P2 (w2): {neuron.w2*100:5.1f}% -> {v2:.2f}V ")
    print(f"  PB (wb): {neuron.w_bias*100:5.1f}% -> {vb:.2f}V")

def train_network(target_table, epochs=500000, lr=0.005):
    
    # --- CONFIGURAÇÃO DA TOPOLOGIA MISTA ---
    
    # Camada 1 (N1, N2): Mundo 9V
    n1 = HardwareNeuron("Oculto 1", v_signal=L1_SIGNAL, v_supply=L1_VCC, v_ref=L1_REF, v_sat=L1_SAT)
    n2 = HardwareNeuron("Oculto 2", v_signal=L1_SIGNAL, v_supply=L1_VCC, v_ref=L1_REF, v_sat=L1_SAT)
    
    # Camada 2 (N3): Mundo 7.5V
    n3 = HardwareNeuron("Saída",    v_signal=L2_SIGNAL, v_supply=L2_VCC, v_ref=L2_REF, v_sat=L2_SAT)
    
    momentum = 0.9
    margem = 0.3
    
    for i in range(epochs):
        errors_count = 0
        exemplos = list(target_table.items())
        random.shuffle(exemplos)
        
        for (x1, x2), y_target in exemplos:
            # Forward
            out_n1 = n1.forward(x1, x2)
            out_n2 = n2.forward(x1, x2)
            n3.forward(out_n1, out_n2)
            
            # --- Lógica Hinge Loss (N3) ---
            z_n3 = n3.last_va - n3.last_bias_v
            y_sign = 1.0 if y_target == 1 else -1.0
            
            
            L = max(0, margem - y_sign * z_n3)
            
            if L > 0:
                errors_count += 1
                
                # Gradiente do Hinge Loss
                # L = margem - y_sign * z
                # dL/dz = -y_sign
                delta_n3 = -y_sign
                
                # IMPORTANTE: Guardar os pesos ANTIGOS de N3 para o Backpropagation.
                # O erro de N1/N2 deve ser calculado com base na rede que gerou a saída atual,
                # não na rede já alterada.
                old_w1_n3 = n3.w1
                old_w2_n3 = n3.w2
                
                # --- Atualização N3 (Linear) ---
                # d(Va)/dw = Gain * (1/n) * V_signal
                # d(Vbias)/dw = V_supply
                
                # Fator de correção usa L2_SIGNAL (7.5V)
                # grad_w = delta * (GAIN/n) * V_signal
                factor_n3 = lr * delta_n3 * (1.0/n3.last_n) * GAIN * n3.v_signal
                
                # Update N3 com Momentum
                step_w1 = factor_n3 if out_n1 else 0
                n3.vel_w1 = momentum * n3.vel_w1 + step_w1
                n3.w1 -= n3.vel_w1 
                
                step_w2 = factor_n3 if out_n2 else 0
                n3.vel_w2 = momentum * n3.vel_w2 + step_w2
                n3.w2 -= n3.vel_w2
                
                # Bias: z = Va - Vbias. d(z)/d(bias) = -1.
                # grad_bias = delta * (-1) * V_supply
                step_bias = lr * delta_n3 * (-1.0) * n3.v_supply
                n3.vel_bias = momentum * n3.vel_bias + step_bias
                n3.w_bias -= n3.vel_bias
                
                # --- Backpropagation para N1/N2 ---
                # Usamos old_w1_n3 e old_w2_n3 aqui!
                
                dist_n1 = (n1.last_va - n1.last_bias_v)
                delta_n1 = (delta_n3 * old_w1_n3) * sigmoid_derivative(dist_n1)
                
                dist_n2 = (n2.last_va - n2.last_bias_v)
                delta_n2 = (delta_n3 * old_w2_n3) * sigmoid_derivative(dist_n2)

                # --- Atualização N1 ---
                factor_n1 = lr * delta_n1 * (1.0/n1.last_n) * GAIN * n1.v_signal
                
                step_w1_n1 = factor_n1 if x1 else 0
                n1.vel_w1 = momentum * n1.vel_w1 + step_w1_n1
                n1.w1 -= n1.vel_w1
                
                step_w2_n1 = factor_n1 if x2 else 0
                n1.vel_w2 = momentum * n1.vel_w2 + step_w2_n1
                n1.w2 -= n1.vel_w2
                
                step_bias_n1 = lr * delta_n1 * (-1.0) * n1.v_supply
                n1.vel_bias = momentum * n1.vel_bias + step_bias_n1
                n1.w_bias -= n1.vel_bias
                
                # --- Atualização N2 ---
                factor_n2 = lr * delta_n2 * (1.0/n2.last_n) * GAIN * n2.v_signal
                
                step_w1_n2 = factor_n2 if x1 else 0
                n2.vel_w1 = momentum * n2.vel_w1 + step_w1_n2
                n2.w1 -= n2.vel_w1
                
                step_w2_n2 = factor_n2 if x2 else 0
                n2.vel_w2 = momentum * n2.vel_w2 + step_w2_n2
                n2.w2 -= n2.vel_w2
                
                step_bias_n2 = lr * delta_n2 * (-1.0) * n2.v_supply
                n2.vel_bias = momentum * n2.vel_bias + step_bias_n2
                n2.w_bias -= n2.vel_bias

                # Manter físico (0-100%)
                for n in [n1, n2, n3]:
                    n.w1 = clip(n.w1, 0, 1)
                    n.w2 = clip(n.w2, 0, 1)
                    max_bias_w = n.v_sat / n.v_supply
                    n.w_bias = clip(n.w_bias, 0, max_bias_w)
        
        if errors_count == 0:
            break
            
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
            
        custom_table = {
            (0, 0): int(s_input[0]),
            (1, 0): int(s_input[1]), 
            (0, 1): int(s_input[2]), 
            (1, 1): int(s_input[3])
        }
        
        known_gates = {
            "0001": "AND", "0111": "OR", "1110": "NAND", "1000": "NOR",
            "0110": "XOR", "1001": "XNOR",
            "0100": "INHIBIT A", "0010": "INHIBIT B"
        }
        
        gate_name = known_gates.get(s_input, f"Custom: {s_input}")
        
        best_n1, best_n2, best_n3 = None, None, None
        min_errors = 999
        
        print(f"--- Treinando {gate_name} (Hinge Loss + Backprop) ---")
        
        # Tentativas múltiplas para evitar mínimos locais (comum em MLP)
        for attempt in range(10): 
            n1, n2, n3 = train_network(custom_table, epochs=50000)
            erros = 0

            for (x1, x2), target in custom_table.items():
                y1 = n1.forward(x1, x2)
                y2 = n2.forward(x1, x2)
                y3 = n3.forward(y1, y2)
                
                m3 = abs(n3.last_va - n3.last_bias_v)
                
                if y3 != target:
                    erros += 1
                elif m3 < 0.3: # Se acertou mas margem ruim, conta como "meio erro" para desempate
                    erros += 0.1
                    
            if erros < min_errors:
                min_errors = erros
                best_n1, best_n2, best_n3 = n1, n2, n3
                print(f"  Tentativa {attempt+1}: Erros={erros:.1f} (Novo Melhor)")
            
            if erros == 0: 
                print(f"  Tentativa {attempt+1}: Convergência Perfeita!")
                break

        print(f"\n=== RESULTADOS PARA {gate_name} ===")
        
        if best_n1:
            print_res(best_n1, "CAMADA 1")
            print_res(best_n2, "CAMADA 1")
            print_res(best_n3, "CAMADA 2 (SAÍDA)")
            
            print("\n--- TESTE FINAL ---")
            final_errors = 0
            for (x1, x2), target in custom_table.items():
                y1 = best_n1.forward(x1, x2)
                y2 = best_n2.forward(x1, x2)
                y3 = best_n3.forward(y1, y2)
                
                if y3 != target:
                    final_errors += 1
                    status = "ERRO"
                else:
                    status = "OK"
                    
                print(f"  In({x1},{x2}) | N1={y1} N2={y2} -> N3={y3} (Meta {target}) | {status}")

            if final_errors == 0:
                print("\nSUCESSO: A rede aprendeu a porta perfeitamente!")
            else:
                print(f"\nFALHA: A rede errou {final_errors} casos.")
