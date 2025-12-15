# Rede Neural Analógica (Simulação)

Este repositório contém simulações em Python de redes neurais projetadas para serem implementadas fisicamente com componentes analógicos (OpAmps LM324, resistores, etc.). O objetivo é calcular os valores de resistores e tensões de bias necessários para configurar o hardware real.

## Estrutura do Projeto

O projeto está dividido em duas abordagens matemáticas distintas para o cálculo dos pesos, organizadas em pastas:

### 1. Pasta `MSE/` (Mean Squared Error)
Utiliza a abordagem clássica de minimização do **Erro Quadrático Médio**.
*   **Objetivo**: Tenta forçar a tensão de saída a ser exatamente igual ao alvo (ex: 7.5V para '1' e 0V para '0').
*   **Características**:
    *   Mais rígido matematicamente.
    *   Pode ter dificuldade de convergência em portas lógicas com margens estreitas (como AND/NAND) devido à saturação dos OpAmps.
    *   Arquivos: `Perceptron.py` (1 Neurônio) e `Perceptron3N.py` (3 Neurônios).

### 2. Pasta `Hinge Loss/` (Margem de Segurança)
Utiliza uma função de perda baseada em **Margem (Hinge Loss)**, similar ao SVM.
*   **Objetivo**: Penaliza apenas se a saída estiver errada ou dentro de uma "zona de risco" (margem de 0.3V). Se a saída estiver correta e segura, o erro é zero.
*   **Características**:
    *   Mais robusto para implementação em hardware.
    *   Permite que o circuito encontre qualquer solução que funcione ("se funciona, não mexe"), gerando maior diversidade de configurações válidas.
    *   Converge mais rápido para problemas complexos como XOR.
    *   Arquivos: `Perceptron_Hinge.py` (1 Neurônio) e `Perceptron3N_Hinge.py` (3 Neurônios).

### Outros
*   **`ltspice/`**: Arquivos de simulação de circuito (.asc) para validação elétrica no LTSpice.
*   **`Perceptron_LogLoss.py`**: (Experimental) Implementação usando Cross-Entropy Loss.

## Como Usar

### Pré-requisitos
*   Python 3.x

### Executando o Perceptron Simples
1.  Execute o script:
    ```bash
    python Perceptron.py
    ```
2.  Digite a tabela verdade desejada (4 bits). Exemplo para NAND: `1110`.
3.  O programa retornará as tensões de ajuste para os potenciômetros P1, P2 e Bias.

### Executando a Rede XOR (3 Neurônios)
1.  Execute o script:
    ```bash
    python Perceptron3N.py
    ```
2.  Digite a tabela verdade desejada. Exemplo para XOR: `0110`.
3.  O programa treinará a rede e exibirá as tensões para os 3 neurônios (9 potenciômetros no total).

## Detalhes Técnicos da Implementação

*   **Hardware Alvo**: Amplificadores Operacionais LM324.
*   **Tensão de Alimentação**: 9V (Assimétrica).
*   **Tensão de Referência**: 4.5V (VCC/2).
*   **Lógica de Treinamento**: Gradient Descent com Hinge Loss adaptada (Margem de 0.3V).
*   **Limitações Físicas**: O código simula explicitamente a saturação dos amplificadores (clipagem em 7.5V) para garantir que os pesos calculados funcionem no mundo real.

## Autor
[Caio D. S. Ribeiro] - Projeto de Graduação (LAB VI)
