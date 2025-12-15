# Rede Neural Analógica (Simulação)

Este repositório contém simulações em Python de redes neurais projetadas para serem implementadas fisicamente com componentes analógicos (OpAmps LM324, resistores, etc.). O objetivo é calcular os valores de resistores e tensões de bias necessários para configurar o hardware real.

## Estrutura do Projeto

*   **`Perceptron.py`**: Simulação de um único neurônio (Perceptron). Capaz de resolver problemas linearmente separáveis como AND, OR, NAND, NOR.
    *   Calcula os pesos (w1, w2) e bias (wb) como tensões de 0V a 9V.
    *   Usa minimização de Erro Quadrático (MSE) com margem de segurança de 0.3V.
    *   Considera a saturação real do LM324 (~7.5V).

*   **`Perceptron3N.py`**: Simulação de uma rede Multi-Layer Perceptron (MLP) com 3 neurônios (2 na camada oculta, 1 na saída).
    *   Capaz de resolver problemas não-lineares como **XOR** e **XNOR**.
    *   Implementa topologia mista de tensão (Camada 1 em 9V, Camada 2 em 7.5V) para casar impedâncias e níveis lógicos.
    *   Usa Momentum (0.9) para estabilidade e convergência.

*   **`ltspice/`**: Arquivos de simulação de circuito (.asc) para validação elétrica no LTSpice.

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
[Caio D.] - Projeto de Graduação (LAB VI)
