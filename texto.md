```mermaid
flowchart TD
    A([Início]) --> B[Inicializar parâmetros]
    B --> C[Carregar dados YAML]
    C --> D[Definir constantes:<br>IND_SIZE=16<br>CIRCLE_RADIUS=1300m<br>VOLTAGE=33kV]
    D --> E[Criar classes DEAP:<br>FitnessMulti<br>Individual]
    E --> F[Configurar toolbox:<br>mate, mutate, select]

    subgraph Algoritmo_Genético["Algoritmo Genético"]
        F --> G[Gerar população inicial<br>(300 indivíduos)]
        G --> H[Avaliar fitness]
        
        subgraph Avaliação["Avaliação"]
            H --> H1[Converter para coordenadas<br>(16 nós)]
            H1 --> H2[Penalizar nós fora do círculo]
            H2 --> H3[Verificar distâncias mínimas]
            H3 --> H4[Calcular AEP]
            H4 --> H5[Clusterizar com KMeans]
            H5 --> H6[Gerar MST clusterizada]
            H6 --> H7[Calcular Plant:<br>Perdas & Custo]
        end

        H --> I[Seleção por torneio]
        I --> J[Crossover<br>(cxBlend α=0.5)]
        J --> K[Mutação Gaussiana<br>(σ=100m)]
        K --> L[Aplicar enforce_circle]
        L --> M[Nova população]
    end

    M --> N{300 gerações<br>completas?}
    N -- Não --> G
    N -- Sim --> O[Resultados]

    O --> P[Plotar layout ótimo]
    P --> Q[Exibir:<br>- Coordenadas<br>- AEP<br>- Perdas<br>- Custo]
    Q --> R([Fim])

    style H fill:#e6f3ff,stroke:#004080
    style Algoritmo_Genético fill:#f5f5f5,stroke:#666
    style Avaliação fill:#fff9e6,stroke:#cc9900
    style A,R fill:#004080,color:#fff
    style O fill:#008000,color:#fff