import time
import csv

class TimeCallback:
    def __init__(self, total_calls, csv_file='hyperparameters_log.csv'):
        self.total_calls = total_calls
        self.start_time = None
        self.iteration = 0
        self.csv_file = csv_file
        
        # Inicializando o arquivo CSV com cabeçalhos
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Escreve os cabeçalhos das colunas (hiperparâmetros e AEP), removendo 'iteration'
            writer.writerow(['cxpb', 'mutpb', 'pop', 'torneio', 'alpha', 'geracoes', 'indpb', 'sigma', 'AEP'])

    def __call__(self, res):
        # Inicializa o tempo de início na primeira chamada
        if self.iteration == 0:
            self.start_time = time.time()
        
        # Incrementa o número de iterações
        self.iteration += 1

        # Calcula o tempo decorrido
        elapsed_time = time.time() - self.start_time
        avg_time_per_iter = elapsed_time / self.iteration

        # Calcula o tempo restante com base nas iterações restantes
        remaining_iters = self.total_calls - self.iteration
        remaining_time = remaining_iters * avg_time_per_iter

        # Exibe o tempo restante
        print(f"Epoch {self.iteration}/{self.total_calls}")
        
        # Obtém os parâmetros e o AEP da última iteração
        params = res.x_iters[-1]  # Parâmetros da última iteração
        aep = -res.func_vals[-1]  # AEP da última iteração (negativo porque estamos minimizando)
        
        # Salva no arquivo CSV os parâmetros e o AEP da iteração atual
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(params + [aep])  # Removendo self.iteration da linha escrita

        # Opção para encerrar se atingir o número total de chamadas
        if self.iteration >= self.total_calls:
            print("Todas as iterações completadas.")
