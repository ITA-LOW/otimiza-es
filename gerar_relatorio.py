from fpdf import FPDF
import os
import sys
import shutil
from datetime import datetime

class PDF(FPDF):
    def header(self):
        if os.path.exists('logos/logo_laia.jpeg'):
            self.image('logos/logo_laia.jpeg', 10, 8, 33)
        if os.path.exists('logos/logo_inep.png'):
            self.image('logos/logo_inep.png', 170, 8, 33)
        
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Relatório de Otimização de Parque Eólico', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 1, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, image_path):
        if os.path.exists(image_path):
            self.image(image_path, x=35, w=95)
        else:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f'[Imagem {image_path} não encontrada]', 0, 1)
        self.ln(10)

def generate_report():
    pdf = PDF()
    pdf.add_page()

    output_dir = 'output'
    charts = {
        "Comparação de Produção de Energia (AEP)": os.path.join(output_dir, "report_aep_comparison.png"),
        "Convergência do Algoritmo Genético": os.path.join(output_dir, "max_fitness_vs_generations.png"),
        "Comparativo de Layouts (Inicial vs. Otimizado)": os.path.join(output_dir, "report_combined_layout.png"),
        "Distribuição de Velocidade Média nas Turbinas": os.path.join(output_dir, "report_velocity_distribution.png"),
        "Campo de Velocidade e Esteiras - Layout Inicial": os.path.join(output_dir, "report_initial_layout_wake.png"),
        "Campo de Velocidade e Esteiras - Layout Otimizado": os.path.join(output_dir, "report_optimized_layout_wake.png"),
    }

    for title, path in charts.items():
        pdf.chapter_title(title)
        pdf.chapter_body(path)

    report_filename = 'relatorio_otimizacao.pdf'
    pdf.output(report_filename)
    print(f"\nRelatório salvo com sucesso como '{report_filename}'")

    # Lógica de Limpeza
    if "--clean" in sys.argv:
        print(f"Deletando o diretório '{output_dir}'...")
        try:
            shutil.rmtree(output_dir)
            print("Diretório de saída limpo com sucesso.")
        except OSError as e:
            print(f"Erro ao deletar o diretório {output_dir}: {e.strerror}")

if __name__ == '__main__':
    try:
        from fpdf import FPDF
    except ImportError:
        print("Biblioteca fpdf2 não encontrada.")
        print("Por favor, instale-a com o comando: pip install fpdf2")
        exit()
    
    generate_report()
