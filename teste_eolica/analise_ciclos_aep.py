import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

def analisar_ciclos_aep(csv_file):
    """Analisa padrões cíclicos em um arquivo CSV"""
    print(f"\n{'='*70}")
    print(f"Analisando: {csv_file.name}")
    print(f"{'='*70}")
    
    try:
        df = pd.read_csv(csv_file)
        
        # Localizar coluna AEP (case-insensitive)
        aep_col = None
        for col in df.columns:
            if 'aep' in col.lower():
                aep_col = col
                break
        
        if aep_col is None:
            print(f"⚠️  Coluna AEP não encontrada. Colunas disponíveis: {df.columns.tolist()}")
            return None
        
        aep = df[aep_col].dropna().values
        
        if len(aep) < 100:
            print(f"⚠️  Dados insuficientes ({len(aep)} valores)")
            return None
        
        print(f"✓ Coluna encontrada: '{aep_col}'")
        print(f"✓ Total de valores: {len(aep)}")
        print(f"✓ Intervalo: [{aep.min():.2f}, {aep.max():.2f}]")
        
        # 1. Análise de Picos (Extremos Locais)
        print(f"\n--- ANÁLISE DE PICOS ---")
        peaks_high, _ = signal.find_peaks(aep, height=np.percentile(aep, 75))
        peaks_low, _ = signal.find_peaks(-aep, height=-np.percentile(aep, 25))
        
        print(f"Picos altos detectados: {len(peaks_high)}")
        print(f"Picos baixos detectados: {len(peaks_low)}")
        
        if len(peaks_high) > 1:
            intervals_high = np.diff(peaks_high)
            print(f"  Intervalo médio entre picos altos: {intervals_high.mean():.1f} ± {intervals_high.std():.1f}")
            print(f"  Intervalo mínimo: {intervals_high.min()}, máximo: {intervals_high.max()}")
        
        if len(peaks_low) > 1:
            intervals_low = np.diff(peaks_low)
            print(f"  Intervalo médio entre picos baixos: {intervals_low.mean():.1f} ± {intervals_low.std():.1f}")
            print(f"  Intervalo mínimo: {intervals_low.min()}, máximo: {intervals_low.max()}")
        
        # 2. Análise de Frequência (FFT)
        print(f"\n--- ANÁLISE DE FREQUÊNCIA (FFT) ---")
        aep_normalized = (aep - aep.mean()) / aep.std()
        fft_values = fft(aep_normalized)
        freqs = fftfreq(len(aep_normalized))
        
        # Pega frequências positivas e correspondentes magnitudes
        positive_freqs = freqs[:len(freqs)//2]
        magnitudes = np.abs(fft_values[:len(freqs)//2])
        
        # Top 5 frequências dominantes
        top_5_idx = np.argsort(magnitudes)[-5:][::-1]
        print("Top 5 frequências dominantes:")
        for i, idx in enumerate(top_5_idx, 1):
            if positive_freqs[idx] > 0:
                periodo = 1 / positive_freqs[idx]
                print(f"  {i}. Frequência: {positive_freqs[idx]:.4f}, Período: {periodo:.1f} pontos, Magnitude: {magnitudes[idx]:.2f}")
        
        # 3. Autocorrelação
        print(f"\n--- AUTOCORRELAÇÃO ---")
        autocorr = np.correlate(aep_normalized, aep_normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        
        # Encontra o primeiro pico de autocorrelação (depois do lag 0)
        peaks_auto, _ = signal.find_peaks(autocorr[1:], height=0.3)
        if len(peaks_auto) > 0:
            first_peak_lag = peaks_auto[0] + 1
            print(f"Primeiro pico de autocorrelação no lag: {first_peak_lag} pontos")
            print(f"Valor da autocorrelação: {autocorr[first_peak_lag]:.4f}")
        else:
            print("Nenhum pico significativo de autocorrelação detectado")
        
        # 4. Estatísticas Globais
        print(f"\n--- ESTATÍSTICAS ---")
        print(f"Média: {aep.mean():.2f}")
        print(f"Desvio Padrão: {aep.std():.2f}")
        print(f"Coeficiente de Variação: {(aep.std()/aep.mean()):.4f}")
        print(f"Skewness: {pd.Series(aep).skew():.4f}")
        print(f"Kurtosis: {pd.Series(aep).kurtosis():.4f}")
        
        return {
            'arquivo': csv_file.name,
            'aep_col': aep_col,
            'n_valores': len(aep),
            'media': aep.mean(),
            'std': aep.std(),
            'picos_altos': len(peaks_high),
            'picos_baixos': len(peaks_low),
            'periodo_dominante': 1 / positive_freqs[top_5_idx[0]] if positive_freqs[top_5_idx[0]] > 0 else None
        }
    
    except Exception as e:
        print(f"❌ Erro ao processar: {str(e)}")
        return None


def main():
    """Processa todos os CSV na pasta"""
    pasta = Path("/home/italo/Área de Trabalho/teste_eolica")
    csv_files = list(pasta.glob("*.csv"))
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado na pasta!")
        return
    
    print(f"Encontrados {len(csv_files)} arquivos CSV")
    
    resultados = []
    for csv_file in sorted(csv_files):
        resultado = analisar_ciclos_aep(csv_file)
        if resultado:
            resultados.append(resultado)
    
    # Resumo Final
    print(f"\n{'='*70}")
    print("RESUMO GERAL")
    print(f"{'='*70}")
    for res in resultados:
        print(f"\n{res['arquivo']}:")
        print(f"  Período dominante: {res['periodo_dominante']:.1f} pontos" if res['periodo_dominante'] else "  Período: Não detectado")
        print(f"  Picos altos/baixos: {res['picos_altos']}/{res['picos_baixos']}")


if __name__ == "__main__":
    main()