#!/usr/bin/env python3
"""
Script para aumentar o tamanho das legendas nas imagens de hypervolume
Usa upscaling inteligente apenas da região da legenda
"""
from PIL import Image, ImageDraw, ImageFont
import sys
import os

def enlarge_legend_region(image_path, output_path, scale_factor=1.5):
    """
    Aumenta a região da legenda (canto superior direito) usando upscaling
    """
    # Abrir imagem
    img = Image.open(image_path)
    width, height = img.size
    
    # A legenda geralmente está no canto superior direito
    # Vamos definir a região aproximada (ajuste conforme necessário)
    # Assumindo legenda no topo direito: últimos 30% da largura, primeiros 25% da altura
    legend_x = int(width * 0.65)  # Começa em 65% da largura
    legend_y = 0
    legend_width = int(width * 0.35)  # 35% da largura
    legend_height = int(height * 0.25)  # 25% da altura
    
    # Criar backup
    backup_path = image_path.replace('.png', '_backup.png')
    img.save(backup_path)
    print(f"✓ Backup criado: {backup_path}")
    
    # Extrair região da legenda
    legend_region = img.crop((legend_x, legend_y, legend_x + legend_width, legend_y + legend_height))
    
    # Fazer upscaling da região da legenda
    new_legend_width = int(legend_width * scale_factor)
    new_legend_height = int(legend_height * scale_factor)
    legend_upscaled = legend_region.resize((new_legend_width, new_legend_height), Image.LANCZOS)
    
    # Criar nova imagem com fundo branco
    new_img = img.copy()
    
    # Colar a legenda aumentada de volta (centralizada na região original)
    paste_x = legend_x - int((new_legend_width - legend_width) / 2)
    paste_y = legend_y - int((new_legend_height - legend_height) / 2)
    
    # Garantir que não saia dos limites
    paste_x = max(0, min(paste_x, width - new_legend_width))
    paste_y = max(0, min(paste_y, height - new_legend_height))
    
    # Criar máscara para transparência suave nas bordas
    mask = Image.new('L', (new_legend_width, new_legend_height), 255)
    draw_mask = ImageDraw.Draw(mask)
    
    # Aplicar fade nas bordas (opcional, para suavizar)
    # Por enquanto, vamos colar diretamente
    
    new_img.paste(legend_upscaled, (paste_x, paste_y), legend_upscaled if legend_upscaled.mode == 'RGBA' else None)
    
    # Salvar
    new_img.save(output_path, 'PNG', dpi=(300, 300))
    print(f"✓ Imagem processada: {output_path}")
    print(f"  Região da legenda aumentada em {int((scale_factor - 1) * 100)}%")
    
    return True

if __name__ == '__main__':
    scales = [16, 36, 64]
    
    for scale in scales:
        input_file = f'hv_{scale}.png'
        output_file = f'hv_{scale}_legend_enlarged.png'
        
        if os.path.exists(input_file):
            print(f"\nProcessando {input_file}...")
            try:
                enlarge_legend_region(input_file, output_file, scale_factor=1.4)
            except Exception as e:
                print(f"✗ Erro ao processar {input_file}: {e}")
        else:
            print(f"⚠ Arquivo não encontrado: {input_file}")
    
    print("\n=== Concluído ===")
    print("Arquivos gerados com sufixo '_legend_enlarged.png'")
    print("Revise as imagens e substitua as originais se estiverem boas.")
