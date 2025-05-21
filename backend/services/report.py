from fpdf import FPDF

def gerar_pdf(resultados, caminho_pdf):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Relatório de Classificação de Soldas", ln=True, align='C')
    pdf.ln(10)

    for r in resultados:
        texto = f"{r['imagem1']} + {r['imagem2']} -> {r['classificacao']} (Confiança: {r['confiança']:.2f})"
        pdf.cell(0, 10, txt=texto, ln=True)

    pdf.output(caminho_pdf)
