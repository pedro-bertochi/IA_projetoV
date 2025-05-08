import os
import matplotlib.pyplot as plt
import pandas as pd

# Caminho da pasta onde estão os logs de cada fold
log_dir = '../results'

def plot_metric(metric: str, ylabel: str, filename: str, folds: int = 5):
    plt.figure(figsize=(10, 6))
    
    for fold in range(1, folds + 1):
        csv_path = os.path.join(log_dir, f'historico_treinamento_fold{fold}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df[metric], label=f'Fold {fold}')
        else:
            print(f"[⚠️] Log do Fold {fold} não encontrado.")

    plt.title(f'{ylabel} por Época (Cross-validation)')
    plt.xlabel('Época')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(log_dir, filename)
    plt.savefig(output_path)
    print(f"[✅] Gráfico salvo em: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_metric('accuracy', 'Acurácia', 'grafico_accuracy_folds.png')
    plot_metric('loss', 'Loss', 'grafico_loss_folds.png')
