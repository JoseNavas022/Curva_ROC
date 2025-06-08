import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, f1_score

# 1. Leer archivo y crear etiquetas
ruta_archivo = r"C:\Users\jnava\OneDrive\Escritorio\Curva_R.xlsx"
datos = pd.read_excel(ruta_archivo)
datos['Etiqueta_Real'] = datos['Clasificación'].apply(lambda x: 1 if str(x).lower().startswith("estandar") else 0)

# 2. Generar set balanceado 70% activos, 30% decoys
activos = datos[datos['Etiqueta_Real'] == 1]
decoys = datos[datos['Etiqueta_Real'] == 0]
num_activos = len(activos)
num_decoys = int(num_activos * 0.3 / 0.7)
decoys_sample = decoys.sample(n=num_decoys, random_state=42)
datos_70_30 = pd.concat([activos, decoys_sample])

# 3. Calcular curva ROC, AUC y Youden
y_true = datos_70_30['Etiqueta_Real']
y_scores = -1 * datos_70_30['Docking_Score']  # Invertir si es necesario

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
best_sens = tpr[best_idx]
best_spec = 1 - fpr[best_idx]

# 4. Clasificar con el umbral óptimo
datos_70_30['Predicción'] = (y_scores >= best_threshold).astype(int)

# 5. Métricas adicionales
precision = precision_score(y_true, datos_70_30['Predicción'])
f1 = f1_score(y_true, datos_70_30['Predicción'])

# 6. Enrichment Factor (EF) en top 1%, 5%, 10%, usando score invertido
def enrichment_factor(df, score_col, label_col, top_frac):
    df_sorted = df.copy()
    df_sorted['ScoreInvertido'] = -1 * df_sorted[score_col]
    df_sorted = df_sorted.sort_values('ScoreInvertido', ascending=False)
    n_total = len(df_sorted)
    n_top = int(np.ceil(n_total * top_frac))
    n_activos_total = df_sorted[label_col].sum()
    n_activos_top = df_sorted.iloc[:n_top][label_col].sum()
    ef = (n_activos_top / n_top) / (n_activos_total / n_total) if n_activos_total > 0 else 0
    return ef, n_activos_top, n_top

efs = []
for frac, nombre in zip([0.01, 0.05, 0.10], ['1%', '5%', '10%']):
    ef, n_top_activos, n_top = enrichment_factor(datos_70_30, 'Docking_Score', 'Etiqueta_Real', frac)
    efs.append((nombre, ef, n_top_activos, n_top))

# 7. Imprimir resultados
print(f"Activos: {len(activos)} | Decoys usados: {len(decoys_sample)} | Total: {len(datos_70_30)}")
print(f"AUC ROC: {roc_auc:.3f}")
print(f"Índice de Youden máximo: {youden_index[best_idx]:.3f}")
print(f"Umbral óptimo de score: {best_threshold:.3f}")
print(f"Sensibilidad (SE): {best_sens*100:.2f}%")
print(f"Especificidad (SP): {best_spec*100:.2f}%")
print(f"Precisión: {precision:.3f}")
print(f"F1-score: {f1:.3f}")
for nombre, ef, n_act, n_t in efs:
    print(f"EF top {nombre}: {ef:.2f} (activos encontrados: {n_act} de {n_t})")

# 8. Graficar la curva ROC y el índice de Youden
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[best_idx], tpr[best_idx], color='black', zorder=5, label='Youden index')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Clasificación Docking')
plt.legend(loc="lower right")
plt.grid()
plt.show()
