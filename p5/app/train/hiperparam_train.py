import json
from pathlib import Path
from .train import train_model

# Listas de hiperparámetros a probar (OPTIMIZADOS SEGÚN ANÁLISIS)
# Total de experimentos: 3 × 3 × 1 × 2 × 1 × 1 × 1 × 2 × 2 = 72 ✅

# PARÁMETROS CRÍTICOS (exploración activa):
vocab_sizes = [128, 256, 350]          # Expandir hacia valores menores (256 fue mejor)
dim_embeddings = [40, 64, 128]         # Aumentar capacidad (64 fue mejor que 32)
dim_attentions = [80, 128, 256]                 # Mejor encontrado, mínimo impacto

# PARÁMETROS CONFIRMADOS (fijar):
num_layers_list = [2]                  # Mejor encontrado (2 > 1)
num_heads_list = [2]                   # Mejor encontrado, mínimo impacto
epochs_list = [5]                      # Más iteraciones para convergencia
batch_sizes = [16]                     # Mantener constante

# 📊 PARÁMETROS CON IMPACTO MÍNIMO (máximo 2 valores):
seq_lens = [64, 128]                   # Mantener los 2 originales

# 🚀 NUEVOS PARÁMETROS A EXPLORAR (antes eran solo 1):
learning_rates = [1e-4, 5e-4]          # Explorar rango de tasas

results = []

# Fors anidados
for vocab_size in vocab_sizes:
    for seq_len in seq_lens:
        for dim_embedding in dim_embeddings:
            for dim_attention in dim_attentions:
                for num_heads in num_heads_list:
                    for num_layers in num_layers_list:
                        for batch_size in batch_sizes:
                            for epochs in epochs_list:
                                for learning_rate in learning_rates:
                                    
                                    config = {
                                        "vocab_size": vocab_size,
                                        "seq_len": seq_len,
                                        "dim_embedding": dim_embedding,
                                        "dim_attention": dim_attention,
                                        "num_heads": num_heads,
                                        "num_layers": num_layers,
                                        "batch_size": batch_size,
                                        "epochs": epochs,
                                        "learning_rate": learning_rate,
                                    }
                                    
                                    print(f"\n🔥 Probando: vocab={vocab_size}, seq={seq_len}, emb={dim_embedding}, attn={dim_attention}, heads={num_heads}, layers={num_layers}, batch={batch_size}, epochs={epochs}, lr={learning_rate}")
                                    
                                    try:
                                        model, tokenizer = train_model(
                                            vocab_size=vocab_size,
                                            seq_len=seq_len,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            learning_rate=learning_rate,
                                            dim_embedding=dim_embedding,
                                            dim_attention=dim_attention,
                                            num_heads=num_heads,
                                            num_layers=num_layers,
                                            device="cuda",
                                        )
                                        
                                        # Leer resultados del archivo guardado
                                        artifacts_path = Path(__file__).resolve().parents[2] / "artifacts"
                                        latest_exp = max(artifacts_path.glob("[0-9]*-[0-9]*"), key=lambda p: p.stat().st_mtime)
                                        results_file = latest_exp / "results.txt"
                                        
                                        if results_file.exists():
                                            epochs_data = json.loads(results_file.read_text(encoding="utf-8"))
                                            best_epoch = min(epochs_data, key=lambda x: x["val_loss"])
                                            
                                            result = {
                                                "config": config,
                                                "best_val_loss": best_epoch["val_loss"],
                                                "best_perplexity": best_epoch["perplexity"],
                                                "best_epoch": best_epoch["epoch"],
                                                "status": "completed"
                                            }
                                            results.append(result)
                                            print(f"✅ Val Loss: {best_epoch['val_loss']:.4f}, PPL: {best_epoch['perplexity']:.2f}")
                                        else:
                                            result = {
                                                "config": config,
                                                "status": "failed",
                                                "error": "No results file found"
                                            }
                                            results.append(result)
                                            print(f"❌ Error: No results file")
                                    
                                    except Exception as e:
                                        result = {
                                            "config": config,
                                            "status": "failed",
                                            "error": str(e)
                                        }
                                        results.append(result)
                                        print(f"❌ Error: {e}")

# Guardar resultados
output_path = Path(__file__).resolve().parents[2] / "artifacts" / "hiperparam_results.json"
output_path.parent.mkdir(parents=True, exist_ok=True)
if output_path.exists():
    existing_results = json.loads(output_path.read_text(encoding="utf-8"))
    existing_results.extend(results)
    output_path.write_text(json.dumps(existing_results, ensure_ascii=False, indent=2), encoding="utf-8")
else:
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 Resultados guardados en: {output_path}")
