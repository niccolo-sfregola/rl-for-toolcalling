# Diario di progresso

Log del progetto, una sezione per giornata di lavoro.

## 2026-09-22

**Fatto:**
- Definito l'ambiente sintetico in `dataset_generator.py`:
  - `generate_random_graph`: genera un knowledge graph casuale (triple entità-relazione-entità), con vincolo di unicità su `(head, relation)` e niente self-loop.
  - `sample_chain`: campiona una catena valida a due hop `(h0, r1, r2, gold_answer)` a partire dal grafo, senza mai esporre l'entità intermedia.
  - `lookup`: tool di lookup `(entity, relation) -> value`, ritorna `None` su chiave mancante invece di sollevare eccezione (per non far crashare il rollout dell'agente più avanti).
  - `build_questions`: template che trasforma `(h0, r1, r2)` in una domanda in linguaggio naturale, mantenendo l'entità intermedia nascosta dietro un placeholder generico ("lo Stato che...").
- Sanity check end-to-end (in `if __name__ == "__main__"`): genera grafo → campiona catena → due lookup in sequenza → confronto con gold answer → reward 0/1. Verificato funzionante.
- Setup ambiente Python: venv locale in `.venv/`, installati `torch` (con supporto CUDA, GPU RTX 3060 6GB rilevata) e `transformers`. `requirements.txt` congelato, `.gitignore` aggiunto.

**Prossimo passo:**
- Scaricare Qwen2.5-0.5B-Instruct e ispezionare il suo chat template (Jinja2) per capire come vengono formattati messaggi e tool call nel formato nativo del modello, prima di scrivere il ciclo agente.
