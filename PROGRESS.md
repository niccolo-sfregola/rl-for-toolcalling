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

## 2026-09-23

**Fatto:**
- Scritto `verify_tokenizer.py`: script per esplorare come Qwen2.5-0.5B-Instruct formatta tool e tool call tramite `apply_chat_template`, senza coinvolgere il modello vero e proprio.
- Risolto un problema di ambiente: lanciando lo script con `python3` invece di `python` (con il venv attivo) veniva usato un interprete diverso da quello del progetto, causando un errore nel rendering del chat template (`jinja2.exceptions.UndefinedError: dict object has no element 0`). Con `python` dentro il venv attivo l'errore sparisce.
- Costruito lo schema del tool `lookup` nel formato standard (`type: "function"`, `function: {name, description, parameters}`, con `parameters` come JSON Schema su `entity` e `relation`), passato a `apply_chat_template` come lista (`tools=[...]`).
- Simulata una tool call dell'assistant (`role: "assistant"`, `tool_calls: [{"type": "function", "function": {"name": ..., "arguments": {...}}}]`) e osservato che viene serializzata come testo puro dentro il tag `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, senza un canale di output strutturato separato.
- Simulata una tool response (`role: "tool"`) e osservato che il template la reincapsula in un turno `<|im_start|>user` con tag `<tool_response>...</tool_response>`, perché Qwen2.5 non ha un ruolo "tool" dedicato nel suo formato nativo.

**Prossimo passo:**
- Scrivere il ciclo agente: costruire il prompt iniziale con `tools`, generare con il modello, fare il parsing del `<tool_call>` prodotto (gestendo il caso di sintassi malformata), eseguire `lookup` sul grafo, appendere la tool response, ripetere fino a risposta finale o a un numero massimo di turni.

## 2026-09-24

**Fatto:**
- Creato `agent.py` con i pezzi del ciclo agente che non dipendono dal modello:
  - `parse_tool_call`: estrae la prima `<tool_call>` dal testo generato (regex non-greedy con `re.DOTALL`), fa `json.loads` gestendo `JSONDecodeError`, e valida la forma (dizionario, `name == "lookup"`, `arguments` dizionario con `entity` e `relation` stringhe). Ritorna `None` in ogni caso di errore. Testato su cinque casi: chiamata corretta, tag di chiusura mancante, JSON con virgolette singole, nessuna chiamata, due chiamate di fila.
  - `parse_answer`: estrae il contenuto di `<answer>...</answer>`, normalizzato con `.strip().lower()`, `None` se manca o è vuoto. Conseguenza: nel reward la gold answer va confrontata in minuscolo (`gold.lower()`).
  - `load_model()`: carica Qwen2.5-0.5B-Instruct in bfloat16 su GPU, dentro una funzione e non a livello di modulo, per non caricare il modello a ogni `import agent`.
  - `lookup_dict`: schema del tool con `description` riscritta e `"enum": relations` sul campo `relation`, importato da `dataset_generator` per avere una sola lista di relazioni.
- Tradotti in inglese codice, commenti e docstring di tutti i file Python (i `.md` restano in italiano). Relazioni rinominate: `ally_of`, `enemy_of`, `governed_by`, `borders`.
- Corretto un bug in `build_questions`: il template invertiva il verso di entrambe le relazioni (la domanda chiedeva il contrario di quello che i lookup trovavano). Ora `relation_phrases` contiene frasi nominali ("the ruler of", "the neighbor of", ...) e la domanda è `Who is {r2} {r1} {h0}?`, che si legge da destra a sinistra nello stesso ordine dei lookup.
- Scritto `try_model.py`: primo contatto con il modello vero. Genera una domanda, costruisce `messages` con un system prompt che chiede di usare il tool e di rispondere in `<answer>`, fa una sola `generate` in greedy e passa l'output ai due parser.
- Primo output osservato (domanda: "Who is the ruler of the neighbor of Fervia?"):
  - entità giusta ma relazione sbagliata: `"neighbor"` invece di `"borders"`, nonostante l'`enum` nello schema
  - due `<tool_call>` nello stesso turno, la seconda con un'entità inventata (`"Ruler"`), perché il modello non ha aspettato il risultato della prima
  - i parser si sono comportati correttamente (prima chiamata estratta, nessuna `<answer>` perché il turno non era finito)

**Prossimo passo:**
- Nel system prompt, aggiungere la corrispondenza esplicita tra frasi della domanda e nomi delle relazioni, generata da `relation_phrases`.
- Fermare la generazione alla prima `</tool_call>` con `stop_strings` (passando `tokenizer=tok` a `generate`).
- Poi scrivere il ciclo multi-turno completo con `MAX_TURNS`, esecuzione di `lookup`, messaggi `assistant`/`tool` e reward finale.
- Committare il lavoro di ieri e di oggi.
