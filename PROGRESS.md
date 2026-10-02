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

## 2026-09-25

**Fatto:**
- `try_model.py`: aggiunto `stop_strings=["</tool_call>"]` (con `tokenizer=tok`) a `generate`. Il modello ora emette una sola tool call per turno, e la seconda chiamata inventata è sparita.
- System prompt reso **agnostico rispetto alle relazioni**: niente mappa "the ruler of" -> `governed_by`, solo regole generali (usare il tool una chiamata alla volta, risolvere la domanda dall'interno verso l'esterno con un esempio astratto `A`/`B`/`X`, rispondere in `<answer>`, e in caso di errore del tool correggere gli argomenti e richiamarlo invece di chiedere aiuto all'utente). Il prompt è stato spostato in `agent.py` (`SYSTEM_PROMPT`).
- `lookup` in `dataset_generator.py` ora solleva eccezioni con messaggi informativi, tutti con il prefisso `Error:`:
  - entità inesistente (`ValueError`), senza elencare le entità valide;
  - relazione non valida (`ValueError`), con l'elenco delle relazioni valide;
  - entità e relazione valide ma nessun fatto (`LookupError`).
- Tolto `rnd.seed(42)` da `dataset_generator.py`: il seed ora lo fissa lo script che usa il modulo.
- `agent.py`:
  - `execute_tool_call(call, facts_dict) -> str`: chiama `lookup` dentro un `try/except (ValueError, LookupError)` e restituisce sempre una stringa (l'entità trovata o il messaggio d'errore), così il rollout non crasha mai.
  - `run_episode(tok, model, facts_dict, question, max_turns, DO_SAMPLE)`: ciclo multi-turno completo. `messages` cresce a ogni turno (assistant con `tool_calls` + tool con il risultato); un turno senza tool call valida (anche malformata) è la risposta finale. Restituisce `(answer, turns_used, messages)`.
- `try_model.py` riscritto per giocare un episodio intero con `run_episode` e stampare traiettoria, risposta, gold e reward.
- Primi episodi osservati a mano (seed 42, 123, 1234, 12). Errori tutti diversi fra loro:
  - nome della relazione copiato dalla domanda (`"neighbor"` invece di `"borders"`);
  - relazione esterna invece di quella interna;
  - parola della domanda messa nel campo `entity` (`"entity": "neighbor"`);
  - entità in minuscolo (`"ravion"`);
  - stop dopo il primo hop, con l'entità intermedia data come risposta finale;
  - dopo un errore del tool, resa con richiesta di chiarimenti all'utente (migliorata, ma non risolta, dalla riga nel prompt);
  - nome copiato male ("Caledrex" invece di "Caldrex") e risposta senza `<answer>`.
- Scritto `evaluate.py`:
  - `build_eval_set(n, seed)`: `n` domande, ognuna con il suo grafo, con seed fisso (`EVAL_SEED = 999`) perché l'eval set sia identico prima e dopo il training. Verificata la riproducibilità con `==` su due chiamate.
  - `episode_stats(messages, answer, gold)`: metriche di un episodio (`has_answer`, `correct`, `n_calls`, `n_ok_lookups`, `two_hops`, `retried`). Testata su una traiettoria scritta a mano.
  - loop greedy e loop con sampling (G = 8 episodi per domanda, `torch.manual_seed` per riprodurre il campionamento), con `pass@8` e "groups with signal".
- **Baseline greedy su 30 domande:**

  | metrica | valore |
  |---|---|
  | has_answer | 0.13 |
  | correct | 0.00 |
  | n_calls | 1.10 |
  | n_ok_lookups | 0.37 |
  | two_hops | 0.00 |
  | retried | 0.03 |

  Il modello quasi sempre fa una sola chiamata e si ferma; circa due chiamate su tre falliscono; non completa mai i due hop.

**Prossimo passo:**
- Lanciare il loop con sampling e leggere `pass@8`. Stima a priori: tra 0 e 0.15.
- In base al risultato scegliere come dare segnale a GRPO: curriculum a un hop, SFT di warm-up su traiettorie perfette generate dal grafo, reward di formato, o una combinazione.
- Aggiungere a `episode_stats` il conteggio per tipo di errore (entità, relazione, fatto inesistente) e la divisione dei risultati per relazione, per verificare l'ipotesi che `ally_of`/`enemy_of` siano più facili.
