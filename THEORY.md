# Diario di teoria

Argomenti di teoria toccati durante il progetto, spiegati in dettaglio. Una sezione per giornata di lavoro, in ordine cronologico.

## 2026-09-22

### 1. Knowledge graph come struttura dati

Un knowledge graph è un insieme di fatti espressi come triple `(head, relation, tail)`: un grafo diretto ed etichettato, dove i nodi sono entità e gli archi sono relazioni tipizzate. `(Roma, capitale_di, Italia)` è un arco che va da "Roma" a "Italia", etichettato "capitale_di".

Per rappresentarlo in codice ci sono due strutture complementari:
- una lista di triple, utile per iterare su tutti i fatti o serializzare il grafo
- un dizionario `{(head, relation): tail}`, utile per il lookup: dato un nodo di partenza e una relazione, trovare il nodo di arrivo in tempo costante O(1), invece di scandire linearmente tutte le triple O(n)

Vincolo importante per questo progetto: ogni coppia `(head, relation)` deve mappare a un solo `tail`. Senza questo vincolo, "qual è la relazione X di Y" non avrebbe una risposta univoca, e un reward verificabile (confronto esatto con una gold answer) non avrebbe senso.

### 2. Ragionamento multi-hop

Una domanda "a un hop" richiede un solo lookup nel grafo: `(h0, r) -> risposta`. Una domanda "a due hop" richiede comporre due lookup in sequenza: `(h0, r1) -> h1`, poi `(h1, r2) -> h2`, dove `h2` è la risposta finale.

Il punto cruciale è che `h1` (l'entità intermedia) non deve mai comparire esplicitamente nella domanda in linguaggio naturale. Se la domanda rivelasse `h1` direttamente, il problema degenererebbe in due domande a un hop indipendenti, e non testerebbe più la capacità di incatenare informazioni recuperate passo per passo, che è esattamente il comportamento agentic che il progetto vuole insegnare al modello.

### 3. Dati sintetici e reward verificabile (RLVR)

In letteratura RL recente si parla spesso di RLVR, Reinforcement Learning from Verifiable Rewards: invece di usare un reward model appreso (come nel classico RLHF), si usa una funzione di verifica deterministica che confronta l'output del modello con una risposta nota per essere corretta, tipicamente 1 se corretto, 0 altrimenti.

Un knowledge graph generato sinteticamente è il caso ideale per questo approccio: la gold answer è nota per costruzione (l'abbiamo generata noi), quindi il reward è verificabile al 100%, senza ambiguità. Questo è il motivo per cui si parte da un ambiente giocattolo invece che da dati reali: dati reali (retrieval da Wikipedia, corpora esterni) introducono rumore e ambiguità nella verifica della risposta, un problema ortogonale a quello che si vuole imparare in questa fase, cioè il meccanismo di tool calling multi-hop e la dinamica di GRPO.

Un dataset sintetico concettualmente simile, citato come riferimento, è CLUTRR: grafi di relazioni familiari generati proceduralmente con domande multi-hop compositive.

### 4. Rejection sampling nei generatori

Diverse funzioni del progetto usano lo stesso pattern: generare un candidato a caso e, se viola un vincolo, ripescarne un altro finché non se ne trova uno valido (un `while` che ricampiona). Questo è un caso semplice di rejection sampling.

Il pattern ha un rischio strutturale da tenere sempre a mente: se i vincoli sono troppo stringenti rispetto allo spazio dei candidati (ad esempio un grafo troppo piccolo o troppo sparso rispetto al numero di entità), la probabilità di trovare un candidato valido crolla, e nel caso limite il loop non termina mai. Nel progetto questo è emerso in due punti: la ricerca di una coppia `(head, relation)` non ancora occupata durante la generazione del grafo, e la ricerca di una catena a due hop che non finisca in un vicolo cieco (un `h1` senza relazioni uscenti) durante il campionamento delle domande.

**Prossimo argomento in coda:** chat template e formato dei messaggi per un modello Instruct (Qwen2.5-0.5B-Instruct), introdotto solo a livello concettuale oggi, da approfondire con il modello in mano nella prossima sessione.

## 2026-09-23

### 5. Formato dei tool e dei tool call in un modello Instruct

Un chat template (Jinja2, dentro `tokenizer_config.json`) trasforma una lista di messaggi Python in un'unica stringa di testo delimitata da token speciali di turno. Per un modello Instruct non è un dettaglio cosmetico: il modello riconosce i confini tra system, user e assistant solo se il testo in input rispetta esattamente lo stesso schema di delimitazione visto in training.

Quando si passano dei tool tramite l'argomento `tools` di `apply_chat_template`, va sempre passata una lista, anche con un solo tool: l'API è pensata per il caso generale di più strumenti disponibili contemporaneamente (lo stesso principio di `messages`, sempre una lista anche con un solo messaggio). Il template inietta poi nel messaggio di sistema la definizione di ciascun tool, in uno schema a due livelli: `type: "function"` e dentro `function: {name, description, parameters}`. `parameters` è a sua volta un JSON Schema standard (`type: "object"`, `properties`, `required`) che descrive i *tipi* degli argomenti accettati.

Punto centrale da non confondere: questo schema descrive il tool in astratto, ed è cosa diversa da una tool call concreta. Una tool call (dentro `message["tool_calls"]`, generata dal modello durante un turno assistant) condivide la stessa forma a due livelli (`type: "function"`, `function: {...}`), ma con chiavi diverse dentro `function`: al posto di `description`/`parameters` (i tipi accettati) c'è `arguments`, un dizionario con i valori concreti passati in quella chiamata specifica.

Un'altra cosa emersa ispezionando l'output: la tool call generata dal modello è **testo puro**, dentro tag XML `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, non un canale di output strutturato separato dal resto della generazione. Questo significa che il parsing dell'output del modello (isolare il contenuto tra i tag, fare `json.loads`, gestire il caso in cui il modello sbagli la sintassi o dimentichi un tag di chiusura) è responsabilità di chi consuma l'output, non è automatico.

Infine, un dettaglio specifico di Qwen2.5: non esiste un ruolo "tool" dedicato nel formato nativo del modello. Un messaggio Python con `role: "tool"` (la risposta della funzione eseguita) viene reincapsulato dal template in un turno `<|im_start|>user`, con tag `<tool_response>...</tool_response>`, perché il modello è stato addestrato a riconoscere solo i ruoli `system`, `user`, `assistant`.

**Prossimo argomento in coda:** come strutturare il ciclo di rollout multi-turno (quando fermare la generazione, come limitare un eventuale loop infinito di tool call, cosa succede al reward se il modello non produce mai una risposta finale entro il budget di turni).

## 2026-09-24

### 6. Struttura di un rollout multi-turno

Un rollout è un episodio completo: domanda, una o più tool call, le risposte dei tool, la risposta finale. Si implementa come un ciclo: a ogni turno si ricostruisce tutto il prompt da `messages` con `apply_chat_template(..., add_generation_prompt=True)` (che aggiunge in fondo `<|im_start|>assistant\n`, per dire al modello che tocca a lui), si genera, si cerca una tool call nel testo prodotto. Se c'è, si esegue il tool e si aggiungono a `messages` il turno `assistant` e il turno `tool`, poi si ricomincia. Se non c'è, il turno è la risposta finale e il ciclo finisce.

Tre scelte di design che fanno parte dell'ambiente, non del modello:
- **Budget di turni (`MAX_TURNS`)**: senza un limite il modello potrebbe chiamare tool all'infinito. Per due hop servono tre turni (due lookup e la risposta), quindi un budget di 4 o 5 lascia un po' di margine. Se il budget si esaurisce, il reward è 0: una penalità implicita che l'RL impara a evitare.
- **Tool che fallisce**: al modello va restituito un messaggio informativo (es. "nessun fatto trovato per (X, r)"), non `None`, così ha una possibilità di correggersi al turno successivo.
- **Formato della risposta finale**: in RLVR la verifica è un confronto esatto, quindi la risposta va chiesta in un formato fisso (`<answer>...</answer>`) ed estratta con un parser. La prosa libera renderebbe il reward fragile.

Una nota che servirà in GRPO: nel rollout ci sono token generati dal modello (tool call, risposta) e token che il modello non ha generato (le risposte dei tool). La loss va calcolata solo sui primi. Inoltre il testo che il template ricostruisce da un messaggio `assistant` strutturato può non coincidere carattere per carattere con quello effettivamente generato (spazi, ordine delle chiavi JSON): per addestrare servono i token esatti prodotti dal modello.

### 7. Parsing robusto dell'output di un modello

Visto che la tool call è testo puro, il parser deve sopravvivere a qualsiasi cosa il modello scriva. Principio guida: un errore del modello non deve mai far crashare il programma, deve diventare un valore (qui `None`) che il ciclo sa gestire.

Regex, i dettagli che contano:
- `re.search` cerca il pattern in qualunque punto del testo. Gli ancoraggi `^` e `$` lo obbligherebbero a coincidere con inizio e fine della stringa, e basterebbe uno spazio o una frase prima della chiamata per farlo fallire.
- Di default `.` non attraversa `\n`. Con `re.DOTALL` sì, ed è necessario perché Qwen mette il JSON su una riga separata dai tag.
- `.*` è **greedy**: prende più testo possibile, quindi con due tool call di fila andrebbe dal primo tag di apertura all'ultimo di chiusura. `.*?` è **non-greedy** e si ferma alla prima chiusura.
- Le parentesi creano un **gruppo di cattura**: `match.group(1)` restituisce solo il contenuto tra i tag, `group(0)` tutto il match.
- I pattern si scrivono come raw string (`r"..."`) perché Python non interpreti i backslash prima di `re`.

Eccezioni, due errori istruttivi emersi oggi:
- `re.search` non solleva eccezioni quando non trova niente, ritorna `None`. Un `try/except` attorno non scatta mai. Il `try/except` serve invece attorno a `json.loads`, che solleva `JSONDecodeError`.
- Dopo `except` va la **classe** dell'eccezione, non un'istanza: `except json.JSONDecodeError:`, non `except json.JSONDecodeError("messaggio"):`. La parte insidiosa è che Python valuta l'espressione dopo `except` solo quando un'eccezione arriva davvero lì: con JSON valido il codice sembra funzionare, e il bug esplode (con un `TypeError`) proprio nel caso che l'`except` doveva gestire. Da qui la regola: testare sempre anche i casi di errore, non solo quello felice. Anche l'`except:` nudo va evitato, perché cattura tutto, compresi errori di battitura e `Ctrl+C`.

Dopo `json.loads` serve comunque validare la forma: può restituire una lista, una stringa, un numero, o un dizionario senza le chiavi attese. `.get("chiave")` al posto di `["chiave"]` evita il `KeyError` quando la chiave manca, e `isinstance(..., str)` garantisce che gli argomenti siano davvero utilizzabili.

### 8. Generare testo con `transformers`

- **`AutoTokenizer` / `AutoModelForCausalLM`**: leggono il `config.json` del modello e scelgono la classe giusta. "Causal LM" è un modello che predice il token successivo guardando solo quelli precedenti.
- **`dtype=torch.bfloat16`**: pesi a 16 bit invece di 32, metà memoria. 0.5B parametri occupano circa 1 GB, e il margine sui 6 GB della GPU servirà in training per gradienti e stato dell'ottimizzatore. (In `transformers` 5 il vecchio nome `torch_dtype` è deprecato.)
- **`model.eval()`** disattiva i comportamenti da training, come il dropout. **`torch.no_grad()`** evita di registrare le operazioni per il calcolo dei gradienti: in generazione non servono, e si risparmiano memoria e tempo.
- **`generate` si ferma** quando il modello produce il token di fine sequenza (`<|im_end|>` per Qwen2.5-Instruct), oppure al raggiungimento di `max_new_tokens`, che limita la lunghezza di un singolo turno. `MAX_TURNS` invece limita il numero di turni.
- **Greedy vs sampling**: con `do_sample=False` si sceglie sempre il token più probabile, e lo stesso prompt dà sempre lo stesso output, comodo per il debug. Con il sampling si pesca dalla distribuzione, e la `temperature` la appiattisce (>1) o la appuntisce (<1). GRPO richiede il sampling: genera più rollout per la stessa domanda e ne confronta i reward, e rollout identici non darebbero niente da confrontare.
- **L'output contiene anche il prompt**: `out` è prompt + token nuovi, quindi va tagliato con `out[0, n_token_prompt:]` prima del `decode`. `skip_special_tokens=True` toglie `<|im_end|>`, ma non i tag `<tool_call>`, che per Qwen2.5 sono testo normale.

### 9. Il verso delle relazioni nelle domande in linguaggio naturale

Una tripla `(head, relation, tail)` ha un verso: `(Aldoria, governed_by, Brevon)` dice che Brevon governa Aldoria, non il contrario. Il template di domanda deve rispettare quel verso, altrimenti la domanda chiede una cosa e i lookup ne trovano un'altra. Il vecchio template ("Who is governed by the state that is governed by Aldoria?") invertiva entrambe le relazioni. Per `ally_of`, `enemy_of` e `borders` nel mondo reale non cambierebbe nulla perché sono simmetriche, ma nel grafo sono memorizzate in una sola direzione, e `lookup` risponde solo in quella.

La soluzione adottata: descrivere ogni relazione con una frase nominale che identifica la coda a partire dalla testa ("the ruler of X" = la coda di `(X, governed_by)`), e comporle: "Who is the ally of the ruler of Aldoria?". La domanda si legge da destra a sinistra nello stesso ordine dei lookup. L'articolo determinativo "the" è legittimo perché il grafo garantisce una sola coda per ogni coppia `(head, relation)`.

### 10. Chiamate parallele e perché nel multi-hop non servono

Il template di Qwen2.5 permette più `<tool_call>` nello stesso turno (chiamate parallele). Hanno senso quando le chiamate sono indipendenti, per esempio il meteo di tre città. Nel multi-hop non lo sono mai: l'argomento della seconda chiamata è il risultato della prima. Nel primo test il modello ha fatto proprio questo, e non conoscendo ancora l'entità intermedia se l'è inventata (`"entity": "Ruler"`). La soluzione più diretta è interrompere la generazione alla prima `</tool_call>` (`stop_strings`), così il modello è costretto ad aspettare il risultato.

### 11. Perché serve un tasso di successo iniziale non nullo (GRPO)

GRPO campiona un gruppo di rollout per la stessa domanda e calcola per ciascuno un **vantaggio** relativo al gruppo: reward meno la media del gruppo (normalizzato per la deviazione standard). I rollout sopra la media vengono rinforzati, quelli sotto scoraggiati. Se tutti i rollout di un gruppo prendono 0, la media è 0 e ogni vantaggio è 0: **il gradiente è nullo** e il modello non impara niente da quella domanda. Lo stesso vale se prendono tutti 1.

Ne segue che l'RL rinforza comportamenti che il modello a volte produce già, non ne crea di nuovi dal nulla. Rendere il compito fattibile per il modello di partenza (un system prompt chiaro, la mappa esplicita tra parole della domanda e nomi delle relazioni, lo stop dopo una tool call) non è barare: è quello che rende possibile il segnale di apprendimento. Il primo test l'ha mostrato concretamente: senza aiuti il modello sbaglia il nome della relazione (`"neighbor"` invece di `"borders"`) anche con l'`enum` nello schema.

**Prossimo argomento in coda:** il ciclo multi-turno completo e la funzione di reward; poi, prima di GRPO, misurare il tasso di successo del modello base su un centinaio di domande.
