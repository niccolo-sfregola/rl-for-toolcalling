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

## 2026-09-25

### 12. Dove mettere le informazioni: system prompt, schema del tool, messaggi d'errore

Nella sezione 11 si diceva che una mappa esplicita tra parole della domanda e nomi delle relazioni nel system prompt "non è barare". Oggi la scelta è stata rivista: il system prompt è stato reso **agnostico** rispetto alle relazioni. Il motivo è che associare "the ruler of" a `governed_by` è proprio una delle abilità che il modello deve imparare. Se la scriviamo nel prompt, il modello la legge invece di impararla, e il prompt resta legato a queste quattro relazioni.

Una regola pratica per decidere dove va un'informazione:
- **system prompt**: *come* si lavora nell'ambiente (usare il tool, una chiamata alla volta, l'ordine di risoluzione di una domanda annidata, il formato della risposta, "nessuno risponderà alle tue domande"). Regole generali, che valgono per qualsiasi relazione.
- **schema del tool**: cosa accetta il tool, come nella documentazione di un'API reale (`enum` sui valori ammessi, eventualmente una `description` per ogni valore). È il posto legittimo dove spiegare le relazioni, se il modello da solo non ce la fa.
- **messaggi d'errore dell'ambiente**: cosa è andato storto, in modo che il modello possa correggersi. Anche qui c'è un limite: il messaggio non deve rivelare il grafo né offrire scorciatoie. Per questo l'errore sulla relazione elenca le relazioni valide (sono il vocabolario del tool), mentre quello sull'entità **non** elenca le entità: la giusta è già nella domanda, l'elenco non scalerebbe su un grafo grande, e soprattutto darebbe al modello la lista di tutte le risposte possibili, da cui tirare a indovinare.

Il limite di tutto questo è la starvation (sezione 11): un prompt più "puro" rende il task più difficile, e se il modello base non ci riesce mai GRPO non ha segnale. La scelta si verifica con i numeri, non a priori.

Un dettaglio pratico: conviene che tutti gli errori dell'ambiente abbiano un formato uniforme (qui il prefisso `Error:`). Semplifica sia il codice che li riconosce (`content.startswith("Error")`) sia quello che il modello deve imparare a riconoscere.

### 13. Il modello non ha memoria: la storia dei messaggi cresce sempre

Tra una chiamata a `generate` e la successiva il modello non ricorda niente. Ogni volta vede solo il prompt che gli viene passato, e `apply_chat_template(messages)` trasforma l'intera lista in un unico testo. Per questo `messages` può solo crescere:

```
turno 0:  [system, user]
turno 1:  [system, user, assistant(call 1), tool(risultato 1)]
turno 2:  [system, user, assistant(call 1), tool(risultato 1), assistant(call 2), tool(risultato 2)]
```

Se si tolgono i messaggi iniziali, al secondo turno il modello riceve un'entità senza sapere a quale domanda si riferisce. La stessa lista è anche la **traiettoria** che GRPO userà per il training: il reward valuta l'episodio intero.

### 14. Seed e riproducibilità

- `random.seed(s)` fissa il generatore del modulo `random` di Python, che è **unico per tutto il processo**: vale anche per i moduli importati (qui `dataset_generator`). Conviene fissarlo nello script che lancia l'esperimento, vicino al punto in cui serve, e non dentro un modulo di libreria al momento dell'import. L'alternativa più robusta è un generatore dedicato (`rng = random.Random(s)`) passato come parametro.
- Il campionamento dei token passa invece da **PyTorch**, che ha un generatore suo: `random.seed` fissa il grafo e la domanda, ma non la traiettoria con `do_sample=True`. Per riprodurre anche quella serve `torch.manual_seed(s)`.
- L'**eval set** deve essere identico prima e dopo il training, altrimenti non si può distinguere un modello migliorato da domande più facili. E il suo seed deve essere diverso da quelli di training, perché il modello non si alleni sulle domande su cui viene valutato.

### 15. Greedy, sampling e temperatura

A ogni passo il modello produce una **distribuzione di probabilità** su tutto il vocabolario (circa 150.000 token per Qwen). Per esempio, dopo `"relation": "` potrebbe dare `"neighbor"` 0.55, `"borders"` 0.30, `"ally_of"` 0.10. Da questa distribuzione va estratto un token:

- **greedy** (`do_sample=False`): sempre il più probabile, qui `"neighbor"`. È deterministico, utile per il debug e per misurare il comportamento "preferito" del modello. Il 30% su `"borders"`, che sarebbe la risposta giusta, non esce mai.
- **sampling** (`do_sample=True`): si estrae secondo le probabilità, quindi `"borders"` esce circa il 30% delle volte. Ogni esecuzione dà una traiettoria diversa.
- **temperature** `T`: prima dell'estrazione i logit vengono divisi per `T`. Con `T < 1` la distribuzione si appuntisce (più vicina al greedy), con `T > 1` si appiattisce (più varia, ma più caotica), con `T -> 0` coincide con il greedy.

In training si usa sempre il sampling: è l'**esplorazione** dell'RL. Senza variabilità fra le traiettorie non c'è niente da confrontare.

### 16. pass@G e "groups with signal"

**pass@G**: per ogni domanda si fanno G episodi con sampling; la domanda è risolta se **almeno uno** ha reward 1. pass@G è la frazione di domande risolte. Con 30 domande e G = 8, se in 3 domande almeno un episodio su 8 è corretto, pass@8 = 0.10. Può essere maggiore di zero anche quando l'accuracy greedy è 0.

Se un singolo episodio ha probabilità `p` di successo e gli episodi sono indipendenti:

```
pass@G = 1 - (1 - p)^G        p = 0.01, G = 8  ->  circa 0.08
```

Il legame con GRPO: il vantaggio di ogni traiettoria è `(r_i - media del gruppo) / std del gruppo`. Un gruppo con reward tutte 0, oppure tutte 1, ha vantaggi tutti nulli e non produce gradiente. La metrica che conta davvero è quindi la frazione di domande con **alcuni** episodi corretti ma **non tutti** (`0 < k < G`), chiamata qui "groups with signal". Con un modello base quasi sempre sbagliato coincide con pass@G. Dopo il training, quando alcune domande avranno gruppi tutti corretti, le due metriche divergono.

In sintesi: il greedy misura cosa fa il modello adesso, il pass@G con sampling misura cosa **può** imparare. **GRPO non crea il segnale da zero**: amplifica successi che il modello già produce ogni tanto. Se pass@G è 0, far partire il training non serve, bisogna prima rendere il task raggiungibile.

### 17. Errori tipici di un modello piccolo in un task agentico multi-hop

Dagli episodi di oggi (Qwen2.5-0.5B-Instruct, senza training):

- **Copiare invece di mappare**: scrive nel JSON la parola della domanda (`"neighbor"`) invece del nome della relazione (`"borders"`), o la mette nel campo sbagliato (`"entity": "neighbor"`). Quando la parola della domanda è contenuta nel nome della relazione ("ally" in `ally_of`) la chiamata tende a essere giusta: la difficoltà del task dipende dalla relazione, e una reward media può salire solo perché il modello migliora sulle relazioni facili.
- **Fermarsi al primo hop**: riceve l'entità intermedia e la dà come risposta ("The neighbor of the ally of Lunara is Quendria"). La frase suona plausibile, ma è sbagliata. È l'errore classico del multi-hop, ed è proprio l'abilità di ragionamento che il training deve insegnare.
- **Arrendersi all'utente dopo un errore del tool**: "Could you please provide more details?". Il motivo è nel formato di Qwen2.5 (sezione 5): la risposta del tool viene messa in un turno `user`, quindi per il modello è l'utente che gli segnala l'errore, e un modello Instruct risponde all'utente in modo cortese invece di correggere la chiamata. Una riga generica nel system prompt ("se il tool restituisce un errore, correggi gli argomenti e richiamalo; nessuno risponderà") lo ha fatto riprovare, ma cambiando anche l'argomento che era già giusto: tira a indovinare più che ragionare sull'errore.
- **Errori di formato e di copia**: risposta senza `<answer>`, oppure nome copiato male ("Caledrex" per "Caldrex"). Il nome inventato viene spezzato dal tokenizer in pezzi poco comuni, e il modello li ricompone male. Con un confronto esatto, anche la catena giusta prende reward 0.

Conviene tenere separati gli errori di **formato** (facili da imparare, eventualmente con una piccola reward dedicata) da quelli di **ragionamento** (il vero obiettivo, che deve restare nella reward principale). Una reward di formato va però usata con cautela: il modello può imparare a raccoglierla senza risolvere il task (reward hacking).

### 18. Come dare segnale a GRPO quando la baseline è a zero

Opzioni, dalla più "pura" alla più efficace:
1. **Curriculum**: partire da domande a un hop, dove il modello ha già un tasso di successo non nullo, e passare ai due hop quando la reward sale.
2. **Reward shaping**: reward parziali (formato, primo lookup corretto). Con il rischio di reward hacking visto sopra.
3. **SFT di warm-up (cold start)**: poiché la catena gold è nota, si possono generare in automatico traiettorie perfette e fare un breve fine-tuning supervisionato prima dell'RL. È l'approccio usato anche in DeepSeek-R1.
4. **Modello più grande**: più capacità di partenza, ma con 6 GB di VRAM il training richiederebbe tecniche come LoRA.

**Prossimo argomento in coda:** leggere pass@8 e scegliere fra queste opzioni; poi la struttura della loss di GRPO e il mascheramento dei token non generati dal modello (le risposte dei tool).
