"""
Ambiente sintetico per il progetto di tool-calling multi-hop con GRPO.

Genera un piccolo knowledge graph casuale (triple entità-relazione-entità),
campiona domande a due hop la cui risposta richiede due lookup consecutivi
nel grafo (l'entità intermedia non viene mai rivelata nella domanda), ed
espone il tool `lookup` che un agente userà per interrogare il grafo passo
per passo. Nessun modello linguistico è coinvolto in questo file: è la base
puramente Python su cui verrà costruito il ciclo agente e il reward
verificabile.
"""

import random as rnd


rnd.seed(42)

#definizione del vocabolario

entities = [
    "Aldoria", "Brevon", "Caldrex", "Dorelia", "Elandor",
    "Fervia", "Galtron", "Hespera", "Istrion", "Jandria",
    "Kelmor", "Lunara", "Mervon", "Norelia", "Orvex",
    "Paldora", "Quendria", "Ravion", "Selmora", "Tervex",
    "Uldria", "Valmor", "Westria", "Xandor", "Yelvia"
]

relations = [
    "alleato_di",
    "nemico_di",
    "governato_da",
    "confina_con"
]


relation_phrases = {
    "alleato_di" : "è alleato di",
    "nemico_di" : "è nemico di",
    "governato_da": "è governato da",
    "confina_con": "confina con"

}


def generate_random_graph(l_dict: int, entities,relations):

    tripla = []
    facts_dict = {}

    for i in range(l_dict):
        head = rnd.choice(entities)
        rel = rnd.choice(relations)

        while (head,rel) in facts_dict:
            head = rnd.choice(entities)
            rel = rnd.choice(relations)

        tail = rnd.choice(entities)

        while tail == head:
            tail = rnd.choice(entities)

        tripla.append((head, rel, tail))
        facts_dict[(head, rel)] = tail


    return tuple(tripla), facts_dict


#Domande a due hop: scegli a caso un'entità
#  di partenza h0 e due relazioni r1, r2 tali che
#  esistano sia (h0, r1) -> h1 che (h1, r2) -> h2.
#  La domanda è "hop-question" su h0, r1, r2, la gold answer è h2.


def sample_chain(relations,facts_dict):

    valid_r2 = []

    while not valid_r2:
        (h0, r1) = rnd.choice(tuple(facts_dict.keys()))
        h1 = facts_dict[(h0, r1)]

        valid_r2 = [r for r in relations if (h1, r) in facts_dict]
        if not valid_r2:
            continue
        
        r2 = rnd.choice(valid_r2)
        h2 = facts_dict[(h1, r2)]

    return (h0, r1, r2, h2)


#quando il modello chiama il tool di lookup
#durante il rollout, se sbaglia chiedendo un'entità
#o una relazine che non esiste, non deve mai
#  far crashare il rollout

# dato entità, relazione cerca in facts_dict e ritorna il valore

def lookup(entity, relation, facts_dict):

    if facts_dict.get((entity, relation)) is None:
        return None #deve essere dato in pasto al programma per non fermare l'agente in rollout

    return facts_dict[(entity, relation)]


def build_questions(h0, r1, r2, relation_phrases):
    return f"Chi {relation_phrases[r2]} quello Stato che {relation_phrases[r1]} {h0}?"


if __name__ == "__main__":
    #check del funzionamento senza alcun LLM.

    _,facts_dict = generate_random_graph(40,entities,relations)
    chain_tuple = sample_chain(relations,facts_dict)

    chain_list = list(chain_tuple)
    print(build_questions(chain_list[0],chain_list[1], chain_list[2], relation_phrases))
    lookup_1 = lookup(chain_list[0],chain_list[1], facts_dict)
    lookup_2 = lookup(lookup_1, chain_list[2], facts_dict)

    check_answer = lookup_2 == chain_list[3]
    print(int(check_answer))