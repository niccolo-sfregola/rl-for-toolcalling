"""
Synthetic environment for the multi-hop tool-calling project with GRPO.

Generates a small random knowledge graph (entity-relation-entity triples),
samples two-hop questions whose answer requires two consecutive lookups
in the graph (the intermediate entity is never revealed in the question),
and exposes the `lookup` tool that an agent will use to query the graph step
by step. No language model is involved in this file: it is the pure Python
base on which the agent loop and the verifiable reward will be built.
"""

import random as rnd


rnd.seed(42)

#vocabulary definition

entities = [
    "Aldoria", "Brevon", "Caldrex", "Dorelia", "Elandor",
    "Fervia", "Galtron", "Hespera", "Istrion", "Jandria",
    "Kelmor", "Lunara", "Mervon", "Norelia", "Orvex",
    "Paldora", "Quendria", "Ravion", "Selmora", "Tervex",
    "Uldria", "Valmor", "Westria", "Xandor", "Yelvia"
]

relations = [
    "ally_of",
    "enemy_of",
    "governed_by",
    "borders"
]


#noun phrases for the tail of (head, relation, tail), given the head:
#  (Aldoria, governed_by, Brevon) -> Brevon is "the ruler of Aldoria"
relation_phrases = {
    "ally_of" : "the ally of",
    "enemy_of" : "the enemy of",
    "governed_by": "the ruler of",
    "borders": "the neighbor of"

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


#Two-hop questions: pick a random
#  starting entity h0 and two relations r1, r2 such that
#  both (h0, r1) -> h1 and (h1, r2) -> h2 exist.
#  The question is a "hop-question" on h0, r1, r2, the gold answer is h2.


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


#when the model calls the lookup tool
#during the rollout, if it makes a mistake by asking for an entity
#or a relation that does not exist, it must never
#  crash the rollout

# given entity, relation, look it up in facts_dict and return the value

def lookup(entity, relation, facts_dict):

    if facts_dict.get((entity, relation)) is None:
        return None #must be fed back to the program so the agent does not stop during rollout

    return facts_dict[(entity, relation)]


def build_questions(h0, r1, r2, relation_phrases):
    #read right to left, in the same order as the lookups:
    #  "Who is the ally of the ruler of Aldoria?" -> (Aldoria, governed_by), then (h1, ally_of)
    return f"Who is {relation_phrases[r2]} {relation_phrases[r1]} {h0}?"


if __name__ == "__main__":
    #check that it works without any LLM.

    _,facts_dict = generate_random_graph(40,entities,relations)
    chain_tuple = sample_chain(relations,facts_dict)

    chain_list = list(chain_tuple)
    print(build_questions(chain_list[0],chain_list[1], chain_list[2], relation_phrases))
    lookup_1 = lookup(chain_list[0],chain_list[1], facts_dict)
    lookup_2 = lookup(lookup_1, chain_list[2], facts_dict)

    check_answer = lookup_2 == chain_list[3]
    print(int(check_answer))
