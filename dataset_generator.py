"""
Synthetic environment for the multi-hop tool-calling project with GRPO.

Generates a small random knowledge graph (entity-relation-entity triples),
samples n-hop questions whose answer requires n consecutive lookups
in the graph (the intermediate entities are never revealed in the question),
and exposes the `lookup` tool that an agent will use to query the graph step
by step. No language model is involved in this file: it is the pure Python
base on which the agent loop and the verifiable reward will be built.
"""

import random as rnd


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


#n-hop questions: pick a random fact (h0, r1) -> h1, then keep walking
#  from the last entity with a random relation that has a fact, until n_hops lookups.
#  If the walk hits a dead end, restart from a new fact.
#  The question is built on h0 and the relations, the gold answer is the last entity.


def sample_chain(relations, facts_dict, n_hops):
    """Returns (path, rels): path = [h0, h1, ..., h_n], rels = [r1, ..., r_n],
    with facts_dict[(path[i], rels[i])] == path[i + 1]. The gold answer is path[-1]."""

    while True:
        (h0, r1) = rnd.choice(tuple(facts_dict.keys()))
        path = [h0, facts_dict[(h0, r1)]]
        rels = [r1]

        while len(rels) < n_hops:
            valid_r = [r for r in relations if (path[-1], r) in facts_dict]
            if not valid_r:
                break   #dead end: restart from a new fact

            r = rnd.choice(valid_r)
            rels.append(r)
            path.append(facts_dict[(path[-1], r)])

        if len(rels) == n_hops:
            return path, rels


#when the model calls the lookup tool
#during the rollout, if it makes a mistake by asking for an entity
#or a relation that does not exist, it must never
#  crash the rollout

# given entity, relation, look it up in facts_dict and return the value

def lookup(entity, relation, facts_dict):

    if facts_dict.get((entity, relation)) is None:
        if entity not in entities:
            raise ValueError(f"Error: unknown entity '{entity}'. Read the question more carefully.")
        if relation not in relations:
            raise ValueError(f"Error: unknown relation '{relation}'. Valid relations: {', '.join(relations)}. Read the question more carefully.")
        
        raise LookupError(f"Error: no entity found for ({entity},{relation})")
    return facts_dict[(entity, relation)]


def build_questions(h0, rels, relation_phrases):
    #rels is in lookup order, the question reads right to left:
    #  rels = [governed_by, ally_of] -> "Who is the ally of the ruler of Aldoria?"
    phrases = " ".join(relation_phrases[r] for r in reversed(rels))
    return f"Who is {phrases} {h0}?"


if __name__ == "__main__":
    #check that it works without any LLM, for one and two hops.

    _,facts_dict = generate_random_graph(40,entities,relations)

    for n_hops in (1, 2):
        path, rels = sample_chain(relations, facts_dict, n_hops)
        print(build_questions(path[0], rels, relation_phrases))

        entity = path[0]
        for rel in rels:
            entity = lookup(entity, rel, facts_dict)

        check_answer = entity == path[-1]
        print(int(check_answer))
