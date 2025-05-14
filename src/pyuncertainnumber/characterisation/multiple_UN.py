from dataclasses import dataclass, field
from PyUncertainNumber.UC import UncertainNumber
from PyUncertainNumber.UC.utils import UNEncoder
from typing import List
import json

""" not-in-use"""

""" 
Create and dump multiple UN objects into JSON data

This is to simulate the behaviour of API from a code where we need to create a bunch of UN object (deck)"""


def make_many_intervals():
    lo = [1, 10, 20, 30, 40]
    hi = [55, 65, 75, 85, 95]
    itvl_lists = [[r, s] for r in lo for s in hi]
    interval_UN = [
        UncertainNumber(
            name="elas_modulus",
            symbol="E",
            units="pa",
            essence="interval",
            interval_initialisation=x,
        )
        for x in itvl_lists
    ]
    print(f"Created {len(interval_UN)} Interval-type UN objects")

    """ back-up code to create (top-level) dict-type of JSON data """
    # json_mulUN_data = {}
    # for un in interval_UN:
    #     k = id(un)
    #     json_mulUN_data[k] = un

    json_mulUN_data = []
    for un in interval_UN:
        k = id(un)
        json_mulUN_data.append({k: un})

    return json_mulUN_data


@dataclass
class Deck:
    """multiple UN objects in a deck

    note:
        - the deck is a list of UN objects, where each UN object is termed as a card;
    """

    cards: List[UncertainNumber] = field(default_factory=make_many_intervals)

    def __repr__(self):
        cards = ", ".join(f"{c!r}" for c in self.cards)
        return f"{self.__class__.__name__}({cards})"

    def JSON_dump(self, filename="./results/mulUN_data.json"):
        with open(filename, "w") as fp:
            json.dump(self.cards, fp, cls=UNEncoder, indent=4)
