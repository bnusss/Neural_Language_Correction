#!/bin/python
# -*- coding: utf-8 -*-


# initialization
g = 0
hash_table = {start}; # close set
BEAM = {start};       # open set

# main loop
while len(BEAM) != 0:
    SET = {}; # input set for beam search

    # generate the SET nodes
    for each state in BEAM:
        for each successor of state:
            if successor == goal:
                return g + 1
            SET = SET & {successor}

    # start beam search
    BEAM = {}
    g = g + 1

    # fill the BEAM for the next loop
    # set is not empty and the number of nodes in BEAM is less than B
    while len(SET) != 0 and B > len(BEAM):
        state = successor in SET with smallest h value
        SET = SET - {state}
        if state not in hash_table:
            if hash_table is full:
                return inf
            hash_table = hash_table & {state}
            BEAM = BEAM & {state}

