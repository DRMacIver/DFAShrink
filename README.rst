DFAShrink
=========

This is an experimental new approach to example shrinking which is based on
regular language theory.

It doesn't yet work *very* well, but I think it shows a lot of promise as an
approach.

The problem to solve is this: Given some predicate over binary strings, and at
least one example for which it returns true, find the minimal string for which
it is true. Minimal here means minimal length, and lexicographically minimal
amongst the minimal length strings.

The core idea of this shrinker is this:

1. Given a deterministic finite automaton representing a regular language, finding the smallest element of the language is just a shortest path problem (e.g. using Dijkstra's algorithm).
2. The set of strings that are simpler than a given string is a finite (albeit potentially very large) set, and thus is a regular language.

So we can use a process of regular language induction (attempting to fit a DFA
to an unknown regular language) to try to shrink the example.

The algorithm we use is based on Rivest and Schapire's "`Inference of Finite
Automata Using Homing Sequences <http://rob.schapire.net/papers/homing.pdf>`_",
in which they present an improvement to
Dana Angluin's L* search from "`Learning Regular Sets From Queries and
Counterexamples <http://www.cs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf>`_"
but with one important difference: We don't make any attempt to
build the whole DFA, but instead allow it to be constructed lazily, potentially
building a new state whenever we try a previously unseen transition. This
avoids doing a potentially large amount of work by only focusing on the areas
of the graph that we actually need to explore.

Using this DFA we can try various operations which if our DFA is correct will
shrink the current best example.

If this shrink succeeds, we've improved our example. If it doesn't, we have
evidence that our current DFA is correct. We improve the DFA and try again.
We stop once we believe that our current best example is also the minimal
example of the regular language matched by our DFA.

We use the DFA construction to attempt to improve our best example in two
ways:

The first is that we walked the DFA via the current best and look for any loops
introduced. We delete these loops. We also look to see if the path passes
through an accepting state before the end and truncate to there.

We then do a "cheapest first" variant of Dijkstra's algorithm where we
prioritize paths where the transition is cheaper to perform.

The way this works is that figuring out the transition to a state will require
us to perform a number of calls to the underlying criterion. These calls are
cached. The cost of the transition is thus the number of cache misses for the
string that we would be transitioning to.

Results
-------

The test suite has some partial results demonstrating that the concept at least
works.

It does not currently work on larger examples. I have attempted to recreate
the shrinking example in Regehr's `Reducers are fuzzers <http://blog.regehr.org/archives/1284>`_
(for no particularly good reason other than that it was an interesting large
example) and it never got anywhere meaningful. The reason for this is that
currently the check() operation which makes the DFA consistent with an example
is O(n^2) in the size of the example, and the current implementation makes use
of check() on the initial example. I believe this to be fixable but have not
yet got the details worked out.

As such this is currently more suitable for a final stage of reduction which
you run after e.g. a more conventional `delta debugging <https://www.st.cs.uni-saarland.de/papers/tse2002/tse2002.pdf>`_
based shrink.
