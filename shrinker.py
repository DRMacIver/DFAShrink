import heapq


BYTES = range(256)
ALPHABET = [bytes([i]) for i in BYTES]
assert len(ALPHABET) == 256


def sort_key(s):
    return (len(s), s)


class Shrinker(object):
    """A Shrinker maintains a DFA that corresponds to its idea of the language
    of things that match the criterion with which it is provided.

    The goal of this is to produce a string which is minimal length and
    lexicographically minimal amongst strings of minimal length."""

    def __init__(self, criterion, *, shrink_callback=None):
        """Initialise with criterion. shrink_callback will be called with an
        example every time one that is strictly better than the previous best
        is found."""
        self.__best = None
        self.__criterion = criterion
        self.__starts = [b'']
        self.__ends = []
        self.__cache = {}
        self.__strings_to_indices = {}
        self.__shrink_callback = shrink_callback or (lambda s: None)
        self.__add_end(b'')
        self.__rows_to_indices = {self.__row(b''): 0}
        self.__row_cache = {}

    @property
    def best(self):
        """The best example positive we've seen so far. Raises ValueError if
        no examples found."""
        if self.__best is None:
            raise ValueError("No positive examples seen yet")
        return self.__best

    @property
    def start(self):
        """The initial state of the current state machine"""
        return 0

    def accepting(self, state):
        """Is this state an accepting one. i.e. do all strings lead to here
        satisfy criterion if we're right about the shape of the language."""
        return self.criterion(self.__starts[state])

    def transition(self, state, byte):
        """Returns the state reached by reading byte while in this state."""
        table = self.__transitions.setdefault(state, {})
        try:
            return table[byte]
        except KeyError:
            pass
        nextstring = self.__starts[state] + bytes([byte])
        try:
            nextstate = self.__strings_to_indices[nextstring]
        except KeyError:
            row = self.__row(nextstring)
            try:
                nextstate = self.__index_for_row(row)
            except KeyError:
                nextstate = len(self.__starts)
                self.__starts.append(nextstring)
                self.__strings_to_indices[nextstring] = nextstate
                self.__rows_to_indices[row] = nextstate
            assert self.__row(self.__starts[nextstate]) == row
        table[byte] = nextstate
        return nextstate

    def shrink(self):
        """Improve self.best to the point where it is the best example that
        satisfies the current DFA. There may be a better example that satisfies
        self.criterion, but finding it requires more evidence about the
        structure of the language."""
        if self.criterion(b''):
            return
        while self.__shrink_step():
            pass

    def check(self, string):
        """Ensure that the current DFA agrees with self.criterion on string.
        Returns True if this results in any change to the structure of the
        graph."""
        return self.__dynamic_check(lambda: string)

    def criterion(self, string):
        """A 'smart' version of the criterion passed in. It is cached and will
        automatically update the current best example if a better one is found.
        """
        try:
            return self.__cache[string]
        except KeyError:
            pass
        result = bool(self.__criterion(string))
        if result and (
            self.__best is None or sort_key(string) < sort_key(self.__best)
        ):
            self.__best = string
            self.__shrink_callback(string)
        self.__cache[string] = result
        return result

    def __repr__(self):
        return "Shrinker(__starts=%r, __ends=%r)" % (
            self.__starts, self.__ends)

    def __check_step(self, string):
        """Do some work towards making the DFA consistent on string. If there
        is more work to do, return True. Else, return False."""

        if not string:
            return False

        history = [0]

        def f(i):
            while i >= len(history):
                history.append(
                    self.transition(history[-1], string[len(history) - 1])
                )
            return self.__starts[history[i]] + string[i:]
        n = len(string)
        lo = 0
        hi = 1
        loval = self.criterion(f(0))
        while self.criterion(f(hi)) == loval:
            lo = hi
            hi *= 2
            if hi >= n:
                if self.criterion(f(n)) == loval:
                    return False
                hi = n
                break
        # Invariant: self.criterion(f(lo)) == self.criterion(f(0))
        # Invariant: self.criterion(f(hi)) == self.criterion(f(n))
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.criterion(f(mid)) == loval:
                lo = mid
            else:
                hi = mid
        r = string[hi:]
        assert r not in self.__ends
        self.__add_end(r)
        return True

    def __dynamic_check(self, f):
        """Adjust the DFA until the current DFA agrees with criterion on f()"""
        start_count = -1
        end_count = -1
        changed = False
        while self.__check_step(f()):
            changed = True
            assert len(self.__ends) > end_count
            end_count = len(self.__ends)
            assert len(self.__starts) > start_count
            start_count = len(self.__starts)
        return changed

    def __row(self, string):
        """A row for a string uses our current set of ends to produce a
        signature that distinguishes it from other strings."""
        try:
            return self.__row_cache[string]
        except KeyError:
            result = tuple(self.criterion(string + e) for e in self.__ends)
            self.__row_cache[string] = result
            return result

    def __add_end(self, e):
        """Add a new end to the list of experiments. This changes the structure
        of the graph."""
        assert e not in self.__ends
        self.__ends.append(e)
        self.__transitions = {}
        self.__row_cache = {}

    def __index_for_row(self, row):
        assert len(row) == len(self.__ends)
        i = self.__index_for_row_no_check(row)
        assert self.__row(self.__starts[i]) == row
        return i

    def __index_for_row_no_check(self, row):
        """Find the index that corresponds to this row, or raise KeyError if we
        do not currently have a state for this row."""
        try:
            return self.__rows_to_indices[row]
        except KeyError:
            pass
        # We might have added ends since the last time we looked for this row.
        # Rather than doing an expensive rebuild every time we add an end we
        # instead "repair" the table by looking for prefixes of this row and
        # adding them back in to the table. All states will have had a presence
        # in the table before, and we only add extensions, so if there is a
        # state that corresponds to this row then it must be present in the
        # table as a prefix.
        for i in range(len(row) - 1, 0, -1):
            trunc = row[:i]
            if trunc in self.__rows_to_indices:
                s = self.__rows_to_indices.pop(trunc)
                self.__rows_to_indices[self.__row(self.__starts[s])] = s
                try:
                    # We must look the row up again. s might not be it! If we
                    # don't find it, continue on to other prefixes.
                    return self.__rows_to_indices[row]
                except KeyError:
                    pass
        raise KeyError()

    def __transition_costs(self, state):
        """Return a cost value for each transition out of state. The cost is
        the number of uncached criterion calls that would be required to figure
        out that transition."""
        costs = []
        base = self.__starts[state]
        for a in ALPHABET:
            s = base + a
            if s in self.__strings_to_indices:
                costs.append(0)
            else:
                string = self.__starts[state]
                c = 0
                for e in self.__ends:
                    if string + e not in self.__cache:
                        c += 1
                costs.append(c)
        return costs

    def __shrink_step(self):
        """Do some shrinking. Return True if more shrinking is required."""

        # Ensure there is a zero cost path to a best example.
        self.__dynamic_check(lambda: self.best)

        initial = self.__best

        # Cost, length of route, route, last state
        queue = [(0, 0, b'', 0)]
        best_route = {}
        while queue:
            _, _, route, state = heapq.heappop(queue)
            if sort_key(route) >= sort_key(self.__best):
                continue
            if route:
                state = self.transition(state, route[-1])
            if (
                state in best_route and
                sort_key(route) >= sort_key(best_route[state])
            ):
                continue
            best_route[state] = route
            if self.accepting(state):
                # We've found a short route, but it doesn't actually work when
                # we try it against our criterion. The shape of the graph now
                # changes and we start again from scratch
                if self.check(route):
                    return True
            costs = self.__transition_costs(state)
            for b in BYTES:
                cost = costs[b]
                newroute = route + ALPHABET[b]
                heapq.heappush(queue, (
                    cost, len(newroute), newroute, state
                ))
        if self.check(self.__best):
            return True
        if not any(self.__best):
            return False
        return self.__best != initial


def shrink(initial, criterion, *, shrink_callback=None):
    """Attempt to find a minimal version of initial that satisfies criterion"""
    shrinker = Shrinker(criterion, shrink_callback=shrink_callback)
    if not shrinker.criterion(initial):
        raise ValueError("Initial example does not satisfy criterion")
    shrinker.shrink()
    return shrinker.best
