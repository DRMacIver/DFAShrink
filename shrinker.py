import heapq
import hashlib


BYTES = range(256)
ALPHABET = [bytes([i]) for i in BYTES]
assert len(ALPHABET) == 256


def sort_key(s):
    return (len(s), s)


def cache_key(s):
    if len(s) < 20:
        return s
    return hashlib.sha1(s).digest()


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
        try:
            table = self.__transitions[state]
        except KeyError:
            table = [None] * 256
            self.__transitions[state] = table
        if table[byte] is not None:
            return table[byte]
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
        key = cache_key(string)
        try:
            return self.__cache[key]
        except KeyError:
            pass
        result = bool(self.__criterion(string))
        if result and (
            self.__best is None or sort_key(string) < sort_key(self.__best)
        ):
            self.__best = string
            self.__shrink_callback(string)
        self.__cache[key] = result
        return result

    def __repr__(self):
        return "Shrinker(%d states, %d experiments)" % (
            len(self.__starts), len(self.__ends))

    def __route_to_best(self):
        while not self.__have_row(self.best):
            print(self)
            self.__route_step()
            # Do a little bit of work building states in the direction of
            # this target. The first one finds an experiment that increases
            # the number of states, the second one actually increases the
            # number of states.

    def __have_row(self, string):
        try:
            self.__index_for_row(self.__row(string))
            return True
        except KeyError:
            return False

    def __probe_for_start(self, lo, target):
        assert not self.__have_row(target)
        n = lo + 1
        while self.__have_row(target[:n]):
            n *= 2
            if n >= len(target):
                n = len(target)
        return n

    def __route_step(self):
        target = self.best
        n = len(target)
        lo = 0
        # Invariants: have_row(target[:lo]), not have_row(target[:hi])
        while lo < n:
            hi = self.__probe_for_start(0, target)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self.__have_row(target[:mid]):
                    lo = mid
                else:
                    hi = mid
            assert lo + 1 == hi
            byte = target[lo]
            curstate = self.__index_for_row(self.__row(target[:lo]))
            nextstate = self.transition(curstate, byte)
            equivstring = self.__starts[nextstate]
            presentrow = self.__row(equivstring)
            targetrow = self.__row(target[:hi])
            if presentrow == targetrow:
                # We have successfully routed to a previously unseen row.
                # This extends the reachable subset of the string. Start again
                # from here.
                if self.__have_row(target):
                    return
                lo = hi
                hi = self.__probe_for_start(lo, target)
            else:
                # There is a mismatch between the structure of our graph and
                # reality witnessed here. Adding a new experiment will fix it.
                # We must then start again from the beginning.
                targetstring = target[:hi]
                assert targetstring not in self.__strings_to_indices
                distinguishers = {
                    e for e in self.__ends
                    if self.criterion(targetstring + e) != self.criterion(
                        equivstring + e
                    )
                }
                assert distinguishers
                best_distinguisher = min(distinguishers, key=sort_key)
                new_experiment = ALPHABET[byte] + best_distinguisher
                self.__add_end(new_experiment)
                return

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
        prefix = self.__starts[history[lo]] + ALPHABET[string[lo]]
        altprefix = self.__starts[history[hi]]
        assert prefix != altprefix
        assert self.__row(prefix) == self.__row(altprefix)
        assert self.criterion(prefix + r) != self.criterion(altprefix + r)
        probe = len(r) - hi
        while probe > 0:
            attempt = hi + probe
            probe //= 2
            r2 = string[attempt:]
            if self.criterion(prefix + r2) != self.criterion(altprefix + r2):
                r = r2
        self.__add_end(r)
        return True

    def __dynamic_check(self, f):
        """Adjust the DFA until the current DFA agrees with criterion on f()"""
        start_count = -1
        end_count = -1
        changed = False
        while self.__check_step(f()):
            changed = True
            assert len(self.__ends) > end_count or (
                len(self.__starts) > start_count)
            end_count = len(self.__ends)
            start_count = len(self.__starts)
        return changed

    def __row(self, string):
        """A row for a string uses our current set of ends to produce a
        signature that distinguishes it from other strings."""
        key = cache_key(string)
        try:
            return self.__row_cache[key]
        except KeyError:
            result = tuple(self.criterion(string + e) for e in self.__ends)
            self.__row_cache[key] = result
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

    def states(self):
        return range(len(self.__starts))

    def __shrink_step(self):
        """Do some shrinking. Return True if more shrinking is required."""

        # Ensure there is a zero cost path to a best example.
        self.__route_to_best()

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
        if self.__best != initial:
            return True
        return self.__dynamic_check(lambda: self.best)


def shrink(initial, criterion, *, shrink_callback=None):
    """Attempt to find a minimal version of initial that satisfies criterion"""
    shrinker = Shrinker(criterion, shrink_callback=shrink_callback)
    if not shrinker.criterion(initial):
        raise ValueError("Initial example does not satisfy criterion")
    shrinker.shrink()
    return shrinker.best
