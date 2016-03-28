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

TOMBSTONE = object()


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
            try:
                nextstate = self.__find_equivalent_state(nextstring)
            except KeyError:
                nextstate = len(self.__starts)
                self.__starts.append(nextstring)
                self.__strings_to_indices[nextstring] = nextstate
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

    def __have_row(self, string):
        try:
            self.__find_equivalent_state(string)
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
        assert self.__are_strings_equivalent(prefix, altprefix)
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

    def __are_strings_equivalent(self, s1, s2):
        if s1 == s2:
            return True
        for e in self.__ends:
            if self.criterion(s1 + e) != self.criterion(s2 + e):
                return False
        return True

    def __is_equivalent_to_state(self, state, string):
        return self.__are_strings_equivalent(self.__starts[state], string)

    def __find_equivalent_state(self, string):
        try:
            return self.__strings_to_indices[string]
        except KeyError:
            pass
        result = None
        key = cache_key(string)
        try:
            result = self.__state_cache[key]
        except KeyError:
            pass
        if result is None:
            result = TOMBSTONE
            assert isinstance(string, bytes)
            states = list(self.states())
            for s in states:
                if self.__is_equivalent_to_state(s, string):
                    result = s
                    break
        self.__state_cache[key] = result
        if result is TOMBSTONE:
            raise KeyError()
        return result

    def __add_end(self, e):
        """Add a new end to the list of experiments. This changes the structure
        of the graph."""
        assert e not in self.__ends
        self.__ends.append(e)
        self.__transitions = {}
        self.__state_cache = {}

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
