import heapq
import hashlib
from random import Random


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

    def __init__(self, initial, criterion, *, shrink_callback=None):
        """Initialise with criterion. shrink_callback will be called with an
        example every time one that is strictly better than the previous best
        is found."""
        self.__criterion = criterion
        self.__starts = [b'']
        self.__experiments = [None]
        self.__live = [True]
        self.__cache = {}
        self.__transitions = {}
        self.__strings_to_indices = {}
        self.__shrink_callback = shrink_callback or (lambda s: None)
        self.__random = Random()
        self.__changes = 0
        self.__best = None
        if not self.criterion(initial):
            raise ValueError("Initial example does not satisfy criterion")
        assert self.__best == initial

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
        if result:
            for i, s in enumerate(self.__starts):
                if not s:
                    continue
                if string.startswith(s):
                    suffix = string[len(s):]
                    if self.__live[i]:
                        if sort_key(suffix) < sort_key(self.__experiments[i]):
                            self.__experiments[i] = suffix
                    else:
                        self.__live[i] = True
                        self.__changes += 1
                        self.__experiments[i] = suffix
            if (
                self.__best is None or sort_key(string) < sort_key(self.__best)
            ):
                self.__best = string
                self.__shrink_callback(string)
                self.__changes += 1
        self.__cache[key] = result
        return result

    @property
    def best(self):
        """The best example positive we've seen so far."""
        return self.__best

    @property
    def start(self):
        """The initial state of the current state machine"""
        return 0

    def states(self):
        return range(len(self.__starts))

    def accepting(self, state):
        """Is this state an accepting one. i.e. do all strings lead to here
        satisfy criterion if we're right about the shape of the language."""
        return self.criterion(self.__starts[state])

    def forbid(self, source, byte, target):
        complete, forbidden = self.__transition_value(source, byte)
        if complete:
            if target == forbidden:
                raise ValueError("Cannot forbid canonical transition" % (
                    source, byte, target
                ))
            else:
                return
        if target not in forbidden:
            self.__changes += 1
            forbidden.add(target)

    def transitions(self, state, byte):
        """Returns the state reached by reading byte while in this state."""
        complete, forbidden_or_state = self.__transition_value(state, byte)
        if complete:
            assert isinstance(forbidden_or_state, int)
            return [forbidden_or_state]
        assert isinstance(forbidden_or_state, set)
        non_forbidden = [
            s for s in self.states() if s not in forbidden_or_state
        ]
        if non_forbidden:
            return non_forbidden
        s = self.__starts[state] + ALPHABET[byte]
        self.__starts.append(s)
        i = len(self.__starts) - 1
        self.__strings_to_indices[s] = i
        self.__transitions[state][byte] = (True, i)
        # Dummy values during criterion calculation
        self.__live.append(False)
        self.__experiments.append(b'')
        # run criterion for the side effects
        for e in (b'', self.__experiment(state)[1:]):
            self.__experiments[i] = e
            if self.criterion(s + e):
                self.__live[i] = True
                break
        assert len(self.__experiments) == len(self.__starts)
        assert len(self.__live) == len(self.__starts)
        return [i]

    def shrink(self):
        changes = -1
        while self.__changes > changes:
            changes = self.__changes
            best_routes = {}
            queue = []
            # Heap format: Presumed dead, Uncertainty, length of path, path,
            # state history
            heapq.heappush(queue, (
                (0, b'', (0,))
            ))
            while queue:
                _, path, history = heapq.heappop(queue)
                assert len(path) + 1 == len(history)
                state = history[-1]
                if not self.__live[state]:
                    continue
                # We already have a better route to this path (this can happen
                # because of prioritizing more certain routes. It can't happen
                # in normal Dijkstra's algorithm). That makes this path not
                # interesting.
                if (
                    state in best_routes and
                    sort_key(best_routes[state][0]) <= sort_key(path)
                ):
                    continue
                # We're only interested in routes that improve on our current
                # best, so if the path is already >= our current best it can't
                # do that and we return early.
                if sort_key(path) >= sort_key(self.best):
                    continue
                # We now need to check if this was a valid transition. We don't
                # do this earlier in the hopes that the above checks can save
                # us the work
                if len(history) > 1:
                    prevstate = history[-2]
                    if not self.__double_check(
                        prevstate, path[-1], state
                    ):
                        continue
                best_routes[state] = (path, history)
                for b, a in enumerate(ALPHABET):
                    newpath = path + a
                    transitions = self.transitions(state, b)
                    for t in transitions:
                        heapq.heappush(queue, (
                            len(newpath), newpath,
                            history + (t,)
                        ))
            # We now have a set of putative best paths to each state that we
            # know about. It's time to find out if they actually work.
            for state, (path, history) in best_routes.items():
                assert len(history) == len(path) + 1
                if not path:
                    continue
                last_state = history[-1]
                altpath = self.__starts[last_state]
                if path == altpath:
                    continue
                experiment = self.__experiment(last_state)
                if (
                    self.criterion(path + experiment) !=
                    self.criterion(altpath + experiment)
                ):
                    # This path is wrong. Lets diagnose why and forbid a
                    # transition that lead to it.
                    lo = 0
                    hi = len(history) - 1
                    while lo + 1 < hi:
                        mid = (lo + hi) // 2
                        prefix = path[:mid - 1]
                        state = history[mid]
                        altprefix = self.__starts[state]
                        experiment = self.__experiment(state)
                        if (
                            self.criterion(prefix + experiment) ==
                            self.criterion(altprefix + experiment)
                        ):
                            lo = mid
                        else:
                            hi = mid
                    # The transition we made when we were in state history[lo]
                    # was bad.
                    self.forbid(history[lo], path[lo], history[hi])

            # We've now got a pretty good guess about how to navigate around
            # the states that we known about, but we don't know much about how
            # to escape them. Try to expand our horizons.
            # We do this by looking for states that we know lead to an
            # accepting state but where our current graph does not evidence
            # that.
            any_changes = False
            for s in reversed(self.states()):
                if not self.__live[s]:
                    continue
                if self.accepting(s):
                    continue
                ex = self.__experiment(s)
                c = ex[0]
                suffix = ex[1:]
                for t in self.transitions(s, c):
                    if not self.criterion(self.__starts[t] + suffix):
                        self.forbid(s, c, t)
                        any_changes = True
                if not any_changes:
                    for c in range(256):
                        self.__double_check(s, c, self.transitions(s, c)[0])
                if any_changes:
                    break
        assert any(self.accepting(s) for s in self.states())

    def __experiment(self, state):
        if state == 0:
            return self.best
        return self.__experiments[state]

    def __double_check(self, source, byte, target):
        confirmed, value = self.__transition_value(source, byte)
        if confirmed:
            return target == value
        else:
            if target in value:
                return False
            s = self.__starts[source]
            t = self.__starts[target]
            extended = s + ALPHABET[byte]
            for e in (b'', self.__experiment(target)):
                if self.criterion(extended + e) != self.criterion(
                    t + e
                ):
                    self.forbid(source, byte, target)
                    return False
            return True

    def __transition_value(self, source, byte):
        return self.__transitions.setdefault(
            source, {}).setdefault(byte, (False, set()))

    def __is_complete(self, source, byte):
        return self.__transition_value(source, byte)[0]

    def __repr__(self):
        return "Shrinker(%d states (%d live), %d bytes in best)" % (
            len(self.__starts), sum(self.__live), len(self.best))


def shrink(initial, criterion, *, shrink_callback=None):
    """Attempt to find a minimal version of initial that satisfies criterion"""
    shrinker = Shrinker(initial, criterion, shrink_callback=shrink_callback)
    shrinker.shrink()
    return shrinker.best
