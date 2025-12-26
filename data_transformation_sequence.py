# from collections import defaultdict
#
#
# class DataTransformationSequence:
#     def __init__(self):
#         self.sequence = defaultdict(list)
#         self.length = 0
#     def append(self, function_name, col_name):
#         self.sequence[col_name].append(function_name)
#         self.length += 1
#     def get_seq_length(self):
#         return self.length
#
import itertools


class SearchContext:
    """
    Holds static data (Functions, Columns, Pre-computed Signatures).
    Created ONCE to avoid re-processing strings during the search.
    """

    def __init__(self, functions, columns):
        # 1. Store raw lists
        self.functions = functions
        self.columns = columns

        # 2. Pre-compute all possible moves and their "Constraint Signature"
        # We flatten the loops here so the search just iterates a single list.
        # Structure: list of tuples -> (func_name, col_name, signature_tuple)
        self.all_moves = []

        for func in functions:
            # Extract "relevant" name (e.g., 'bin_1' -> 'bin')
            # Doing this here saves millions of .split() calls later.
            relevant_name = func.split('_')[0]

            for col in columns:
                # The signature blocks future uses of this relevant_name on this col
                signature = (relevant_name, col)
                self.all_moves.append((func, col, signature))


class SequenceState:
    """
    Represents a specific path in the search tree.
    Immutable-ish: expand() returns NEW instances, never modifies self.
    """
    __slots__ = ['path', 'used_signatures', 'context']

    def __init__(self, context, path=None, used_signatures=None):
        self.context = context
        self.path = path if path is not None else []
        # specific set of (relevant_name, col) that are "burnt" for this path
        self.used_signatures = used_signatures if used_signatures is not None else set()

    def expand(self):
        """
        Iterator that yields ONLY valid next SequenceState objects.
        Fast O(1) filtering using pre-computed signatures.
        """
        # Iterate through the pre-computed list from Context
        for func, col, signature in self.context.all_moves:

            # FAST CHECK: If this (relevant_name, col) combo exists, skip.
            if signature in self.used_signatures:
                continue

            # --- PREPARE NEW STATE ---

            # 1. Copy the set (Fast C-level copy) and add new signature
            new_signatures = self.used_signatures.copy()
            new_signatures.add(signature)

            # 2. Extend path (New list creation)
            new_step = (func, col)
            new_path = self.path + [new_step]

            # 3. Yield new independent object
            yield SequenceState(self.context, new_path, new_signatures)

    def __repr__(self):
        return f"Seq(len={len(self.path)}, last={self.path[-1] if self.path else 'Start'})"