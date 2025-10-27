"""
huffman.py: Huffman coding implementation for integer symbol sequences.
"""

from collections import Counter
import heapq

class HuffmanNode:
    """
    Node in the Huffman tree.  Contains a symbol (or None for internal nodes), frequency, 
    and left/right children.  Implements comparison by frequency for priority queue.
    """
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        # Needed for heapq (min-heap) to sort by frequency
        return self.freq < other.freq

def build_huffman_tree(frequencies):
    """
    Build a Huffman tree given a dict of symbol frequencies.
    Returns the root node of the Huffman tree.
    """
    heap = [HuffmanNode(sym, freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)
    # Combine nodes until one tree remains
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq, node1, node2)
        heapq.heappush(heap, merged)
    return heap[0] if heap else None

def generate_codes(node, prefix="", code_map=None):
    """
    Traverse the Huffman tree to build a symbol-to-code map.
    `prefix` accumulates the bit string; left edges add '0', right edges '1'.
    """
    if code_map is None:
        code_map = {}
    if node is not None:
        if node.symbol is not None:
            # Leaf node: assign current prefix as code for this symbol
            code_map[node.symbol] = prefix
        generate_codes(node.left, prefix + "0", code_map)
        generate_codes(node.right, prefix + "1", code_map)
    return code_map

def huffman_encode(data):
    """
    Encode a list of symbols (hashable, e.g. ints or tuples) into a bitstring.
    Returns (code_map, encoded_string).  code_map maps symbol->bitstring.
    """
    freqs = Counter(data)
    # Build tree and codes
    root = build_huffman_tree(freqs)
    code_map = generate_codes(root)
    # Handle the case of only one unique symbol:
    # Assign it code '0' and output that bit for each symbol.
    if len(freqs) == 1:
        only_symbol = next(iter(freqs))
        code_map = {only_symbol: '0'}
        encoded = '0' * len(data)
    else:
        encoded = ''.join(code_map[s] for s in data)
    return code_map, encoded

def huffman_decode(encoded, code_map):
    """
    Decode a bitstring into the list of symbols given a code_map (symbol->code).
    """
    # If only one symbol was used in encoding, simply repeat it
    if len(code_map) == 1:
        symbol = next(iter(code_map))
        return [symbol] * len(encoded)
    # Invert the map to decode bits to symbols
    inv_map = {code: sym for sym, code in code_map.items()}
    decoded = []
    temp = ""
    for bit in encoded:
        temp += bit
        if temp in inv_map:
            decoded.append(inv_map[temp])
            temp = ""
    return decoded
