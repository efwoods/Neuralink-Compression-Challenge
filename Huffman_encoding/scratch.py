import heapq

chars = ['a','b','c','d','e','f']
freq = [5, 9, 12, 13, 16, 45]
nodes = []

class node:
    def __init__(self, freq, symbol, left = None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''
        
    def __lt__(self, nxt):
        return self.freq < nxt.freq

for x in range(len(chars)):
    heapq.heappush(nodes, node(freq[x], chars[x]))
    
    
    