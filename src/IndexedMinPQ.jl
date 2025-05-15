# ChatGPT 04-mini-high Dijkstra Priority Queue
# Worked out-of-the-box! But slightly less efficient than QuickHeap (11.4 s vs 10.2 s)
# module IndexedMinPQ

# export IndexedMinPQ, is_empty, size, contains, insert!, decrease_key!, min_index, min_key, pop_min!

# A highly optimized indexed min-priority queue for Dijkstra's algorithm
# Based on the "pq", "qp", and "keys" arrays technique for O(log N) operations

mutable struct IndexedMinPQ{T}
    maxN::Int        # maximum number of elements
    N::Int           # current number of elements
    pq::Vector{Int}  # binary heap: pq[1..N] holds the indices
    qp::Vector{Int}  # inverse: qp[i] gives position of i in pq (0 if not present)
    keys::Vector{T}  # keys[i] is the priority of index i
    function IndexedMinPQ{T}(maxN::Int) where T
        new{T}(maxN, 0,
            Vector{Int}(undef, maxN+1),
            fill(0, maxN+1),
            Vector{T}(undef, maxN+1))
    end
end

# Check if empty
is_empty(pq::IndexedMinPQ) = (pq.N == 0)
Base.size(pq::IndexedMinPQ) = pq.N

# Does the queue contain index i?
contains(pq::IndexedMinPQ, i::Int) = pq.qp[i] != 0

# Insert index i with priority key
function insert!(pq::IndexedMinPQ{T}, i::Int, key::T) where T
    @assert pq.qp[i] == 0 "Index $i already in priority queue"
    pq.N += 1
    pq.qp[i] = pq.N
    pq.pq[pq.N] = i
    pq.keys[i] = key
    swim!(pq, pq.N)
end

# Decrease the key (priority) of index i
function decrease_key!(pq::IndexedMinPQ{T}, i::Int, key::T) where T
    @assert contains(pq, i) "Index $i not in priority queue"
    @assert key < pq.keys[i] "New key is not smaller than current key"
    pq.keys[i] = key
    swim!(pq, pq.qp[i])
end

# Peek minimum index and key
min_index(pq::IndexedMinPQ) = pq.pq[1]
min_key(pq::IndexedMinPQ) = pq.keys[pq.pq[1]]

# Remove and return the index with minimum key
function pop_min!(pq::IndexedMinPQ{T}) where T
    @assert pq.N > 0 "Priority queue underflow"
    min = pq.pq[1]
    exch!(pq, 1, pq.N)
    pq.N -= 1
    sink!(pq, 1)
    pq.qp[min] = 0
    pq.pq[pq.N+1] = 0
    return min
end
function dequeue!(pq::IndexedMinPQ{T}) where T
    return pop_min!(pq)
end

# Internal: swim up
@inline function swim!(pq::IndexedMinPQ{T}, k::Int) where T
    while k > 1 && pq.keys[pq.pq[k]] < pq.keys[pq.pq[k >>> 1]]
        exch!(pq, k, k >>> 1)
        k >>>= 1
    end
end

# Internal: sink down
@inline function sink!(pq::IndexedMinPQ{T}, k::Int) where T
    N = pq.N
    while (j = k << 1) <= N
        if j < N && pq.keys[pq.pq[j+1]] < pq.keys[pq.pq[j]]
            j += 1
        end
        pq.keys[pq.pq[j]] >= pq.keys[pq.pq[k]] && break
        exch!(pq, k, j)
        k = j
    end
end

# Internal: exchange positions i and j in the heap
@inline function exch!(pq::IndexedMinPQ, i::Int, j::Int)
    @inbounds tmp = pq.pq[i]
    @inbounds pq.pq[i] = pq.pq[j]
    @inbounds pq.pq[j] = tmp
    @inbounds pq.qp[pq.pq[i]] = i
    @inbounds pq.qp[pq.pq[j]] = j
end

# Base overloads: array-style access and isempty
# import Base: getindex, setindex!, isempty

function Base.getindex(pq::IndexedMinPQ{T}, i::Int) where T
    @assert contains(pq, i) "Index $i not in priority queue"
    return pq.keys[i]
end

function Base.setindex!(pq::IndexedMinPQ{T}, key::T, i::Int) where T
    if contains(pq, i)
        decrease_key!(pq, i, key)
    else
        insert!(pq, i, key)
    end
end

function Base.isempty(pq::IndexedMinPQ)
    return is_empty(pq)
end

# end # module
