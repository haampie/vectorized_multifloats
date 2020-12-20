using MultiFloats, VectorizationBase

using VectorizationBase: extractelement, pick_vector_width
using MultiFloats: renormalize

using DoubleFloats

import DoubleFloats: DoubleFloat

DoubleFloat(x::MultiFloat{T,2}) where {T} = DoubleFloat{T}(x._limbs[1], x._limbs[2])

using VectorizationBase: VecUnroll, vtranspose, unrolleddata

@generated function MultiFloatOfVec(fs::NTuple{M,MultiFloat{T,N}}) where {T,M,N}
    if M == N == pick_vector_width(T)
        to_vecs = [:(Vec(fs[$i]._limbs...)) for i = 1:M]
        # do a fast transpose.
        return quote
            $(Expr(:meta, :inline))
            MultiFloat(unrolleddata(vtranspose(VecUnroll(tuple($(to_vecs...))))))
        end
    else
        exprs = [:(Vec($([:(fs[$j]._limbs[$i]) for j=1:M]...))) for i=1:N]

        return quote
            $(Expr(:meta, :inline))
            MultiFloat(tuple($(exprs...)))
        end
    end
end

@generated function TupleOfMultiFloat(fs::MultiFloat{Vec{M,T},N}) where {T,M,N}
    exprs = [:(MultiFloat(tuple($([:(extractelement(fs._limbs[$j], $i)) for j=1:N]...)))) for i=0:M-1]
    return quote
        tuple($(exprs...))
    end
end

function trivial_sum(xs)
    t = zero(eltype(xs))
    @inbounds @simd for x in xs
        t += x
    end
    return t
end

function vectorized_sum(xs::Vector{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    t = zero(MultiFloat{Vec{M,T},N})

    for i = 1:M:length(xs)-M+1
        t += MultiFloatOfVec(ntuple(k -> @inbounds(xs[i + k - 1]), M))
    end

    return +(TupleOfMultiFloat(t)...)
end

using VectorizationBase: Unroll, vload, vtranspose, unrolleddata

function handwritten_sum(xs::Vector{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    @assert M == N

    t = zero(MultiFloat{Vec{N,T},N})

    p = stridedpointer(reinterpret(T, xs))

    # load N Multifloats or N*N T's of at a time.
    for i = 1:N:length(xs)
        t += MultiFloat(unrolleddata(vtranspose(vload(p, Unroll{1,1,N,1,N,0x0000000000000000}(((i-1)*N+1,))))))
    end

    return +(TupleOfMultiFloat(t)...)
end

function trivial_dot(xs, ys)
    t = zero(eltype(xs))
    @inbounds for i = 1:length(xs)
        t += xs[i] * ys[i]
    end
    return t
end

function vectorized_dot(xs::Vector{MultiFloat{T,N}}, ys::Vector{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    t = zero(MultiFloat{Vec{M,T},N})

    for i = 1:M:length(xs)-M+1
        x = MultiFloatOfVec(ntuple(k -> @inbounds(xs[i + k - 1]), M))
        y = MultiFloatOfVec(ntuple(k -> @inbounds(ys[i + k - 1]), M))
        t += x * y
    end
  
    return +(TupleOfMultiFloat(t)...)
end

using BenchmarkTools

random_vec(::Type{MultiFloat{T,N}}, k) where {T,N} =
    [renormalize(MultiFloat(ntuple(i -> rand(T) * eps(T)^(i-1), N))) for _ = 1:k]

function benchmark_dot(::Type{T}) where {T<:MultiFloat}
    xs = random_vec(T, 2^13)
    ys = random_vec(T, 2^13)

    @show vectorized_dot(xs, ys) - trivial_dot(xs, ys)
    
    vectorized = @benchmark vectorized_dot($xs, $ys)
    trivial = @benchmark trivial_dot($xs, $ys)

    return vectorized, trivial
end

function benchmark_sum(::Type{T}) where {T<:MultiFloat}
    xs = random_vec(T, 2^13)

    @show vectorized_sum(xs) - trivial_sum(xs)
    
    vectorized = @benchmark vectorized_sum($xs)
    trivial = @benchmark trivial_sum($xs)

    return vectorized, trivial
end
