using MultiFloats, VectorizationBase

using VectorizationBase: extractelement, pick_vector_width
using MultiFloats: renormalize

using DoubleFloats

import DoubleFloats: DoubleFloat

# struct of array type.
struct MultiFloatSoA{T,N}
    limbs::NTuple{N,Vector{T}}
end

function MultiFloatSoA(xs::Vector{MultiFloat{T,N}}) where {T,N}
    limbs = ntuple(_ -> Vector{T}(undef, length(xs)), Val{N}())

    for i in eachindex(xs)
        for j = 1:N
            limbs[j][i] = xs[i]._limbs[j]
        end
    end

    return MultiFloatSoA(limbs)
end

DoubleFloat(x::MultiFloat{T,2}) where {T} = DoubleFloat{T}(x._limbs[1], x._limbs[2])

using VectorizationBase: VecUnroll, vtranspose, unrolleddata

@generated function MultiFloatOfVec(fs::NTuple{M,MultiFloat{T,N}}) where {T,M,N}
    exprs = [:(Vec($([:(fs[$j]._limbs[$i]) for j=1:M]...))) for i=1:N]

    return quote
        $(Expr(:meta, :inline))
        MultiFloat(tuple($(exprs...)))
    end
end

@generated function TupleOfMultiFloat(fs::MultiFloat{Vec{M,T},N}) where {T,M,N}
    exprs = [:(MultiFloat(tuple($([:(extractelement(fs._limbs[$j], $i)) for j=1:N]...)))) for i=0:M-1]
    return quote
        $(Expr(:meta, :inline))
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

using VectorizationBase: Unroll, vload, vtranspose, unrolleddata

@generated function vectorized_sum(xs::AbstractArray{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    load_xs = ntuple(k -> :(xs[i + $(k - 1)]), M)

    quote
        @inbounds begin
            vec_total = zero(MultiFloat{Vec{$M,$T},$N})

            # iterate in steps of M
            step, i = 1, 1
            while step ≤ length(xs) ÷ $M
                vec_total += MultiFloatOfVec(tuple($(load_xs...)))
                step += 1
                i += $M
            end
        
            # sum the remainder
            total = +(TupleOfMultiFloat(vec_total)...)
            while i ≤ length(xs)
                total += xs[i]
                i += 1
            end

            return total
        end
    end
end

@generated function handwritten_sum(xs::MultiFloatSoA{T,N}) where {T,N}
    M = pick_vector_width(T)

    quote
        len = length(xs.limbs[1])
        t = zero(MultiFloat{Vec{$M,$T},$N})

        # create N pointers
        Base.Cartesian.@nexprs $N i -> px_i = stridedpointer(xs.limbs[i])
        
        j = 1

        while j < len
            # N loads.
            t += MultiFloat(Base.Cartesian.@ntuple $N i -> vload(px_i, (MM{$M}(j),)))
            j += $M
        end

        return +(TupleOfMultiFloat(t)...)
    end
end

function handwritten_sum(xs::Vector{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    t = zero(MultiFloat{Vec{M,T},N})

    p = stridedpointer(reinterpret(T, xs))

    for j = Base.OneTo(length(xs)÷M)
        idx = (j - 1) * M * N + 1
        t += MultiFloat(unrolleddata(vtranspose(vload(p, Unroll{1,1,N,1,M,0x0000000000000000}((idx,))), Val{N}())))
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

@generated function vectorized_dot(xs::AbstractArray{MultiFloat{T,N}}, ys::AbstractArray{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    load_xs = ntuple(k -> :(xs[i + $(k - 1)]), M)
    load_ys = ntuple(k -> :(ys[i + $(k - 1)]), M)

    quote
        vec_total = zero(MultiFloat{Vec{$M,$T},$N})

        # iterate in steps of M
        step, i = 1, 1
        @inbounds while step ≤ length(xs) ÷ $M
            x = MultiFloatOfVec(tuple($(load_xs...)))
            y = MultiFloatOfVec(tuple($(load_ys...)))
            vec_total += x * y
            step += 1
            i += $M
        end
    
        # sum the remainder
        @inbounds total = +(TupleOfMultiFloat(vec_total)...)
        while i ≤ length(xs)
            @inbounds total += xs[i] * ys[i]
            i += 1
        end

        return total
    end
end

@generated function handwritten_dot(xs::MultiFloatSoA{T,N}, ys::MultiFloatSoA{T,N}) where {T,N}
    M = pick_vector_width(T)

    quote
        len = length(xs.limbs[1])
        t = zero(MultiFloat{Vec{$M,$T},$N})

        # create N pointers
        Base.Cartesian.@nexprs $N i -> px_i = stridedpointer(xs.limbs[i])
        Base.Cartesian.@nexprs $N i -> py_i = stridedpointer(ys.limbs[i])
        
        j = 1

        while j < len
            # N loads.
            x = MultiFloat(Base.Cartesian.@ntuple $N i -> vload(px_i, (MM{$M}(j),)))
            y = MultiFloat(Base.Cartesian.@ntuple $N i -> vload(py_i, (MM{$M}(j),)))
            t += x * y
            j += $M
        end

        return +(TupleOfMultiFloat(t)...)
    end
end

function handwritten_dot(xs::Vector{MultiFloat{T,N}}, ys::Vector{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    t = zero(MultiFloat{Vec{M,T},N})

    px = stridedpointer(reinterpret(T, xs))
    py = stridedpointer(reinterpret(T, ys))

    # load M Multifloats at a time.
    for j = Base.OneTo(length(xs)÷M)
        idx = (j - 1) * M * N + 1
        mx = MultiFloat(unrolleddata(vtranspose(vload(px, Unroll{1,1,N,1,M,0x0000000000000000}((idx,))), Val{N}())))
        my = MultiFloat(unrolleddata(vtranspose(vload(py, Unroll{1,1,N,1,M,0x0000000000000000}((idx,))), Val{N}())))
        
        t += mx * my
    end

    return +(TupleOfMultiFloat(t)...)
end

using BenchmarkTools

random_vec(::Type{MultiFloat{T,N}}, k) where {T,N} =
    [renormalize(MultiFloat(ntuple(i -> rand(T) * eps(T)^(i-1), N))) for _ = 1:k]

function benchmark_dot(::Type{T}) where {T<:MultiFloat}
    xs = random_vec(T, 2^13)
    ys = random_vec(T, 2^13)

    xs_soa = StructArray(xs)
    ys_soa = StructArray(ys)

    @show handwritten_dot(xs, ys) - trivial_dot(xs, ys)
    @show vectorized_dot(xs_soa, ys_soa) - trivial_dot(xs, ys)

    handwritten = @benchmark handwritten_dot($xs, $ys)
    vectorized = @benchmark vectorized_dot($xs_soa, $ys_soa)
    trivial = @benchmark trivial_dot($xs, $ys)

    return handwritten, vectorized, trivial
end

function benchmark_sum(::Type{T}) where {T<:MultiFloat}
    xs = random_vec(T, 2^13)
    xs_soa = StructArray(xs)

    @show handwritten_sum(xs) - trivial_sum(xs)
    @show vectorized_sum(xs_soa) - trivial_sum(xs)

    handwritten = @benchmark handwritten_sum($xs)
    vectorized = @benchmark vectorized_sum($xs_soa)
    trivial = @benchmark trivial_sum($xs)

    return handwritten, vectorized, trivial
end

using StructArrays, MultiFloats

import StructArrays: staticschema, createinstance
import Base: getproperty, propertynames

propertynames(::MultiFloat{T,N}) where {T,N} = ntuple(i -> Symbol(:idx_, i), Val{N}())

@generated function getproperty(x::MultiFloat{T,N}, s::Symbol) where {N,T}
    symbols = [Symbol(:idx_, i) for i = 1:N]
    quote
        $(Expr(:meta, :inline))
        $([:(s === $(QuoteNode(symbols[i])) && return getfield(x, 1)[$i]) for i = 1:N]...)
        return getfield(x, s)
    end
end

@generated function staticschema(::Type{MultiFloat{T,N}}) where {N,T}
    symbols = [Symbol(:idx_, i) for i = 1:N]
    quote
        NamedTuple{$(QuoteNode(ntuple(i -> symbols[i], Val{N}()))), NTuple{$N,$T}}
    end
end

createinstance(::Type{MultiFloat{T,N}}, args...) where {N,T} = MultiFloat{T,N}(values(args))