using MultiFloats, VectorizationBase
using VectorizationBase: extractelement, pick_vector_width
using MultiFloats: renormalize
using VectorizationBase: VecUnroll, Unroll, vtranspose, unrolleddata, vload
using DoubleFloats
using BenchmarkTools
using Random: AbstractRNG, SamplerType
using Base: IEEEFloat

import DoubleFloats: DoubleFloat
import Random: rand
import LinearAlgebra: dot, axpy!, norm
import Base: sum


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

DoubleFloat(x::MultiFloat{T,2}) where {T} = DoubleFloat{T}(x._limbs[1], x._limbs[2])

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
    @inbounds @simd ivdep for x in xs
        t += x
    end
    return t
end

function trivial_dot(xs, ys)
    t = zero(eltype(xs))
    @inbounds @simd ivdep for i = 1:length(xs)
        t += xs[i] * ys[i]
    end
    return t
end

function trivial_axpy!(a, xs, ys)
    @inbounds @simd ivdep for i = 1:length(xs)
        ys[i] = xs[i] * a
    end
    return ys
end

@generated function sum(xs::AbstractArray{MultiFloat{T,N}}) where {T,N}
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

@generated function dot(xs::AbstractArray{MultiFloat{T,N}}, ys::AbstractArray{MultiFloat{T,N}}) where {T,N}
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

@generated function axpy!(a::MultiFloat{T,N}, xs::AbstractArray{MultiFloat{T,N}}, ys::AbstractArray{MultiFloat{T,N}}) where {T,N}
    M = pick_vector_width(T)

    load_xs = ntuple(k -> :(xs[i + $(k - 1)]), M)
    load_ys = ntuple(k -> :(ys[i + $(k - 1)]), M)
    assign = [:($(load_ys[i]) = t[$i]) for i = 1:M]
    broadcast_a = ntuple(i -> :a, M)

    quote
        # iterate in steps of M
        step, i = 1, 1
        av = MultiFloatOfVec(tuple($(broadcast_a...)))
        @inbounds while step ≤ length(xs) ÷ $M
            x = MultiFloatOfVec(tuple($(load_xs...)))
            y = MultiFloatOfVec(tuple($(load_ys...)))
            result = y + x * av
            t = TupleOfMultiFloat(result)
            $(assign...)
            step += 1
            i += $M
        end
    
        # sum the remainder
        while i ≤ length(xs)
            @inbounds ys[i] += xs[i] * a
            i += 1
        end

        return ys
    end
end

norm(xs::AbstractArray{<:MultiFloat}) = sqrt(dot(xs, xs))

rand(rng::AbstractRNG, ::SamplerType{MultiFloat{T,N}}) where {T<:IEEEFloat,N} =
    renormalize(MultiFloat(ntuple(i -> rand(T) * eps(T)^(i-1), Val{N}())))

function benchmark_dot(::Type{T}) where {T<:MultiFloat}
    xs, ys = rand(T, 2^13), rand(T, 2^13)
    xs_soa, ys_soa = StructArray(xs), StructArray(ys)

    @show dot(xs, ys) - trivial_dot(xs, ys)

    vectorized = @benchmark dot($xs_soa, $ys_soa)
    trivial = @benchmark trivial_dot($xs, $ys)

    return vectorized, trivial
end

function benchmark_axpy(::Type{T}) where {T<:MultiFloat}
    xs, ys = rand(T, 2^13), rand(T, 2^13)
    xs_soa, ys_soa = StructArray(xs), StructArray(ys)

    @show norm(axpy!(T(2.0), xs, ys) - trivial_axpy!(T(2.0), xs, ys))

    vectorized = @benchmark axpy!(a, xs_soa, ys_soa) setup=(a=$T(1.0); xs_soa=StructArray(rand($T, 2^13)); ys_soa=StructArray(rand($T, 2^13)))
    trivial = @benchmark trivial_axpy!(a, xs, ys) setup=(a=$T(1.0); xs=rand($T, 2^13); ys=rand($T, 2^13))

    return vectorized, trivial
end

function benchmark_sum(::Type{T}) where {T<:MultiFloat}
    xs = rand(T, 2^13)
    xs_soa = StructArray(xs)

    @show sum(xs_soa) - trivial_sum(xs)

    vectorized = @benchmark sum($xs_soa)
    trivial = @benchmark trivial_sum($xs)

    return vectorized, trivial
end