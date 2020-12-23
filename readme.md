some experiments with vectorized multifloat things.

this package does type piracy, and also requires the manifest for patches to deps.

start using it with

```julia
$ cd VectorizedMultiFloats.jl
$ julia -O3 --project=.
```

as an example on an AVX-512 machine:

```
julia> using VectorizedMultiFloats, MultiFloats

julia> eps(Float64x4)
2.4308653429145085e-63

julia> setprecision(213); eps(BigFloat)
1.51929083932156779959571876312991314467535453337919119460476479853e-64

julia> mat = rand(Float64x4, 1000, 1000);

julia> @time Matrix(LinearAlgebra.qrfactUnblocked!(StructArray(mat)).factors);
  8.141877 seconds (16 allocations: 61.066 MiB)

julia> @time LinearAlgebra.qrfactUnblocked!(mat).factors;
 33.060760 seconds (4 allocations: 31.391 KiB)

julia> @time LinearAlgebra.qrfactUnblocked!(z).factors;
228.394904 seconds (2.67 G allocations: 139.130 GiB, 46.52% gc time)
```

just `norm`, `dot`, `sum` and `axpy!` are vectorized.

without this package, `axpy!` can auto-vectorize for MultiFloats types, but it fails to do so for reductions like `dot` and `sum`. by using `StructArray` you still need to help the compiler a little bit to auto-vectorize, so that's implemented here.