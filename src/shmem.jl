__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

__shmem(D::Device, args...) = throw(MethodError(__shmem, (D, args...)))
@inline __shmem(::CPU, ::Type{T}, ::Val{dims}, ::Val) where {T, dims} =MArray{__size(dims), T}(undef)

@init @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    using .CUDA

    @inline function __shmem(::CUDA, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
        ptr = CUDAnative._shmem(Val(id), T, Val(prod(dims)))
        CUDAnative.CuDeviceArray(dims, CUDAnative.DevicePtr{T, CUDAnative.AS.Shared}(ptr))
    end
end

shmem_id = 0
macro shmem(T, dims)
    global shmem_id
    id = shmem_id::Int += 1

    quote
        $__shmem($backend(), $(esc(T)), Val($(esc(dims))), Val($id))
    end
end
