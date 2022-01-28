module irutils

using NPZ

"""
Loads IR data.

* input
	* `base_dir` : path to the directory of the data

* output
	* `pos_mic` : Positions of microphones, Vector{Vector{Float64}}.
	* `pos_src` : Positions of sound sources, Vector{Vector{Float64}}.
	* `full_ir` : Impulse signal recored at microphones, Array{Float64, 3}.
	            First index is for each sound source.
				Second index is for each microphone.
"""
function loadIR(base_dir::String)
    pos_mic = npzread("$(base_dir)/pos_mic.npy")
    pos_src = npzread("$(base_dir)/pos_src.npy")

    pos_mic = [pos_mic[i, :] for i = 1:size(pos_mic, 1)]
    pos_src = [pos_src[i, :] for i = 1:size(pos_src, 1)]

    num_mic = size(pos_mic, 1)

    tmp_ir = Array{Array{Float64,2}}(undef, num_mic)

    for f in readdir(base_dir)
        if isfile("$(base_dir)/$(f)") && startswith(f, "ir_") && endswith(f, ".npy")
            idx = parse(Int, last(split(first(split(f, ".")), "_"))) + 1
            tmp_ir[idx] = npzread("$(base_dir)/$(f)")
        end
    end

    num_src = size(tmp_ir[1], 1)
    ir_len = size(tmp_ir[1], 2)
    full_ir = zeros(num_src, num_mic, ir_len)
    for idx = 1:num_mic
        full_ir[:, idx, :] = tmp_ir[idx]
    end

    return pos_mic, pos_src, full_ir
end

function loadIR(base_dir::String, src_idices::Vector{Int})
    pos_mic = npzread("$(base_dir)/pos_mic.npy")
    pos_src = npzread("$(base_dir)/pos_src.npy")

    pos_mic = [pos_mic[i, :] for i = 1:size(pos_mic, 1)]
    pos_src = [pos_src[i, :] for i in src_idices]

    num_mic = size(pos_mic, 1)

    tmp_ir = Array{Array{Float64,2}}(undef, num_mic)

    for f in readdir(base_dir)
        if isfile("$(base_dir)/$(f)") && startswith(f, "ir_") && endswith(f, ".npy")
            idx = parse(Int, last(split(first(split(f, ".")), "_"))) + 1
            tmp_ir[idx] = npzread("$(base_dir)/$(f)")[src_idices, :]
        end
    end

    ir_len = size(tmp_ir[1], 2)
    num_src = length(src_idices)
    full_ir = zeros(num_src, num_mic, ir_len)
    for idx = 1:num_mic
        full_ir[:, idx, :] = tmp_ir[idx]
    end

    return pos_mic, pos_src, full_ir
end

end

