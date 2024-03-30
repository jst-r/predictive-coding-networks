module Test

using Flux, CUDA, Statistics, ProgressMeter

noisy = rand(Float32, 2, 1000)
xor_value = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]

model = Chain(
    Dense(2 => 3, tanh)
)

end