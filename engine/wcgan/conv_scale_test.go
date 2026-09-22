package wcgan

import (
	"math/rand/v2"
	"testing"
)

func TestConvTrainerConfig(t *testing.T) {
	shapeFrames, shapeBytes := 8, 64
	m := New(Config{
		Arch:          "conv",
		InputDim:      shapeFrames * shapeBytes * 2,
		InputChannels: 2,
		Frames:        shapeFrames,
		BytesPerFrame: shapeBytes,
		SeedChannels:  32,
		GenChannels:   []int{16, 8},
		CritChannels:  []int{8, 16, 32},
		Kernel:        3,
		LatentDim:     32,
		NumClasses:    2,
		NCritic:       1,
	})
	const batch = 8
	rng := rand.New(rand.NewPCG(1, 2))
	x := make([]float32, m.cfg.InputDim*batch)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	y := OneHot([]int{0, 1, 0, 1, 1, 0, 1, 0}, 2)
	for step := 0; step < 5; step++ {
		m.CriticStep(x, y, batch)
		m.GeneratorStep(batch)
	}
}
