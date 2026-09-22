package wcgan

import (
	"bytes"
	"math"
	"testing"

	"github.com/cookiengineer/goganet/engine/autodiff"
)

func convConfig() Config {
	return Config{
		Arch:          "conv",
		Frames:        8,
		BytesPerFrame: 16,
		InputChannels: 1,
		SeedChannels:  8,
		GenChannels:   []int{8, 4},
		CritChannels:  []int{8, 16},
		Kernel:        3,
		InputDim:      8 * 16,
		LatentDim:     8,
		NumClasses:    2,
		Lambda:        1,
		NCritic:       1,
		GenLR:         2e-4,
		CritLR:        2e-4,
		Seed:          9,
	}
}

func TestConvClassifierLearns(t *testing.T) {
	m := New(convConfig())
	if m.cfg.Arch != "conv" {
		t.Fatal("expected conv architecture")
	}
	const batch = 8
	const dim = 8 * 16
	x := make([]float32, dim*batch)
	y := make([]int, batch)
	for b := 0; b < batch; b++ {
		cls := b % 2
		y[b] = cls
		sign := float32(1)
		if cls == 1 {
			sign = -1
		}
		for i := 0; i < dim; i++ {
			x[i*batch+b] = sign * (0.5 + float32(i%3)/10)
		}
	}
	for iter := 0; iter < 600; iter++ {
		dLoss := m.CriticStep(x, OneHot(y, 2), batch)
		gLoss := m.GeneratorStep(batch)
		if math.IsNaN(float64(dLoss)) || math.IsNaN(float64(gLoss)) {
			t.Fatalf("NaN loss at %d: %v %v", iter, dLoss, gLoss)
		}
	}
	correct := 0
	sample := make([]float32, dim)
	for b := 0; b < batch; b++ {
		for i := 0; i < dim; i++ {
			sample[i] = x[i*batch+b]
		}
		probs, _ := m.Classify(sample, 1)
		pred := 0
		if probs[1] > 0.5 {
			pred = 1
		}
		if pred == y[b] {
			correct++
		}
	}
	if acc := float64(correct) / batch; acc < 0.8 {
		t.Fatalf("conv classifier accuracy too low: %.2f", acc)
	}
}

func TestConvGenerateShapeAndSaveLoad(t *testing.T) {
	m := New(convConfig())
	z := m.noise(4)
	zNode := autodiff.Const(autodiff.S(m.cfg.LatentDim, 4), z)
	cNode := autodiff.Const(autodiff.S(2, 4), OneHot([]int{0, 1, 0, 1}, 2))
	img := m.Generate(zNode, cNode)
	if got := img.Shape(); got.R != m.cfg.InputDim || got.C != 4 {
		t.Fatalf("conv generator output shape %v", got)
	}

	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if loaded.cfg.Arch != "conv" {
		t.Fatal("loaded model lost conv architecture")
	}
}
