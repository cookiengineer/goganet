package wcgan

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/cookiengineer/goganet/engine/autodiff"
)

// makeData builds a simple two-class linearly separable dataset laid out as
// (dim x samples) row-major, together with the class indices.
func makeData(dim, perClass int, seed uint64) (x []float32, labels []int) {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	total := perClass * 2
	x = make([]float32, dim*total)
	labels = make([]int, total)
	for b := 0; b < total; b++ {
		cls := b % 2
		labels[b] = cls
		sign := float32(1)
		if cls == 1 {
			sign = -1
		}
		for d := 0; d < dim; d++ {
			x[d*total+b] = sign*1.0 + float32(rng.NormFloat64())*0.3
		}
	}
	return x, labels
}

func TestClassifierLearns(t *testing.T) {
	const dim = 8
	const perClass = 32
	const batch = 16

	cfg := Config{
		InputDim:   dim,
		LatentDim:  4,
		GenHidden:  []int{8},
		CritHidden: []int{8},
		NumClasses: 2,
		NCritic:    1,
		GenLR:      1e-3,
		CritLR:     1e-3,
		Seed:       42,
	}
	m := New(cfg)
	x, labels := makeData(dim, perClass, 7)
	total := perClass * 2

	for iter := 0; iter < 300; iter++ {
		// Slice a batch of real samples.
		xb := make([]float32, dim*batch)
		yb := make([]int, batch)
		for b := 0; b < batch; b++ {
			src := (iter*batch + b) % total
			yb[b] = labels[src]
			for d := 0; d < dim; d++ {
				xb[d*batch+b] = x[d*total+src]
			}
		}
		cl := OneHot(yb, 2)

		dLoss := m.CriticStep(xb, cl, batch)
		gLoss := m.GeneratorStep(batch)
		if math.IsNaN(float64(dLoss)) || math.IsNaN(float64(gLoss)) {
			t.Fatalf("NaN loss at iteration %d: d=%v g=%v", iter, dLoss, gLoss)
		}
	}

	probs, scores := m.Classify(x, total)
	correct := 0
	for b := 0; b < total; b++ {
		p1 := probs[1*total+b]
		pred := 0
		if p1 > 0.5 {
			pred = 1
		}
		if pred == labels[b] {
			correct++
		}
		if math.IsNaN(float64(scores[b])) || math.IsInf(float64(scores[b]), 0) {
			t.Fatalf("bad critic score at %d: %v", b, scores[b])
		}
	}
	acc := float64(correct) / float64(total)
	if acc < 0.8 {
		t.Fatalf("classifier accuracy too low: %.2f", acc)
	}
	t.Logf("accuracy %.2f", acc)
}

func TestGenerateFinite(t *testing.T) {
	m := New(Config{InputDim: 12, LatentDim: 3, GenHidden: []int{6}, CritHidden: []int{6}, NumClasses: 2, Seed: 1})
	c := OneHot([]int{0, 1, 0, 1}, 2)
	zNode := autodiff.Const(autodiff.S(m.cfg.LatentDim, 4), m.noise(4))
	cNode := autodiff.Const(autodiff.S(2, 4), c)
	img := m.Generate(zNode, cNode)
	if len(img.Value()) != 12*4 {
		t.Fatalf("unexpected generator output length %d", len(img.Value()))
	}
	for i, v := range img.Value() {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite generator output at %d: %v", i, v)
		}
	}
}
