package autodiff

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestConvClassifierGrad numerically checks the gradient of a cross-entropy
// loss through conv -> leaky -> global mean -> softmax.
func TestConvClassifierGrad(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	B, inC, outC, H, W, K, stride, pad := 2, 1, 3, 4, 4, 3, 2, 1
	cs := ConvShape{B: B, InH: H, InW: W, K: K, Stride: stride, Pad: pad}.withOutput()
	poolC := outC

	xData := make([]float32, inC*B*H*W)
	wData := make([]float32, outC*inC*K*K)
	clsData := make([]float32, 2*poolC)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	for i := range clsData {
		clsData[i] = float32(rng.NormFloat64())
	}
	labels := []float32{1, 0, 0, 1} // (2 x B)

	x := Const(S(inC, B*H*W), xData)
	w := Var(S(outC, inC*K*K), wData)
	b := Var(S(outC, 1), make([]float32, outC))
	clsW := Var(S(2, poolC), clsData)
	clsB := Var(S(2, 1), make([]float32, 2))
	lab := Const(S(2, B), labels)

	build := func() *Node {
		h := LeakyReLU(Conv2D(x, w, b, cs), 0.2)
		pooled := GlobalMean(h, B, cs.OutH, cs.OutW, outC)
		logits := AddBias(MatMul(clsW, pooled), clsB)
		return SoftmaxCrossEntropy(logits, lab, B)
	}
	loss := build()
	gw := loss.Grad(w)

	const eps = 1e-3
	for i := range wData {
		orig := wData[i]
		wData[i] = orig + eps
		copy(w.Value(), wData)
		lp := build().Value()[0]
		wData[i] = orig - eps
		copy(w.Value(), wData)
		lm := build().Value()[0]
		wData[i] = orig
		copy(w.Value(), wData)
		num := (lp - lm) / (2 * eps)
		if math.Abs(float64(num-gw.Value()[i])) > 1e-1*math.Max(1, math.Abs(float64(num))) {
			t.Fatalf("conv classifier grad[%d]: analytic %v numeric %v", i, gw.Value()[i], num)
		}
	}
}
