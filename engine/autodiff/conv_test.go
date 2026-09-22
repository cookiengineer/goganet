package autodiff

import (
	"math"
	"math/rand/v2"
	"testing"
)

// naiveConv computes a zero-padded convolution for B=1 with nested loops.
func naiveConv(x, w []float32, inC, outC, H, W, K, stride, pad int) ([]float32, int, int) {
	oh := (H+2*pad-K)/stride + 1
	ow := (W+2*pad-K)/stride + 1
	out := make([]float32, outC*oh*ow)
	for co := 0; co < outC; co++ {
		for c := 0; c < inC; c++ {
			for kh := 0; kh < K; kh++ {
				for kw := 0; kw < K; kw++ {
					wk := w[co*inC*K*K+c*K*K+kh*K+kw]
					for y := 0; y < oh; y++ {
						for xx := 0; xx < ow; xx++ {
							iy := y*stride - pad + kh
							ix := xx*stride - pad + kw
							if iy < 0 || iy >= H || ix < 0 || ix >= W {
								continue
							}
							out[co*oh*ow+y*ow+xx] += wk * x[c*H*W+iy*W+ix]
						}
					}
				}
			}
		}
	}
	return out, oh, ow
}

func TestConv2DForward(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	w := []float32{1, 0, 0, 1} // 2x2 identity-ish
	xN := Var(S(1, 9), x)
	wN := Var(S(1, 4), w)
	bN := Var(S(1, 1), []float32{0})
	y := Conv2D(xN, wN, bN, ConvShape{B: 1, InH: 3, InW: 3, K: 2, Stride: 1, Pad: 0})
	want, _, _ := naiveConv(x, w, 1, 1, 3, 3, 2, 1, 0)
	check(t, "conv", y.Value(), want)
}

func TestConv2DAdjoint(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	B, inC, outC, H, W, K, stride, pad := 2, 2, 3, 5, 6, 3, 2, 1
	cs := ConvShape{B: B, InH: H, InW: W, K: K, Stride: stride, Pad: pad}.withOutput()

	xData := make([]float32, inC*B*H*W)
	wData := make([]float32, outC*inC*K*K)
	yData := make([]float32, outC*B*cs.OutH*cs.OutW)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	for i := range yData {
		yData[i] = float32(rng.NormFloat64())
	}

	x := Var(S(inC, B*H*W), xData)
	w := Var(S(outC, inC*K*K), wData)
	b := Var(S(outC, 1), make([]float32, outC))
	y := Var(S(outC, B*cs.OutH*cs.OutW), yData)

	conv := Conv2D(x, w, b, ConvShape{B: B, InH: H, InW: W, K: K, Stride: stride, Pad: pad})

	// Conv2DTranspose of y with a (inC x outC*K*K) weight gives an (inC, B*H*W)
	// map that must satisfy the adjoint identity <conv(x,w), y> = <x, deconv(y,w)>.
	wT := Var(S(outC, inC*K*K), wData) // same numbers, deconv weight (inC_deconv=outC)
	bd := Var(S(inC, 1), make([]float32, inC))
	deconvCS := ConvShape{B: B, InH: H, InW: W, OutH: cs.OutH, OutW: cs.OutW, K: K, Stride: stride, Pad: pad}
	deconv := Conv2DTranspose(y, wT, bd, deconvCS)

	lhs := dot(y.Value(), conv.Value())
	rhs := dot(xValue(x), deconv.Value())
	if math.Abs(float64(lhs-rhs)) > 1e-2*math.Max(1, math.Abs(float64(lhs))) {
		t.Fatalf("adjoint identity failed: lhs=%v rhs=%v", lhs, rhs)
	}
	_ = outC
}

func xValue(n *Node) []float32 { return n.Value() }

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func TestConv2DDoubleBackprop(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	B, inC, outC, H, W, K, stride, pad := 1, 1, 2, 4, 4, 3, 1, 1
	cs := ConvShape{B: B, InH: H, InW: W, K: K, Stride: stride, Pad: pad}.withOutput()

	xData := make([]float32, inC*B*H*W)
	wData := make([]float32, outC*inC*K*K)
	for i := range xData {
		xData[i] = float32(rng.NormFloat64())
	}
	for i := range wData {
		wData[i] = float32(rng.NormFloat64())
	}
	x := Var(S(inC, B*H*W), xData)
	w := Var(S(outC, inC*K*K), wData)
	b := Var(S(outC, 1), make([]float32, outC))

	d := SumAll(Conv2D(x, w, b, cs))
	g := d.Grad(x)
	penalty := SumAll(Mul(g, g))
	gp := penalty.Grad(w)

	// Finite-difference check of d(penalty)/d(w).
	const eps = 1e-2
	for i := range wData {
		orig := wData[i]
		wData[i] = orig + eps
		autodiffSet(w, wData)
		pp := evalPenalty(x, w, b, cs)

		wData[i] = orig - eps
		autodiffSet(w, wData)
		pm := evalPenalty(x, w, b, cs)

		wData[i] = orig
		autodiffSet(w, wData)

		num := (pp - pm) / (2 * eps)
		if math.Abs(float64(num-gp.Value()[i])) > 5e-2*math.Max(1, math.Abs(float64(num))) {
			t.Fatalf("gp[%d]: analytic %v numeric %v", i, gp.Value()[i], num)
		}
	}
}

func autodiffSet(n *Node, data []float32) {
	copy(n.val, data)
}

func evalPenalty(x, w, b *Node, cs ConvShape) float32 {
	d := SumAll(Conv2D(x, w, b, cs))
	g := d.Grad(x)
	return SumAll(Mul(g, g)).Value()[0]
}

func TestFlattenGlobalMean(t *testing.T) {
	B, C, H, W := 2, 3, 2, 2
	data := make([]float32, C*B*H*W)
	for i := range data {
		data[i] = float32(i)
	}
	x := Var(S(C, B*H*W), data)
	flat := FlattenSpatial(x, B, H, W, C)
	if flat.Shape().R != C*H*W || flat.Shape().C != B {
		t.Fatalf("flatten shape %v", flat.Shape())
	}
	back := unflattenSpatial(flat, B, H, W, C)
	for i := range data {
		if back.Value()[i] != data[i] {
			t.Fatalf("flatten roundtrip failed at %d", i)
		}
	}
	gm := GlobalMean(x, B, H, W, C)
	if gm.Shape().R != C || gm.Shape().C != B {
		t.Fatalf("global mean shape %v", gm.Shape())
	}
	// Each mean should be the average of the corresponding H*W block.
	for c := 0; c < C; c++ {
		for b := 0; b < B; b++ {
			var want float32
			for s := 0; s < H*W; s++ {
				want += data[c*B*H*W+b*H*W+s]
			}
			want /= float32(H * W)
			if math.Abs(float64(gm.Value()[c*B+b]-want)) > 1e-5 {
				t.Fatalf("global mean mismatch")
			}
		}
	}
}
