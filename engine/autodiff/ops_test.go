package autodiff

import (
	"math"
	"testing"
)

func TestColSumRowSumForward(t *testing.T) {
	x := Const(S(3, 2), []float32{1, 2, 3, 4, 5, 6})
	check(t, "ColSum", ColSum(x).Value(), []float32{9, 12})
	check(t, "RowSum", RowSum(x).Value(), []float32{3, 7, 11})
}

func TestBroadcastAndMulRowForward(t *testing.T) {
	r := Const(S(1, 2), []float32{2, 3})
	bc := BroadcastCol(r, 3)
	check(t, "BroadcastCol", bc.Value(), []float32{2, 3, 2, 3, 2, 3})
	br := BroadcastRow(r, 2)
	check(t, "BroadcastRow", br.Value(), []float32{2, 3, 2, 3})

	x := Const(S(2, 2), []float32{1, 1, 1, 1})
	check(t, "MulRow", MulRow(x, r).Value(), []float32{2, 3, 2, 3})
}

func TestColMaxForwardAndGrad(t *testing.T) {
	x := Var(S(3, 2), []float32{1, 2, 3, 4, 5, 6})
	c := ColMax(x)
	check(t, "ColMax", c.Value(), []float32{5, 6})
	g := SumAll(c).Grad(x)
	check(t, "ColMax grad", g.Value(), []float32{0, 0, 0, 0, 1, 1})
}

func TestExpLogForward(t *testing.T) {
	x := Const(S(2, 1), []float32{0, 1})
	e := Exp(x)
	if math.Abs(float64(e.Value()[0])-1) > 1e-5 || math.Abs(float64(e.Value()[1])-math.E) > 1e-4 {
		t.Fatalf("exp = %v", e.Value())
	}
	l := Log(e)
	check(t, "log(exp(x))", l.Value(), []float32{0, 1})
}

func TestTransposeReshapeRoundTrip(t *testing.T) {
	x := Const(S(2, 3), []float32{1, 2, 3, 4, 5, 6})
	tr := Transpose(x)
	if tr.Shape() != (Shape{3, 2}) {
		t.Fatalf("transpose shape = %v", tr.Shape())
	}
	check(t, "transpose", tr.Value(), []float32{1, 4, 2, 5, 3, 6})
	back := Transpose(tr)
	check(t, "transpose^2", back.Value(), x.Value())
	rs := Reshape(x, 3, 2)
	check(t, "reshape", rs.Value(), x.Value())
}

// TestSoftmaxCrossEntropyGrad verifies the classifier loss gradient numerically.
func TestSoftmaxCrossEntropyGrad(t *testing.T) {
	data := []float32{0.5, -1.0, 2.0, 1.5, 0.0, -0.5}
	logits := Var(S(2, 3), append([]float32(nil), data...))
	labels := Const(S(2, 3), []float32{1, 0, 0, 0, 0, 1}) // class 0 for col0, class 1 for col2

	loss := SoftmaxCrossEntropy(logits, labels, 3)
	g := loss.Grad(logits)

	const eps = 1e-3
	for i := range data {
		orig := data[i]
		data[i] = orig + eps
		copy(logits.Value(), data)
		lp := SoftmaxCrossEntropy(logits, labels, 3).Value()[0]

		data[i] = orig - eps
		copy(logits.Value(), data)
		lm := SoftmaxCrossEntropy(logits, labels, 3).Value()[0]

		data[i] = orig
		copy(logits.Value(), data)

		num := (lp - lm) / (2 * eps)
		if math.Abs(float64(num-g.Value()[i])) > 1e-2*math.Max(1, math.Abs(float64(num))) {
			t.Fatalf("softmax ce grad[%d]: analytic %v numeric %v", i, g.Value()[i], num)
		}
	}
}
