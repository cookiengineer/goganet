package autodiff

import (
	"math"
	"testing"
)

func check(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", name, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Fatalf("%s[%d]: got %v, want %v", name, i, got[i], want[i])
		}
	}
}

func TestPolynomialFirstAndSecondOrder(t *testing.T) {
	x := Var(Shape{4, 1}, []float32{-1, 0.5, 2, -3})

	// f = sum(x^3)
	f := SumAll(Mul(Mul(x, x), x))
	g := f.Grad(x)
	wantG := []float32{3 * -1 * -1, 3 * 0.25, 3 * 4, 3 * 9}
	check(t, "f'", g.Value(), wantG)

	// d/dx sum(f') = 6x
	h := SumAll(g).Grad(x)
	wantH := []float32{-6, 3, 12, -18}
	check(t, "f''", h.Value(), wantH)
}

func TestMatMulGrad(t *testing.T) {
	// y = W x with W (2x3), x (3x1). Loss = sum(y).
	w := Var(Shape{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	x := Var(Shape{3, 1}, []float32{1, 1, 1})
	y := MatMul(w, x)
	loss := SumAll(y)
	gw := loss.Grad(w)
	// dLoss/dW = ones(2x1) * x^T = ones(2x3)
	check(t, "dW", gw.Value(), []float32{1, 1, 1, 1, 1, 1})
	gx := loss.Grad(x)
	// dLoss/dx = W^T * ones = column sums of W
	check(t, "dx", gx.Value(), []float32{5, 7, 9})
}

// TestGradientPenaltyGrad mimics the WGAN-GP penalty: D(x) = w . x, the penalty
// is (||dD/dx|| - 1)^2, and we check the derivative with respect to w.
func TestGradientPenaltyGrad(t *testing.T) {
	w := Var(Shape{3, 1}, []float32{3, 4, 0}) // ||w|| = 5
	x := Var(Shape{3, 1}, []float32{1, 1, 1})

	d := SumAll(Mul(w, x)) // w.x
	g := d.Grad(x)         // w
	norm := Sqrt(ColSumSquares(g))
	penalty := MeanAll(Mul(AddConst(norm, -1), AddConst(norm, -1)))

	gp := penalty.Grad(w)
	// penalty = (||w||-1)^2, d/dw = 2(||w||-1) * w/||w|| = 8*w/5
	want := []float32{8.0 * 3 / 5, 8.0 * 4 / 5, 0}
	check(t, "gp", gp.Value(), want)
}
