package vector

import (
	"math"
	"testing"
)

func TestAdamFirstStep(t *testing.T) {
	param := []float32{1, 1}
	grad := []float32{0.5, -0.5}
	m := []float32{0, 0}
	v := []float32{0, 0}
	Adam(param, grad, m, v, 0.1, 0.9, 0.999, 1e-8, 1)
	// On the first step the bias correction cancels the (1-beta) factors, so
	// the update is lr * g / (|g| + eps) ~= lr * sign(g).
	want := []float32{0.9, 1.1}
	for i := range param {
		if math.Abs(float64(param[i]-want[i])) > 1e-3 {
			t.Fatalf("param[%d] = %v, want %v", i, param[i], want[i])
		}
	}
}

func TestFillZeroAndEmptyReductions(t *testing.T) {
	dst := make([]float32, 3)
	Fill(dst, 2.5)
	for _, v := range dst {
		if v != 2.5 {
			t.Fatal("Fill failed")
		}
	}
	Zero(dst)
	for _, v := range dst {
		if v != 0 {
			t.Fatal("Zero failed")
		}
	}
	if Sum(nil) != 0 || Max(nil) != 0 || Min(nil) != 0 {
		t.Fatal("empty reductions should return 0")
	}
	// Softmax on an empty slice must not panic.
	Softmax(nil, nil)
}

func TestAdamPanicsOnLengthMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	Adam([]float32{1}, []float32{1, 2}, []float32{0}, []float32{0}, 0.1, 0.9, 0.999, 1e-8, 1)
}
