package classifiers

import (
	"math"
	"testing"
)

func approxEq(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-6 }

func TestEvaluateConfusion(t *testing.T) {
	probs := []float32{0.9, 0.6, 0.4, 0.1}
	labels := []int{1, 0, 1, 0}
	m := Evaluate(probs, labels, 0.5)
	if m.Confusion.TP != 1 || m.Confusion.FP != 1 || m.Confusion.TN != 1 || m.Confusion.FN != 1 {
		t.Fatalf("confusion = %+v", m.Confusion)
	}
	if !approxEq(m.Accuracy, 0.5) || !approxEq(m.Precision, 0.5) || !approxEq(m.Recall, 0.5) || !approxEq(m.Specificity, 0.5) {
		t.Fatalf("metrics = %+v", m)
	}
	if !approxEq(m.F1, 0.5) || !approxEq(m.BalancedAccuracy, 0.5) {
		t.Fatalf("f1/bal = %+v", m)
	}
	if !approxEq(m.AUC, 0.75) {
		t.Fatalf("auc = %v, want 0.75", m.AUC)
	}
}

func TestEvaluatePerfect(t *testing.T) {
	m := Evaluate([]float32{0.9, 0.8, 0.2, 0.1}, []int{1, 1, 0, 0}, 0.5)
	if !approxEq(m.Accuracy, 1) || !approxEq(m.F1, 1) || !approxEq(m.AUC, 1) || !approxEq(m.Specificity, 1) {
		t.Fatalf("perfect metrics = %+v", m)
	}
}

func TestAUC(t *testing.T) {
	if got := AUC([]float32{0.9, 0.8, 0.2, 0.1}, []int{1, 1, 0, 0}); !approxEq(got, 1) {
		t.Fatalf("perfect auc = %v", got)
	}
	if got := AUC([]float32{0.1, 0.2, 0.8, 0.9}, []int{1, 1, 0, 0}); !approxEq(got, 0) {
		t.Fatalf("reversed auc = %v", got)
	}
	// All scores equal: no discrimination.
	if got := AUC([]float32{0.5, 0.5, 0.5, 0.5}, []int{1, 0, 1, 0}); !approxEq(got, 0.5) {
		t.Fatalf("tied auc = %v", got)
	}
	// Single class present: undefined, reports 0.5.
	if got := AUC([]float32{0.1, 0.9}, []int{1, 1}); !approxEq(got, 0.5) {
		t.Fatalf("single-class auc = %v", got)
	}
}

func TestBestThreshold(t *testing.T) {
	probs := []float32{0.9, 0.8, 0.7, 0.6}
	labels := []int{1, 1, 0, 0}
	m := BestThreshold(probs, labels)
	if !approxEq(m.F1, 1) {
		t.Fatalf("best f1 = %v", m.F1)
	}
	if m.Threshold < 0.7 || m.Threshold >= 0.8 {
		t.Fatalf("threshold = %v", m.Threshold)
	}
	if !approxEq(m.Precision, 1) || !approxEq(m.Recall, 1) {
		t.Fatalf("best metrics = %+v", m)
	}
}

func TestEvaluateEmpty(t *testing.T) {
	m := Evaluate(nil, nil, 0.5)
	if m.N != 0 || m.AUC != 0.5 {
		t.Fatalf("empty metrics = %+v", m)
	}
}
