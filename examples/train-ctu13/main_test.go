package main

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
)

func TestOversampleBalancesAndTerminates(t *testing.T) {
	y := []int{1, 0, 0, 0}
	idx := []int{0, 1, 2, 3}
	out := oversample(y, idx, rand.New(rand.NewPCG(1, 2)))
	var pos, neg int
	for _, i := range out {
		if y[i] == 1 {
			pos++
		} else {
			neg++
		}
	}
	if pos != neg || pos != 3 {
		t.Fatalf("balanced counts = %d/%d, want 3/3", pos, neg)
	}
}

func TestOversampleSingleClass(t *testing.T) {
	y := []int{1, 1, 1}
	out := oversample(y, []int{0, 1, 2}, rand.New(rand.NewPCG(1, 2)))
	if len(out) != 3 {
		t.Fatalf("single class should be unchanged, got %d", len(out))
	}
}

func TestBatchMatrixLayout(t *testing.T) {
	x := [][]float32{{1, 2}, {3, 4}}
	got := batchMatrix(x, []int{0, 1})
	want := []float32{1, 3, 2, 4}
	if len(got) != len(want) {
		t.Fatalf("len = %d", len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("batchMatrix[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestScenarioNumber(t *testing.T) {
	if scenarioNumber("datasets/ctu-13/12") != 12 {
		t.Fatal("scenarioNumber failed")
	}
	if scenarioNumber("not-a-number") < 1<<29 {
		t.Fatal("non-numeric should sort last")
	}
}

func TestDiscoverScenarios(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"4", "10", "empty"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "4", "a.pcap"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "10", "b.pcap"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := discoverScenarios(root)
	if len(got) != 2 {
		t.Fatalf("scenarios = %v", got)
	}
	// A directory that itself contains a capture is returned directly.
	if got := discoverScenarios(filepath.Join(root, "4")); len(got) != 1 {
		t.Fatalf("direct scenario = %v", got)
	}
}
