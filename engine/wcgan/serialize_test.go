package wcgan

import (
	"bytes"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	m := New(Config{InputDim: 10, LatentDim: 4, GenHidden: []int{6}, CritHidden: []int{6}, NumClasses: 2, Seed: 3})
	x, labels := makeData(10, 8, 11)
	batch := 8
	xb := x[:10*batch]
	cl := OneHot(labels[:batch], 2)
	m.CriticStep(xb, cl, batch)

	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatal(err)
	}

	probsA, scoresA := m.Classify(x, 16)
	probsB, scoresB := loaded.Classify(x, 16)
	for i := range probsA {
		if probsA[i] != probsB[i] {
			t.Fatalf("probability mismatch at %d: %v != %v", i, probsA[i], probsB[i])
		}
	}
	for i := range scoresA {
		if scoresA[i] != scoresB[i] {
			t.Fatalf("score mismatch at %d: %v != %v", i, scoresA[i], scoresB[i])
		}
	}
}
