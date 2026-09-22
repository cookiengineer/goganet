package classifiers

import (
	"testing"

	"github.com/cookiengineer/goganet/adapter/image"
)

// TestEmbeddedWeights verifies that a pretrained model committed under
// classifiers/weights is loaded through go:embed and can classify. The
// selftest fixture is produced by examples/make-weights and is deliberately
// excluded from the public protocol list.
func TestEmbeddedWeights(t *testing.T) {
	clf, err := New(selftestName)
	if err != nil {
		t.Fatalf("loading embedded weights: %v", err)
	}
	if clf.Name() != selftestName {
		t.Fatalf("name = %q", clf.Name())
	}

	shape := clf.Shape()
	img := image.New(shape)
	// Fill with a plausible signal; the point is to exercise the embedded
	// model, not to assert a specific classification.
	for i := range img.Data {
		img.Data[i] = float32(i%11) / 11
	}
	res, err := clf.Classify(img)
	if err != nil {
		t.Fatalf("classify: %v", err)
	}
	if res.Probability < 0 || res.Probability > 1 {
		t.Fatalf("probability out of range: %v", res.Probability)
	}
	t.Logf("embedded %s: p(malicious)=%.3f score=%+.3f", res.Protocol, res.Probability, res.Score)

	// The synthetic fixture must not appear as a real protocol.
	for _, name := range Available() {
		if name == selftestName {
			t.Fatalf("selftest should not be listed in Available()")
		}
	}

	// The committed HTTP/1 model (trained on CTU-13) must load and classify.
	http1, err := New("http1")
	if err != nil {
		t.Fatalf("embedded http1 weights: %v", err)
	}
	img2 := image.New(http1.Shape())
	res2, err := http1.Classify(img2)
	if err != nil {
		t.Fatalf("http1 classify: %v", err)
	}
	t.Logf("embedded http1: shape=%dx%d p(malicious)=%.3f score=%+.3f",
		http1.Shape().Frames, http1.Shape().BytesPerFrame, res2.Probability, res2.Score)
}
