package classifiers

import (
	"testing"

	"github.com/cookiengineer/goganet/adapter/image"
)

// TestBundledWeightLoads verifies that the bundled model committed under
// classifiers/weights is loaded through go:embed and can classify.
func TestBundledWeightLoads(t *testing.T) {
	clf, err := New("bundled")
	if err != nil {
		t.Fatalf("loading embedded bundled weights: %v", err)
	}
	if clf.Name() != "bundled" {
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
	t.Logf("embedded %s: shape=%dx%d arch=%s p(malicious)=%.3f score=%+.3f",
		res.Protocol, shape.Frames, shape.BytesPerFrame, clf.Model().Config().Arch, res.Probability, res.Score)
}

// TestAllEmbeddedWeightsLoad loads and runs every embedded model, so a corrupt
// or stale weight file fails the build.
func TestAllEmbeddedWeightsLoad(t *testing.T) {
	names := Available()
	if len(names) == 0 {
		t.Fatal("no embedded weights found")
	}
	for _, name := range names {
		clf, err := New(name)
		if err != nil {
			t.Fatalf("embedded %s: %v", name, err)
		}
		res, err := clf.Classify(image.New(clf.Shape()))
		if err != nil {
			t.Fatalf("%s classify: %v", name, err)
		}
		t.Logf("%-24s arch=%s %dx%d p(malicious)=%.3f",
			name, clf.Model().Config().Arch, clf.Shape().Frames, clf.Shape().BytesPerFrame, res.Probability)
	}
}
