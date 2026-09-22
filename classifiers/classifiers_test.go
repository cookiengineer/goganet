package classifiers

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

func mustOpen(t *testing.T, path string) *os.File {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}

func TestWrapAndClassify(t *testing.T) {
	cfg := image.Config{Frames: 4, BytesPerFrame: 16}
	model := wcgan.New(wcgan.Config{
		InputDim:      cfg.Len(),
		LatentDim:     8,
		GenHidden:     []int{16},
		CritHidden:    []int{16},
		NumClasses:    2,
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
	})
	clf := Wrap("dns", model, cfg)
	img := image.New(cfg)
	for i := range img.Data {
		img.Data[i] = float32(i%7) / 7
	}
	res, err := clf.Classify(img)
	if err != nil {
		t.Fatal(err)
	}
	if res.Probability < 0 || res.Probability > 1 {
		t.Fatalf("probability out of range: %v", res.Probability)
	}

	// Round-trip through the weight format.
	path := filepath.Join(t.TempDir(), "dns.ggnt")
	if err := Save(path, model); err != nil {
		t.Fatal(err)
	}
	loaded, err := wcgan.Load(mustOpen(t, path))
	if err != nil {
		t.Fatal(err)
	}
	clf2 := Wrap("dns", loaded, cfg)
	res2, err := clf2.Classify(img)
	if err != nil {
		t.Fatal(err)
	}
	if res2.Probability != res.Probability {
		t.Fatalf("probability changed after reload: %v != %v", res2.Probability, res.Probability)
	}
}

func TestNewWithoutWeights(t *testing.T) {
	_, err := New("dns")
	if !errors.Is(err, ErrUntrained) {
		t.Fatalf("expected ErrUntrained, got %v", err)
	}
}

func TestClassifyShapeMismatch(t *testing.T) {
	cfg := image.Config{Frames: 4, BytesPerFrame: 16}
	model := wcgan.New(wcgan.Config{InputDim: cfg.Len(), LatentDim: 4, GenHidden: []int{8}, CritHidden: []int{8}, NumClasses: 2})
	clf := Wrap("dns", model, cfg)
	wrong := image.New(image.Config{Frames: 3, BytesPerFrame: 16})
	if _, err := clf.Classify(wrong); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}
