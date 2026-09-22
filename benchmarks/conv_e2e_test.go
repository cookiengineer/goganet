package benchmarks

import (
	"bytes"
	"math"
	"testing"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/pipeline"
	"github.com/cookiengineer/goganet/adapter/session"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

func TestConvEndToEnd(t *testing.T) {
	dir := t.TempDir()
	buildDataset(t, dir)

	capture, _ := pipeline.FindCapture(dir)
	sessions, err := pipeline.ReadSessions(capture, 8)
	if err != nil {
		t.Fatal(err)
	}
	bf, _ := pipeline.FindBinetflow(dir)
	labels, _ := session.LoadBinetflow(bf)
	session.JoinLabels(sessions, session.BuildLabelMap(labels))

	cfg := image.Config{Frames: 4, BytesPerFrame: 64}
	planar := image.Config{Frames: 4, BytesPerFrame: 64}.Channels()
	x, y := pipeline.DatasetPlanar(sessions, "dns", cfg)
	if len(x) != 4 {
		t.Fatalf("expected 4 planar samples, got %d", len(x))
	}
	if len(x[0]) != cfg.Len() {
		t.Fatalf("planar length %d, want %d", len(x[0]), cfg.Len())
	}

	model := wcgan.New(wcgan.Config{
		Arch:          "conv",
		InputDim:      cfg.Len(),
		InputChannels: planar,
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
		SeedChannels:  16,
		GenChannels:   []int{8, 4},
		CritChannels:  []int{8, 16},
		Kernel:        3,
		LatentDim:     8,
		NumClasses:    2,
		NCritic:       1,
		Lambda:        1,
		GenLR:         5e-4,
		CritLR:        5e-4,
	})
	batch := len(x)
	for epoch := 0; epoch < 400; epoch++ {
		dLoss := model.CriticStep(flatten(x, nil), wcgan.OneHot(y, 2), batch)
		gLoss := model.GeneratorStep(batch)
		if math.IsNaN(float64(dLoss)) || math.IsNaN(float64(gLoss)) {
			t.Fatalf("NaN loss at epoch %d", epoch)
		}
	}

	correct := 0
	for b := 0; b < batch; b++ {
		probs, _ := model.Classify(x[b], 1)
		pred := 0
		if probs[1] > 0.5 {
			pred = 1
		}
		if pred == y[b] {
			correct++
		}
	}
	if acc := float64(correct) / float64(batch); acc < 0.75 {
		t.Fatalf("conv end-to-end accuracy too low: %.2f", acc)
	}

	var buf bytes.Buffer
	if err := model.Save(&buf); err != nil {
		t.Fatal(err)
	}
	if _, err := wcgan.Load(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
}
