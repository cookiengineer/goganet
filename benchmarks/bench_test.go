package benchmarks

import (
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/net"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

func BenchmarkDecodeAndRender(b *testing.B) {
	frame := udpFrame("10.0.0.1", "8.8.8.8", 40000, 53, []byte("payload-bytes-here"))
	frames := make([]*net.Frame, 32)
	for i := range frames {
		f, err := net.Decode(1, frame, time.Now())
		if err != nil {
			b.Fatal(err)
		}
		frames[i] = f
	}
	cfg := image.Config{Frames: 32, BytesPerFrame: 1480}
	ad := adapter.ByName("dns")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = image.RenderNet(cfg, frames, ad.Bytes)
	}
}

func BenchmarkClassify(b *testing.B) {
	cfg := image.Config{Frames: 8, BytesPerFrame: 64}
	model := wcgan.New(wcgan.Config{
		InputDim:      cfg.Len(),
		LatentDim:     16,
		GenHidden:     []int{64},
		CritHidden:    []int{64},
		NumClasses:    2,
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
	})
	img := image.New(cfg)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Classify(img.Data, 1)
	}
}

func BenchmarkCriticStep(b *testing.B) {
	cfg := image.Config{Frames: 4, BytesPerFrame: 32}
	model := wcgan.New(wcgan.Config{
		InputDim:      cfg.Len(),
		LatentDim:     8,
		GenHidden:     []int{32},
		CritHidden:    []int{32},
		NumClasses:    2,
		NCritic:       1,
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
	})
	x := make([]float32, cfg.Len()*4)
	y := wcgan.OneHot([]int{0, 1, 0, 1}, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = model.CriticStep(x, y, 4)
	}
}

func BenchmarkConvClassify(b *testing.B) {
	cfg := image.Config{Frames: 8, BytesPerFrame: 64}
	model := wcgan.New(wcgan.Config{
		Arch:          "conv",
		InputDim:      cfg.Len(),
		InputChannels: cfg.Channels(),
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
		SeedChannels:  16,
		GenChannels:   []int{8, 4},
		CritChannels:  []int{8, 16},
		LatentDim:     16,
		NumClasses:    2,
	})
	img := image.New(cfg)
	planar := img.Planar()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = model.Classify(planar, 1)
	}
}
