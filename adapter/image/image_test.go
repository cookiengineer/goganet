package image

import (
	"testing"

	"github.com/cookiengineer/goganet/adapter/net"
)

func TestConfigDefaults(t *testing.T) {
	c := Config{}.WithDefaults()
	if c.Frames != 32 || c.BytesPerFrame != 1480 {
		t.Fatalf("defaults = %+v", c)
	}
	if c.Channels() != 2 {
		t.Fatalf("channels = %d, want 2", c.Channels())
	}
	if c.Len() != 32*1480*2 {
		t.Fatalf("len = %d", c.Len())
	}
	bits := Config{BitPlanes: true}.WithDefaults()
	if bits.Channels() != 10 {
		t.Fatalf("bitplane channels = %d, want 10", bits.Channels())
	}
}

func TestAtSet(t *testing.T) {
	im := New(Config{Frames: 2, BytesPerFrame: 3})
	if len(im.Data) != 2*3*2 {
		t.Fatalf("allocated %d", len(im.Data))
	}
	im.Set(1, 2, 1, 0.75)
	if got := im.At(1, 2, 1); got != 0.75 {
		t.Fatalf("At/Set = %v", got)
	}
}

func TestRenderPresenceAndPadding(t *testing.T) {
	cfg := Config{Frames: 3, BytesPerFrame: 4}
	im := Render(cfg, [][]byte{[]byte("ab"), nil, []byte("cd")})
	if im.Channels != 2 {
		t.Fatalf("channels = %d", im.Channels)
	}
	// Frame 0 present, bytes a,b then padding.
	if im.At(0, 0, 1) != 1 || im.At(0, 1, 1) != 1 || im.At(0, 2, 1) != 1 {
		t.Fatal("frame 0 presence should be 1 across the row")
	}
	if im.At(0, 0, 0) != float32('a')/255 || im.At(0, 1, 0) != float32('b')/255 {
		t.Fatal("frame 0 intensity wrong")
	}
	if im.At(0, 2, 0) != 0 {
		t.Fatal("frame 0 padding should be zero intensity")
	}
	// Frame 1 is padding.
	if im.At(1, 0, 1) != 0 {
		t.Fatal("frame 1 presence should be 0")
	}
	// Frame 2 present.
	if im.At(2, 0, 0) != float32('c')/255 {
		t.Fatal("frame 2 intensity wrong")
	}
}

func TestRenderBitPlanes(t *testing.T) {
	im := Render(Config{Frames: 1, BytesPerFrame: 1, BitPlanes: true}, [][]byte{{0b10110001}})
	if im.Channels != 10 {
		t.Fatalf("channels = %d", im.Channels)
	}
	want := []float32{1, 0, 0, 0, 1, 1, 0, 1}
	for bit := 0; bit < 8; bit++ {
		if got := im.At(0, 0, 2+bit); got != want[bit] {
			t.Fatalf("bit %d = %v, want %v", bit, got, want[bit])
		}
	}
}

func TestRenderTruncatesAtCapacity(t *testing.T) {
	cfg := Config{Frames: 1, BytesPerFrame: 2}
	im := Render(cfg, [][]byte{{1, 2, 3, 4}})
	if im.At(0, 1, 0) != 2.0/255 {
		t.Fatal("extra bytes should be truncated")
	}
}

func TestPlanar(t *testing.T) {
	im := New(Config{Frames: 2, BytesPerFrame: 2, BitPlanes: false})
	// Fill interleaved [frame][byte][channel].
	for i := range im.Data {
		im.Data[i] = float32(i)
	}
	p := im.Planar()
	// Planar layout is [channel][frame*bytes+byte].
	if len(p) != len(im.Data) {
		t.Fatalf("planar len %d", len(p))
	}
	for f := 0; f < 2; f++ {
		for b := 0; b < 2; b++ {
			for c := 0; c < 2; c++ {
				want := im.Data[(f*2+b)*2+c]
				got := p[c*2*2+f*2+b]
				if got != want {
					t.Fatalf("planar[%d,%d,%d] = %v, want %v", f, b, c, got, want)
				}
			}
		}
	}
}

func TestPlanarSingleChannelIsIdentity(t *testing.T) {
	im := New(Config{Frames: 2, BytesPerFrame: 2})
	im.Channels = 1
	im.Data = im.Data[:4]
	for i := range im.Data {
		im.Data[i] = float32(i)
	}
	p := im.Planar()
	if &p[0] != &im.Data[0] {
		t.Fatal("single-channel Planar should return Data directly")
	}
}

func TestRenderNetSelector(t *testing.T) {
	f := &net.Frame{Protocol: net.ProtoUDP, SrcPort: 1, DstPort: 53, Payload: []byte{9, 9, 9}}
	// nil selector uses the payload.
	im := RenderNet(Config{Frames: 1, BytesPerFrame: 4}, []*net.Frame{f}, nil)
	if im.At(0, 0, 0) != 9.0/255 {
		t.Fatal("nil selector should render payload")
	}
	// Custom selector.
	im2 := RenderNet(Config{Frames: 1, BytesPerFrame: 4}, []*net.Frame{f}, func(*net.Frame) []byte {
		return []byte{3}
	})
	if im2.At(0, 0, 0) != 3.0/255 {
		t.Fatal("custom selector not used")
	}
	// nil frame is padding.
	im3 := RenderNet(Config{Frames: 1, BytesPerFrame: 4}, []*net.Frame{nil}, nil)
	if im3.At(0, 0, 1) != 0 {
		t.Fatal("nil frame should be padding")
	}
}
