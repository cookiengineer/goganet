// Package image renders a window of up to Frames network frames into a
// multi-channel image suitable for the WGAN engine. Each frame occupies one row
// of BytesPerFrame pixels; frames that are absent are zero padded and flagged by
// the presence channel so that padding cannot be mistaken for real zero bytes.
package image

import "github.com/cookiengineer/goganet/adapter/net"

// Config controls the image geometry.
type Config struct {
	Frames        int  // number of frame rows (default 32)
	BytesPerFrame int  // bytes per row (default 1480)
	BitPlanes     bool // if true, add 8 bit-plane channels
}

// WithDefaults fills unset fields.
func (c Config) WithDefaults() Config {
	if c.Frames <= 0 {
		c.Frames = 32
	}
	if c.BytesPerFrame <= 0 {
		c.BytesPerFrame = 1480
	}
	return c
}

// Channels returns the number of channels produced by the config.
func (c Config) Channels() int {
	n := 2 // byte intensity + presence
	if c.BitPlanes {
		n += 8
	}
	return n
}

// Len returns the total number of float32 values in an image.
func (c Config) Len() int {
	c = c.WithDefaults()
	return c.Frames * c.BytesPerFrame * c.Channels()
}

// Image is a dense multi-channel frame window. The data is stored row-major as
// [frame][byte][channel].
type Image struct {
	Frames   int
	Bytes    int
	Channels int
	Data     []float32
}

// New allocates an empty image for the config.
func New(cfg Config) *Image {
	cfg = cfg.WithDefaults()
	return &Image{
		Frames:   cfg.Frames,
		Bytes:    cfg.BytesPerFrame,
		Channels: cfg.Channels(),
		Data:     make([]float32, cfg.Frames*cfg.BytesPerFrame*cfg.Channels()),
	}
}

// At returns the value at (frame, byte, channel).
func (im *Image) At(f, b, c int) float32 {
	return im.Data[(f*im.Bytes+b)*im.Channels+c]
}

// Set stores a value at (frame, byte, channel).
func (im *Image) Set(f, b, c int, v float32) {
	im.Data[(f*im.Bytes+b)*im.Channels+c] = v
}

// Render writes frames into an image. nil or missing frames are zero padded.
// The bytePlane callback may return a nil slice to render padding for that row.
func Render(cfg Config, frames [][]byte) *Image {
	cfg = cfg.WithDefaults()
	im := New(cfg)
	bits := cfg.BitPlanes
	for f := 0; f < cfg.Frames; f++ {
		present := f < len(frames) && frames[f] != nil
		for b := 0; b < cfg.BytesPerFrame; b++ {
			base := (f*cfg.BytesPerFrame + b) * im.Channels
			if present && b < len(frames[f]) {
				v := frames[f][b]
				im.Data[base] = float32(v) / 255
				if bits {
					for k := 0; k < 8; k++ {
						im.Data[base+2+k] = float32((v >> uint(k)) & 1)
					}
				}
			} else if bits {
				// padding planes stay zero
			}
			if present {
				im.Data[base+1] = 1
			}
		}
	}
	return im
}

// Planar returns the image data in planar layout (Channels, Frames*Bytes),
// which is what the convolutional engine consumes. For a single channel the
// result is identical to Data.
func (im *Image) Planar() []float32 {
	if im.Channels == 1 {
		return im.Data
	}
	out := make([]float32, len(im.Data))
	for f := 0; f < im.Frames; f++ {
		for b := 0; b < im.Bytes; b++ {
			for c := 0; c < im.Channels; c++ {
				out[c*im.Frames*im.Bytes+f*im.Bytes+b] = im.Data[(f*im.Bytes+b)*im.Channels+c]
			}
		}
	}
	return out
}

// RenderNet converts decoded frames to byte slices using the supplied adapter
// byte selector and renders the image. A nil selector uses the transport
// payload.
func RenderNet(cfg Config, frames []*net.Frame, bytesOf func(*net.Frame) []byte) *Image {
	rows := make([][]byte, len(frames))
	for i, f := range frames {
		if f == nil {
			continue
		}
		if bytesOf != nil {
			rows[i] = bytesOf(f)
		} else {
			rows[i] = f.Payload
		}
	}
	return Render(cfg, rows)
}
