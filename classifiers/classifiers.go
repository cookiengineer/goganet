// Package classifiers exposes ready-to-use malicious/non-malicious traffic
// classifiers built on the WGAN-GP engine. A classifier wraps a trained model
// together with the image geometry it expects. Models can be embedded at build
// time from the weights directory or wrapped at run time.
package classifiers

import (
	"bytes"
	"embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

//go:embed weights/*.ggnt
var weightsFS embed.FS

// ErrUntrained is returned when no embedded weights exist for a protocol.
var ErrUntrained = errors.New("classifiers: no embedded weights for protocol")

// selftestName is the synthetic fixture used by the embed test. It is hidden
// from the public protocol list.
const selftestName = "selftest"

// Result is the outcome of classifying one session image.
type Result struct {
	Protocol    string
	Malicious   bool
	Probability float32 // P(malicious)
	Score       float32 // critic realness / anomaly score
	Label       string  // optional human-readable label
}

// Classifier classifies rendered session images.
type Classifier struct {
	name  string
	model *wcgan.Model
	shape image.Config
}

// Wrap builds a classifier around an in-memory model.
func Wrap(protocol string, model *wcgan.Model, shape image.Config) *Classifier {
	return &Classifier{name: protocol, model: model, shape: shape.WithDefaults()}
}

// New loads the embedded weights for the protocol and returns a classifier.
func New(protocol string) (*Classifier, error) {
	name := strings.ToLower(protocol)
	data, err := weightsFS.ReadFile("weights/" + name + ".ggnt")
	if err != nil {
		return nil, fmt.Errorf("%w: %s", ErrUntrained, name)
	}
	model, err := wcgan.Load(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("classifiers: loading %s: %w", name, err)
	}
	cfg := model.Config()
	shape := image.Config{Frames: cfg.Frames, BytesPerFrame: cfg.BytesPerFrame, BitPlanes: cfg.BitPlanes}.WithDefaults()
	return &Classifier{name: name, model: model, shape: shape}, nil
}

// Available lists the protocols with embedded weights.
func Available() []string {
	entries, err := weightsFS.ReadDir("weights")
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".ggnt") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".ggnt")
		if name == selftestName {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Name returns the protocol name.
func (c *Classifier) Name() string { return c.name }

// Model returns the underlying model.
func (c *Classifier) Model() *wcgan.Model { return c.model }

// Shape returns the expected image geometry.
func (c *Classifier) Shape() image.Config { return c.shape }

// Classify renders must already be produced by the caller; img.Data must match
// the model input length. Class index 1 is treated as malicious.
func (c *Classifier) Classify(img *image.Image) (Result, error) {
	if img == nil {
		return Result{}, errors.New("classifiers: nil image")
	}
	data := img.Data
	if c.model.Config().Arch == "conv" {
		data = img.Planar()
	}
	if len(data) != c.model.Config().InputDim {
		return Result{}, fmt.Errorf("classifiers: image length %d does not match model input %d", len(data), c.model.Config().InputDim)
	}
	probs, scores := c.model.Classify(data, 1)
	r := Result{Protocol: c.name}
	if len(probs) >= 2 {
		r.Probability = probs[1]
		r.Malicious = probs[1] >= 0.5
	}
	if len(scores) >= 1 {
		r.Score = scores[0]
	}
	return r, nil
}

// Save writes a trained model to disk for later embedding.
func Save(path string, model *wcgan.Model) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return model.Save(f)
}
