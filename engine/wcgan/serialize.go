package wcgan

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/cookiengineer/goganet/engine/autodiff"
)

const (
	serialMagic   = "GGNT"
	serialVersion = 1
)

// serialHeader is the JSON header written before the parameter data.
type serialHeader struct {
	Version    int
	Config     Config
	GenParams  []paramHeader
	CritParams []paramHeader
}

type paramHeader struct {
	R, C int
}

// Save writes the model to w in the GoGANet binary format.
func (m *Model) Save(w io.Writer) error {
	gen, crit := m.Params()
	hdr := serialHeader{
		Version:    serialVersion,
		Config:     m.cfg,
		GenParams:  shapeHeaders(gen),
		CritParams: shapeHeaders(crit),
	}
	body, err := json.Marshal(hdr)
	if err != nil {
		return err
	}
	bw := bufio.NewWriter(w)
	if _, err := bw.WriteString(serialMagic); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, uint32(serialVersion)); err != nil {
		return err
	}
	if err := binary.Write(bw, binary.LittleEndian, uint32(len(body))); err != nil {
		return err
	}
	if _, err := bw.Write(body); err != nil {
		return err
	}
	for _, n := range append(append([]*autodiff.Node{}, gen...), crit...) {
		s := n.Shape()
		if err := binary.Write(bw, binary.LittleEndian, int32(s.R)); err != nil {
			return err
		}
		if err := binary.Write(bw, binary.LittleEndian, int32(s.C)); err != nil {
			return err
		}
		vals := n.Value()
		buf := make([]byte, 4)
		for _, v := range vals {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
			if _, err := bw.Write(buf); err != nil {
				return err
			}
		}
	}
	return bw.Flush()
}

func shapeHeaders(nodes []*autodiff.Node) []paramHeader {
	out := make([]paramHeader, len(nodes))
	for i, n := range nodes {
		s := n.Shape()
		out[i] = paramHeader{R: s.R, C: s.C}
	}
	return out
}

// Load reads a model written by Save.
func Load(r io.Reader) (*Model, error) {
	br := bufio.NewReader(r)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(br, magic); err != nil {
		return nil, err
	}
	if string(magic) != serialMagic {
		return nil, fmt.Errorf("wcgan: bad magic %q", magic)
	}
	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != serialVersion {
		return nil, fmt.Errorf("wcgan: unsupported version %d", version)
	}
	var hdrLen uint32
	if err := binary.Read(br, binary.LittleEndian, &hdrLen); err != nil {
		return nil, err
	}
	body := make([]byte, hdrLen)
	if _, err := io.ReadFull(br, body); err != nil {
		return nil, err
	}
	var hdr serialHeader
	if err := json.Unmarshal(body, &hdr); err != nil {
		return nil, err
	}
	m := New(hdr.Config)
	gen, crit := m.Params()
	want := append(append([]paramHeader{}, hdr.GenParams...), hdr.CritParams...)
	all := append(append([]*autodiff.Node{}, gen...), crit...)
	if len(want) != len(all) {
		return nil, fmt.Errorf("wcgan: parameter count mismatch: file %d, model %d", len(want), len(all))
	}
	for i, n := range all {
		var r, c int32
		if err := binary.Read(br, binary.LittleEndian, &r); err != nil {
			return nil, err
		}
		if err := binary.Read(br, binary.LittleEndian, &c); err != nil {
			return nil, err
		}
		if int(r) != want[i].R || int(c) != want[i].C {
			return nil, fmt.Errorf("wcgan: shape mismatch at parameter %d", i)
		}
		vals := make([]float32, int(r)*int(c))
		buf := make([]byte, 4)
		for j := range vals {
			if _, err := io.ReadFull(br, buf); err != nil {
				return nil, err
			}
			vals[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf))
		}
		autodiff.SetVar(n, vals)
	}
	return m, nil
}
