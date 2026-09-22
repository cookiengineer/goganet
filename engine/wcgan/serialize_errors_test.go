package wcgan

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func savedModel(t *testing.T) []byte {
	t.Helper()
	m := New(Config{InputDim: 8, LatentDim: 4, GenHidden: []int{8}, CritHidden: []int{8}, NumClasses: 2, Seed: 5})
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestLoadErrors(t *testing.T) {
	data := savedModel(t)

	if _, err := Load(bytes.NewReader(nil)); err == nil {
		t.Fatal("expected error on empty input")
	}

	badMagic := append([]byte(nil), data...)
	badMagic[0] = 'X'
	if _, err := Load(bytes.NewReader(badMagic)); err == nil {
		t.Fatal("expected bad magic error")
	}

	badVersion := append([]byte(nil), data...)
	binary.LittleEndian.PutUint32(badVersion[4:8], 99)
	if _, err := Load(bytes.NewReader(badVersion)); err == nil {
		t.Fatal("expected bad version error")
	}

	truncated := data[:len(data)/2]
	if _, err := Load(bytes.NewReader(truncated)); err == nil {
		t.Fatal("expected truncated error")
	}
}

func TestLoadRejectsShapeMismatch(t *testing.T) {
	// A model saved with one hidden width must not load into a differently
	// shaped architecture: the header declares parameter shapes.
	m := New(Config{InputDim: 8, LatentDim: 4, GenHidden: []int{8}, CritHidden: []int{8}, NumClasses: 2})
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	// Corrupt the very first parameter's declared row count in the binary
	// payload by flipping a byte after the JSON header.
	raw := buf.Bytes()
	// magic(4) + version(4) + hdrLen(4) + header
	hdrLen := binary.LittleEndian.Uint32(raw[8:12])
	paramStart := 12 + int(hdrLen)
	raw[paramStart] ^= 0xff
	if _, err := Load(bytes.NewReader(raw)); err == nil {
		t.Fatal("expected parameter shape mismatch error")
	}
}

func TestParamsStableOrder(t *testing.T) {
	m := New(Config{InputDim: 8, LatentDim: 4, GenHidden: []int{8}, CritHidden: []int{8}, NumClasses: 2})
	gen1, crit1 := m.Params()
	gen2, crit2 := m.Params()
	if len(gen1) == 0 || len(crit1) == 0 {
		t.Fatal("expected parameters")
	}
	if len(gen1) != len(gen2) || len(crit1) != len(crit2) {
		t.Fatal("parameter counts not stable")
	}
	for i := range gen1 {
		if gen1[i] != gen2[i] {
			t.Fatal("generator parameter order not stable")
		}
	}
	for i := range crit1 {
		if crit1[i] != crit2[i] {
			t.Fatal("critic parameter order not stable")
		}
	}
}
