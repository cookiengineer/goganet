package pcap

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestPCAPNGBadByteOrderMagic(t *testing.T) {
	var b bytes.Buffer
	le := binary.LittleEndian
	binary.Write(&b, le, uint32(pcapngSectionHeader))
	binary.Write(&b, le, uint32(28))
	binary.Write(&b, le, uint32(0xDEADBEEF)) // wrong byte-order magic
	binary.Write(&b, le, uint16(1))
	binary.Write(&b, le, uint16(0))
	binary.Write(&b, le, int64(-1))
	binary.Write(&b, le, uint32(28))
	if _, err := NewReader(bytes.NewReader(b.Bytes())); err == nil {
		t.Fatal("expected bad byte order magic error")
	}
}

func TestPCAPNGShortSectionHeader(t *testing.T) {
	var b bytes.Buffer
	le := binary.LittleEndian
	binary.Write(&b, le, uint32(pcapngSectionHeader))
	binary.Write(&b, le, uint32(10)) // too short
	binary.Write(&b, le, uint32(pcapngByteOrderMagic))
	binary.Write(&b, le, uint32(0))
	if _, err := NewReader(bytes.NewReader(b.Bytes())); err == nil {
		t.Fatal("expected short section header error")
	}
}

func TestPCAPNGInvalidBlockLength(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	// A block declaring a total length below the 12-byte minimum.
	b.u32(0xFFFFFFFF)
	b.u32(4)
	if _, err := NewReader(bytes.NewReader(b.bytes())); err == nil {
		t.Fatal("expected invalid block length error")
	}
}

func TestPCAPNGShortEnhancedPacket(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.block(pcapngEnhancedPacket, make([]byte, 8)) // body too short for EPB
	r, err := NewReader(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err == nil {
		t.Fatal("expected short enhanced packet error")
	}
}

func TestPCAPNGShortSimplePacket(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.block(pcapngSimplePacket, nil) // empty body
	r, err := NewReader(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err == nil {
		t.Fatal("expected short simple packet error")
	}
}
