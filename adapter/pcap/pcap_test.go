package pcap

import (
	"bytes"
	"encoding/binary"
	"io"
	"testing"
)

func buildClassic() []byte {
	var b bytes.Buffer
	le := binary.LittleEndian
	w := func(v uint32) { binary.Write(&b, le, v) }
	wh := func(v uint16) { binary.Write(&b, le, v) }

	w(0xa1b2c3d4) // magic
	wh(2)
	wh(4)
	w(0) // thiszone
	w(0) // sigfigs
	w(65535)
	w(1) // ethernet

	// packet 1
	w(100) // sec
	w(500) // usec
	w(4)   // incl
	w(4)   // orig
	b.WriteString("abcd")
	// packet 2
	w(101)
	w(0)
	w(2)
	w(2)
	b.WriteString("hi")
	return b.Bytes()
}

func TestClassicReader(t *testing.T) {
	r, err := NewReader(bytes.NewReader(buildClassic()))
	if err != nil {
		t.Fatal(err)
	}
	if r.LinkType() != LinkTypeEthernet {
		t.Fatalf("link type %d", r.LinkType())
	}
	p1, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p1.Data) != "abcd" || p1.OrigLen != 4 {
		t.Fatalf("packet 1 = %q orig=%d", p1.Data, p1.OrigLen)
	}
	if p1.Time.Unix() != 100 {
		t.Fatalf("packet 1 time %v", p1.Time)
	}
	p2, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p2.Data) != "hi" {
		t.Fatalf("packet 2 = %q", p2.Data)
	}
	if _, err := r.Read(); err != io.EOF {
		t.Fatalf("expected EOF, got %v", err)
	}
}

func buildPCAPNG() []byte {
	var b bytes.Buffer
	le := binary.LittleEndian
	w := func(v uint32) { binary.Write(&b, le, v) }
	wh := func(v uint16) { binary.Write(&b, le, v) }
	wl := func(v uint64) { binary.Write(&b, le, v) }

	// SHB
	w(pcapngSectionHeader)
	w(28)
	w(pcapngByteOrderMagic)
	wh(1)
	wh(0)
	wl(0xffffffffffffffff)
	w(28)

	// IDB
	w(pcapngInterfaceDesc)
	w(20)
	wh(1) // link type ethernet
	wh(0)
	w(65535)
	w(20)

	// EPB
	w(pcapngEnhancedPacket)
	w(36)
	w(0)       // interface id
	w(0)       // ts high
	w(1000000) // ts low -> 1 second at 1e6
	w(4)       // caplen
	w(4)       // origlen
	b.WriteString("wxyz")
	w(36)

	return b.Bytes()
}

func TestPCAPNGReader(t *testing.T) {
	r, err := NewReader(bytes.NewReader(buildPCAPNG()))
	if err != nil {
		t.Fatal(err)
	}
	if r.LinkType() != LinkTypeEthernet {
		t.Fatalf("link type %d", r.LinkType())
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "wxyz" {
		t.Fatalf("data = %q", p.Data)
	}
	if p.Time.Unix() != 1 {
		t.Fatalf("time = %v", p.Time)
	}
}
