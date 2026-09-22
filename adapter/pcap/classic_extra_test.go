package pcap

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func buildClassicOrdered(order binary.ByteOrder, magic uint32, link uint32, packets [][]byte) []byte {
	var b bytes.Buffer
	w := func(v uint32) { binary.Write(&b, order, v) }
	wh := func(v uint16) { binary.Write(&b, order, v) }
	w(magic)
	wh(2)
	wh(4)
	w(0)
	w(0)
	w(65535)
	w(link)
	for i, p := range packets {
		w(uint32(100 + i))
		w(uint32(10))
		w(uint32(len(p)))
		w(uint32(len(p)))
		b.Write(p)
	}
	return b.Bytes()
}

func TestClassicBigEndian(t *testing.T) {
	data := buildClassicOrdered(binary.BigEndian, 0xa1b2c3d4, LinkTypeRaw, [][]byte{[]byte("be")})
	r, err := NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if r.LinkType() != LinkTypeRaw {
		t.Fatalf("link = %d", r.LinkType())
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "be" {
		t.Fatalf("data = %q", p.Data)
	}
}

func TestClassicNanosecondMagic(t *testing.T) {
	data := buildClassicOrdered(binary.LittleEndian, 0xa1b23c4d, LinkTypeEthernet, [][]byte{nil})
	r, err := NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if p.Time.Nanosecond() != 10 {
		t.Fatalf("nanos = %d", p.Time.Nanosecond())
	}
}

func TestClassicGzip(t *testing.T) {
	data := buildClassicOrdered(binary.LittleEndian, 0xa1b2c3d4, LinkTypeEthernet, [][]byte{[]byte("gz")})
	var z bytes.Buffer
	zw := gzip.NewWriter(&z)
	zw.Write(data)
	zw.Close()
	r, err := NewReader(bytes.NewReader(z.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "gz" {
		t.Fatalf("data = %q", p.Data)
	}
}

func TestClassicTruncatedPacket(t *testing.T) {
	data := buildClassicOrdered(binary.LittleEndian, 0xa1b2c3d4, LinkTypeEthernet, [][]byte{[]byte("abcd")})
	// Chop off all but the first two bytes of the packet payload.
	data = data[:len(data)-2]
	r, err := NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err == nil {
		t.Fatal("expected error for truncated packet data")
	}
}

func TestClassicBadMagic(t *testing.T) {
	if _, err := NewReader(bytes.NewReader([]byte{1, 2, 3, 4, 5, 6, 7, 8})); err == nil {
		t.Fatal("expected bad magic error")
	}
}

func TestOpenFileAndClose(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.pcap")
	data := buildClassicOrdered(binary.LittleEndian, 0xa1b2c3d4, LinkTypeEthernet, [][]byte{[]byte("f")})
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Read(); err == nil {
		t.Fatal("read after close should fail")
	}
}
