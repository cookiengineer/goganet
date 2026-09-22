package pcap

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"testing"
)

// ngBuilder assembles PCAPNG blocks in a chosen byte order.
type ngBuilder struct {
	buf   bytes.Buffer
	order binary.ByteOrder
}

func newNGBuilder(order binary.ByteOrder) *ngBuilder { return &ngBuilder{order: order} }

func (b *ngBuilder) u16(v uint16) { binary.Write(&b.buf, b.order, v) }
func (b *ngBuilder) u32(v uint32) { binary.Write(&b.buf, b.order, v) }
func (b *ngBuilder) u64(v uint64) { binary.Write(&b.buf, b.order, v) }
func (b *ngBuilder) raw(p []byte) { b.buf.Write(p) }

// block writes type, body (padded to 4 bytes) and the trailing length.
func (b *ngBuilder) block(typ uint32, body []byte) {
	pad := (4 - len(body)%4) % 4
	total := uint32(12 + len(body) + pad)
	b.u32(typ)
	b.u32(total)
	b.raw(body)
	for i := 0; i < pad; i++ {
		b.buf.WriteByte(0)
	}
	b.u32(total)
}

func (b *ngBuilder) shb() {
	var body bytes.Buffer
	binary.Write(&body, b.order, uint32(pcapngByteOrderMagic))
	binary.Write(&body, b.order, uint16(1))
	binary.Write(&body, b.order, uint16(0))
	binary.Write(&body, b.order, int64(-1))
	b.block(pcapngSectionHeader, body.Bytes())
}

// idb writes an interface description with an optional if_tsresol option.
func (b *ngBuilder) idb(linkType uint32, tsresol byte, hasResol bool) {
	var body bytes.Buffer
	binary.Write(&body, b.order, uint16(linkType))
	binary.Write(&body, b.order, uint16(0))
	binary.Write(&body, b.order, uint32(65535))
	if hasResol {
		binary.Write(&body, b.order, uint16(pcapngOptionTsResol))
		binary.Write(&body, b.order, uint16(1))
		body.WriteByte(tsresol)
		body.Write([]byte{0, 0, 0})
		binary.Write(&body, b.order, uint16(pcapngOptionEndOfOpt))
		binary.Write(&body, b.order, uint16(0))
	}
	b.block(pcapngInterfaceDesc, body.Bytes())
}

func (b *ngBuilder) epb(iface uint32, ts uint64, data []byte) {
	var body bytes.Buffer
	binary.Write(&body, b.order, iface)
	binary.Write(&body, b.order, uint32(ts>>32))
	binary.Write(&body, b.order, uint32(ts))
	binary.Write(&body, b.order, uint32(len(data)))
	binary.Write(&body, b.order, uint32(len(data)))
	body.Write(data)
	b.block(pcapngEnhancedPacket, body.Bytes())
}

func (b *ngBuilder) spb(data []byte) {
	var body bytes.Buffer
	binary.Write(&body, b.order, uint32(len(data)))
	body.Write(data)
	b.block(pcapngSimplePacket, body.Bytes())
}

func (b *ngBuilder) packetBlock(iface uint16, ts uint64, data []byte) {
	var body bytes.Buffer
	binary.Write(&body, b.order, iface)
	binary.Write(&body, b.order, uint16(0))
	binary.Write(&body, b.order, uint32(ts>>32))
	binary.Write(&body, b.order, uint32(ts))
	binary.Write(&body, b.order, uint32(len(data)))
	binary.Write(&body, b.order, uint32(len(data)))
	body.Write(data)
	b.block(pcapngPacketBlock, body.Bytes())
}

func (b *ngBuilder) bytes() []byte { return b.buf.Bytes() }

func TestPCAPNGSimplePacket(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.spb([]byte("hello"))
	r, err := NewReader(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "hello" {
		t.Fatalf("spb data = %q", p.Data)
	}
}

func TestPCAPNGPacketBlockObsolete(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.packetBlock(0, 2_000_000, []byte("wxyz"))
	r, err := NewReader(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "wxyz" || p.Time.Unix() != 2 {
		t.Fatalf("obsolete packet block = %q at %v", p.Data, p.Time)
	}
}

func TestPCAPNGTimestampResolutionDecimal(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 9, true) // 10^9 ticks per second
	b.epb(0, 1_500_000_000, []byte("x"))
	r, _ := NewReader(bytes.NewReader(b.bytes()))
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if p.Time.UnixNano() != 1_500_000_000 {
		t.Fatalf("time = %v", p.Time)
	}
}

func TestPCAPNGTimestampResolutionBinary(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0x80|10, true) // 2^10 ticks per second
	b.epb(0, 1024, []byte("x"))
	r, _ := NewReader(bytes.NewReader(b.bytes()))
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if p.Time.Unix() != 1 {
		t.Fatalf("time = %v", p.Time)
	}
}

func TestPCAPNgBigEndian(t *testing.T) {
	b := newNGBuilder(binary.BigEndian)
	b.shb()
	b.idb(LinkTypeRaw, 0, false)
	b.epb(0, 1_000_000, []byte("be"))
	r, err := NewReader(bytes.NewReader(b.bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if r.LinkType() != LinkTypeRaw {
		t.Fatalf("link type = %d", r.LinkType())
	}
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if string(p.Data) != "be" {
		t.Fatalf("data = %q", p.Data)
	}
}

func TestPCAPNGMultipleInterfaces(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.idb(LinkTypeRaw, 0, false)
	b.epb(1, 1_000_000, []byte("raw"))
	r, _ := NewReader(bytes.NewReader(b.bytes()))
	p, err := r.Read()
	if err != nil {
		t.Fatal(err)
	}
	if p.Link != LinkTypeRaw {
		t.Fatalf("interface link = %d, want raw", p.Link)
	}
}

func TestPCAPNGNewSection(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.epb(0, 1_000_000, []byte("one"))
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.epb(0, 1_000_000, []byte("two"))
	r, _ := NewReader(bytes.NewReader(b.bytes()))
	for _, want := range []string{"one", "two"} {
		p, err := r.Read()
		if err != nil {
			t.Fatal(err)
		}
		if string(p.Data) != want {
			t.Fatalf("data = %q, want %q", p.Data, want)
		}
	}
}

func TestPCAPNGGzip(t *testing.T) {
	b := newNGBuilder(binary.LittleEndian)
	b.shb()
	b.idb(LinkTypeEthernet, 0, false)
	b.epb(0, 1_000_000, []byte("gz"))
	var z bytes.Buffer
	zw := gzip.NewWriter(&z)
	zw.Write(b.bytes())
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
		t.Fatalf("gzip data = %q", p.Data)
	}
}
