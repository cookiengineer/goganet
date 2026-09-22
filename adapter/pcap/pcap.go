// Package pcap implements readers for the classic libpcap capture format and
// the newer PCAPNG format using only the Go standard library. It supports
// transparent gzip decompression so that compressed captures can be read
// directly.
//
// The package is deliberately independent of any third-party packet library:
// it exposes raw link-layer frames and their capture timestamps, and leaves
// protocol decoding to the adapter packages.
package pcap

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"time"
)

// Link types from the libpcap registry. Only the common ones are named.
const (
	LinkTypeNull      uint32 = 0
	LinkTypeEthernet  uint32 = 1
	LinkTypeRaw       uint32 = 101
	LinkTypeLinuxSLL  uint32 = 113
	LinkTypeLinuxSLL2 uint32 = 276
)

// Packet is a single captured frame.
type Packet struct {
	Time    time.Time // capture timestamp
	Link    uint32    // link-layer type of the containing capture
	Data    []byte    // link-layer frame bytes
	OrigLen uint32    // original on-wire length
}

// Reader reads packets sequentially from a capture.
type Reader struct {
	src      source
	linkType uint32
	closed   bool
	f        *os.File
}

type source interface {
	next() (*Packet, error)
}

var (
	errBadMagic = errors.New("pcap: unrecognized capture magic")
)

// Open opens a capture file. Gzip-compressed files are detected and
// decompressed transparently.
func Open(path string) (*Reader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	r, err := newReader(f)
	if err != nil {
		f.Close()
		return nil, err
	}
	r.f = f
	return r, nil
}

// NewReader wraps an existing stream.
func NewReader(rd io.Reader) (*Reader, error) {
	return newReader(rd)
}

func newReader(rd io.Reader) (*Reader, error) {
	br := bufio.NewReader(rd)
	magic, err := br.Peek(4)
	if err != nil {
		return nil, fmt.Errorf("pcap: reading magic: %w", err)
	}
	if magic[0] == 0x1f && magic[1] == 0x8b {
		gz, err := gzip.NewReader(br)
		if err != nil {
			return nil, fmt.Errorf("pcap: gzip: %w", err)
		}
		br = bufio.NewReader(gz)
		if magic, err = br.Peek(4); err != nil {
			return nil, fmt.Errorf("pcap: reading magic after gzip: %w", err)
		}
	}

	if isPCAPNG(magic) {
		s, err := newPcapngSource(br)
		if err != nil {
			return nil, err
		}
		return &Reader{src: s, linkType: s.linkType}, nil
	}

	s, err := newClassicSource(br)
	if err != nil {
		return nil, err
	}
	return &Reader{src: s, linkType: s.linkType}, nil
}

// LinkType returns the link-layer type of the capture.
func (r *Reader) LinkType() uint32 { return r.linkType }

// Read returns the next packet, or io.EOF when the capture is exhausted.
func (r *Reader) Read() (*Packet, error) {
	if r.closed {
		return nil, errors.New("pcap: reader closed")
	}
	return r.src.next()
}

// Close closes the underlying file if the reader was created with Open.
func (r *Reader) Close() error {
	r.closed = true
	if r.f != nil {
		return r.f.Close()
	}
	return nil
}

// classicSource parses the original libpcap format.
type classicSource struct {
	br       *bufio.Reader
	order    binary.ByteOrder
	nanos    bool
	linkType uint32
}

func newClassicSource(br *bufio.Reader) (*classicSource, error) {
	var hdr [4]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return nil, fmt.Errorf("pcap: reading magic: %w", err)
	}
	le := binary.LittleEndian.Uint32(hdr[:])
	be := binary.BigEndian.Uint32(hdr[:])

	var order binary.ByteOrder
	var nanos bool
	switch {
	case le == 0xa1b2c3d4:
		order, nanos = binary.LittleEndian, false
	case be == 0xa1b2c3d4:
		order, nanos = binary.BigEndian, false
	case le == 0xa1b23c4d:
		order, nanos = binary.LittleEndian, true
	case be == 0xa1b23c4d:
		order, nanos = binary.BigEndian, true
	default:
		return nil, errBadMagic
	}

	var rest [20]byte
	if _, err := io.ReadFull(br, rest[:]); err != nil {
		return nil, fmt.Errorf("pcap: reading global header: %w", err)
	}
	linkType := order.Uint32(rest[16:20])
	return &classicSource{br: br, order: order, nanos: nanos, linkType: linkType}, nil
}

func (s *classicSource) next() (*Packet, error) {
	var hdr [16]byte
	if _, err := io.ReadFull(s.br, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, io.EOF
		}
		return nil, fmt.Errorf("pcap: reading packet header: %w", err)
	}
	sec := s.order.Uint32(hdr[0:4])
	frac := s.order.Uint32(hdr[4:8])
	incl := s.order.Uint32(hdr[8:12])
	orig := s.order.Uint32(hdr[12:16])

	data := make([]byte, incl)
	if _, err := io.ReadFull(s.br, data); err != nil {
		return nil, fmt.Errorf("pcap: reading packet data: %w", err)
	}

	var t time.Time
	if s.nanos {
		t = time.Unix(int64(sec), int64(frac))
	} else {
		t = time.Unix(int64(sec), int64(frac)*1000)
	}
	return &Packet{Time: t, Link: s.linkType, Data: data, OrigLen: orig}, nil
}
