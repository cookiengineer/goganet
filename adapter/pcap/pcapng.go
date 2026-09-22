package pcap

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"time"
)

const (
	pcapngSectionHeader   = 0x0A0D0D0A
	pcapngInterfaceDesc   = 0x00000001
	pcapngPacketBlock     = 0x00000002
	pcapngSimplePacket    = 0x00000003
	pcapngEnhancedPacket  = 0x00000006
	pcapngByteOrderMagic  = 0x1A2B3C4D
	pcapngOptionTsResol   = 9
	pcapngOptionEndOfOpt  = 0
	pcapngDefaultTsPerSec = 1_000_000
)

func isPCAPNG(magic []byte) bool {
	return len(magic) >= 4 &&
		magic[0] == 0x0A && magic[1] == 0x0D && magic[2] == 0x0D && magic[3] == 0x0A
}

type ngInterface struct {
	linkType    uint32
	tsPerSecond uint64
}

func (i ngInterface) toTime(ts uint64) time.Time {
	per := i.tsPerSecond
	if per == 0 {
		per = pcapngDefaultTsPerSec
	}
	secs := ts / per
	rem := ts % per
	nanos := rem * 1_000_000_000 / per
	return time.Unix(int64(secs), int64(nanos))
}

// pcapngSource parses the PCAPNG format.
type pcapngSource struct {
	br         *bufio.Reader
	order      binary.ByteOrder
	interfaces []ngInterface
	linkType   uint32
	pending    *Packet
}

func newPcapngSource(br *bufio.Reader) (*pcapngSource, error) {
	s := &pcapngSource{br: br, order: binary.LittleEndian}
	// The first block must be a Section Header Block.
	typ, body, err := s.readBlock()
	if err != nil {
		return nil, err
	}
	if typ != pcapngSectionHeader {
		return nil, fmt.Errorf("pcapng: first block is not a section header")
	}
	_ = body

	// Advance to the first Interface Description Block (or a packet, which is
	// kept pending).
	for {
		typ, body, err = s.readBlock()
		if err != nil {
			return nil, err
		}
		switch typ {
		case pcapngInterfaceDesc:
			s.parseInterface(body)
			return s, nil
		case pcapngEnhancedPacket:
			p, err := s.parseEnhanced(body)
			if err != nil {
				return nil, err
			}
			s.pending = p
			return s, nil
		case pcapngSimplePacket:
			p, err := s.parseSimple(body)
			if err != nil {
				return nil, err
			}
			s.pending = p
			return s, nil
		case pcapngSectionHeader:
			// Keep looking.
		}
	}
}

// readBlock reads the next block. A Section Header Block resets the interface
// table and establishes the byte order for the new section.
func (s *pcapngSource) readBlock() (uint32, []byte, error) {
	var hdr [8]byte
	if _, err := io.ReadFull(s.br, hdr[:]); err != nil {
		return 0, nil, err
	}
	typ := s.order.Uint32(hdr[0:4])

	if typ == pcapngSectionHeader {
		bom, err := s.br.Peek(4)
		if err != nil {
			return 0, nil, fmt.Errorf("pcapng: reading byte order magic: %w", err)
		}
		switch {
		case binary.LittleEndian.Uint32(bom) == pcapngByteOrderMagic:
			s.order = binary.LittleEndian
		case binary.BigEndian.Uint32(bom) == pcapngByteOrderMagic:
			s.order = binary.BigEndian
		default:
			return 0, nil, fmt.Errorf("pcapng: bad byte order magic")
		}
		total := s.order.Uint32(hdr[4:8])
		if total < 28 {
			return 0, nil, fmt.Errorf("pcapng: section header too short")
		}
		body := make([]byte, total-12)
		if _, err := io.ReadFull(s.br, body); err != nil {
			return 0, nil, fmt.Errorf("pcapng: reading SHB body: %w", err)
		}
		var trailer [4]byte
		if _, err := io.ReadFull(s.br, trailer[:]); err != nil {
			return 0, nil, fmt.Errorf("pcapng: reading SHB trailer: %w", err)
		}
		s.interfaces = nil
		return typ, body, nil
	}

	total := s.order.Uint32(hdr[4:8])
	if total < 12 {
		return 0, nil, fmt.Errorf("pcapng: invalid block length %d", total)
	}
	body := make([]byte, total-12)
	if _, err := io.ReadFull(s.br, body); err != nil {
		return 0, nil, fmt.Errorf("pcapng: reading block body: %w", err)
	}
	var trailer [4]byte
	if _, err := io.ReadFull(s.br, trailer[:]); err != nil {
		return 0, nil, fmt.Errorf("pcapng: reading block trailer: %w", err)
	}
	return typ, body, nil
}

func (s *pcapngSource) next() (*Packet, error) {
	if s.pending != nil {
		p := s.pending
		s.pending = nil
		return p, nil
	}
	for {
		typ, body, err := s.readBlock()
		if err != nil {
			if err == io.EOF {
				return nil, io.EOF
			}
			return nil, err
		}
		switch typ {
		case pcapngInterfaceDesc:
			s.parseInterface(body)
		case pcapngEnhancedPacket:
			return s.parseEnhanced(body)
		case pcapngSimplePacket:
			return s.parseSimple(body)
		case pcapngPacketBlock:
			return s.parsePacketBlock(body)
		}
	}
}

func (s *pcapngSource) parseInterface(body []byte) {
	if len(body) < 8 {
		return
	}
	iface := ngInterface{
		linkType:    uint32(s.order.Uint16(body[0:2])),
		tsPerSecond: pcapngDefaultTsPerSec,
	}
	for off := 8; off+4 <= len(body); {
		code := s.order.Uint16(body[off : off+2])
		length := int(s.order.Uint16(body[off+2 : off+4]))
		off += 4
		if code == pcapngOptionEndOfOpt {
			break
		}
		if off+length > len(body) {
			break
		}
		if code == pcapngOptionTsResol && length >= 1 {
			v := body[off]
			if v&0x80 != 0 {
				if exp := v & 0x7f; exp < 64 {
					iface.tsPerSecond = uint64(1) << exp
				}
			} else {
				per := uint64(1)
				for i := uint8(0); i < v && i < 19; i++ {
					per *= 10
				}
				iface.tsPerSecond = per
			}
		}
		off += (length + 3) &^ 3
	}
	s.interfaces = append(s.interfaces, iface)
	if s.linkType == 0 {
		s.linkType = iface.linkType
	}
}

func (s *pcapngSource) parseEnhanced(body []byte) (*Packet, error) {
	if len(body) < 20 {
		return nil, fmt.Errorf("pcapng: short enhanced packet block")
	}
	ifaceID := s.order.Uint32(body[0:4])
	tsHigh := s.order.Uint32(body[4:8])
	tsLow := s.order.Uint32(body[8:12])
	caplen := s.order.Uint32(body[12:16])
	origlen := s.order.Uint32(body[16:20])
	if int(20+caplen) > len(body) {
		return nil, fmt.Errorf("pcapng: packet data exceeds block")
	}
	data := make([]byte, caplen)
	copy(data, body[20:20+caplen])
	link := s.linkType
	ts := (uint64(tsHigh) << 32) | uint64(tsLow)
	t := time.Now()
	if int(ifaceID) < len(s.interfaces) {
		link = s.interfaces[ifaceID].linkType
		t = s.interfaces[ifaceID].toTime(ts)
	}
	return &Packet{Time: t, Link: link, Data: data, OrigLen: origlen}, nil
}

func (s *pcapngSource) parseSimple(body []byte) (*Packet, error) {
	if len(body) < 4 {
		return nil, fmt.Errorf("pcapng: short simple packet block")
	}
	origlen := s.order.Uint32(body[0:4])
	dataLen := len(body) - 4
	if dataLen > int(origlen) {
		dataLen = int(origlen)
	}
	data := make([]byte, dataLen)
	copy(data, body[4:4+dataLen])
	link := s.linkType
	if len(s.interfaces) > 0 {
		link = s.interfaces[0].linkType
	}
	return &Packet{Time: time.Now(), Link: link, Data: data, OrigLen: origlen}, nil
}

// parsePacketBlock handles the obsolete Packet Block block type.
func (s *pcapngSource) parsePacketBlock(body []byte) (*Packet, error) {
	if len(body) < 20 {
		return nil, fmt.Errorf("pcapng: short packet block")
	}
	ifaceID := int(s.order.Uint16(body[0:2]))
	tsHigh := s.order.Uint32(body[4:8])
	tsLow := s.order.Uint32(body[8:12])
	caplen := s.order.Uint32(body[12:16])
	origlen := s.order.Uint32(body[16:20])
	if int(20+caplen) > len(body) {
		return nil, fmt.Errorf("pcapng: packet data exceeds block")
	}
	data := make([]byte, caplen)
	copy(data, body[20:20+caplen])
	link := s.linkType
	ts := (uint64(tsHigh) << 32) | uint64(tsLow)
	t := time.Now()
	if ifaceID < len(s.interfaces) {
		link = s.interfaces[ifaceID].linkType
		t = s.interfaces[ifaceID].toTime(ts)
	}
	return &Packet{Time: t, Link: link, Data: data, OrigLen: origlen}, nil
}
