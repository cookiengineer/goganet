// Package net decodes link-layer frames from a capture into transport-level
// views. It supports Ethernet (including a single VLAN tag), raw IP, and the
// Linux cooked-capture headers, and extracts IPv4/IPv6 plus TCP/UDP/ICMP
// metadata. No third-party packet library is used.
package net

import (
	"encoding/binary"
	"fmt"
	"net/netip"
	"time"
)

// Transport protocol numbers from the IANA registry.
const (
	ProtoICMP   uint8 = 1
	ProtoTCP    uint8 = 6
	ProtoUDP    uint8 = 17
	ProtoICMPv6 uint8 = 58
)

// Frame is a decoded network frame.
type Frame struct {
	Time     time.Time
	Src      netip.Addr
	Dst      netip.Addr
	SrcPort  uint16
	DstPort  uint16
	Protocol uint8
	TCPFlags uint8
	Payload  []byte // transport payload
	Raw      []byte // original link-layer frame
}

// SessionKey is the direction-independent bidirectional flow key.
type SessionKey struct {
	A, B netip.AddrPort
	P    uint8
}

// Key returns the normalized bidirectional flow key for the frame.
func (f *Frame) Key() SessionKey {
	a := netip.AddrPortFrom(f.Src, f.SrcPort)
	b := netip.AddrPortFrom(f.Dst, f.DstPort)
	if addrLess(b, a) {
		a, b = b, a
	}
	return SessionKey{A: a, B: b, P: f.Protocol}
}

func addrLess(a, b netip.AddrPort) bool {
	if c := a.Addr().Compare(b.Addr()); c != 0 {
		return c < 0
	}
	return a.Port() < b.Port()
}

func (f *Frame) String() string {
	return fmt.Sprintf("%s:%d -> %s:%d proto=%d", f.Src, f.SrcPort, f.Dst, f.DstPort, f.Protocol)
}

// Link-layer types, duplicated here to avoid importing the pcap package from
// callers that only need decoding.
const (
	linkEthernet  = 1
	linkRaw       = 101
	linkLinuxSLL  = 113
	linkLinuxSLL2 = 276
)

// Decode parses a link-layer frame of the given link type.
func Decode(linkType uint32, data []byte, ts time.Time) (*Frame, error) {
	var l3 []byte
	switch linkType {
	case linkEthernet:
		var err error
		l3, err = stripEthernet(data)
		if err != nil {
			return nil, err
		}
	case linkRaw:
		l3 = data
	case linkLinuxSLL:
		if len(data) < 16 {
			return nil, fmt.Errorf("net: short linux sll header")
		}
		l3 = data[16:]
	case linkLinuxSLL2:
		if len(data) < 20 {
			return nil, fmt.Errorf("net: short linux sll2 header")
		}
		l3 = data[20:]
	default:
		return nil, fmt.Errorf("net: unsupported link type %d", linkType)
	}
	if len(l3) == 0 {
		return nil, fmt.Errorf("net: empty network layer")
	}

	switch l3[0] >> 4 {
	case 4:
		return decodeIPv4(l3, ts, data)
	case 6:
		return decodeIPv6(l3, ts, data)
	default:
		return nil, fmt.Errorf("net: unknown IP version")
	}
}

func stripEthernet(data []byte) ([]byte, error) {
	if len(data) < 14 {
		return nil, fmt.Errorf("net: short ethernet header")
	}
	etherType := binary.BigEndian.Uint16(data[12:14])
	offset := 14
	for etherType == 0x8100 || etherType == 0x88a8 {
		if len(data) < offset+4 {
			return nil, fmt.Errorf("net: short vlan header")
		}
		etherType = binary.BigEndian.Uint16(data[offset+2 : offset+4])
		offset += 4
	}
	if etherType != 0x0800 && etherType != 0x86dd {
		return nil, fmt.Errorf("net: unsupported ethertype 0x%04x", etherType)
	}
	return data[offset:], nil
}

func decodeIPv4(b []byte, ts time.Time, raw []byte) (*Frame, error) {
	if len(b) < 20 {
		return nil, fmt.Errorf("net: short ipv4 header")
	}
	ihl := int(b[0]&0x0f) * 4
	if ihl < 20 || len(b) < ihl {
		return nil, fmt.Errorf("net: bad ipv4 ihl")
	}
	total := int(binary.BigEndian.Uint16(b[2:4]))
	if total == 0 || total > len(b) {
		total = len(b)
	}
	proto := b[9]
	src, _ := netip.AddrFromSlice(b[12:16])
	dst, _ := netip.AddrFromSlice(b[16:20])
	// Reject fragments other than the first for transport parsing.
	fragOff := binary.BigEndian.Uint16(b[6:8]) & 0x1fff
	transport := b[ihl:total]
	if fragOff != 0 {
		return &Frame{Time: ts, Src: src, Dst: dst, Protocol: proto, Raw: raw}, nil
	}
	return decodeTransport(proto, src, dst, transport, ts, raw)
}

func decodeIPv6(b []byte, ts time.Time, raw []byte) (*Frame, error) {
	if len(b) < 40 {
		return nil, fmt.Errorf("net: short ipv6 header")
	}
	proto := b[6]
	src, _ := netip.AddrFromSlice(b[8:24])
	dst, _ := netip.AddrFromSlice(b[24:40])
	transport := b[40:]
	// Walk a few common extension headers.
	next := proto
	off := 0
	for i := 0; i < 4; i++ {
		switch next {
		case 0, 43, 60: // hop-by-hop, routing, destination options
			if off+2 > len(transport) {
				return nil, fmt.Errorf("net: short ipv6 extension")
			}
			next = transport[off]
			off += (int(transport[off+1]) + 1) * 8
		case 44: // fragment
			next = transport[off]
			off += 8
		default:
			proto = next
			transport = transport[off:]
			return decodeTransport(proto, src, dst, transport, ts, raw)
		}
		if off > len(transport) {
			return nil, fmt.Errorf("net: bad ipv6 extension chain")
		}
	}
	proto = next
	transport = transport[off:]
	return decodeTransport(proto, src, dst, transport, ts, raw)
}

func decodeTransport(proto uint8, src, dst netip.Addr, b []byte, ts time.Time, raw []byte) (*Frame, error) {
	f := &Frame{Time: ts, Src: src, Dst: dst, Protocol: proto, Payload: b, Raw: raw}
	switch proto {
	case ProtoTCP:
		if len(b) < 20 {
			return nil, fmt.Errorf("net: short tcp header")
		}
		f.SrcPort = binary.BigEndian.Uint16(b[0:2])
		f.DstPort = binary.BigEndian.Uint16(b[2:4])
		doff := int(b[12]>>4) * 4
		f.TCPFlags = b[13]
		if doff >= 20 && doff <= len(b) {
			f.Payload = b[doff:]
		} else {
			f.Payload = nil
		}
	case ProtoUDP:
		if len(b) < 8 {
			return nil, fmt.Errorf("net: short udp header")
		}
		f.SrcPort = binary.BigEndian.Uint16(b[0:2])
		f.DstPort = binary.BigEndian.Uint16(b[2:4])
		end := len(b)
		if l := int(binary.BigEndian.Uint16(b[4:6])); l >= 8 && l <= len(b) {
			end = l
		}
		f.Payload = b[8:end]
	case ProtoICMP, ProtoICMPv6:
		f.Payload = b
	default:
		f.Payload = b
	}
	return f, nil
}
