package adapter

import (
	"bytes"

	"github.com/cookiengineer/goganet/adapter/net"
)

// http2Adapter matches cleartext HTTP/2 by its connection preface. HTTP/2 over
// TLS is not decrypted and therefore remains an opaque payload; header
// compression (HPACK) is not decoded.
type http2Adapter struct{}

func (http2Adapter) Name() string { return "http2" }

var http2Preface = []byte("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")

func (http2Adapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoTCP {
		return false
	}
	if bytes.HasPrefix(f.Payload, http2Preface) {
		return true
	}
	// An HTTP/2 frame has a 9-byte header: 24-bit length, 8-bit type, 8-bit
	// flags, 31-bit stream id. On the usual cleartext port we accept a frame
	// whose declared length and type are plausible.
	if f.SrcPort == 8080 || f.DstPort == 8080 {
		if len(f.Payload) >= 9 {
			length := int(f.Payload[0])<<16 | int(f.Payload[1])<<8 | int(f.Payload[2])
			typ := f.Payload[3]
			if length <= len(f.Payload)-9 && typ <= 0x09 {
				return true
			}
		}
	}
	return false
}

func (http2Adapter) Bytes(f *net.Frame) []byte { return f.Payload }
