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
	// flags, 31-bit stream id. The type must be one of the known values.
	if len(f.Payload) >= 9 {
		typ := f.Payload[3]
		if typ <= 0x09 || typ == 0x0a || typ == 0x0b || typ == 0x0c {
			if f.SrcPort == 8080 || f.DstPort == 8080 {
				return len(f.Payload) >= 9
			}
		}
	}
	return false
}

func (http2Adapter) Bytes(f *net.Frame) []byte { return f.Payload }
