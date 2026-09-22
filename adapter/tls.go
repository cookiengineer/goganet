package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// tlsAdapter matches TLS records by port or record content type. The record
// layer is not decrypted; the handshake and application records are rendered as
// opaque bytes.
type tlsAdapter struct{}

func (tlsAdapter) Name() string { return "tls" }

func (tlsAdapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoTCP {
		return false
	}
	if len(f.Payload) >= 3 {
		if f.Payload[0] >= 0x14 && f.Payload[0] <= 0x18 && f.Payload[1] == 0x03 {
			return true
		}
	}
	switch f.SrcPort {
	case 443, 8443, 993, 995, 465, 990:
		return true
	}
	switch f.DstPort {
	case 443, 8443, 993, 995, 465, 990:
		return true
	}
	return false
}

func (tlsAdapter) Bytes(f *net.Frame) []byte { return f.Payload }
