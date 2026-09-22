package adapter

import (
	"bytes"

	"github.com/cookiengineer/goganet/adapter/net"
)

// http1Adapter matches cleartext HTTP/1.x by port or by a recognizable request
// or response line. It intentionally does not attempt to match TLS.
type http1Adapter struct{}

func (http1Adapter) Name() string { return "http1" }

var http1Methods = [][]byte{
	[]byte("GET "), []byte("POST "), []byte("PUT "), []byte("DELETE "),
	[]byte("HEAD "), []byte("OPTIONS "), []byte("PATCH "), []byte("TRACE "),
	[]byte("CONNECT "), []byte("HTTP/1."),
}

func (http1Adapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoTCP {
		return false
	}
	switch f.SrcPort {
	case 80, 8080, 8000, 8008, 591:
		return true
	}
	switch f.DstPort {
	case 80, 8080, 8000, 8008, 591:
		return true
	}
	for _, m := range http1Methods {
		if bytes.HasPrefix(f.Payload, m) {
			return true
		}
	}
	return false
}

func (http1Adapter) Bytes(f *net.Frame) []byte { return f.Payload }
