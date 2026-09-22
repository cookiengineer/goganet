// Package adapter translates protocol-specific capture frames into the byte
// rows consumed by the image renderer. Each protocol implements the Adapter
// interface: it decides which frames belong to it and which bytes represent the
// protocol data unit to render.
package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// Adapter is a protocol-specific frame adapter.
type Adapter interface {
	// Name is the stable protocol identifier, e.g. "dns".
	Name() string
	// Match reports whether the frame belongs to this protocol.
	Match(f *net.Frame) bool
	// Bytes returns the byte slice to render into one image row.
	Bytes(f *net.Frame) []byte
}

var (
	registry []Adapter
	byName   = map[string]Adapter{}
)

// Register adds an adapter to the registry. Adapters registered earlier take
// precedence when matching.
func Register(a Adapter) {
	if _, ok := byName[a.Name()]; ok {
		return
	}
	registry = append(registry, a)
	byName[a.Name()] = a
}

// All returns the registered adapters in match order.
func All() []Adapter { return registry }

// ByName returns the adapter with the given name, or nil.
func ByName(name string) Adapter { return byName[name] }

// For returns the first adapter that matches the frame, or nil.
func For(f *net.Frame) Adapter {
	for _, a := range registry {
		if a.Match(f) {
			return a
		}
	}
	return nil
}

func init() {
	// Order matters: more specific matchers first.
	Register(dnsAdapter{})
	Register(snmpAdapter{})
	Register(icmpAdapter{})
	Register(http2Adapter{})
	Register(http1Adapter{})
	Register(http3Adapter{})
	Register(tlsAdapter{})
	// The generic transport fallback is registered last.
	Register(rawAdapter{})
}
