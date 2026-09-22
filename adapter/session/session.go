// Package session groups decoded frames into bidirectional socket sessions and
// joins them with the per-flow labels found in CTU-13 binetflow files.
package session

import (
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

// Session is a bidirectional socket session.
type Session struct {
	Key    net.SessionKey
	Frames []*net.Frame
	Start  time.Time
	End    time.Time
	Bytes  int
	// Label is filled by JoinLabels.
	Label     string
	Malicious bool
	Labeled   bool
}

// Group groups frames by their bidirectional flow key. It preserves first-seen
// order and keeps at most maxFrames frames per session (the earliest ones), which
// is what early-detection windows need. If maxFrames <= 0 a default of 32 is
// used.
func Group(frames []*net.Frame, maxFrames int) []*Session {
	g := NewGrouper(maxFrames)
	for _, f := range frames {
		g.Add(f)
	}
	return g.Sessions()
}

// Grouper incrementally groups frames by flow key without retaining the whole
// capture in memory. It is the streaming counterpart of Group.
type Grouper struct {
	index map[net.SessionKey]*Session
	order []*Session
	max   int
}

// NewGrouper creates a grouper that keeps at most maxFrames frames per session.
func NewGrouper(maxFrames int) *Grouper {
	if maxFrames <= 0 {
		maxFrames = 32
	}
	return &Grouper{index: map[net.SessionKey]*Session{}, max: maxFrames}
}

// Add incorporates one decoded frame.
func (g *Grouper) Add(f *net.Frame) {
	if f == nil {
		return
	}
	key := f.Key()
	s := g.index[key]
	if s == nil {
		s = &Session{Key: key, Start: f.Time, End: f.Time}
		g.index[key] = s
		g.order = append(g.order, s)
	}
	if len(s.Frames) < g.max {
		s.Frames = append(s.Frames, f)
	}
	if f.Time.Before(s.Start) {
		s.Start = f.Time
	}
	if f.Time.After(s.End) {
		s.End = f.Time
	}
	s.Bytes += len(f.Payload)
}

// Sessions returns the sessions in first-seen order.
func (g *Grouper) Sessions() []*Session { return g.order }
