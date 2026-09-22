package session

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"net/netip"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

// FlowLabel is one labelled flow from a CTU-13 binetflow file.
type FlowLabel struct {
	Key       net.SessionKey
	Start     time.Time
	End       time.Time
	Raw       string
	Malicious bool
}

// binetflowLayouts lists the timestamp formats seen across CTU-13 scenarios.
var binetflowLayouts = []string{
	"2006-01-02 15:04:05.000000",
	"2006/01/02 15:04:05.000000",
	"2006-01-02 15:04:05",
	"2006/01/02 15:04:05",
}

func protoNumber(s string) uint8 {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "TCP":
		return net.ProtoTCP
	case "UDP":
		return net.ProtoUDP
	case "ICMP":
		return net.ProtoICMP
	default:
		return 0
	}
}

// MaliciousLabel reports whether a CTU-13 label denotes botnet/malicious
// traffic. Everything else (Background, Normal) is treated as benign.
func MaliciousLabel(label string) bool {
	l := strings.ToLower(label)
	switch {
	case strings.Contains(l, "botnet"):
		return true
	case strings.Contains(l, "background"):
		return false
	case strings.Contains(l, "normal"):
		return false
	default:
		return false
	}
}

// LoadBinetflow parses a CTU-13 binetflow CSV file.
func LoadBinetflow(path string) ([]FlowLabel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReader(f))
	r.FieldsPerRecord = -1
	r.TrimLeadingSpace = true

	var labels []FlowLabel
	first := true
	for {
		rec, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("binetflow: %w", err)
		}
		if first {
			first = false
			if len(rec) > 0 && strings.EqualFold(strings.TrimSpace(rec[0]), "StartTime") {
				continue
			}
		}
		if len(rec) < 8 {
			continue
		}
		label, ok := parseBinetflowRecord(rec)
		if ok {
			labels = append(labels, label)
		}
	}
	return labels, nil
}

func parseBinetflowRecord(rec []string) (FlowLabel, bool) {
	var start time.Time
	var err error
	rawTime := strings.TrimSpace(rec[0])
	for _, layout := range binetflowLayouts {
		start, err = time.Parse(layout, rawTime)
		if err == nil {
			break
		}
	}
	if err != nil {
		return FlowLabel{}, false
	}
	dur, _ := strconv.ParseFloat(strings.TrimSpace(rec[1]), 64)
	proto := protoNumber(rec[2])
	src, err := netip.ParseAddr(strings.TrimSpace(rec[3]))
	if err != nil {
		return FlowLabel{}, false
	}
	sport, _ := strconv.ParseUint(strings.TrimSpace(rec[4]), 10, 16)
	dst, err := netip.ParseAddr(strings.TrimSpace(rec[6]))
	if err != nil {
		return FlowLabel{}, false
	}
	dport, _ := strconv.ParseUint(strings.TrimSpace(rec[7]), 10, 16)

	raw := strings.TrimSpace(rec[len(rec)-1])
	l := FlowLabel{
		Key:       makeKey(src, uint16(sport), dst, uint16(dport), proto),
		Start:     start,
		End:       start.Add(time.Duration(dur * float64(time.Second))),
		Raw:       raw,
		Malicious: MaliciousLabel(raw),
	}
	return l, true
}

func makeKey(src netip.Addr, sport uint16, dst netip.Addr, dport uint16, proto uint8) net.SessionKey {
	a := netip.AddrPortFrom(src, sport)
	b := netip.AddrPortFrom(dst, dport)
	if less(b, a) {
		a, b = b, a
	}
	return net.SessionKey{A: a, B: b, P: proto}
}

func less(a, b netip.AddrPort) bool {
	if c := a.Addr().Compare(b.Addr()); c != 0 {
		return c < 0
	}
	return a.Port() < b.Port()
}

// JoinBinetflowFile streams a binetflow file and labels only the sessions whose
// keys it encounters, avoiding retention of the entire label table. It returns
// the number of sessions labelled.
func JoinBinetflowFile(sessions []*Session, path string) (int, error) {
	index := make(map[net.SessionKey]*Session, len(sessions))
	for _, s := range sessions {
		if _, ok := index[s.Key]; !ok {
			index[s.Key] = s
		}
	}

	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r := csv.NewReader(bufio.NewReaderSize(f, 1<<20))
	r.FieldsPerRecord = -1
	r.TrimLeadingSpace = true

	matched := 0
	first := true
	for {
		rec, err := r.Read()
		if err != nil {
			if err == io.EOF {
				break
			}
			return matched, fmt.Errorf("binetflow: %w", err)
		}
		if first {
			first = false
			if len(rec) > 0 && strings.EqualFold(strings.TrimSpace(rec[0]), "StartTime") {
				continue
			}
		}
		if len(rec) < 8 {
			continue
		}
		label, ok := parseBinetflowRecord(rec)
		if !ok {
			continue
		}
		s := index[label.Key]
		if s == nil || s.Labeled {
			continue
		}
		s.Label = label.Raw
		s.Malicious = label.Malicious
		s.Labeled = true
		matched++
	}
	return matched, nil
}

// LabelMap indexes labels by flow key.
type LabelMap map[net.SessionKey][]FlowLabel

// BuildLabelMap indexes labels for fast session matching.
func BuildLabelMap(labels []FlowLabel) LabelMap {
	m := make(LabelMap, len(labels))
	for _, l := range labels {
		m[l.Key] = append(m[l.Key], l)
	}
	return m
}

// JoinLabels assigns a label to each session whose key matches a labelled flow
// and whose start time falls inside (or close to) the label interval. It returns
// the number of sessions that were labelled.
func JoinLabels(sessions []*Session, m LabelMap) int {
	matched := 0
	for _, s := range sessions {
		candidates := m[s.Key]
		if len(candidates) == 0 {
			continue
		}
		var best *FlowLabel
		for i := range candidates {
			c := &candidates[i]
			if !s.Start.Before(c.Start.Add(-time.Second)) && !s.Start.After(c.End.Add(time.Second)) {
				best = c
				break
			}
		}
		if best == nil {
			best = &candidates[0]
		}
		s.Label = best.Raw
		s.Malicious = best.Malicious
		s.Labeled = true
		matched++
	}
	return matched
}
