// Command analyze-ctu13 reports the protocol and label distribution of a
// labelled CTU-13 scenario. It is used to design training and benchmark splits.
//
//	go run ./examples/analyze-ctu13 -dir datasets/ctu-13/4
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/net"
	"github.com/cookiengineer/goganet/adapter/pipeline"
	"github.com/cookiengineer/goganet/adapter/session"
)

func main() {
	dir := flag.String("dir", "", "scenario directory")
	protos := flag.String("protocols", "dns,icmp,http1,http2,http3,snmp,tls", "comma-separated protocol filters")
	maxFrames := flag.Int("frames", 4, "frames per session to retain")
	flag.Parse()

	if *dir == "" {
		fmt.Fprintln(os.Stderr, "usage: analyze-ctu13 -dir <scenario>")
		os.Exit(2)
	}

	want := map[string]bool{}
	for _, p := range strings.Split(*protos, ",") {
		want[strings.TrimSpace(p)] = true
	}

	capture, err := pipeline.FindCapture(*dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "analyze-ctu13:", err)
		os.Exit(1)
	}
	sessions, err := pipeline.ReadSessionsFiltered(capture, *maxFrames, func(f *net.Frame) bool {
		a := adapter.For(f)
		return a != nil && want[a.Name()]
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "analyze-ctu13:", err)
		os.Exit(1)
	}
	if bf, err := pipeline.FindBinetflow(*dir); err == nil {
		labels, err := session.LoadBinetflow(bf)
		if err != nil {
			fmt.Fprintln(os.Stderr, "analyze-ctu13:", err)
			os.Exit(1)
		}
		matched := session.JoinLabels(sessions, session.BuildLabelMap(labels))
		fmt.Printf("sessions: %d, labelled: %d\n", len(sessions), matched)
	} else {
		fmt.Printf("sessions: %d (no binetflow)\n", len(sessions))
	}

	type counts struct{ mal, ben, unlab int }
	byProto := map[string]*counts{}
	for _, s := range sessions {
		name := "raw"
		if a := adapter.For(s.Frames[0]); a != nil {
			name = a.Name()
		}
		c := byProto[name]
		if c == nil {
			c = &counts{}
			byProto[name] = c
		}
		switch {
		case !s.Labeled:
			c.unlab++
		case s.Malicious:
			c.mal++
		default:
			c.ben++
		}
	}

	names := make([]string, 0, len(byProto))
	for n := range byProto {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Printf("%-8s %10s %10s %10s\n", "proto", "malicious", "benign", "unlabelled")
	for _, n := range names {
		c := byProto[n]
		fmt.Printf("%-8s %10d %10d %10d\n", n, c.mal, c.ben, c.unlab)
	}
}
