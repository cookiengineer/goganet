package session

import (
	"net/netip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

func TestMaliciousLabel(t *testing.T) {
	cases := map[string]bool{
		"flow=From-Botnet-V42-UDP":    true,
		"flow=To-Botnet":              true,
		"flow=Botnet":                 true,
		"flow=Background":             false,
		"flow=Normal":                 false,
		"flow=Background-TCP-Attempt": false,
	}
	for label, want := range cases {
		if got := MaliciousLabel(label); got != want {
			t.Errorf("MaliciousLabel(%q) = %v, want %v", label, got, want)
		}
	}
}

func TestLoadBinetflowSlashDatesAndMalformed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.binetflow")
	content := "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n" +
		"2011/08/15 11:00:30.431541,1.0,tcp,10.0.0.1,1000,   ->,10.0.0.2,80,S_RA,0,0,4,252,132,flow=Normal\n" +
		"not-a-date,1.0,tcp,10.0.0.1,1000,->,10.0.0.2,80,CON,0,0,1,1,1,flow=Botnet\n" +
		"short,row\n" +
		"2011/08/15 11:01:10.000000,1.0,udp,10.0.0.3,53,->,10.0.0.4,5000,CON,0,0,1,1,1,flow=From-Botnet-V42-UDP\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	labels, err := LoadBinetflow(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != 2 {
		t.Fatalf("parsed %d labels, want 2 (malformed skipped)", len(labels))
	}
	if labels[0].Malicious || !labels[1].Malicious {
		t.Fatalf("labels = %+v", labels)
	}
	if labels[0].Key.P != net.ProtoTCP {
		t.Fatalf("proto = %d", labels[0].Key.P)
	}
}

func TestJoinBinetflowFileStreaming(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "x.binetflow")
	content := "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n" +
		"2011-08-15 11:00:00.000000,1.0,udp,10.0.0.1,40000,->,8.8.8.8,53,CON,0,0,1,1,1,flow=Normal\n" +
		"2011-08-15 11:05:00.000000,1.0,udp,192.168.1.5,50000,->,1.2.3.4,53,CON,0,0,1,1,1,flow=From-Botnet-V42-UDP\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	benign := &Session{Key: makeKey(netip.MustParseAddr("10.0.0.1"), 40000, netip.MustParseAddr("8.8.8.8"), 53, net.ProtoUDP), Start: time.Date(2011, 8, 15, 11, 0, 30, 0, time.UTC)}
	mal := &Session{Key: makeKey(netip.MustParseAddr("192.168.1.5"), 50000, netip.MustParseAddr("1.2.3.4"), 53, net.ProtoUDP), Start: time.Date(2011, 8, 15, 11, 5, 30, 0, time.UTC)}
	unmatched := &Session{Key: makeKey(netip.MustParseAddr("9.9.9.9"), 1, netip.MustParseAddr("8.8.4.4"), 2, net.ProtoTCP), Start: time.Now()}

	n, err := JoinBinetflowFile([]*Session{benign, mal, unmatched}, path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("matched = %d, want 2", n)
	}
	if !benign.Labeled || benign.Malicious {
		t.Fatalf("benign = %+v", benign)
	}
	if !mal.Labeled || !mal.Malicious {
		t.Fatalf("mal = %+v", mal)
	}
	if unmatched.Labeled {
		t.Fatal("unmatched session should not be labelled")
	}
}

func TestJoinLabelsFallback(t *testing.T) {
	// A label with a non-overlapping time interval still matches by key.
	lbl := FlowLabel{
		Key:       makeKey(netip.MustParseAddr("10.0.0.1"), 1, netip.MustParseAddr("10.0.0.2"), 2, net.ProtoUDP),
		Start:     time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC),
		End:       time.Date(2020, 1, 1, 0, 0, 1, 0, time.UTC),
		Raw:       "flow=Normal",
		Malicious: false,
	}
	s := &Session{Key: lbl.Key, Start: time.Date(2011, 1, 1, 0, 0, 0, 0, time.UTC)}
	if n := JoinLabels([]*Session{s}, BuildLabelMap([]FlowLabel{lbl})); n != 1 {
		t.Fatalf("matched = %d", n)
	}
	if !s.Labeled || s.Label != "flow=Normal" {
		t.Fatalf("session = %+v", s)
	}
}
