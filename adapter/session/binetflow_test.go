package session

import (
	"net/netip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

func TestBinetflowParseAndJoin(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "capture.binetflow")
	content := "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n" +
		"2011-08-10 10:00:00.000000,5.0,udp,10.0.0.1,40000,->,8.8.8.8,53,CON,0,0,4,100,50,flow=Normal\n" +
		"2011-08-10 10:01:00.000000,5.0,udp,192.168.1.5,50000,->,1.2.3.4,53,CON,0,0,4,100,50,flow=From-Botnet-V42-UDP\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	labels, err := LoadBinetflow(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(labels) != 2 {
		t.Fatalf("expected 2 labels, got %d", len(labels))
	}
	if labels[0].Malicious {
		t.Fatalf("first label should be benign")
	}
	if !labels[1].Malicious {
		t.Fatalf("second label should be malicious")
	}

	start := time.Date(2011, 8, 10, 10, 0, 1, 0, time.UTC)
	s := &Session{
		Key:   makeKey(netip.MustParseAddr("10.0.0.1"), 40000, netip.MustParseAddr("8.8.8.8"), 53, net.ProtoUDP),
		Start: start,
	}
	if n := JoinLabels([]*Session{s}, BuildLabelMap(labels)); n != 1 {
		t.Fatalf("matched %d", n)
	}
	if !s.Labeled || s.Malicious {
		t.Fatalf("session label = %q malicious=%v", s.Label, s.Malicious)
	}
}
