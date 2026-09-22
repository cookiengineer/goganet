package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNegate(t *testing.T) {
	got := negate([]float32{1, -2, 0})
	want := []float32{-1, 2, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("negate[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestBenchDiscover(t *testing.T) {
	root := t.TempDir()
	for _, name := range []string{"1", "8", "empty"} {
		if err := os.MkdirAll(filepath.Join(root, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "1", "a.pcap"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "8", "b.pcapng"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := discover(root); len(got) != 2 {
		t.Fatalf("discover = %v", got)
	}
	if got := discover(filepath.Join(root, "1")); len(got) != 1 {
		t.Fatalf("direct discover = %v", got)
	}
}
