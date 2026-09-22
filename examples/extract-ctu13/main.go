// Command extract-ctu13 extracts a single scenario folder from the CTU-13
// dataset tarball using only the Go standard library (compress/bzip2 and
// archive/tar). No password is required: the archive is plain bzip2.
//
//	go run ./examples/extract-ctu13 -scenario 4
package main

import (
	"archive/tar"
	"compress/bzip2"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	archive := flag.String("archive", "datasets/CTU-13-Dataset.tar.bz2", "path to the CTU-13 tarball")
	scenario := flag.String("scenario", "4", "scenario folder to extract")
	out := flag.String("out", "datasets/ctu-13", "output directory")
	flag.Parse()

	if err := extract(*archive, *scenario, *out); err != nil {
		fmt.Fprintln(os.Stderr, "extract-ctu13:", err)
		os.Exit(1)
	}
}

func extract(archive, scenario, out string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()

	prefix := "CTU-13-Dataset/" + scenario + "/"
	dest := filepath.Join(out, scenario)
	tr := tar.NewReader(bzip2.NewReader(f))

	extracted := 0
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("reading tar: %w", err)
		}
		if !strings.HasPrefix(hdr.Name, prefix) {
			continue
		}
		rel := strings.TrimPrefix(hdr.Name, prefix)
		if rel == "" {
			continue
		}
		clean := filepath.Clean(rel)
		if clean == "." || strings.HasPrefix(clean, "..") {
			continue
		}
		target := filepath.Join(dest, clean)

		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := writeFile(target, tr, os.FileMode(hdr.Mode)&0o777); err != nil {
				return err
			}
			extracted++
			fmt.Println("extracted", target)
		}
	}
	if extracted == 0 {
		return fmt.Errorf("no files found for scenario %q", scenario)
	}
	fmt.Printf("done: %d files into %s\n", extracted, dest)
	return nil
}

func writeFile(path string, r io.Reader, mode os.FileMode) error {
	if mode == 0 {
		mode = 0o644
	}
	w, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	defer w.Close()
	_, err = io.Copy(w, r)
	return err
}
