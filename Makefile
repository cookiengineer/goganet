GO ?= go

# Go 1.27 SIMD is experimental and must be enabled explicitly.
# Without it the engine/vector package falls back to the scalar path.
SIMD_FLAGS = GOEXPERIMENT=simd

.PHONY: all build test test-scalar bench vet fmt clean

all: build

build:
	$(SIMD_FLAGS) $(GO) build ./...

test:
	$(SIMD_FLAGS) $(GO) test ./...

# Verify that the scalar fallback path builds and passes without SIMD.
test-scalar:
	$(GO) test ./...

bench:
	$(SIMD_FLAGS) $(GO) test -run '^$$' -bench . -benchmem ./benchmarks/...

vet:
	$(SIMD_FLAGS) $(GO) vet ./...

fmt:
	$(GO) fmt ./...

clean:
	$(GO) clean ./...
