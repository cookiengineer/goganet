# GoGANet

GoGANet trains and embeds weighted network-traffic classifiers. It converts
labelled socket-session PCAP recordings into multi-frame pixel images and uses a
conditional Wasserstein GAN with gradient penalty (WGAN-GP) plus an auxiliary
classifier head to discriminate malicious from non-malicious traffic.

## Model

- Conditional WGAN-GP core: Wasserstein critic with gradient penalty,
  `lambda = 10`, `n_critic = 5`, LayerNorm instead of BatchNorm in the critic.
- Auxiliary classifier head on the critic (AC-GAN / GACN style): the critic
  exposes both a scalar realness score and class logits. The generator is also
  trained with a classification loss so it preserves class-discriminative
  structure.
- Session/socket labels come from CTU-13 `*.binetflow` files.

### Architectures

`engine/wcgan` provides two architectures selected by `Config.Arch`.

Dense (`dense`, default). An MLP matching the traffic-specific WCGAN-GP
reference: generator 256-512 hidden units, critic 512-256, LeakyReLU 0.2, no
normalization. Best for small feature vectors.

Convolutional (`conv`). Follows the ByteSGAN / DCGAN and WGAN-GP image
references:

- Critic: `len(CritChannels)` strided `3x3` convolutions (stride 2, pad 1,
  channels default `16,32,64`), LeakyReLU 0.2, **no BatchNorm** (per WGAN-GP),
  global mean pooling, then the score and class heads. This scales the
  ByteSGAN 128-64-32 and WGAN-GP 64-128-256-512 stacks to the image size.
- Generator: a dense seed projected to a planar feature map, then
  `len(GenChannels)` transposed convolutions with `4x4` kernel, stride 2,
  pad 1 (each exactly doubles the spatial grid, matching ByteSGAN's `4x4`
  deconvolution), channels default `32,16,8`, then a `3x3` convolution to the
  image channels with `tanh` output.

The `conv` architecture consumes the **planar** layout `(channels,
frames*bytes)`; use `image.Image.Planar()` (or `pipeline.DatasetPlanar`) for
multi-channel images. For a single channel the interleaved and planar layouts
are identical.

Both architectures support the exact gradient penalty because the convolution
operators are composed from linear primitives (im2col, col2im, matmul,
transpose) on the second-order autodiff tape.

## Image representation

A sample is a fixed window of up to **32 frames**. Each frame contributes a row
of **1480 bytes**, zero-padded when fewer frames are available, giving a
`32 x 1480` image per channel. A dedicated frame-presence channel lets the
model distinguish padding from real zero bytes, which is what makes
"early detection" windows trainable.

## Build

Go 1.27 or newer is required. The AVX-512 path is experimental and must be
opted into:

```sh
make build          # GOEXPERIMENT=simd go build ./...
make test           # GOEXPERIMENT=simd go test ./...
make test-scalar    # plain go test ./... (scalar fallback)
```

## Repository Layout

```
engine/vector/   AVX-512 float32 kernels and scalar fallback
engine/autodiff/ second-order reverse-mode autodiff tape
engine/wcgan/    generator, critic, gradient penalty, trainer, inference, weights
adapter/pcap/    stdlib-only classic PCAP, PCAPNG and gzip readers
adapter/net/     Ethernet/IPv4/IPv6/TCP/UDP/ICMP decoding
adapter/image/   multi-frame image rendering and presence mask
adapter/         DNS, HTTP/1, HTTP/2, HTTP/3, SNMP, ICMP and TLS adapters
adapter/session/ session grouping and CTU-13 binetflow label joining
adapter/pipeline/ end-to-end read -> decode -> group -> render helper
classifiers/     embedded ready-to-use weights and registry
examples/        extract-ctu13, analyze-ctu13, train-ctu13, train-iot23,
                 benchmark-ctu13, make-weights, classify-session, inspect-image
benchmarks/      correctness, throughput and convergence tests
```

## Workflow

Download and extract both datasets with the helper script. It skips downloads
whose files already exist and extraction whose directories are already
populated, so it is safe to re-run:

```sh
./download-datasets.sh                # CTU-13 + IoT-23
./download-datasets.sh --only ctu     # just CTU-13
./download-datasets.sh --force        # re-download
./download-datasets.sh --verify       # gzip/bzip2 integrity check before extracting
```

A single CTU-13 scenario can also be extracted with the stdlib-only extractor:

```sh
go run ./examples/extract-ctu13 -scenario 4
```

Inspect the protocol and label balance of a scenario, then train and benchmark:

```sh
go run ./examples/analyze-ctu13 -dir datasets/ctu-13/4

# Whole CTU-13 dataset, one protocol, saved to classifiers/weights/http1.ggnt
go run ./examples/train-ctu13 -dir datasets/ctu-13 -protocol http1 -arch conv -epochs 10

# Benchmark a saved or embedded model on a cross-scenario holdout
go run ./examples/benchmark-ctu13 -dir datasets/ctu-13 -protocol http1 -holdout 1,8
```

IoT-23 stores one directory per capture containing PCAPs and a Zeek
`conn.log.labeled` file. `train-iot23` discovers captures recursively and joins
labels from the Zeek log, so the same flags apply:

```sh
go run ./examples/train-iot23 -dir datasets/iot-23 -protocol http1 -arch conv -epochs 10
go run ./examples/train-iot23 -dir datasets/iot-23 -protocol http1 -holdout Capture-1
```

Protocol adapters currently cover `dns`, `http1`, `http2`, `http3` (QUIC,
opaque), `snmp`, `icmp` and `tls` (opaque). Encrypted transports are not
decrypted; their records are rendered as bytes.

## Benchmarking Pitfalls

The public CTU-13 tarball used here contains **botnet-filtered** captures
(`botnet-capture-*.pcap`) plus full per-flow labels (`capture*.binetflow`); it
does **not** include the full mixed-traffic captures.

Benign raw packets are therefore scarce (a few hundred sessions across the
extracted scenarios), which caps how far a binary packet-image classifier
can generalise.

Measured on the seven extracted scenarios with the convolutional model
(`http1`, `8x64x2`, 4000 sessions, 10 epochs):

| Split | AUC(classifier) | AUC(critic score) | Balanced accuracy |
|---|---|---|---|
| Random session split | 0.998 | 0.591 | 0.999 |
| Cross-scenario holdout `{1,8}` | 0.537 | 0.738 | 0.746 |

The random split is optimistic (the same scenarios appear in train and test).
On unseen scenarios the classifier head overfits, while the WGAN critic's
realness score transfers better (AUC 0.738) and is the more trustworthy signal.
For a rigorous benign-vs-malicious benchmark, supply the full CTU-13 captures
or an additional benign corpus and use `-holdout`.

## Embedding pretrained weights

Models train in pure Go (dense or conv, exact WGAN-GP) and serialize to a
versioned `GGNT` binary via `wcgan.Save` / `wcgan.Load`. The `classifiers`
package embeds every `classifiers/weights/*.ggnt` file with `go:embed`, so a
trained classifier can be used without any external files:

```sh
# Train a per-protocol model; it is written to classifiers/weights/dns.ggnt.
go run ./examples/train-ctu13 -dir datasets/ctu-13/4 -protocol dns -arch conv

# Rebuild so the new weights are baked into the binary, then classify.
go run ./examples/classify-session -pcap capture.pcap -protocol dns
```

- `classifiers.New("dns")` returns the embedded model
- `classifiers.Available()` lists the embedded protocols
- `classifiers.Wrap()` uses a model held in memory instead

Embedding is resolved at compile time, so rebuild after adding or updating a
weight file. `examples/make-weights` produces the synthetic `selftest` fixture
used by the embed test.

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt).

