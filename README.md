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
opaque), `snmp`, `icmp`, `tls` (opaque) and a generic `raw` fallback that
matches any TCP/UDP/ICMP flow. Encrypted transports are not decrypted; their
records are rendered as bytes. Use `-protocol raw` to train a single classifier
across heterogeneous traffic (as needed for IoT-23).

## Dataset comparison

Both datasets were trained with the convolutional model on `8x64x2` window
images, LeakyReLU 0.2, no BatchNorm, exact WGAN-GP penalty.

CTU-13 ships **botnet-filtered** captures plus full flow labels but **no full
mixed-traffic captures**, so benign raw packets are scarce. A random session
split looks excellent, but that is optimistic: the same scenarios appear in
train and test, and under a cross-scenario holdout the model degrades:

| Dataset | Adapter | Split | AUC(clf) | Bal. acc | Specificity |
|---|---|---|---|---|---|
| CTU-13 | http1 | random | 0.998 | 0.999 | 1.000 |
| CTU-13 | http1 | holdout `{1,8}` | 0.537 | 0.746 | 0.500 |
| CTU-13 | raw | random | 0.989 | 0.833 | 0.667 |
| CTU-13 | raw | holdout `{1,8}` | 0.767 | 0.580 | 0.162 |
| **IoT-23** | **raw** | **random** | **0.978** | **0.962** | **0.947** |
| **IoT-23** | **raw** | **holdout `Capture-1-1`** | **0.978** | **0.978** | **0.957** |

CTU-13 has ~99% malicious sessions in its filtered captures (only tens of benign
sessions survive), so a classifier learns to predict "malicious" for almost
everything and specificity collapses. IoT-23 contains large benign volumes
(device background traffic and honeypot captures, ~20-30% benign in the sampled
set), which is enough to train a genuinely balanced classifier that still
generalises to a completely held-out capture. The key difference is the data,
not the model.

For a rigorous CTU-13 benchmark, supply the full captures or an additional
benign corpus and use `-holdout`.

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

