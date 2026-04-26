FROM rust:bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git pkg-config libssl-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git

WORKDIR /build/gst-plugins-rs/net/webrtc/signalling
RUN cargo build --release --bin gst-webrtc-signalling-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/gst-plugins-rs/target/release/gst-webrtc-signalling-server /usr/local/bin/gst-webrtc-signalling-server

EXPOSE 8443
CMD ["gst-webrtc-signalling-server"]