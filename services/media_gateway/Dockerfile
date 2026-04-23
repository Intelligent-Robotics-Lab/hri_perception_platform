FROM debian:bookworm-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    curl \
    pkg-config \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    python3-gi \
    python3-gi-cairo \
    libssl-dev \
    libglib2.0-dev \
    libgirepository1.0-dev \
    libcairo2-dev \
    libudev-dev \
    # Added for WebRTC Support
    libnice10 \
    libnice-dev \
    gstreamer1.0-nice \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install cargo-c

RUN git clone --depth 1 https://github.com/GStreamer/gst-plugins-rs.git /build/gst-plugins-rs

WORKDIR /build/gst-plugins-rs
RUN cargo cinstall -p gst-plugin-webrtc --release --prefix=/usr/local

# --- Runtime Stage ---
FROM debian:bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-gi \
    python3-gi-cairo \
    ca-certificates \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # Added for WebRTC Runtime Support
    libnice10 \
    gstreamer1.0-nice \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/usr/lib/python3/dist-packages
ENV GST_PLUGIN_PATH=/usr/local/lib/x86_64-linux-gnu/gstreamer-1.0:/usr/local/lib/gstreamer-1.0
ENV LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu:/usr/local/lib

COPY --from=builder /usr/local /usr/local

COPY services/media_gateway/requirements.txt .
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY services/media_gateway/app /app/app

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]