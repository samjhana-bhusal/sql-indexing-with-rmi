# Use standard Ubuntu LTS base image
FROM ubuntu:22.04

# Avoid interactive timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install core build utilities, compilers, and Linux perf
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    clang \
    git \
    python3 \
    python3-pip \
    linux-tools-generic \
    linux-tools-common \
    linux-tools-5.15.0-76-generic \
    matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Note on perf:
# The Linux kernel version inside the container matches the host kernel.
# To make perf run correctly inside Docker, you must:
# 1. Run the container with privileges: --privileged or --cap-add=SYS_ADMIN
# 2. Adjust host kernel permissions: sudo sysctl -w kernel.perf_event_paranoid=1

# Install PyTorch (CPU version is sufficient for CPU benchmarks and RMI training)
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Copy the project files
COPY . /app

# Compile C++ benchmarks and run unit tests with GCC
RUN make clean && make CXX=g++ && make test CXX=g++

# Set default command
CMD ["bash"]
