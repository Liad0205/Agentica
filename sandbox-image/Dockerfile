# Sandbox container image for AI agent code execution
# Provides a secure, isolated Node.js environment for running agent-generated code

FROM node:20-slim

# Install system dependencies
# - git: for cloning repos if needed
# - curl: for downloading resources (will be blocked by command validation)
# - jq: for JSON processing in scripts
RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    git \
    curl \
    jq \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Enable Corepack so pnpm/yarn are available out of the box.
RUN corepack enable && corepack prepare pnpm@latest --activate

# Pre-install common global tools used by agents
# These are available system-wide for all projects
# Pinning ESLint to v8 to match .eslintrc.json
RUN npm install -g \
    vite@latest \
    create-vite@latest \
    create-next-app@latest \
    eslint@^8.57.0 \
    prettier \
    typescript \
    tsx \
    @tailwindcss/cli@latest

# Create workspace directory
RUN mkdir -p /workspace

# Set working directory
WORKDIR /workspace

# Copy full template project into workspace
COPY template/ /workspace/

# Pre-install node_modules (biggest time savings: ~30-40s per session)
RUN cd /workspace && npm install

# Set ownership for the node user (non-root)
RUN chown -R node:node /workspace

# Run as non-root user for security
USER node

# Default command keeps container alive for exec commands
# The actual work is done via docker exec from the SandboxManager
CMD ["tail", "-f", "/dev/null"]
