"""Docker-based sandbox manager for isolated code execution.

This module provides the SandboxManager class that manages Docker containers
for safe, isolated execution of AI agent-generated code.
"""

import asyncio
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC
from io import BytesIO

import docker
import structlog
from docker.errors import APIError, ImageNotFound, NotFound

from config import settings
from sandbox.security import sanitize_output, validate_command, validate_path

logger = structlog.get_logger()

# Directories to skip when building a file tree for the UI.
# These can easily contain thousands of files and make the UI unusable.
DEFAULT_IGNORED_DIR_NAMES: set[str] = {
    "node_modules",
    ".git",
    ".next",
    "dist",
    "build",
    "out",
    "coverage",
    ".turbo",
    ".cache",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
}


@dataclass
class SandboxInfo:
    """Information about a created sandbox container."""

    sandbox_id: str
    container_id: str
    port: int
    workspace_path: str
    status: str = "created"


@dataclass
class FileInfo:
    """Information about a file or directory in the sandbox."""

    name: str
    path: str
    is_directory: bool
    size: int = 0


@dataclass
class CommandResult:
    """Result of executing a command in the sandbox."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


# Container security configuration
CONTAINER_CONFIG: dict[str, object] = {
    "mem_limit": "2048m",
    "cpu_period": 100000,
    "cpu_quota": 50000,  # 50% of one CPU core
    "network_mode": "bridge",  # Allow npm install; use "none" for stricter security
    "security_opt": ["no-new-privileges"],
    "cap_drop": ["ALL"],
    "cap_add": ["CHOWN", "SETUID", "SETGID"],  # Minimum needed for npm
    "user": "node",
}


class SandboxManager:
    """Manages Docker sandbox lifecycles for agent code execution.

    Each sandbox is an isolated Docker container with Node.js pre-installed.
    Multiple sandboxes can run concurrently:
    - One per agent in hypothesis mode
    - One shared in react mode
    - One per sub-agent + one integration in decomp mode

    Attributes:
        image_name: The Docker image to use for sandbox containers.
        base_port: The starting port number for dev server allocation.
        max_sandboxes: Maximum number of concurrent sandboxes allowed.
    """

    def __init__(
        self,
        image_name: str = "agent-swarm-sandbox:latest",
        base_port: int = 5173,
        max_sandboxes: int = 10,
    ) -> None:
        """Initialize the SandboxManager.

        Args:
            image_name: Docker image name for sandbox containers.
            base_port: Base port for dev server allocation (default: 5173).
            max_sandboxes: Maximum concurrent sandboxes (default: 10).
        """
        self.image_name = image_name
        self.base_port = base_port
        self.max_sandboxes = max_sandboxes
        self._client: docker.DockerClient | None = None
        self._sandboxes: dict[str, SandboxInfo] = {}
        self._allocated_ports: set[int] = set()
        self._lock = asyncio.Lock()

    @property
    def client(self) -> docker.DockerClient:
        """Lazy initialization of Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _allocate_port(self) -> int:
        """Allocate the next available port for a sandbox.

        Returns:
            An available port number.

        Raises:
            RuntimeError: If no ports are available.
        """
        for offset in range(self.max_sandboxes):
            port = self.base_port + offset
            if port not in self._allocated_ports:
                self._allocated_ports.add(port)
                return port
        raise RuntimeError(
            f"No available ports. Maximum {self.max_sandboxes} concurrent sandboxes reached."
        )

    def _release_port(self, port: int) -> None:
        """Release a previously allocated port."""
        self._allocated_ports.discard(port)

    async def create_sandbox(self, sandbox_id: str) -> SandboxInfo:
        """Create and start a new sandbox container.

        Args:
            sandbox_id: Unique identifier for the sandbox.

        Returns:
            SandboxInfo with container details including assigned port.

        Raises:
            ValueError: If sandbox_id already exists.
            RuntimeError: If container creation fails or max sandboxes reached.
        """
        async with self._lock:
            if sandbox_id in self._sandboxes:
                raise ValueError(f"Sandbox '{sandbox_id}' already exists")

            if len(self._sandboxes) >= self.max_sandboxes:
                raise RuntimeError(
                    f"Maximum sandboxes ({self.max_sandboxes}) reached"
                )

            port = self._allocate_port()

        try:
            # Run container creation in executor with timeout to avoid blocking
            container = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    self._create_container,
                    sandbox_id,
                    port,
                ),
                timeout=30,
            )

            sandbox_info = SandboxInfo(
                sandbox_id=sandbox_id,
                container_id=container.id,
                port=port,
                workspace_path="/workspace",
                status="running",
            )

            async with self._lock:
                self._sandboxes[sandbox_id] = sandbox_info

            logger.info(
                "sandbox_created",
                sandbox_id=sandbox_id,
                container_id=container.id[:12],
                port=port,
            )

            return sandbox_info

        except (APIError, ImageNotFound, TimeoutError) as e:
            self._release_port(port)
            logger.error("sandbox_creation_failed", sandbox_id=sandbox_id, error=str(e))
            raise RuntimeError(f"Failed to create sandbox: {e}") from e

    def _create_container(self, sandbox_id: str, port: int) -> docker.models.containers.Container:
        """Create the Docker container (blocking operation).

        Args:
            sandbox_id: Unique identifier for the sandbox.
            port: The port to expose for the dev server.

        Returns:
            The created and started container.
        """
        container = self.client.containers.run(
            self.image_name,
            name=f"sandbox-{sandbox_id}",
            detach=True,
            remove=False,
            ports={f"{port}/tcp": port},
            mem_limit=CONTAINER_CONFIG["mem_limit"],
            cpu_period=CONTAINER_CONFIG["cpu_period"],
            cpu_quota=CONTAINER_CONFIG["cpu_quota"],
            network_mode=CONTAINER_CONFIG["network_mode"],
            security_opt=CONTAINER_CONFIG["security_opt"],
            cap_drop=CONTAINER_CONFIG["cap_drop"],
            cap_add=CONTAINER_CONFIG["cap_add"],
            user=CONTAINER_CONFIG["user"],
            working_dir="/workspace",
            environment={
                "NODE_ENV": "development",
                "PORT": str(port),
            },
        )
        return container

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """Stop and remove a sandbox container.

        Args:
            sandbox_id: The sandbox to destroy.

        Raises:
            KeyError: If sandbox doesn't exist.
        """
        async with self._lock:
            if sandbox_id not in self._sandboxes:
                raise KeyError(f"Sandbox '{sandbox_id}' not found")

            sandbox_info = self._sandboxes.pop(sandbox_id)
            self._release_port(sandbox_info.port)

        try:
            await asyncio.get_running_loop().run_in_executor(
                None,
                self._destroy_container,
                sandbox_info.container_id,
            )
            logger.info("sandbox_destroyed", sandbox_id=sandbox_id)
        except NotFound:
            logger.warning("sandbox_already_removed", sandbox_id=sandbox_id)
        except APIError as e:
            logger.error("sandbox_destroy_failed", sandbox_id=sandbox_id, error=str(e))
            raise

    def _destroy_container(self, container_id: str) -> None:
        """Stop and remove a container (blocking operation)."""
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=5)
            container.remove(force=True)
        except NotFound:
            pass  # Already removed

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """Write a file inside the sandbox's /workspace.

        Creates parent directories automatically if they don't exist.

        Args:
            sandbox_id: The sandbox to write to.
            path: Relative path within /workspace.
            content: File content to write.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If path validation fails.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate the path
        is_valid, error_msg, resolved_path = validate_path(
            sandbox_info.workspace_path, path
        )
        if not is_valid:
            raise ValueError(error_msg)

        # Create a tar archive with the file content (with timeout)
        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                self._write_file_to_container,
                sandbox_info.container_id,
                path,
                content,
            ),
            timeout=30,
        )

        logger.debug("file_written", sandbox_id=sandbox_id, path=path)

    def _write_file_to_container(
        self, container_id: str, path: str, content: str
    ) -> None:
        """Write file to container using tar archive (blocking operation)."""
        container = self.client.containers.get(container_id)

        # Create parent directories first (use list form to avoid shell injection)
        parent_dir = os.path.dirname(path)
        if parent_dir:
            container.exec_run(
                ["mkdir", "-p", f"/workspace/{parent_dir}"], user="node"
            )

        # Create tar archive with the file
        tar_stream = BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            file_data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=path)
            tarinfo.size = len(file_data)
            tarinfo.mode = 0o644
            tar.addfile(tarinfo, BytesIO(file_data))

        tar_stream.seek(0)
        container.put_archive("/workspace", tar_stream)

        # Fix ownership to ensure node user can modify files
        container.exec_run(
            ["chown", "node:node", f"/workspace/{path}"],
            user="root"
        )

    async def read_file(self, sandbox_id: str, path: str) -> str:
        """Read a file from the sandbox's /workspace.

        Args:
            sandbox_id: The sandbox to read from.
            path: Relative path within /workspace.

        Returns:
            The file content as a string.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If path validation fails.
            FileNotFoundError: If file doesn't exist.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate the path
        is_valid, error_msg, _ = validate_path(sandbox_info.workspace_path, path)
        if not is_valid:
            raise ValueError(error_msg)

        content = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                self._read_file_from_container,
                sandbox_info.container_id,
                path,
            ),
            timeout=30,
        )

        return content

    def _read_file_from_container(self, container_id: str, path: str) -> str:
        """Read file from container using tar archive (blocking operation)."""
        container = self.client.containers.get(container_id)

        try:
            bits, _ = container.get_archive(f"/workspace/{path}")
        except NotFound as err:
            raise FileNotFoundError(f"File not found: {path}") from err

        # Extract content from tar archive
        tar_stream = BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)
        tar_stream.seek(0)

        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            member = tar.getmembers()[0]
            extracted = tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(f"Cannot read file: {path}")
            return extracted.read().decode("utf-8")

    async def list_files(self, sandbox_id: str, path: str = ".") -> list[FileInfo]:
        """List files and directories in the sandbox.

        Args:
            sandbox_id: The sandbox to list files in.
            path: Relative path within /workspace (default: root).

        Returns:
            List of FileInfo objects for each entry.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If path validation fails.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate the path
        is_valid, error_msg, _ = validate_path(sandbox_info.workspace_path, path)
        if not is_valid:
            raise ValueError(error_msg)

        files = await asyncio.get_running_loop().run_in_executor(
            None,
            self._list_files_in_container,
            sandbox_info.container_id,
            path,
        )

        return files

    def _list_files_in_container(self, container_id: str, path: str) -> list[FileInfo]:
        """List files in container directory (blocking operation)."""
        container = self.client.containers.get(container_id)

        # Use ls -la to get detailed file listing (list form to avoid injection)
        result = container.exec_run(
            ["ls", "-la", "--full-time", f"/workspace/{path}"],
            user="node",
        )

        if result.exit_code != 0:
            return []

        files: list[FileInfo] = []
        output = result.output.decode("utf-8")

        for line in output.strip().split("\n")[1:]:  # Skip "total" line
            parts = line.split()
            if len(parts) < 9:
                continue

            permissions = parts[0]
            size = int(parts[4]) if parts[4].isdigit() else 0
            name = " ".join(parts[8:])  # Handle filenames with spaces

            # Skip . and ..
            if name in (".", ".."):
                continue

            is_directory = permissions.startswith("d")
            file_path = f"{path}/{name}" if path != "." else name

            files.append(
                FileInfo(
                    name=name,
                    path=file_path,
                    is_directory=is_directory,
                    size=size,
                )
            )

        return files

    async def list_files_recursive(
        self,
        sandbox_id: str,
        path: str = ".",
        *,
        ignored_dir_names: set[str] | None = None,
        max_entries: int = 5000,
    ) -> list[FileInfo]:
        """Recursively list files and directories under a path.

        This is intended for powering the frontend file tree. It prunes
        common large directories like node_modules to avoid huge payloads.

        Args:
            sandbox_id: The sandbox to list files in.
            path: Relative path within /workspace to list from (default: ".").
            ignored_dir_names: Directory basenames to prune (default:
                DEFAULT_IGNORED_DIR_NAMES).
            max_entries: Soft cap on entries returned (default: 5000).

        Returns:
            A flat list of FileInfo entries with paths relative to /workspace.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If path validation fails.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate the base path
        is_valid, error_msg, resolved_path = validate_path(
            sandbox_info.workspace_path, path
        )
        if not is_valid:
            raise ValueError(error_msg)

        normalized_prefix = path.strip()
        if normalized_prefix in ("", ".", "./"):
            normalized_prefix = ""
        else:
            normalized_prefix = normalized_prefix.lstrip("./").strip("/")

        ignored = ignored_dir_names or DEFAULT_IGNORED_DIR_NAMES

        entries = await asyncio.get_running_loop().run_in_executor(
            None,
            self._find_files_in_container,
            sandbox_info.container_id,
            resolved_path,
            normalized_prefix,
            ignored,
            max_entries,
        )

        return entries

    async def export_archive(self, sandbox_id: str, path: str = ".") -> bytes:
        """Export a workspace path as a tar archive.

        Args:
            sandbox_id: The sandbox to export from.
            path: Relative path within /workspace to archive (default: ".").

        Returns:
            Tar archive bytes for the requested path.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If path validation fails.
            FileNotFoundError: If the requested path does not exist.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate export path is inside the sandbox workspace.
        is_valid, error_msg, resolved_path = validate_path(
            sandbox_info.workspace_path, path
        )
        if not is_valid:
            raise ValueError(error_msg)

        archive_bytes = await asyncio.get_running_loop().run_in_executor(
            None,
            self._get_archive_bytes_from_container,
            sandbox_info.container_id,
            resolved_path,
            path,
        )

        return archive_bytes

    def _get_archive_bytes_from_container(
        self,
        container_id: str,
        absolute_path: str,
        requested_path: str,
    ) -> bytes:
        """Read a tar archive stream from the container (blocking operation)."""
        container = self.client.containers.get(container_id)

        try:
            bits, _ = container.get_archive(absolute_path)
        except NotFound as e:
            raise FileNotFoundError(f"Path not found: {requested_path}") from e

        tar_stream = BytesIO()
        for chunk in bits:
            tar_stream.write(chunk)

        return tar_stream.getvalue()

    def _find_files_in_container(
        self,
        container_id: str,
        absolute_path: str,
        prefix: str,
        ignored_dir_names: set[str],
        max_entries: int,
    ) -> list[FileInfo]:
        """Find files/directories via `find` (blocking operation)."""
        container = self.client.containers.get(container_id)

        cmd: list[str] = [
            "find",
            absolute_path,
            "-mindepth",
            "1",
        ]

        # Prune common large dirs anywhere under the target path
        if ignored_dir_names:
            cmd.extend(["-type", "d", "("])
            first = True
            for name in sorted(ignored_dir_names):
                if not first:
                    cmd.append("-o")
                cmd.extend(["-name", name])
                first = False
            cmd.extend([")", "-prune", "-o"])

        # %y=type, %P=path relative to starting-point, %s=size (bytes)
        cmd.extend(["-printf", "%y\t%P\t%s\n"])

        result = container.exec_run(
            cmd,
            user="node",
            workdir="/workspace",
        )

        if result.exit_code != 0:
            return []

        output = result.output.decode("utf-8", errors="replace") if result.output else ""
        files: list[FileInfo] = []

        for line in output.splitlines():
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue

            type_char, rel_path, size_str = parts
            if not rel_path:
                continue

            full_path = f"{prefix}/{rel_path}" if prefix else rel_path
            name = full_path.split("/")[-1]
            is_directory = type_char == "d"

            size = 0
            if size_str.isdigit():
                try:
                    size = int(size_str)
                except ValueError:
                    size = 0

            files.append(
                FileInfo(
                    name=name,
                    path=full_path,
                    is_directory=is_directory,
                    size=size,
                )
            )

            if max_entries > 0 and len(files) >= max_entries:
                logger.warning(
                    "list_files_recursive_truncated",
                    container_id=container_id[:12],
                    path=absolute_path,
                    max_entries=max_entries,
                )
                break

        return files

    async def execute_command(
        self, sandbox_id: str, command: str, timeout: int = 60
    ) -> CommandResult:
        """Execute a shell command inside the sandbox.

        Validates the command against security policy before execution.

        Args:
            sandbox_id: The sandbox to execute in.
            command: The shell command to run.
            timeout: Maximum execution time in seconds (default: 60).

        Returns:
            CommandResult with stdout, stderr, and exit code.

        Raises:
            KeyError: If sandbox doesn't exist.
            ValueError: If command validation fails.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Validate command according to configured policy.
        is_valid, error_msg = validate_command(
            command,
            unrestricted=settings.sandbox_allow_unrestricted_commands,
        )
        if not is_valid:
            return CommandResult(
                stdout="",
                stderr=f"Command rejected: {error_msg}",
                exit_code=1,
                timed_out=False,
            )

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    self._execute_in_container,
                    sandbox_info.container_id,
                    command,
                ),
                timeout=timeout,
            )
            logger.debug(
                "command_executed",
                sandbox_id=sandbox_id,
                command=command[:50],
                exit_code=result.exit_code,
            )
            return result

        except TimeoutError:
            logger.warning(
                "command_timeout",
                sandbox_id=sandbox_id,
                command=command[:50],
                timeout=timeout,
            )
            return CommandResult(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=124,
                timed_out=True,
            )

    def _execute_in_container(self, container_id: str, command: str) -> CommandResult:
        """Execute command in container (blocking operation)."""
        container = self.client.containers.get(container_id)

        result = container.exec_run(
            ["/bin/bash", "-lc", command],
            user="node",
            workdir="/workspace",
            demux=True,
        )

        stdout_bytes: bytes = b""
        stderr_bytes: bytes = b""
        if isinstance(result.output, tuple):
            stdout_bytes = result.output[0] or b""
            stderr_bytes = result.output[1] or b""
        elif result.output:
            # Fallback for older docker-py behavior when demux is unsupported.
            stdout_bytes = result.output

        return CommandResult(
            stdout=sanitize_output(stdout_bytes.decode("utf-8", errors="replace")),
            stderr=sanitize_output(stderr_bytes.decode("utf-8", errors="replace")),
            exit_code=result.exit_code,
            timed_out=False,
        )

    async def start_dev_server(self, sandbox_id: str) -> str:
        """Start Vite dev server in the sandbox.

        Args:
            sandbox_id: The sandbox to start the dev server in.

        Returns:
            The preview URL (e.g., "http://localhost:5173").

        Raises:
            KeyError: If sandbox doesn't exist.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        # Start the dev server in the background (with timeout)
        await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                None,
                self._start_dev_server_in_container,
                sandbox_info.container_id,
                sandbox_info.port,
            ),
            timeout=30,
        )

        preview_url = f"http://localhost:{sandbox_info.port}"
        logger.info(
            "dev_server_started",
            sandbox_id=sandbox_id,
            url=preview_url,
        )

        return preview_url

    def _start_dev_server_in_container(self, container_id: str, port: int) -> None:
        """Start dev server in container (blocking operation)."""
        container = self.client.containers.get(container_id)

        # If it's already responding, don't start a second dev server.
        if self._is_http_responding(container, port):
            return

        # Start a dev server in the background (detached).
        # Prefer project-defined `npm run dev`, then fallback to global vite.
        start_cmd = f"""
if [ -f package.json ] && npm run dev -- --host 0.0.0.0 --port {port}; then
  :
elif command -v vite >/dev/null 2>&1; then
  vite --host 0.0.0.0 --port {port}
else
  exit 1
fi >/tmp/devserver.log 2>&1 &
"""
        container.exec_run(
            ["/bin/bash", "-lc", start_cmd],
            user="node",
            workdir="/workspace",
            detach=True,
        )

        # Wait until the server responds (best-effort readiness check)
        deadline = time.time() + 20
        while time.time() < deadline:
            if self._is_http_responding(container, port):
                return
            time.sleep(1)

        raise RuntimeError(f"Dev server did not become ready on port {port}")

    async def stop_dev_server(self, sandbox_id: str) -> bool:
        """Stop the dev server in the sandbox.

        Args:
            sandbox_id: The sandbox to stop the dev server in.

        Returns:
            True if a server was stopped, False if nothing was running.

        Raises:
            KeyError: If sandbox doesn't exist.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        return await asyncio.get_running_loop().run_in_executor(
            None,
            self._stop_dev_server_in_container,
            sandbox_info.container_id,
            sandbox_info.port,
        )

    def _stop_dev_server_in_container(self, container_id: str, port: int) -> bool:
        """Stop dev server using the port (blocking operation)."""
        container = self.client.containers.get(container_id)

        # Check if anything is listening on the port
        if not self._is_http_responding(container, port):
            return False

        # Kill process using the port. Try fuser first,
        # then fallback to pkill common dev-server processes.
        stop_cmd = (
            f"fuser -k -n tcp {port}"
            " || pkill -f 'vite|next|webpack|react-scripts'"
        )

        container.exec_run(
            ["/bin/bash", "-c", stop_cmd],
            user="root",  # Ensures we can kill the process
        )

        # Wait a moment for it to die
        time.sleep(1)
        return True

    def _is_http_responding(self, container: docker.models.containers.Container, port: int) -> bool:
        """Return True if the container responds on http://127.0.0.1:{port}."""
        try:
            result = container.exec_run(
                [
                    "/bin/bash",
                    "-lc",
                    f"curl -sSf --max-time 1 http://127.0.0.1:{port}/ >/dev/null",
                ],
                user="node",
                workdir="/workspace",
            )
            return result.exit_code == 0
        except Exception:
            return False

    async def get_preview_url(self, sandbox_id: str) -> str | None:
        """Get the current preview URL if dev server is running.

        Args:
            sandbox_id: The sandbox to check.

        Returns:
            The preview URL if server is running, None otherwise.
        """
        try:
            sandbox_info = self._get_sandbox(sandbox_id)
            return f"http://localhost:{sandbox_info.port}"
        except KeyError:
            return None

    async def copy_files_between(
        self, source_id: str, target_id: str, paths: list[str]
    ) -> None:
        """Copy files from one sandbox to another.

        Used in decomposition mode to aggregate work from sub-agents.

        Args:
            source_id: The source sandbox ID.
            target_id: The target sandbox ID.
            paths: List of relative paths to copy.

        Raises:
            KeyError: If either sandbox doesn't exist.
            ValueError: If path validation fails.
        """
        source_info = self._get_sandbox(source_id)
        target_info = self._get_sandbox(target_id)

        for path in paths:
            # Validate paths
            is_valid, error_msg, _ = validate_path(
                source_info.workspace_path, path
            )
            if not is_valid:
                raise ValueError(f"Invalid source path '{path}': {error_msg}")

            is_valid, error_msg, _ = validate_path(
                target_info.workspace_path, path
            )
            if not is_valid:
                raise ValueError(f"Invalid target path '{path}': {error_msg}")

            # Read from source and write to target
            try:
                content = await self.read_file(source_id, path)
                await self.write_file(target_id, path, content)
                logger.debug(
                    "file_copied",
                    source_id=source_id,
                    target_id=target_id,
                    path=path,
                )
            except FileNotFoundError:
                logger.debug(
                    "file_copy_skipped",
                    source_id=source_id,
                    path=path,
                    reason="not_found",
                )

    def _get_sandbox(self, sandbox_id: str) -> SandboxInfo:
        """Get sandbox info or raise KeyError."""
        if sandbox_id not in self._sandboxes:
            raise KeyError(f"Sandbox '{sandbox_id}' not found")
        return self._sandboxes[sandbox_id]

    async def cleanup_all(self) -> None:
        """Clean up all sandbox containers.

        Called during shutdown to ensure no containers are left running.
        """
        sandbox_ids = list(self._sandboxes.keys())
        for sandbox_id in sandbox_ids:
            try:
                await self.destroy_sandbox(sandbox_id)
            except Exception as e:
                logger.error(
                    "cleanup_failed",
                    sandbox_id=sandbox_id,
                    error=str(e),
                )

    async def get_sandbox_status(self, sandbox_id: str) -> dict[str, object]:
        """Get detailed status of a sandbox.

        Args:
            sandbox_id: The sandbox to check.

        Returns:
            Dictionary with sandbox status details.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        try:
            container = await asyncio.get_running_loop().run_in_executor(
                None,
                self.client.containers.get,
                sandbox_info.container_id,
            )

            return {
                "sandbox_id": sandbox_id,
                "container_id": sandbox_info.container_id[:12],
                "port": sandbox_info.port,
                "status": container.status,
                "running": container.status == "running",
            }
        except NotFound:
            return {
                "sandbox_id": sandbox_id,
                "container_id": sandbox_info.container_id[:12],
                "port": sandbox_info.port,
                "status": "not_found",
                "running": False,
            }

    async def health_check(self, sandbox_id: str) -> dict[str, object]:
        """Perform a health check on a sandbox container.

        Checks container status, runs a liveness probe (``echo ok``), and
        gathers basic resource information (memory usage, uptime).

        Args:
            sandbox_id: The sandbox to health-check.

        Returns:
            Dictionary with health details including:
            - healthy (bool): Whether the container passed the liveness check
            - status (str): Docker container status
            - memory_usage_mb (float | None): Current memory usage in MB
            - uptime_seconds (float | None): Container uptime
            - liveness (str): Result of the liveness probe

        Raises:
            KeyError: If sandbox doesn't exist.
        """
        sandbox_info = self._get_sandbox(sandbox_id)

        try:
            result = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    self._health_check_blocking,
                    sandbox_info.container_id,
                ),
                timeout=30,
            )
            return {
                "sandbox_id": sandbox_id,
                **result,
            }
        except NotFound:
            return {
                "sandbox_id": sandbox_id,
                "healthy": False,
                "status": "not_found",
                "memory_usage_mb": None,
                "uptime_seconds": None,
                "liveness": "container_not_found",
            }
        except Exception as e:
            logger.error(
                "health_check_error",
                sandbox_id=sandbox_id,
                error=str(e),
            )
            return {
                "sandbox_id": sandbox_id,
                "healthy": False,
                "status": "error",
                "memory_usage_mb": None,
                "uptime_seconds": None,
                "liveness": f"error: {e}",
            }

    def _health_check_blocking(self, container_id: str) -> dict[str, object]:
        """Perform a blocking health check on a container.

        Args:
            container_id: Docker container ID.

        Returns:
            Health check results dictionary.
        """
        container = self.client.containers.get(container_id)

        # Container status
        container_status = container.status

        # Liveness probe: execute 'echo ok' inside the container
        liveness = "unknown"
        try:
            result = container.exec_run(
                ["echo", "ok"],
                user="node",
            )
            if result.exit_code == 0 and b"ok" in (result.output or b""):
                liveness = "ok"
            else:
                liveness = f"exit_code={result.exit_code}"
        except Exception as e:
            liveness = f"exec_failed: {e}"

        # Memory usage from container stats (one-shot, non-streaming)
        memory_usage_mb: float | None = None
        uptime_seconds: float | None = None
        try:
            stats = container.stats(stream=False)
            memory_stats = stats.get("memory_stats", {})
            usage_bytes = memory_stats.get("usage")
            if usage_bytes is not None:
                memory_usage_mb = round(usage_bytes / (1024 * 1024), 2)

            # Uptime from container start time
            attrs = container.attrs
            if attrs:
                state = attrs.get("State", {})
                started_at_str = state.get("StartedAt", "")
                if started_at_str:
                    from datetime import datetime
                    # Docker timestamps like "2024-01-15T10:30:00.123456789Z"
                    try:
                        started_at = datetime.fromisoformat(
                            started_at_str.replace("Z", "+00:00")
                        )
                        now = datetime.now(UTC)
                        uptime_seconds = round(
                            (now - started_at).total_seconds(), 1
                        )
                    except (ValueError, TypeError):
                        pass
        except Exception:
            # Stats collection is best-effort
            pass

        healthy = container_status == "running" and liveness == "ok"

        return {
            "healthy": healthy,
            "status": container_status,
            "memory_usage_mb": memory_usage_mb,
            "uptime_seconds": uptime_seconds,
            "liveness": liveness,
        }

    async def is_healthy(self, sandbox_id: str) -> bool:
        """Quick check if a sandbox container is running and responsive.

        Args:
            sandbox_id: The sandbox to check.

        Returns:
            True if the container is running and the liveness probe passes.
        """
        try:
            result = await self.health_check(sandbox_id)
            return bool(result.get("healthy", False))
        except KeyError:
            return False

    def is_docker_available(self) -> bool:
        """Check if the Docker daemon is reachable.

        Returns:
            True if the Docker daemon responds to a ping.
        """
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_active_sandbox_count(self) -> int:
        """Return the number of currently tracked sandboxes.

        Returns:
            Number of sandboxes in the internal registry.
        """
        return len(self._sandboxes)

    def get_active_sandbox_ids(self) -> list[str]:
        """Return the IDs of all currently tracked sandboxes.

        Returns:
            List of active sandbox IDs.
        """
        return list(self._sandboxes.keys())
