import docker
from docker.models.containers import Container
import logging
import os
import shutil
from typing import Optional, Tuple
import uuid
from docker.errors import ImageNotFound, NotFound
import json

logger = logging.getLogger(__name__)

import tempfile

SANDBOX_WORKSPACE_DIR = os.environ.get("SANDBOX_WORKSPACE_DIR", tempfile.gettempdir() + "/qagi_sandboxes")

class SandboxManager:
    """
    Manages the lifecycle of secure, isolated sandbox environments for agent code execution.
    """
    def __init__(self, base_image: str = "qagi/agent-sandbox:latest"):
        self.client = docker.from_env()
        self.base_image = base_image
        if not os.path.exists(SANDBOX_WORKSPACE_DIR):
            os.makedirs(SANDBOX_WORKSPACE_DIR)
        
        # Load the default seccomp profile
        seccomp_profile_path = os.path.join(os.path.dirname(__file__), 'seccomp_profile.json')
        if os.path.exists(seccomp_profile_path):
            with open(seccomp_profile_path, 'r') as f:
                self.seccomp_profile = json.load(f)
        else:
            self.seccomp_profile = None

        logger.info(f"SandboxManager initialized. Workspaces in: {SANDBOX_WORKSPACE_DIR}")

    def create_sandbox(self, network_enabled: bool = False) -> Container:
        """
        Creates a new sandbox container with a dedicated workspace volume and security constraints.
        """
        workspace_id = str(uuid.uuid4())
        host_workspace_path = os.path.join(SANDBOX_WORKSPACE_DIR, workspace_id)
        os.makedirs(host_workspace_path)

        security_opts = [f"seccomp={json.dumps(self.seccomp_profile)}"] if self.seccomp_profile else []

        try:
            container = self.client.containers.run(
                self.base_image,
                detach=True,
                tty=True,
                volumes={host_workspace_path: {'bind': '/workspace', 'mode': 'rw'}},
                labels={'qagi_sandbox_workspace_id': workspace_id},
                # Resource limits
                mem_limit="512m",
                cpu_shares=512, # Relative weight, 1024 is default
                # Security options
                network_disabled=not network_enabled,
                security_opt=security_opts
            )
            logger.info(f"Created sandbox container: {container.id} with workspace: {workspace_id}")
            return container
        except ImageNotFound:
            logger.error(f"Base image not found: {self.base_image}. Please build it first.")
            shutil.rmtree(host_workspace_path)
            raise
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            shutil.rmtree(host_workspace_path)
            raise

    def upload_file_to_sandbox(self, container: Container, host_path: str, sandbox_path: str):
        """
        Uploads a file to the sandbox's workspace.
        """
        workspace_id = container.labels.get('qagi_sandbox_workspace_id')
        if not workspace_id:
            raise ValueError("Container is not a valid sandbox (missing workspace label).")
        
        destination = os.path.join(SANDBOX_WORKSPACE_DIR, workspace_id, sandbox_path)
        shutil.copy(host_path, destination)
        logger.info(f"Uploaded {host_path} to {container.id}:{sandbox_path}")

    def download_file_from_sandbox(self, container: Container, sandbox_path: str, host_path: str):
        """
        Downloads a file from the sandbox's workspace.
        """
        workspace_id = container.labels.get('qagi_sandbox_workspace_id')
        if not workspace_id:
            raise ValueError("Container is not a valid sandbox (missing workspace label).")

        source = os.path.join(SANDBOX_WORKSPACE_DIR, workspace_id, sandbox_path)
        shutil.copy(source, host_path)
        logger.info(f"Downloaded {container.id}:{sandbox_path} to {host_path}")

    def execute_in_sandbox(self, container: Container, command: str) -> Tuple[int, str]:
        """
        Executes a command inside a running sandbox container.

        Args:
            container: The container to execute the command in.
            command: The command to execute.

        Returns:
            A tuple containing the exit code and the output of the command.
        """
        exit_code, output = container.exec_run(command)
        logger.info(f"Executed command in {container.id} with exit code {exit_code}")
        return exit_code, output.decode('utf-8')

    def remove_sandbox(self, container: Container):
        """
        Stops and removes a sandbox container and its associated workspace.
        """
        workspace_id = container.labels.get('qagi_sandbox_workspace_id')
        host_workspace_path = os.path.join(SANDBOX_WORKSPACE_DIR, workspace_id) if workspace_id else None

        try:
            container.stop()
            container.remove()
            logger.info(f"Removed sandbox container: {container.id}")
        except Exception as e:
            logger.error(f"Failed to remove sandbox container {container.id}: {e}")
        finally:
            if host_workspace_path and os.path.exists(host_workspace_path):
                shutil.rmtree(host_workspace_path)
                logger.info(f"Cleaned up workspace: {host_workspace_path}")

    def get_sandbox(self, container_id: str) -> Optional[Container]:
        """
        Retrieves a container by its ID.
        """
        try:
            return self.client.containers.get(container_id)
        except NotFound:
            return None

# Singleton instance
sandbox_manager = SandboxManager() 