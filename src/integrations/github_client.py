"""
GitHub Client for Facilitair
Deploy generated code directly to GitHub repositories
Uses GitHub CLI (gh) for simple, reliable GitHub integration
"""
import os
import subprocess
from typing import Dict, Any, Optional, List
import logging
import tempfile
import shutil

logger = logging.getLogger(__name__)


class GitHubClient:
    """
    Client for GitHub repository operations using GitHub CLI (gh)

    Provides:
    1. Authentication check
    2. Repository creation
    3. Code deployment to new/existing repos
    4. Branch management
    5. Pull request creation

    Requires: GitHub CLI (gh) installed and authenticated
    Install: https://cli.github.com/
    Auth: gh auth login
    """

    def __init__(self):
        """Initialize GitHub client"""
        self.gh_available = self._check_gh_cli()
        self.authenticated_user = self._get_authenticated_user() if self.gh_available else None

    def _check_gh_cli(self) -> bool:
        """Check if GitHub CLI is installed and authenticated"""
        try:
            # Check if gh is installed
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.warning("[GITHUB] GitHub CLI (gh) not installed")
                return False

            # Check if authenticated
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.warning("[GITHUB] GitHub CLI not authenticated. Run: gh auth login")
                return False

            logger.info("[GITHUB] GitHub CLI available and authenticated ✓")
            return True

        except FileNotFoundError:
            logger.warning("[GITHUB] GitHub CLI (gh) not found in PATH")
            return False
        except Exception as e:
            logger.warning(f"[GITHUB] Error checking GitHub CLI: {e}")
            return False

    def _get_authenticated_user(self) -> Optional[str]:
        """Get authenticated GitHub username"""
        try:
            result = subprocess.run(
                ["gh", "api", "user", "-q", ".login"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                username = result.stdout.strip()
                logger.info(f"[GITHUB] Authenticated as: {username}")
                return username
            return None
        except Exception as e:
            logger.warning(f"[GITHUB] Could not get authenticated user: {e}")
            return None

    def is_authenticated(self) -> bool:
        """Check if user is authenticated with GitHub"""
        return self.gh_available

    async def deploy_code(
        self,
        repo_name: str,
        files: Dict[str, str],
        description: str,
        task: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False,
        branch_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy generated code to GitHub repository

        Args:
            repo_name: Repository name (e.g., "my-api" or "username/my-api")
            files: Dict of {filepath: content}
            description: Repository/commit description
            task: Original task description (for commit message)
            private: Make repository private (new repos only)
            create_pr: Create pull request instead of pushing to main
            branch_name: Custom branch name (default: facilitair-TIMESTAMP)

        Returns:
            Dict with {success, url, pr_url, error, branch}
        """
        if not self.gh_available:
            return {
                "success": False,
                "error": "GitHub CLI not available or not authenticated. Run: gh auth login",
                "url": None,
                "pr_url": None,
                "branch": None
            }

        # Create temporary directory for git operations
        temp_dir = tempfile.mkdtemp(prefix="facilitair_")

        try:
            logger.info(f"[GITHUB] Deploying code to repository: {repo_name}")

            # Write files to temp directory
            for filepath, content in files.items():
                full_path = os.path.join(temp_dir, filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(content)

            logger.info(f"[GITHUB] Created {len(files)} files in temp directory")

            # Initialize git repo
            subprocess.run(
                ["git", "init"],
                cwd=temp_dir,
                capture_output=True,
                check=True
            )

            # Configure git user using GitHub CLI authenticated user
            username = self.authenticated_user or "Facilitair"
            email = f"{username}@users.noreply.github.com"

            subprocess.run(
                ["git", "config", "user.name", username],
                cwd=temp_dir,
                capture_output=True,
                check=True
            )
            subprocess.run(
                ["git", "config", "user.email", email],
                cwd=temp_dir,
                capture_output=True,
                check=True
            )

            logger.info(f"[GITHUB] Configured git identity: {username} <{email}>")

            # Add all files
            subprocess.run(
                ["git", "add", "."],
                cwd=temp_dir,
                capture_output=True,
                check=True
            )

            # Verify files were added
            status_result = subprocess.run(
                ["git", "status", "--short"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"[GITHUB] Files staged: {len(status_result.stdout.splitlines())} files")

            # Create commit
            commit_msg = self._build_commit_message(task or description)
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"[GITHUB] Commit created successfully")

            # Check if repo exists
            repo_exists = self._check_repo_exists(repo_name)

            if repo_exists:
                logger.info(f"[GITHUB] Repository exists, deploying to existing repo")
                return await self._deploy_to_existing_repo(
                    temp_dir, repo_name, create_pr, branch_name
                )
            else:
                logger.info(f"[GITHUB] Repository does not exist, creating new repo")
                return await self._create_new_repo(
                    temp_dir, repo_name, description, private
                )

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else str(e))
            logger.error(f"[GITHUB] Git command failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "url": None,
                "pr_url": None,
                "branch": None
            }
        except Exception as e:
            logger.error(f"[GITHUB] Error deploying code: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": None,
                "pr_url": None,
                "branch": None
            }
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
                logger.debug(f"[GITHUB] Cleaned up temp directory")
            except Exception as e:
                logger.warning(f"[GITHUB] Could not clean up temp dir: {e}")

    def _check_repo_exists(self, repo_name: str) -> bool:
        """Check if repository exists"""
        try:
            result = subprocess.run(
                ["gh", "repo", "view", repo_name],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _create_new_repo(
        self,
        temp_dir: str,
        repo_name: str,
        description: str,
        private: bool
    ) -> Dict[str, Any]:
        """Create new GitHub repository and push code"""

        # Create GitHub repository
        visibility = "--private" if private else "--public"
        result = subprocess.run(
            ["gh", "repo", "create", repo_name, visibility,
             "--description", description,
             "--source", temp_dir,
             "--push"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            logger.error(f"[GITHUB] Failed to create repository: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "url": None,
                "pr_url": None,
                "branch": None
            }

        # Get repository URL
        result = subprocess.run(
            ["gh", "repo", "view", repo_name, "--json", "url", "-q", ".url"],
            capture_output=True,
            text=True,
            timeout=10
        )

        repo_url = result.stdout.strip() if result.returncode == 0 else None

        logger.info(f"[GITHUB] ✓ Repository created: {repo_url}")

        return {
            "success": True,
            "url": repo_url,
            "pr_url": None,
            "branch": "main",
            "error": None
        }

    async def _deploy_to_existing_repo(
        self,
        temp_dir: str,
        repo_name: str,
        create_pr: bool,
        branch_name: Optional[str]
    ) -> Dict[str, Any]:
        """Deploy code to existing repository"""

        # Generate branch name if not provided
        if not branch_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            branch_name = f"facilitair-{timestamp}"

        # Add remote
        subprocess.run(
            ["gh", "repo", "set-default", repo_name],
            cwd=temp_dir,
            capture_output=True,
            check=True
        )

        # Fetch to get latest
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=temp_dir,
            capture_output=True,
            check=True
        )

        # Create and checkout new branch
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=temp_dir,
            capture_output=True,
            check=True
        )

        # Push to remote
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"[GITHUB] ✓ Pushed to branch: {branch_name}")

        # Get repository URL
        result = subprocess.run(
            ["gh", "repo", "view", repo_name, "--json", "url", "-q", ".url"],
            capture_output=True,
            text=True,
            timeout=10
        )
        repo_url = result.stdout.strip() if result.returncode == 0 else None

        pr_url = None
        if create_pr:
            # Create pull request
            result = subprocess.run(
                ["gh", "pr", "create",
                 "--title", f"Facilitair: {branch_name}",
                 "--body", "🤖 Code generated by Facilitair AI\n\nReview and merge when ready.",
                 "--head", branch_name],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                pr_url = result.stdout.strip()
                logger.info(f"[GITHUB] ✓ Pull request created: {pr_url}")
            else:
                logger.warning(f"[GITHUB] Could not create PR: {result.stderr}")

        return {
            "success": True,
            "url": repo_url,
            "pr_url": pr_url,
            "branch": branch_name,
            "error": None
        }

    def _build_commit_message(self, task: str) -> str:
        """Build commit message with Facilitair branding"""
        return f"""{task}

🤖 Generated with Facilitair AI

This code was automatically generated by Facilitair's multi-agent
collaborative system using sequential workflow architecture.
"""

    def get_status(self) -> Dict[str, Any]:
        """Get GitHub client status"""
        return {
            "available": self.gh_available,
            "authenticated": self.is_authenticated(),
            "username": self.authenticated_user
        }


# Singleton instance
_github_client: Optional[GitHubClient] = None


def get_github_client() -> GitHubClient:
    """Get singleton GitHub client instance"""
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    return _github_client
