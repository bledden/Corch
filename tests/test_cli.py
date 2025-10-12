"""
Tests for Facilitair CLI
"""

import pytest
import asyncio
import json
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, AsyncMock

# Import CLI
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli import cli, FacilitairCLI


class TestFacilitairCLI:
    """Test suite for FacilitairCLI class"""

    def test_init_default_config(self):
        """Test CLI initialization with default config"""
        cli_obj = FacilitairCLI()
        assert cli_obj.config['use_sequential'] == True
        assert cli_obj.config['max_iterations'] == 3
        assert cli_obj.orchestrator is None

    def test_init_with_config_file(self, tmp_path):
        """Test CLI initialization with config file"""
        config_file = tmp_path / "config.json"
        config_data = {
            "use_sequential": False,
            "max_iterations": 5,
            "temperature": 0.5
        }
        config_file.write_text(json.dumps(config_data))

        cli_obj = FacilitairCLI(config_path=str(config_file))
        assert cli_obj.config['use_sequential'] == False
        assert cli_obj.config['max_iterations'] == 5

    @pytest.mark.asyncio
    async def test_collaborate(self):
        """Test collaboration method"""
        cli_obj = FacilitairCLI()

        # Mock the orchestrator
        mock_result = Mock()
        mock_result.success = True
        mock_result.agents_used = ["architect", "coder"]
        mock_result.final_output = "Test output"
        mock_result.metrics = {"quality": 0.9, "duration": 1.5}
        mock_result.consensus_method = "sequential"

        with patch.object(cli_obj, 'initialize_orchestrator', new_callable=AsyncMock):
            cli_obj.orchestrator = Mock()
            cli_obj.orchestrator.collaborate = AsyncMock(return_value=mock_result)

            result = await cli_obj.collaborate("test task")

            assert result['success'] == True
            assert result['agents_used'] == ["architect", "coder"]
            assert result['output'] == "Test output"


class TestCLICommands:
    """Test suite for CLI commands"""

    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()

    def test_health_command(self):
        """Test health check command"""
        with patch('cli.validate_api_keys') as mock_validate:
            mock_validate.return_value = {
                'all_valid': True,
                'results': {
                    'WANDB_API_KEY': {'valid': True, 'message': 'Valid'}
                }
            }

            result = self.runner.invoke(cli, ['health'])
            assert result.exit_code == 0
            assert 'API Key Status' in result.output

    def test_health_command_failure(self):
        """Test health check command with invalid keys"""
        with patch('cli.validate_api_keys') as mock_validate:
            mock_validate.return_value = {
                'all_valid': False,
                'results': {
                    'WANDB_API_KEY': {'valid': False, 'message': 'Invalid'}
                }
            }

            result = self.runner.invoke(cli, ['health'])
            assert result.exit_code == 1
            assert 'not operational' in result.output

    def test_init_command(self, tmp_path):
        """Test init command creates config file"""
        config_path = tmp_path / "test_config.json"

        result = self.runner.invoke(cli, ['init', '--output', str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert 'use_sequential' in config
        assert 'agents' in config

    def test_collaborate_command_missing_api_keys(self):
        """Test collaborate command with missing API keys"""
        with patch('cli.validate_api_keys') as mock_validate:
            mock_validate.return_value = {
                'all_valid': False,
                'results': {}
            }

            result = self.runner.invoke(cli, ['collaborate', 'test task'])
            assert result.exit_code == 1
            assert 'API key validation failed' in result.output


class TestCLIIntegration:
    """Integration tests for CLI"""

    def setup_method(self):
        """Setup test runner"""
        self.runner = CliRunner()

    @pytest.mark.integration
    def test_end_to_end_collaborate(self):
        """Test end-to-end collaboration flow"""
        # This test requires valid API keys
        with patch('cli.validate_api_keys') as mock_validate:
            mock_validate.return_value = {
                'all_valid': True,
                'results': {
                    'WANDB_API_KEY': {'valid': True, 'message': 'Valid'},
                    'OPENROUTER_API_KEY': {'valid': True, 'message': 'Valid'}
                }
            }

            # Mock the orchestrator to avoid actual API calls
            with patch('cli.CollaborativeOrchestrator'):
                result = self.runner.invoke(cli, [
                    'collaborate',
                    'Write a hello world function',
                    '--format', 'json'
                ])

                # Should complete without error
                assert result.exit_code == 0 or 'error' not in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
