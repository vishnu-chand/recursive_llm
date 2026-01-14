import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.app import load_context_from_file


class TestAppUtils(unittest.TestCase):
    def test_load_context_from_file_success(self):
        # Create a temporary file with test content
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            test_content = "This is a test file content."
            temp_file.write(test_content)
            temp_file_path = temp_file.name

        try:
            # Test loading the file
            result, status = load_context_from_file(temp_file_path)
            self.assertEqual(result, test_content)
            self.assertIn("File loaded successfully", str(status))
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_load_context_from_file_not_found(self):
        # Test with non-existent file
        result, status = load_context_from_file("/nonexistent/file.txt")
        self.assertIsNone(result)
        self.assertIn("Failed to read file", str(status))

    def test_load_context_from_file_permission_error(self):
        # Create a temporary file and make it unreadable
        with tempfile.NamedTemporaryFile(mode='w+') as temp_file:
            temp_file.write("test")
            temp_file_path = temp_file.name
            # Make the file unreadable
            os.chmod(temp_file_path, 0o000)

            try:
                # Test loading the file with permission error
                result, status = load_context_from_file(temp_file_path)
                self.assertIsNone(result)
                self.assertIn("Failed to read file", str(status))
            finally:
                # Make the file deletable again and clean up
                os.chmod(temp_file_path, 0o644)
                os.unlink(temp_file_path)

    @patch('src.app.RecursiveLanguageModel')
    def test_generate_bot_response_no_context(self, mock_rlm_class):
        from src.app import generate_bot_response

        # Test with no context
        history = [{"role": "user", "content": "Test message"}]
        result = generate_bot_response(history, None, "gpt-3.5-turbo")

        # The response should indicate no context is loaded
        self.assertIn("No Context Loaded", result[0]["content"])
        mock_rlm_class.assert_not_called()

    @patch('src.app.RecursiveLanguageModel')
    def test_generate_bot_response_with_context(self, mock_rlm_class):
        from src.app import generate_bot_response

        # Setup mock
        mock_rlm = MagicMock()
        mock_rlm.run.return_value = "Test response"
        mock_rlm_class.return_value = mock_rlm

        # Test with context
        history = [{"role": "user", "content": "Test message"}]
        context = "Test context"
        result = generate_bot_response(history, context, "gpt-3.5-turbo")

        # Verify the RLM was called with the correct parameters
        mock_rlm_class.assert_called_once_with(model_name="gpt-3.5-turbo")
        mock_rlm.run.assert_called_once()
        self.assertEqual(len(result), 2)  # Original message + response
        self.assertEqual(result[1]["content"], "Test response")


if __name__ == '__main__':
    unittest.main()
