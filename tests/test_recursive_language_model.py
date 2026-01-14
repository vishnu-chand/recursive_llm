import unittest
from unittest.mock import MagicMock, patch, mock_open
from src.rlm import RecursiveLanguageModel, RLMEnvironment


class TestRecursiveLanguageModel(unittest.TestCase):
    def setUp(self):
        self.model_name = "gpt-3.5-turbo"
        self.rlm = RecursiveLanguageModel(model_name=self.model_name)
        self.context = """This is a test context with some information.
        It contains multiple lines of text for testing purposes.
        The quick brown fox jumps over the lazy dog.
        """
        self.messages = [{"role": "user", "content": "What is the test about?"}]
        self.max_chunk_size = 1000
        self.max_output_chars = 1000
        self.max_steps = 5

    @patch("src.rlm.RLMEnvironment")
    @patch("src.rlm.litellm.completion")
    def test_run_success(self, mock_completion, mock_rlm_env_class):
        # Setup mock environment
        mock_env = MagicMock()
        mock_env.execute.side_effect = [None, SystemExit(0)]  # First call does nothing, second call exits
        mock_rlm_env_class.return_value = mock_env

        # Mock the LLM response for the system prompt
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"thought": "Test thought", "code": "FINAL("Test answer")"}'
        mock_completion.return_value = mock_response

        # Run the test
        result = self.rlm.run(context=self.context, messages=self.messages, max_chunk_size=self.max_chunk_size, max_output_chars=self.max_output_chars, max_steps=self.max_steps)

        # Verify the result
        self.assertEqual(result, "Test answer")
        mock_rlm_env_class.assert_called_once()
        self.assertEqual(mock_env.execute.call_count, 2)  # One for the code, one for the exit

    @patch("src.rlm.RLMEnvironment")
    @patch("src.rlm.litellm.completion")
    def test_run_max_steps_reached(self, mock_completion, mock_rlm_env_class):
        # Setup mock environment to never exit
        mock_env = MagicMock()
        mock_rlm_env_class.return_value = mock_env

        # Mock the LLM response to not call FINAL
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"thought": "Test thought", "code": "print("Test")"}'
        mock_completion.return_value = mock_response

        # Run the test with max_steps=1
        with self.assertRaises(RuntimeError) as context:
            self.rlm.run(context=self.context, messages=self.messages, max_chunk_size=self.max_chunk_size, max_output_chars=self.max_output_chars, max_steps=1)  # Will exit after 1 step without FINAL

        # Verify the error message
        self.assertIn("Maximum number of steps reached", str(context.exception))
        self.assertEqual(mock_env.execute.call_count, 1)

    @patch("src.rlm.RLMEnvironment")
    @patch("src.rlm.litellm.completion")
    def test_run_with_invalid_json_response(self, mock_completion, mock_rlm_env_class):
        # Setup mock environment
        mock_env = MagicMock()
        mock_rlm_env_class.return_value = mock_env

        # Mock the LLM response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_completion.return_value = mock_response

        # Run the test
        with self.assertRaises(ValueError) as context:
            self.rlm.run(context=self.context, messages=self.messages, max_chunk_size=self.max_chunk_size, max_output_chars=self.max_output_chars, max_steps=self.max_steps)

        # Verify the error message
        self.assertIn("Failed to parse LLM response as JSON", str(context.exception))

    @patch("src.rlm.RLMEnvironment")
    @patch("src.rlm.litellm.completion")
    def test_run_with_missing_fields_in_response(self, mock_completion, mock_rlm_env_class):
        # Setup mock environment
        mock_env = MagicMock()
        mock_rlm_env_class.return_value = mock_env

        # Mock the LLM response with missing fields
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"thought": "Only thought"}'  # Missing 'code' field
        mock_completion.return_value = mock_response

        # Run the test
        with self.assertRaises(ValueError) as context:
            self.rlm.run(context=self.context, messages=self.messages, max_chunk_size=self.max_chunk_size, max_output_chars=self.max_output_chars, max_steps=self.max_steps)

        # Verify the error message
        self.assertIn("Missing required fields in LLM response", str(context.exception))


if __name__ == "__main__":
    unittest.main()
