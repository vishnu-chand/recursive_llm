import unittest
from unittest.mock import MagicMock, patch
from src.rlm import RLMEnvironment, SnippetAnalysis


class TestRLMEnvironment(unittest.TestCase):
    def setUp(self):
        self.context = """This is a test context with some information.
        It contains multiple lines of text for testing purposes.
        The quick brown fox jumps over the lazy dog.
        """
        self.model_name = "gpt-3.5-turbo"
        self.max_output_chars = 1000
        self.env = RLMEnvironment(self.context, self.model_name, self.max_output_chars)

    def test_initialization(self):
        self.assertEqual(self.env.context, self.context)
        self.assertEqual(self.env.model_name, self.model_name)
        self.assertEqual(self.env.max_output_chars, self.max_output_chars)
        self.assertIsNone(self.env.answer)
        self.assertIn("CONTEXT", self.env.globals)
        self.assertIn("llm_query", self.env.globals)
        self.assertIn("FINAL", self.env.globals)

    @patch("src.rlm.litellm.completion")
    def test_llm_query_success(self, mock_completion):
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"is_relevant": true, "answer": "Test answer"}'
        mock_completion.return_value = mock_response

        prompt = "Test prompt"
        snippet = "Test snippet"
        result = self.env.llm_query(prompt, snippet)

        self.assertEqual(result, "Test answer")
        mock_completion.assert_called_once()

    def test_final(self):
        answer = "Final test answer"
        with self.assertRaises(SystemExit) as cm:
            self.env.final(answer)
        self.assertEqual(self.env.answer, answer)
        self.assertEqual(cm.exception.code, 0)

    def test_execute_valid_code(self):
        code = "result = 2 + 2"
        result = self.env.execute(code)
        self.assertEqual(result, 4)
        self.assertEqual(self.env.globals["result"], 4)

    def test_execute_with_error(self):
        code = "result = 2 + '2'"  # Type error
        with self.assertRaises(Exception):
            self.env.execute(code)

    def test_snippet_analysis_model(self):
        # Test that the SnippetAnalysis Pydantic model works as expected
        analysis = SnippetAnalysis(is_relevant=True, answer="Test answer")
        self.assertTrue(analysis.is_relevant)
        self.assertEqual(analysis.answer, "Test answer")


if __name__ == "__main__":
    unittest.main()
