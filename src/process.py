"""Text preprocessing helpers for document and code token normalization."""

import string


class Process:
    """Provide static preprocessing utilities for retrieval indexing."""
    translation1 = str.maketrans({"-": " ", ".": " ", "'": " "})
    translation2 = str.maketrans("", "", string.punctuation)
    translation1_code = str.maketrans(
        {"_": " ", ".": " ", '"': ' ',
         "'": " ", ":": " ", "(": " ", ")": " "})
    translation2_code = str.maketrans("", "", string.punctuation)

    @staticmethod
    def preprocess_doc(text: str) -> list[str]:
        """Normalize and tokenize document text for lexical matching."""
        return (text.lower()
                .translate(Process.translation1)
                .translate(Process.translation2))

    @staticmethod
    def preprocess_code(text: str) -> str:
        """Normalize and tokenize code text for lexical matching."""
        return (text.lower()
                .translate(Process.translation1_code)
                .translate(Process.translation2_code))
