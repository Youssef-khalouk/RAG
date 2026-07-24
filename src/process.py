import string


class Process:
    translation1 = str.maketrans({"-": " ", ".": " ", "'": " "})
    translation2 = str.maketrans("", "", string.punctuation)
    translation1_code = str.maketrans(
        {"_": " ", ".": " ", '"': ' ',
         "'": " ", ":": " ", "(": " ", ")": " "})
    translation2_code = str.maketrans("", "", string.punctuation)

    @staticmethod
    def preprocess_doc(text: str) -> list[str]:
        return (text.lower()
                .translate(Process.translation1)
                .translate(Process.translation2))

    @staticmethod
    def preprocess_code(text: str) -> str:
        return (text.lower()
                .translate(Process.translation1_code)
                .translate(Process.translation2_code))
