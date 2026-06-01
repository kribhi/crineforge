import os
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TikaExtractor:
    """
    Extracts text from document formats unsupported by the core DataExtractor
    using Apache Tika. Handles DOCX, PPTX, HTML, Email (.eml, .msg), and
    source code files (.py, .js, .ts, .java, .cpp, .go).

    All output is normalized to UTF-8 text.
    """

    SUPPORTED_EXTENSIONS = {
        ".docx", ".pptx", ".html", ".htm", ".eml", ".msg",
        ".py", ".js", ".ts", ".java", ".cpp", ".go",
    }

    # Source code files can be read directly without Tika
    _SOURCE_CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".go"}

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Checks if the file extension is supported by TikaExtractor."""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in TikaExtractor.SUPPORTED_EXTENSIONS

    @staticmethod
    def extract(file_path: str) -> str:
        """
        Extracts raw text from the given file path.

        For source code files, reads directly as UTF-8 text.
        For all other supported formats, delegates to Apache Tika.

        Returns:
            Extracted text content as a UTF-8 string.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
            RuntimeError: If Tika extraction fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}\n"
                f"Reason: The specified path does not exist.\n"
                f"Suggested fix: Verify the file path and try again."
            )

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in TikaExtractor.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {ext}\n"
                f"Reason: TikaExtractor does not handle '{ext}' files.\n"
                f"Suggested fix: Use one of: {', '.join(sorted(TikaExtractor.SUPPORTED_EXTENSIONS))}"
            )

        logger.info(f"[Ingestion] Extracting text from {ext} file: {file_path}")

        # Source code files — read directly for speed and reliability
        if ext in TikaExtractor._SOURCE_CODE_EXTENSIONS:
            return TikaExtractor._extract_source_code(file_path)

        # All other formats — delegate to Tika
        return TikaExtractor._extract_via_tika(file_path)

    @staticmethod
    def _extract_source_code(file_path: str) -> str:
        """Reads source code files directly as UTF-8 text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"[Ingestion] Source code extracted: {len(content)} characters")
            return content
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.warning(
                    f"[Ingestion] File {file_path} was not valid UTF-8. "
                    f"Read with latin-1 fallback."
                )
                return content
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read source code file: {file_path}\n"
                    f"Reason: {str(e)}\n"
                    f"Suggested fix: Ensure the file is a valid text file."
                ) from e

    @staticmethod
    def _extract_via_tika(file_path: str) -> str:
        """Uses Apache Tika to extract text from documents."""
        try:
            from tika import parser as tika_parser
        except ImportError:
            raise ImportError(
                "Apache Tika is not installed.\n"
                "Reason: The 'tika' Python package is required for DOCX/PPTX/HTML/Email extraction.\n"
                "Suggested fix: Run `pip install tika`"
            )

        try:
            parsed = tika_parser.from_file(file_path)

            if parsed is None or "content" not in parsed or parsed["content"] is None:
                logger.warning(
                    f"[Ingestion] Tika returned no content for {file_path}. "
                    f"File may be empty or corrupted."
                )
                return ""

            content = parsed["content"]

            # Normalize to clean UTF-8 text
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')

            content = content.strip()
            logger.info(f"[Ingestion] Tika extraction complete: {len(content)} characters")
            return content

        except Exception as e:
            raise RuntimeError(
                f"Tika extraction failed for: {file_path}\n"
                f"Reason: {str(e)}\n"
                f"Suggested fix: Ensure Java is installed (Tika requires JRE) "
                f"and the file is not corrupted."
            ) from e
