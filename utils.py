from langchain_community.document_loaders import PyPDFLoader

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def parse_resume(path_to_resume) -> str:
    loader = PyPDFLoader(path_to_resume)
    pages = loader.load()
    content = ""
    for page in pages:
        content += page.page_content

    return content
