"""Search the full bundled document collection with cited answers."""

from examples.rag_basic import create_pipeline

if __name__ == "__main__":
    response = create_pipeline().query(
        "Compare causes of tunnel defects and surface cracks in FSW."
    )
    print(response.answer)
    for source in response.sources:
        print(source.doc_name, source.page_num)
