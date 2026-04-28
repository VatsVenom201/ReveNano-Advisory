from openai import OpenAI

client = OpenAI()

vector_store_id = "vs_69f05dba90ec81918fe438df14e8fcab"

files = client.vector_stores.files.list(
    vector_store_id=vector_store_id
)

for f in files.data:
    print(f.id, f.status)