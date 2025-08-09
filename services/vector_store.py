import boto3, json
from utils.similarity import top_n_similar
import os

s3 = boto3.client("s3")
BUCKET = os.getenv("AWS_S3_BUCKET")

def upload_document(filename, chunks, embeddings):
    """
    Uploads multiple text chunks and their embeddings to S3 as a JSON file.
    """
    data = [
        {"text": chunk, "embedding": emb.tolist()}
        for chunk, emb in zip(chunks, embeddings)
    ]
    s3.put_object(Bucket=BUCKET, Key=f"vectors/{filename}.json", Body=json.dumps(data))


def search_vectors(query_embedding, top_n=3):
    objs = s3.list_objects_v2(Bucket=BUCKET, Prefix="vectors/").get("Contents", [])
    all_texts = []
    all_embeddings = []

    for obj in objs:
        file_data = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
        items = json.loads(file_data)
        for item in items:
            all_texts.append(item["text"])
            all_embeddings.append(item["embedding"])

    if not all_embeddings:
        return []

    top_matches = top_n_similar(query_embedding, all_embeddings, n=top_n)
    return [all_texts[idx] for idx, _ in top_matches]
