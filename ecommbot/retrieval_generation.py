from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModel, pipeline
from ecommbot.ingest import ingestdata
import numpy as np

def mistral_generate_embedding(text):
    # Load the Mistral model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling to get a single vector per input
    return embeddings.detach().numpy()

def ingestdata(status):
    # Your logic to create the vstore and add documents
    # Here we simulate using Mistral embeddings instead of OpenAI
    vstore = AstraDBVectorStore(
            embedding=mistral_generate_embedding,
            collection_name="chatbotecomm",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
            namespace=ASTRA_DB_KEYSPACE,
        )
    
    storage = status
    
    if storage == None:
        docs = dataconveter()  # Assuming this converts your data to the required format
        inserted_ids = vstore.add_documents(docs)
    else:
        return vstore
    return vstore, inserted_ids

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    Your ecommerce bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    # Load the Mistral model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

    # Create a text generation pipeline using the Mistral model
    mistral_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)

    def mistral_generate(inputs):
        # Extract the prompt from the inputs dictionary
        prompt_text = inputs["input"]
        # Generate the text using the pipeline
        generated = mistral_pipeline(prompt_text, num_return_sequences=1)
        # Return the generated text
        return generated[0]["generated_text"]

    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | mistral_generate
        | StrOutputParser()
    )

    return chain

if __name__ == '__main__':
    vstore, inserted_ids = ingestdata(None)
    chain = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))
