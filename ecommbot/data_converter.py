import pandas as pd
from langchain_core.documents import Document

def dataconveter():
    product_data=pd.read_csv("/Users/neetikashree/Documents/GenAi-Sunny-projects/end-to-end-chatbot/data/flipkart_product_review.csv")

    data=product_data[["product_title","review"]]

    product_list = []

    # Iterate over the rows of the DataFrame
    for index, row in data.iterrows():
        # Construct an object with 'product_name' and 'review' attributes
        obj = {
                'product_name': row['product_title'],
                'review': row['review']
            }
        # Append the object to the list
        product_list.append(obj)




    docs= []
    for item in product_list:
        metadata={"product_name":item['product_name']}
        doc = Document(page_content=item['review'],metadata=metadata)
        docs.append(doc)
    return docs