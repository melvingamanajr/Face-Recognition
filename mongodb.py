import pymongo

client = pymongo.MongoClient('mongodb://localhost:27017/')

# Access a database
db = client['Payroll']

# Access a collection
collection = db['registration']

# Insert a document
doc = {"name": "John Doe", "age": 30, "email": "johndoe@example.com"}
collection.insert_one(doc)


# Find a document
query = {"name": "John Doe"}
result = collection.find_one(query)
print(result)

# Close the MongoDB connection
client.close()
