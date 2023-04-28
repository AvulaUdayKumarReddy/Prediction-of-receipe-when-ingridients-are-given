import project2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# this function tests functionality of load_prepare_vectorize_data() function 
def test_load_prepare_vectorize_data_positive():
    test_data=['rice']
    top_n_items=2
    dict,predicted_cuisine=project2.load_prepare_vectorize_data(test_data,top_n_items)
    assert len(dict)>0
# this function checks whrther the output is formatted , it tests the functionality of format_output() function
def test_format_output():
    predicted_cuisine="japanese"
    output_dict={44076: 0.7636428981435683, 21333: 0.7011572019101071, 15180: 0.6572830447330928, 15448: 0.6209323715286515, 12082: 0.5917865975091795, 46714: 0.5887853530954195}
    data=project2.format_output(predicted_cuisine,output_dict)
    assert len(data)>0

# this function tests the functionality of the tfidf vectorizer used in load_prepare_vectorize_data() function
# Note : I didn't write a function for tfidf vectorizer, but it was part of load_prepare_vectorize_data function.
def test_vectorize():
    test_data=['rice','banana']
    vector_tf=TfidfVectorizer()
    data=vector_tf.fit_transform(test_data)
    assert data is not None

# this function tests the functionality of the cosine similarity used in load_prepare_vectorize_data() function
# Note : I didn't write a function for cosine similarity, but it was part of load_prepare_vectorize_data function.
def test_similarity():
    test_data=['rice','banana']
    CV=TfidfVectorizer().fit(test_data)
    array=CV.transform(test_data).toarray()
    data_r=cosine_similarity(array)
    assert data_r is not None