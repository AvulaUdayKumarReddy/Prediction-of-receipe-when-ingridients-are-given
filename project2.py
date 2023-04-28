from sklearn.model_selection import train_test_split
import json
import argparse
from sklearn.linear_model import LogisticRegression
import pandas as pan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import numpy as np

def load_prepare_vectorize_data(test_data,top_n_items):
    json_file=open('yummly.json')
    raw_data=json.load(json_file)
    #print(test_data)
    #print(type(raw_data))
    trans_data=pan.DataFrame.from_dict(raw_data)
    #print(trans_data)
    #print(type(trans_data))
    li=[]
    
    for i in trans_data['ingredients']:
        li.append(", ".join(map(str,i)))
    vector_tf=TfidfVectorizer()
   
    finale_li=vectorized=vector_tf.fit_transform(li)
    #print(finale_li)
    #print(np.shape(li))
    #vector_tf=TfidfVectorizer(tokenizer=lambda doc: doc,lowercase=False)
    labels=trans_data['cuisine'].values.tolist()
    #print(labels)
    #print(vectorized.toarray())
    #test_data=['rice']
    test_tf_trans=vector_tf.transform(test_data)
    lg=LogisticRegression(max_iter=1500).fit(finale_li,labels)
    #pred=lg.predict(test_tf_trans)
    accuracies=cosine_similarity(test_tf_trans,finale_li)
    temp_arr=[]
    
    list_of_id=trans_data['id']
    list_of_cuisine=trans_data['cuisine']
    for i in range(len(accuracies[0])):
        temp_arr.append([accuracies[0][i],i])
    temp_arr.sort()
    
    stop=len(temp_arr)-top_n_items-2
    #print(len(temp_arr))
    #print(stop)
    dict={}
    for i in range(len(temp_arr)-1,stop,-1):
        val=temp_arr[i]
        ind=trans_data['id'][val[1]]
        if i==len(temp_arr)-1:
            predicted_cuisine=trans_data['cuisine'][val[1]]
        dict[ind]=val[0]
    return dict,predicted_cuisine


    #print(temp_arr)
    '''
    print(dict)
    print(top_n_accuracies)
    '''

def format_output(predicted_cuisine,output_dict):
    '''list_of_scores=[]
    for i in output_dict.values():
        list_of_scores.append(i)
    #list_of_ids.astype(in)
    print(list_of_scores)
    list_of_ids=[]
    for i in output_dict.keys():
        list_of_ids.append(i)
    print(list_of_ids)'''
    closest=[]
    for i in output_dict:
        closest.append({'id':int(i),'Score':round(output_dict[i],5)})
    closest.pop(0)
    out={}
    out['Cuisine']=predicted_cuisine
    out['Score']=round(list(output_dict.values())[0],5)
    out['Closest']=closest
   
    try:
        json_data=json.dumps(out,indent=6)
    except Exception as e:
        print(e)
    return json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True, help="closest number of predictions")
    parser.add_argument("--ingredient",action='append' ,required=True)
    args = parser.parse_args()
    top_n_items=args.N
    items=args.ingredient
    output_dict,predicted_cuisine=load_prepare_vectorize_data(items,top_n_items)
    #print(output_dict)
    print(format_output(predicted_cuisine,output_dict))
    