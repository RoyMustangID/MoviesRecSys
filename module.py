import numpy as np
import pandas as pd


def get_unrated_item(userid, rating_data):
    
    unique_item_id = set(rating_data['item_id'])
    rated_item_id = set(rating_data.loc[rating_data['user_id']==userid, 'item_id'])

    unrated_item_id = unique_item_id.difference(rated_item_id)

    return unrated_item_id

def get_pred_unrated_item(userid, estimator, unrated_item_id):
    
    pred_dict = {
        'user_id': userid,
        'item_id': [],
        'predicted_rating': []
    }

   
    for id in unrated_item_id:
        pred_id = estimator.predict(uid = pred_dict['user_id'],
                                    iid = id)

        pred_dict['item_id'].append(id)
        pred_dict['predicted_rating'].append(pred_id.est)

    # Create a dataframe
    pred_data = pd.DataFrame(pred_dict).sort_values('predicted_rating',
                                                     ascending = False)

    return pred_data

def get_top_highest_unrated(estimator, k, userid, rating_data, metadata):
    
    unrated_item_id = get_unrated_item(userid=userid, rating_data=rating_data)
    
    predicted_unrated_item = get_pred_unrated_item(userid = userid,
                                                   estimator = estimator,
                                                   unrated_item_id = unrated_item_id)

    # Sort & add metadata
    top_item_pred = predicted_unrated_item.head(k).copy()
    top_item_pred_detail = metadata.loc[top_item_pred['item_id'], :]
    

    return top_item_pred_detail
