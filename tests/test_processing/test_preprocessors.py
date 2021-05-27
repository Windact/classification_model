from classification_model.config import core
from classification_model.processing import preprocessors as pp


def test_FeatureKeeper(pipeline_inputs):
    """ Testing FeatureKeeper function """
    
    # Given
    feature_keeper = pp.FeatureKeeper(variables_to_keep=core.config.model_config.VARIABLES_TO_KEEP)

    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    subject = feature_keeper.fit_transform(X_train,y_train)

    # Then
    assert list(subject.columns) == core.config.model_config.VARIABLES_TO_KEEP


def test_CategoricalGrouping(pipeline_inputs):
    """ Testing CategoricalGrouping function """
    
    # Given
    categorical_grouping = pp.CategoricalGrouping(config_dict=core.config.model_config.VARIABLES_TO_GROUP)
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    subject = categorical_grouping.fit_transform(X_train,y_train)
    cat_len = []
    t_cat_len = []
    for k in core.config.model_config.VARIABLES_TO_GROUP.keys():
        for i,v in core.config.model_config.VARIABLES_TO_GROUP[k].items():
            cat_len.append(sum([X_train.loc[X_train[k] == j].shape[0] for j in v]))
            t_cat_len.append(subject.loc[subject[k] == i].shape[0])
            
    assert_list = [cat_len[i] == t_cat_len[i] for i in range(len(cat_len))]
    # Then
    
    assert sum(assert_list) == len(cat_len) 




def test_RareCategoriesGrouping(pipeline_inputs):
    """ Testing RareCategoriesGrouping function """
    
    # Given
    rare_categories_grouping = pp.RareCategoriesGrouping(threshold=core.config.model_config.VARIABLES_THRESHOLD)
    X_train, X_test, y_train, y_test = pipeline_inputs

    # When
    subject = rare_categories_grouping.fit_transform(X_train,y_train)
    cat_dict_lower = {}
    cat_dict_subject = {}
    print(rare_categories_grouping.threshold)
    for k,v in rare_categories_grouping.threshold.items():
        cat_list = X_train[k].value_counts(normalize=True)
        cat_dict_lower[k] = cat_list[cat_list<float(v)].index

        cat_dict_subject[k] = subject[k].value_counts(normalize=True).index

    print(cat_dict_subject)
    print('********')
    print(cat_dict_lower)
    # Then
    for k in rare_categories_grouping.threshold.keys():
        assert "Rare" in list(cat_dict_subject[k])
        for i in cat_dict_lower[k]:
            assert i not in cat_dict_subject[k]

    