def get_model():
    pass


def get_data():
    pass


def train(a, d):
    pass


def predict(a, x):
    pass


def combine_data(data_list):
    pass


def copy_model(target_model, model_definition):
    pass


def topk(list, dim):
    pass


def loss_fn(probas, label):
    pass


a = get_model()
d, d_unlabelled = get_data()
d_pseudo = list()

while True:
    train(a, d)
    loss_prev, out_softmax = loss_fn(predict(a, d.x), d.y), predict(a, d_unlabelled.x)

    for x_idx, out_row in enumerate(out_softmax):
        x = d_unlabelled.x[x_idx]
        top_5, top_5_idx = topk(out_row, 5)
        losses = []
        for label_candidate in top_5_idx:
            a_copy = copy_model(a, get_model)
            train(a_copy, {'data': x, 'label': label_candidate})
            out_softmax = predict(a, d.x)
            loss_now = loss_fn(out_softmax, d_unlabelled.y)  # cross_entropy or mse_loss or BCE or AUC
            loss_post = loss_now - loss_prev  # this part is depend on loss_fn
            losses.append(loss_post)
        lowest_entropy, index = losses.min(dim=0)
        if lowest_entropy < 0:
            d_unlabelled.remove(top_5_idx[index[0]])
            d_pseudo.append({'data': x, 'label': top_5_idx[index[0]]})

    d = combine_data([d, d_pseudo])
    d_pseudo = list()
