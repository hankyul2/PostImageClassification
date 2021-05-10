from src.post_train_step import PostTrain


def test_train():
    tool = PostTrain()
    assert tool.train_fn()

def test_post_train():
    tool = PostTrain()
    assert tool.post_train() >= 0.8662

