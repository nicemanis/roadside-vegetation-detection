from model import load_model, test_model, query_model


if __name__ == "__main__":
    model = load_model((448, 728, 3), weights="./data/weights/vegdec_weights.h5")
    # results = test_model(model)
    # print(results)
    query_model(model, dataset="ffd_test")
