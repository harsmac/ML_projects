import flwr as fl
import logging

# Define Flower server
def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting Flower server")
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(strategy=strategy, config={"num_rounds": 5})
    logger.info("Flower server started")

if __name__ == "__main__":
    main()
