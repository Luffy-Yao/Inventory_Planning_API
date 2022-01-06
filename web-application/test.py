from model import Inventory_Simulation
import logging
import collections


# Defining different types of records used throughout the simulation.
EndOfTimeStepRecord = collections.namedtuple(
    "EndOfTimeStepRecord", ("date", "ending_stock")
)
OrderPlacementRecord = collections.namedtuple(
    "OrderPlacementRecord", ("arrival_date", "units_ordered", "order_placement_date")
)
SalesRecord = collections.namedtuple("SalesRecord", ("date", "units_sold"))
LossRecord = collections.namedtuple("LossRecord", ("date", "units_lost"))

# A generic object used for some visualizations
HorizontalLineInfo = collections.namedtuple("HorizontalLineInfo", ("label", "value"))

if __name__ == '__main__':
    simulation = Inventory_Simulation()
    INVENTORY_CHECK_FREQUENCY = 1 # A value of 1 indicates that the stock is checked at the end of each day.

    # Main User Inputs
    STARTING_STOCK = 50
    REORDER_THRESHOLD = 30 # Use a value of float("inf") in order to simulate the behavior of periodic checks
    ORDER_FILL_AMOUNT = 100 # This strategy uses a constant value when placing an order.
    LEAD_TIME = 5

    FAKE_PRODUCT_INFO = {
        "unit_cost": 3,
        "selling_price": 10,
        "holding_cost": 0.01,
        "order_cost": 1,
    }


    strategy_params = {
        "lead_time": LEAD_TIME,
        "reorder_threshold": REORDER_THRESHOLD,
        "order_fill_amount_function": lambda : ORDER_FILL_AMOUNT,
    }

    historical_data = simulation.generate_historical_data() # NOTE: This is currently a placeholder that will be needed in a real environment; it is not used for the purposes of this simulation.
    forecasted_data = simulation.generate_forecasted_data()


    simulation_data = simulation.simulate(
    STARTING_STOCK,
    forecasted_data,
    inventory_check_frequency=INVENTORY_CHECK_FREQUENCY,
    strategy_params=strategy_params,
)
    restock_line = HorizontalLineInfo(label="Restock Threshold", value=REORDER_THRESHOLD)
    simulation.plot_inventory_graph(simulation_data, restock_line)

