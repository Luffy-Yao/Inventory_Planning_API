import collections
import heapq
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logging.basicConfig(
    # filename="log.log",
    # filemode="a",
    format="%(asctime)s %(levelname)-8s [%(name)s] : %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)



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


class Inventory_Simulation(object):
    # Generate / create an input signal
    def generate_normal_input(self, mean=10, std=3, size=100, floor_value=0, round_input=True):
        normal_input = np.random.normal(loc=mean, scale=std, size=size)
        
        # Truncate all entries using the given floor_value (e.g., floor_value=0 means negative values are not allowed).
        if floor_value is not None:
            normal_input[normal_input <= floor_value] = floor_value
        
        # A boolean flag used to round values.  The default rounding strategy will round inputs with decimal values of 0.5 and below down to the nearest integer; inputs with higher decimal values will be rounded up to the next highest integer value.
        if round_input:
            normal_input = np.round(normal_input)
        return normal_input


    def generate_historical_data(self, *args, **kwargs):
        historical_data = self.generate_normal_input(*args, **kwargs)
        return historical_data

    def generate_forecasted_data(self, *args, **kwargs):
        forecasted_data = self.generate_normal_input(*args, **kwargs)
        return forecasted_data


    def calculate_profit(self, simulation_data, product_info):
        unit_cost = product_info["unit_cost"]
        selling_price = product_info["selling_price"]
        holding_cost = product_info["holding_cost"]
        order_cost = product_info["order_cost"]

        revenue = sum(r.units_sold for r in simulation_data["units_sold"]) * selling_price
        order_costs = len(simulation_data["orders_placed"]) * order_cost
        holding_costs = (
            sum(r.ending_stock for r in simulation_data["time_step_stock_counts"])
            * holding_cost
        )
        unit_costs = (
            sum(r.units_ordered for r in simulation_data["orders_placed"]) * unit_cost
        )

        profit = revenue - order_costs - holding_costs - unit_costs

        return profit
    
    def calculate_revenue(self, simulation_data, product_info):
        
        return (sum(record.units_sold for record in simulation_data["units_sold"])
        * product_info["selling_price"]
    )

    def calculate_losses(self, simulation_data, product_info):
        unit_cost = product_info["unit_cost"]
        selling_price = product_info["selling_price"]

        units_lost = sum(r.units_lost for r in simulation_data["units_lost"])

        loss = units_lost * (selling_price - unit_cost)
        return loss

    def calculate_proportion_orders_lost(self, simulation_data, product_info): 
        total_units_lost = sum(
            record.units_lost for record in simulation_data["units_lost"]
        )
        total_units_sold = sum(
            record.units_sold for record in simulation_data["units_sold"]
        )

        return total_units_lost / (total_units_sold + total_units_lost)


    def plot_inventory_graph(self, simulation_data, horizontal_line_info=None):
        G = simulation_data["time_step_stock_counts"]
        plt.plot(list(p.ending_stock for p in G), label="Inventory")

        if horizontal_line_info is not None:
            plt.plot([horizontal_line_info.value] * len(G), label=horizontal_line_info.label)
        plt.legend(loc="upper left")
        plt.title("Inventory Levels")
        plt.xlabel("Time Step")
        plt.ylabel("Inventory")
        plt.show()

    def threshold_restock_strategy(self, 
        current_stock,
        reorder_threshold,
        order_fill_amount_function,
        lead_time,
        current_time_step,
        order_fill_params=None,
    ):
        """
        When `current_stock` falls below the `reorder_threshold` create an order for `order_fill_amount` units.
        """
        order_placed = None

        if order_fill_params is None:
            order_fill_amount = order_fill_amount_function()
        else:
            order_fill_amount = order_fill_amount_function(current_stock, **order_fill_params)
        if (current_stock <= reorder_threshold) and (order_fill_amount > 0):

            order_placed = OrderPlacementRecord(
                arrival_date=current_time_step + lead_time,
                units_ordered=order_fill_amount,
                order_placement_date=current_time_step,
            )

        return order_placed


    # Simulate activity using the restocking strategy

    def simulate(self,
        starting_stock,
        sales_data,
        inventory_check_frequency,
        strategy_params,
    ):
        """
        Simulate the activity based on the given parameters.
        """
        simulation_data = {
            "time_step_stock_counts": [],
            "orders_placed": [],
            "units_sold": [],
            "units_lost": [],
        }
        current_stock = starting_stock
        open_orders_heap = []

        for time_step, sales in enumerate(sales_data):

            # Check if any orders have been delievered and add them to the stock
            # NOTE: This current model makes the assumption that orders arrive at the start of the time period.
            if len(open_orders_heap) > 0:
                # Check that there are not errenous orders lingering
                while (
                    len(open_orders_heap) > 0
                    and open_orders_heap[0].arrival_date < time_step
                ):
                    logger.error(
                        f"A stale order was found in the system of {open_orders_heap[0]}.  Please look into what process placed this order."
                    )
                    heapq.heappop(open_orders_heap)

                # Check if there are any orders which are arriving at the current time step.
                while (
                    len(open_orders_heap) > 0
                    and open_orders_heap[0].arrival_date == time_step
                ):
                    received_order = heapq.heappop(open_orders_heap)
                    logger.debug(f"Received the following order: {received_order}.")
                    current_stock += received_order.units_ordered

            # Perform the activity of sales for the given time period

            # Record information about sales activity that occured
            # This could include potential for losses.
            if current_stock >= sales:
                current_stock -= sales
                sales_record = SalesRecord(date=time_step, units_sold=sales)
                simulation_data["units_sold"].append(sales_record)

            else:

                sales_record = SalesRecord(date=time_step, units_sold=current_stock)
                simulation_data["units_sold"].append(sales_record)

                loss_record = LossRecord(
                    date=time_step, units_lost=abs(sales - current_stock)
                )
                simulation_data["units_lost"].append(loss_record)
                logger.debug(
                    f"Failed to complete {loss_record.units_lost} sales on time_step {loss_record.date}."
                )

                current_stock = 0

            # Add the end of time step status to the records.
            end_of_time_step_record = EndOfTimeStepRecord(
                date=time_step, ending_stock=current_stock
            )
            simulation_data["time_step_stock_counts"].append(end_of_time_step_record)

            # Perform a check at the end of the day to see if an order should be placed.

            if time_step % inventory_check_frequency == 0:
                # NOTE: For now we will limit to only allowing 1 order to be placed at a time.
                if len(open_orders_heap) > 0:
                    continue
                restock_order = self.threshold_restock_strategy(
                    current_stock,
                    current_time_step=time_step,
                    **strategy_params,
                )
                if restock_order is not None:
                    logger.debug(f"The following order has been placed {restock_order}.")
                    simulation_data["orders_placed"].append(restock_order)
                    heapq.heappush(open_orders_heap, restock_order)

        return simulation_data

  