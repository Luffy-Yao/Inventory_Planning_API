from flask import Flask, render_template, flash, request, jsonify, Markup
from model import Inventory_Simulation
import collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, os
import numpy as np


# default User Input
INVENTORY_CHECK_FREQUENCY = 1 # A value of 1 indicates that the stock is checked at the end of each day.

# Main User Inputs
STARTING_STOCK = 50
REORDER_THRESHOLD = 30 # Use a value of float("inf") in order to simulate the behavior of periodic checks
ORDER_FILL_AMOUNT = 100 # This strategy uses a constant value when placing an order.
LEAD_TIME = 5
UNIT_COST = 3
SELLING_PRICE = 10
HOLDING_COST = 0.01
ORDER_COST = 1
INVENTORY_CHECK_FREQUENCY = 14
TARGET_RESTOCK_LEVEL = 200

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

app = Flask(__name__)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')



# Continuous Review System
@app.route("/cReview", methods=['POST', 'GET'])
def continuous_review_system():
    model_results = ''
    if request.method == 'POST':
        selected_startingStock = request.form['starting_stock']
        selected_reorderThreshold = request.form['reorder_threshold']
        selected_orderFillAmount = request.form['order_fill_amount']
        selected_leadTime = request.form['lead_time']
        selected_unitCost = request.form['unit_cost']
        selected_sellingPrice = request.form['selling_price']
        selected_holdingCost = request.form['holding_cost']
        selected_orderCost = request.form['order_cost']

 
        # build new array to be in same format as modeled data so we can feed it right into the predictor
        simulation = Inventory_Simulation()
        INVENTORY_CHECK_FREQUENCY = 1 # A value of 1 indicates that the stock is checked at the end of each day.
 
        strategy_params = {
        "lead_time": float(selected_leadTime),
        "reorder_threshold": float(selected_reorderThreshold),
        "order_fill_amount_function": lambda : float(selected_orderFillAmount),
    }
        # add user desinged passenger to predict function
        historical_data = simulation.generate_historical_data() # NOTE: This is currently a placeholder that will be needed in a real environment; it is not used for the purposes of this simulation.
        forecasted_data = simulation.generate_forecasted_data()

        
        simulation_data = simulation.simulate(
        float(selected_startingStock),
        forecasted_data,
        inventory_check_frequency=INVENTORY_CHECK_FREQUENCY,
        strategy_params=strategy_params,
)

        
        FAKE_PRODUCT_INFO = {
                "unit_cost": int(selected_unitCost),
                "selling_price": int(selected_sellingPrice),
                "holding_cost": float(selected_holdingCost),
                "order_cost": int(selected_orderCost),
            }
        simulation_profit = simulation.calculate_profit(simulation_data, FAKE_PRODUCT_INFO)
        simulation_revenue = simulation.calculate_revenue(simulation_data, FAKE_PRODUCT_INFO)
        simulation_loss = simulation.calculate_losses(simulation_data, FAKE_PRODUCT_INFO)
        simulation_proportion_orders_lost = simulation.calculate_proportion_orders_lost(simulation_data, FAKE_PRODUCT_INFO)
        restock_line = HorizontalLineInfo(label="Restock Threshold", value=REORDER_THRESHOLD)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        G = simulation_data["time_step_stock_counts"]

        plt.plot(list(p.ending_stock for p in G), label="Inventory")
        horizontal_line_info = restock_line
        if horizontal_line_info is not None:
            plt.plot([horizontal_line_info.value] * len(G), label=horizontal_line_info.label)
        plt.legend(loc="upper left")
        plt.title("Inventory Levels")
        plt.xlabel("Time Step")
        plt.ylabel("Inventory")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('cReview.html',
            simulated_profit = simulation_profit,
            simulated_revenue = simulation_revenue,
            simulated_loss = simulation_loss,
            simulated_proportion_orders_lost = simulation_proportion_orders_lost,
            model_plot = Markup('<img src="data:image/png;base64,{}">'.format(plot_url)),
            selected_startingStock = request.form['starting_stock'],
            selected_reorderThreshold = request.form['reorder_threshold'],
            selected_orderFillAmount = request.form['order_fill_amount'],
            selected_leadTime = request.form['lead_time'],
            selected_unitCost = request.form['unit_cost'],
            selected_sellingPrice = request.form['selling_price'],
            selected_holdingCost = request.form['holding_cost'],
            selected_orderCost = request.form['order_cost'])
    else:
        # set default passenger settings
        return render_template('cReview.html',
            model_plot = '',
            selected_startingStock = STARTING_STOCK ,
            selected_reorderThreshold = REORDER_THRESHOLD,
            selected_orderFillAmount = ORDER_FILL_AMOUNT,
            selected_leadTime = LEAD_TIME,
            selected_unitCost = UNIT_COST,
            selected_sellingPrice = SELLING_PRICE,
            selected_holdingCost = HOLDING_COST,
            selected_orderCost = ORDER_COST)

# Periodic Review System
@app.route("/pReview", methods=['POST', 'GET'])
def periodic_review_system():
    model_results = ''
    if request.method == 'POST':
        selected_startingStock = request.form['starting_stock']
        selected_inventoryCheckFrequency = request.form['inventory_check_frequency']
        selected_targetRestockLevel = request.form['target_restock_level']
        selected_leadTime = request.form['lead_time']




        selected_unitCost = request.form['unit_cost']
        selected_sellingPrice = request.form['selling_price']
        selected_holdingCost = request.form['holding_cost']
        selected_orderCost = request.form['order_cost']

 
        # build new array to be in same format as modeled data so we can feed it right into the predictor
        simulation = Inventory_Simulation()
        REORDER_THRESHOLD = float("inf") # A value of 1 indicates that the stock is checked at the end of each day.
 
        # add user desinged passenger to predict function
        historical_data = simulation.generate_historical_data() # NOTE: This is currently a placeholder that will be needed in a real environment; it is not used for the purposes of this simulation.
        forecasted_data = simulation.generate_forecasted_data()


        # NOTE: This is an example of a value that is derived from historical data, but this could also be adjusted by the user if desired.
        EXPECTED_LEAD_TIME_DEMAND = np.round(historical_data.mean() * float(selected_leadTime))

        def periodic_order_function(current_stock, target_restock_level, expected_lead_time_demand):
                return target_restock_level - current_stock + min(expected_lead_time_demand, current_stock) # NOTE: The third term here is used to replenish inventory that is used while waiting for the order to arrive.  The adjustment is based on what is expected to happen (based on historical observations) and it is also limited by the current_stock available. (e.g., In the event that the target_restock_level=100, current_stock=2, and expected_lead_time_demand=5 the order placed should be for 100 units of stock because even though the expectation is that 5 units will be used during the lead time, there are only 2 available to be used up that we would want to replenish while waiting for the order to arrive; note this is in contrast to placing an order for 103 units if the `expected_lead_time_demand` was always used for the third term.).

        strategy_params = {
            "lead_time": int(selected_leadTime),
            "reorder_threshold": REORDER_THRESHOLD,
            "order_fill_amount_function": periodic_order_function,
            "order_fill_params": {
                "target_restock_level" : int(selected_targetRestockLevel),
                "expected_lead_time_demand": EXPECTED_LEAD_TIME_DEMAND,
    }
    
}
        
        simulation_data = simulation.simulate(
            float(selected_startingStock),
            forecasted_data,
            inventory_check_frequency=int(selected_inventoryCheckFrequency),
            strategy_params=strategy_params,
)

        
        FAKE_PRODUCT_INFO = {
                "unit_cost": int(selected_unitCost),
                "selling_price": int(selected_sellingPrice),
                "holding_cost": float(selected_holdingCost),
                "order_cost": int(selected_orderCost),
            }
        simulation_profit = simulation.calculate_profit(simulation_data, FAKE_PRODUCT_INFO)
        simulation_revenue = simulation.calculate_revenue(simulation_data, FAKE_PRODUCT_INFO)
        simulation_loss = simulation.calculate_losses(simulation_data, FAKE_PRODUCT_INFO)
        simulation_proportion_orders_lost = simulation.calculate_proportion_orders_lost(simulation_data, FAKE_PRODUCT_INFO)
        restock_line = HorizontalLineInfo(label="Restock Threshold", value=REORDER_THRESHOLD)

        fig = plt.figure()

        ax = fig.add_subplot(111)
        G = simulation_data["time_step_stock_counts"]

        plt.plot(list(p.ending_stock for p in G), label="Inventory")
        horizontal_line_info = restock_line
        if horizontal_line_info is not None:
            plt.plot([horizontal_line_info.value] * len(G), label=horizontal_line_info.label)
        plt.legend(loc="upper left")
        plt.title("Inventory Levels")
        plt.xlabel("Time Step")
        plt.ylabel("Inventory")
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('pReview.html',
            simulated_profit = simulation_profit,
            simulated_revenue = simulation_revenue,
            simulated_loss = simulation_loss,
            simulated_proportion_orders_lost = simulation_proportion_orders_lost,
            model_plot = Markup('<img src="data:image/png;base64,{}">'.format(plot_url)),
            selected_startingStock = request.form['starting_stock'],
            selected_inventoryCheckFrequency = request.form['inventory_check_frequency'],
            selected_targetRestockLevel = request.form['target_restock_level'],
            selected_leadTime = request.form['lead_time'],
            selected_unitCost = request.form['unit_cost'],
            selected_sellingPrice = request.form['selling_price'],
            selected_holdingCost = request.form['holding_cost'],
            selected_orderCost = request.form['order_cost'])
    else:
        # set default passenger settings
        return render_template('pReview.html',
            model_plot = '',
            selected_startingStock = STARTING_STOCK ,
            selected_inventoryCheckFrequency = INVENTORY_CHECK_FREQUENCY,
            selected_targetRestockLevel = TARGET_RESTOCK_LEVEL,
            selected_leadTime = LEAD_TIME,
            selected_unitCost = UNIT_COST,
            selected_sellingPrice = SELLING_PRICE,
            selected_holdingCost = HOLDING_COST,
            selected_orderCost = ORDER_COST)





if __name__=='__main__':
	app.run(debug=False)