from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)

# Variables to store the cumulative data
total_cost = 0.0
total_queries = 0

# API Resource to handle JSON data submission
class DataSubmission(Resource):
    def post(self):
        global total_cost, total_queries

        data = request.get_json()

        # Extract cost and queries from the request
        cost = data.get('cost', 0.0)
        queries = data.get('queries', 0)

        # Update the total cost and queries
        total_cost += cost
        total_queries += queries

        # Print the data for verification (optional)
        print("Received Data:")
        print(f"Questions: {data.get('question')}")
        print(f"Contexts: {data.get('contexts')}")
        print(f"Answer: {data.get('answer')}")
        print(f"Ground Truth: {data.get('ground_truth')}")
        print(f"Cost: {cost}")
        print(f"Queries: {queries}")

        return jsonify({"message": "Data received successfully!"}), 200

# Route to display the dashboard in HTML
@app.route('/', methods=['GET'])
def dashboard():
    global total_cost, total_queries
    
    # Render the HTML template with total cost and queries
    return render_template('dashboard.html', total_cost=total_cost, total_queries=total_queries)

# Add the resource to the API
api.add_resource(DataSubmission, '/submit_data')

if __name__ == '__main__':
    app.run(debug=True)
