"""
This script is a Flask web application that provides several endpoints for extracting price patterns and values from web pages.
"""
from flask import Flask, jsonify, request, session
from compare import compare_product
from target_compare import compare_tracking_product
from Client_compare import compare_client_product
from Client_tracking_compare import compare_client_tracking_product
from flask_session import Session
from Pattern_extractor import extract_pattern
from urllib.parse import urlparse
import os
from Values_extractor import DOMExtractor


app = Flask(__name__) # Initialize Flask app
app.config['SESSION_TYPE'] = 'filesystem' # Configure session type
Session(app) # Initialize session
extractor = DOMExtractor() # Initialize DOMExtractor
@app.route('/api/extract-patterns', methods=['GET'])
def extract_patterns():
    """
    Extract price patterns from the given URL using extract_pattern function (from Pattern_extractor.py).
    Returns: jsonify: JSON response containing the extracted patterns

    """
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Validate URL
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return jsonify({'error': 'Invalid URL format'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid URL'}), 400

        # Extract prices
        prices,description,stock = extract_pattern(url)

        if not prices:
            return jsonify({'error': 'No prices found'}), 404

        # Format response
        response = {
            'url': url,
            'description': [{
                'text_content': item['text_content'],
                'tag': item['tag_name'],
                'attributes': item['attributes']
            }
            for item in description],
            'prices': [
                {
                    'price': item['price'],
                    'tag': item['tag_name'],
                    'attributes': item['attributes']
                }
                for item in prices
            ],
            'stock': [
                {
                    'stock': item['text_content'],
                    'tag': item['tag_name'],
                    'attributes': item['attributes']
                }
                for item in stock
            ]
        }

        # Save interaction in session (it's possible to make a log file out of it)
        if 'interactions' not in session:
            session['interactions'] = []
        session['interactions'].append(response)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract-price', methods=['GET'])
def extract_price():
    """
    Extract price from the given URL using DOMExtractor class (from Values_extractor.py).
    Returns: jsonify: JSON response containing the extracted data

    """
    try:
        url = request.args.get('url') # Get URL from request
        param = request.args.get('param') # Get param from request
        descr_param = request.args.get('descr_param') # Get descr_param from request
        stock_param = request.args.get('stock_param') or None # Get stock_param from request or None if not provided
        # Validate URL
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Validate URL
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return jsonify({'error': 'Invalid URL format'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid URL'}), 400


        # Extract values using DOMExtractor
        if not param:
            price, title, description, stock = extractor.extract_values(url)
        elif param and not descr_param:
            if (stock_param):
                price, title, description, stock = extractor.extract_values(url, param, stock_param)
            else:
                price, title, description, stock = extractor.extract_values(url, param)
        else:
            if (stock_param):
                price, title, description, stock = extractor.extract_values(url, param, descr_param, stock_param)
            else:
                price, title, description, stock = extractor.extract_values(url, param, descr_param)

        if not price:
            return jsonify({'error': 'No price found'}), 404

        # Save interaction in session (you can also save it in a log file)
        if 'interactions' not in session:
            session['interactions'] = []
        # Before saving to session, ensure data is serializable
        session['interactions'].append({
            'url': url,
            'title': str(title) if title else '',
            'price': str(price) if price else '',
            'description': str(description) if description else '',
            'stock': str(stock) if stock else ''
        })
        # Format response
        return jsonify({
            'success': True,
            'title': title,
            'price': price.replace("TTC", ""), # Remove "TTC" from price if present
            'description': description,
            'descr_param': descr_param,
            'stock': stock,
            'url': url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/interactions', methods=['GET'])
def get_interactions():
    """
    Get the list of interactions stored in the session. Possible to use it to make a log file
    Returns: jsonify: JSON response containing the list of interactions

    """
    return jsonify(session.get('interactions', []))

@app.route('/api/compare', methods=['GET'])
def compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_product(host, user, password, database, database_prefix)
        compare_tracking_product(host, user, password, database, database_prefix)

        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/tracking_compare', methods=['GET'])
def tracking_compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_tracking_product(host, user, password, database, database_prefix)
        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestion_compare', methods=['GET'])
def suggestion_compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_product(host, user, password, database, database_prefix)
        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/api/client_compare', methods=['GET'])
def client_compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_client_product(host, user, password, database, database_prefix)
        compare_client_tracking_product(host, user, password, database, database_prefix)

        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/client_tracking_compare', methods=['GET'])
def client_tracking_compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_client_tracking_product(host, user, password, database, database_prefix)
        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/client_suggestion_compare', methods=['GET'])
def client_suggestion_compare():
    """
    Compare products using the compare_product function (from compare.py).
    Returns: jsonify: JSON response containing the comparison result

    """
    try:
        host = request.args.get('host')
        user = request.args.get('user')
        password = request.args.get('passwd')
        database = request.args.get('database')
        database_prefix = request.args.get('database_prefix')

        compare_client_product(host, user, password, database, database_prefix)
        return jsonify({
            'success': True,
            'message': 'Comparison completed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    # change the port value (default to 8000)
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=True)
