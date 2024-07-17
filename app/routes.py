from flask import Blueprint, request, jsonify, current_app
import os
import logging
from app.Constants import CHROMA_PATH, DATA_PATH
from app.create_database import ProcessInputCreateDatabase
from app.query_data import query_rag

main = Blueprint('main', __name__)

@main.route('/create-database', methods=['POST'])
def create_database():
    try:
        current_dir = current_app.root_path
        input_dir = os.path.abspath(os.path.join(current_dir, '..', DATA_PATH))
        db_dir = os.path.abspath(os.path.join(current_dir, '..', CHROMA_PATH))
        
        logging.debug(f"current_dir: {current_dir}")
        logging.debug(f"input_dir: {input_dir}")
        logging.debug(f"db_dir: {db_dir}")

        create_database = ProcessInputCreateDatabase(input_dir, db_dir)
        create_database.main()
        message = 'Database created successfully.'   
        
        logging.debug(message)    
        
        return jsonify({'message': message}), 200
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    

@main.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query_text = data.get('query_text')
    
    if not query_text:
        return jsonify({'error': 'query_text is required'}), 400
    
    current_dir = current_app.root_path
    db_dir = os.path.abspath(os.path.join(current_dir, '..', CHROMA_PATH))
    
    if not os.path.exists(db_dir):
        return jsonify({ 'error': 'Database not found! '}), 404

    try:
        results = query_rag(query_text)
        
        return jsonify({'results': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
