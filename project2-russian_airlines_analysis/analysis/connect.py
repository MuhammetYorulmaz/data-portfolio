from sqlalchemy import create_engine

def connect_sqlalchemy(config):
    """ Connect to PostgreSQL using SQLAlchemy """
    try:
        # Create the SQLAlchemy engine
        connection_string = f"postgresql://{config['user']}:{config['password']}@{config['host']}/{config['database']}"
        engine = create_engine(connection_string)
        print('Connected to PostgreSQL using SQLAlchemy.')
        return engine
    except Exception as error:
        print(f"Error: {error}")
        return None
    