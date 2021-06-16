from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from math import floor, log

from sqlalchemy.orm.session import Session

USER = "test"
PASS = "test"

from db.tables import *


def humanFormat(number):
    """
    Shorten a long number with abbreviations
    """
    if type(number) == str:
        return number
    elif number <= 0:
        return number
    units = ["", "K", "M", "G", "T", "P"]
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return "%.2f%s" % (number / k ** magnitude, units[magnitude])


def build_engine(
    connection, username, password, host, port, database, autocommit=False, echo=False
):
    """
    Takes string params and returns an engine.
    props[sym] call: buildEngine("mysql", "test", "test", "localhost", "3306", "samwise")
    """
    engine = create_engine(
        f"{connection}://{username}:{password}@{host}:{port}/{database}",
        echo=echo,
    )
    return engine


def reset_tables(engine):
    """
    Drop all tables and create them again
    """
    print(("Resetting database ..."))
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def create_session(engine):
    """
    Create a session from an engine
    """
    Session = sessionmaker(bind=engine)
    session = Session()
    session._model_changes = {}

    return session


def add(engine, data):
    session = create_session(engine)
    session.add(data)
    session.commit()


def bulk_add(engine, data: list):
    session = create_session(engine)
    session.bulk_save_objects(data)
    session.commit()


def get_DB_size(session, dbname):
    """
    Get dbname size
    """
    x = session.execute(
        f'SELECT table_name AS "Table", (data_length + index_length) AS "Size (B)" FROM information_schema.TABLES WHERE table_schema = "{dbname}" ORDER BY (data_length + index_length) DESC'
    ).fetchall()
    db_size = 0
    for tup in x:
        db_size += tup[1]
    formatted = humanFormat(db_size) + "B"
    return formatted


def get_table_length(session, table):
    # get table length
    number_of_rows = session.execute(
        f"SELECT id FROM {table} ORDER BY id DESC LIMIT 1"
    ).fetchall()[0][0]
    return number_of_rows
