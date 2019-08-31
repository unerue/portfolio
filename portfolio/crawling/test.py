from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

from sqlalchemy import create_engine, Column, Integer, String, DateTime

class DataBase(DeclarativeBase):
    __tablename__ = 'dbpia'

    ids = Column(Integer, primary_key=True)
    titles = Column('titles', String)
    authors = Column('authors', String)
    date = Column('publication_date', String)
    abstracts = Column('abstracts', String)
    keywards = Column('keywards', String)

def db_connect():
    return create_engine(URL(**DATABASE))

def create_deals_table(engine):
    DeclarativeBase.metadata.create_all(engine)

DATABASE = {
    'drivername': 'sqlite3',
    'host': 'localhost',
    'port': '8088',
    'username': 'admin',
    'password': 'admin',
    'database': 'scrape'
}

class ScrappingPapers:
    def __init__(self):
        engine = db_connect()
        create_deals_table(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        session = self.Session()
        deal = Deals(**item)

        try:
            session.add(deal)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item