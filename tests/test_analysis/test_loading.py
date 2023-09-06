from os import environ

from dotenv import load_dotenv

from hybrid_jp.analysis.loading import load_async


def test_load_async():
    load_dotenv()
    test_data_dir = environ["TEST_DATA_DIR"]
    load_async(test_data_dir, stop=2)
