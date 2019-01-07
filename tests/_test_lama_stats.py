"""
Run all the config files in the test config directory
Currently just running and making sure there's no uncaught exceptions.
TODO: check that the correct output is generated too

To run these tests, the test data needs to be fechted from bit/dev/lama_stats_test_data
In future we should put this in a web-accessible place
"""

# Import the paths for the test data from tests/__init__.py
from . import stats_config_dir, wt_registration_dir, mut_registration_dir, target_dir, stats_output_dir
from lama.stats.standard_stats import lama_stats_new


# @nottest
def test_all():
    """
    Run the stats module. The data requirted for this to work must be initially made
    by running tests/test_lama.py:test_lama_job_runner()
    """
    config = stats_config_dir / 'new_stats_config.toml'
    lama_stats_new.run(config, wt_registration_dir, mut_registration_dir, stats_output_dir, target_dir)


# @nottest
# def test_organ_vols():
#     config = join(CONFIG_DIR, 'organ_vols.yaml')
#     run_lama_stats.run(config)
#
# @nottest
# def test_glcm():
#     config = join(CONFIG_DIR, 'test_glcm.yaml')
#     run_lama_stats.run(config)
#
# @nottest
# def test_from_arbitrary_directory():
#     config = join(CONFIG_DIR, 'all_specimens.yaml')
#     os.chdir(os.path.expanduser('~'))
#     run_lama_stats.run(config)
#
# @nottest
# def test_missing_config():
#     config = join(CONFIG_DIR, 'all_specimens.flaml')
#     assert_raises(Exception, run_lama_stats.run, config)
#
# @nottest
# def test_misc():
#     config = join(CONFIG_DIR, 'misc_test.yaml')
#     run_lama_stats.run(config)
#
#
# @nottest
# def test_incorrect_staging_file():
#     pass

if __name__ == '__main__':
	test_all()
