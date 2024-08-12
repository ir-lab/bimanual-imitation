import os

from absl import logging


def filter_warnings():
    import logging as logging_orig
    import warnings

    # Suppress tf user warnings and specific syntax warning
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    warnings.filterwarnings(
        "ignore", category=SyntaxWarning, message='"is not" with a literal\. Did you mean "!="\?'
    )

    logging.set_verbosity(logging.INFO)

    class AbslInfoFilter(logging_orig.Filter):
        def filter(self, record):
            # Omit printing out "Saved checkpoint" during logging and only show info messages from absl
            return (record.levelno == logging_orig.INFO) and (
                "Saved checkpoint" not in record.getMessage()
            )

    class InfoFilter(logging_orig.Filter):
        def filter(self, record):
            return record.levelno == logging_orig.INFO

    logging_orig.getLogger("tensorflow").addFilter(InfoFilter())
    logging.get_absl_logger().addFilter(AbslInfoFilter())
