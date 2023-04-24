from enum import Enum


class VoteTallyingStrategy(Enum):
    """
    Vote tallying strategy.

    MAJORITY means majority must vote that an ROI is a cell to consider it a
    cell

    CONSENSUS means that all must vote that an ROI is a cell to consider it a
    cell

    ANY means that only 1 must vote that an ROI is a cell to consider it a
    cell
    """
    MAJORITY = 'majority'
    CONSENSUS = 'consensus'
    ANY = 'any'
