1. need to strengthen phase-1 learning since illegal extents are more numerous (harder task)
2. am thinking of making the first 10-20 epochs only legal, then doing curriculm learning by gradually expanding 
3. need to incorporate the turn number of the game (cuz like turn<25, can pretty safely choose small extents; but once turn>25, there are many more larger extents since "holes" have been cleared)

4. currently, the sum-prediction (170-dim ??) is concatenated with features (256-dim). also gradient interference.