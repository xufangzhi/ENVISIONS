BLCOKSWORLD_PROMPT = '''
Please use the following pre-defined atom format to form the PDDL: (clear ?x) means the top of x is clear. (on-table ?x) means x is on the table. (arm-empty) means the arm of the robot is empty. (holding ?x) means x is held by the robot. (on ?x ?y) means x is on the top of y.

An example problem in natural language is:
You have 5 blocks.
Your arm is empty.
b1 is on top of b4.
b2 is on top of b5.
b3 is on top of b2. 
b4 is on the table.
b5 is on top of b1.  
b3 is clear.  
Your goal is to move the blocks. 
b4 should be on top of b3. 

The corresponding PDDL file to this problem is:
(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects b1 b2 b3 b4 b5 )
(:init
(arm-empty)
(on b1 b4)
(on b2 b5)
(on b3 b2)
(on-table b4)
(on b5 b1)
(clear b3)
)
(:goal
(and
(on b4 b3))
)
)

Now I have a new planning scene.   
'''.strip()