FOLIOLR_PROMPT = '''
The following is one example for reference.

The context is:
All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.

The question is: 
Based on the above information, is the following statement true, false, or uncertain? Rina is either a person who regularly drinks coffee or a person who is unaware that caffeine is a drug.
Based on the above information, is the following statement true, false, or uncertain? If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.

The logical representation is:
Facts:
Forall('$x1', Implies(Atom('RegularlyDrinkCoffee', '$x1'), Atom('DependentOnCaffeine', '$x1'))) ::: All people who regularly drink coffee are dependent on caffeine.
Forall('$x1', Xor(Atom('RegularlyDrinkCoffee', '$x1'), Atom('JokeAboutBeingAddictedToCaffeine', '$x1'))) ::: People either regularly drink coffee or joke about being addicted to caffeine.
Forall('$x1', Implies(Atom('JokeAboutBeingAddictedToCaffeine', '$x1'), Not(Atom('UnawareThatCaffeineIsADrug', '$x1')))) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.
Xor(And(Atom('Student', 'rina'), Atom('UnawareThatCaffeineIsADrug', 'rina')), And(Not(Atom('Student', 'rina')), Not(Atom('UnawareThatCaffeineIsADrug', 'rina')))) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug.
Implies(Not(And(Atom('DependentOnCaffeine', 'rina'), Atom('Student', 'rina'))), Xor(And(Atom('DependentOnCaffeine', 'rina'), Atom('Student', 'rina')), And(Not(Atom('DependentOnCaffeine', 'rina')), Not(Atom('Student', 'rina'))))) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
Query:
Xor(Atom('JokeAboutBeingAddictedToCaffeine', 'rina'), Atom('UnawareThatCaffeineIsADrug', 'rina')) ::: Rina is either a person who jokes about being addicted to caffeine or is unaware that caffeine is a drug.
Xor(Atom('RegularlyDrinkCoffee', 'rina'), Atom('UnawareThatCaffeineIsADrug', 'rina')) ::: Rina is either a person who regularly drinks coffee or a person who is unaware that caffeine is a drug.
Implies(Xor(And(Atom('JokeAboutBeingAddictedToCaffeine', 'rina'), Atom('UnawareThatCaffeineIsADrug', 'rina')), And(Not(Atom('JokeAboutBeingAddictedToCaffeine', 'rina')), Not(Atom('UnawareThatCaffeineIsADrug', 'rina')))), And(Atom('JokeAboutBeingAddictedToCaffeine', 'rina'), Atom('RegularlyDrinkCoffee', 'rina'))) ::: If Rina is either a person who jokes about being addicted to caffeine and a person who is unaware that caffeine is a drug, or neither a person who jokes about being addicted to caffeine nor a person who is unaware that caffeine is a drug, then Rina jokes about being addicted to caffeine and regularly drinks coffee.


Next, you will be given one sample for test.
'''.strip() + '\n'