PROOFWRITER_PROMPT = '''
The following is one example for reference.

The context is: 
Anne is quiet. Erin is furry. Erin is green. Fiona is furry. Fiona is quiet. Fiona is red. Fiona is rough. Fiona is white. Harry is furry. Harry is quiet. Harry is white. Young people are furry. If Anne is quiet then Anne is red. Young, green people are rough. If someone is green then they are white. If someone is furry and quiet then they are white. If someone is young and white then they are rough. All red people are young.
The question is:
Based on the above information, is the following statement true, false, or unknown? Anne is white.

The logical representation is:
Predicates:
Quiet($x, bool) ::: Is x quiet?
Furry($x, bool) ::: Is x furry?
Green($x, bool) ::: Is x green?
Red($x, bool) ::: Is x red?
Rough($x, bool) ::: Is x rough?
White($x, bool) ::: Is x white?
Young($x, bool) ::: Is x young?
Facts:
Quite(Anne, True) ::: Anne is quiet.
Furry(Erin, True) ::: Erin is furry.
Green(Erin, True) ::: Erin is green.
Furry(Fiona, True) ::: Fiona is furry.
Quite(Fiona, True) ::: Fiona is quiet.
Red(Fiona, True) ::: Fiona is red.
Rough(Fiona, True) ::: Fiona is rough.
White(Fiona, True) ::: Fiona is white.
Furry(Harry, True) ::: Harry is furry.
Quite(Harry, True) ::: Harry is quiet.
White(Harry, True) ::: Harry is white.
Rules:
Young($x, True) >>> Furry($x, True) ::: Young people are furry.
Quite(Anne, True) >>> Red($x, True) ::: If Anne is quiet then Anne is red.
Young($x, True) >>> Rough($x, True) ::: Young, green people are rough.
Green($x, True) >>> Rough($x, True) ::: Young, green people are rough.
Green($x, True) >>> White($x, True) ::: If someone is green then they are white.
Furry($x, True) && Quite($x, True) >>> White($x, True) ::: If someone is furry and quiet then they are white.
Young($x, True) && White($x, True) >>> Rough($x, True) ::: If someone is young and white then they are rough.
Red($x, True) >>> Young($x, True) ::: All red people are young.
Query:
White(Anne, True) ::: Anne is white.


Next, you will be given one sample for test.
'''.strip() + '\n'