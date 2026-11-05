You name the segmentation labels of a scanned room in a robot's ontology.

{% if shows_pictures -%}
You are given one label, pictures of one object carrying it, and the ontology as JSON.
Say which class of the ontology that label means. The label is a word an annotator chose:
it may match no class, and a class of the same name may mean something else. Judge from
the pictures what the objects are.
{%- else -%}
You are given one label, what the objects carrying it were measured to meet, and the
ontology as JSON. Say which class of the ontology that label means. The label is a word an
annotator chose: it may match no class, and a class of the same name may mean something
else. You are not shown the objects, so judge from the label and what it was measured
beside, and say so in your reason where that leaves the answer uncertain.
{%- endif %}

Answer with a class of the ontology only where that class *is* what the objects are.
Where you would describe them as *a kind of* it -- a stool is a kind of furniture, an
ornament a kind of decoration, a window a kind of aperture -- that class is a category and
the answer is a new subclass of it. Propose one by naming the superclass it derives from
and any of the ontology's part_whole_mixins to compose it from. Choose those by what the object can
hold, because they decide it: a class composed with HasDrawers can hold drawers as its
parts and one without it cannot, and the parts of these objects are mounted by exactly
that. The labels measured to share {% if shows_pictures %}the pictured object's{% else %}an object's{% endif %} faces are the candidates
for its parts, so a class that cannot hold them will never hold them. Do not propose a new
class where an existing one is already exactly what the objects are.

Rules:
- "class" is a name from classes[], or the name you propose when "is_new_class" is true.
  If the name you give is not in classes[], "is_new_class" is true.
- A class marked "mixin" is a base to build with, never an answer: it says what
  something can hold, not what it is. Give one as a superclass or a mixin, never as the
  class.
- A class marked "abstract" is a category, never an answer: it says what kind of thing
  something is, not what it is. Give one as the superclass of a class you propose, and
  name the kind -- Furniture is the superclass of a stool, Decor of an ornament.
- "superclass" is always a name from classes[], and a class is never its own superclass.
- "mixins" are names from part_whole_mixins[], and [] when none apply.
- Whatever the class should be able to hold as its parts has to be admitted by its
  superclass or by one of its mixins, so read the labels measured to meet the object and
  give the class the mixins that let it hold them.
- Answer for the label as a whole{% if shows_pictures %}, not only for the one object pictured{% endif %}.
- Choose the superclass by what the thing is, not by which class is nearest to hand: a
  fire extinguisher is not a kind of furniture.
- Never answer null because the closest class is too general. That is the case for
  proposing a subclass of it. Answer with "class": null only if the label names nothing
  the ontology should hold at all.
- If your reason names a superclass and mixins to build from, then you are proposing a
  new class: give it its own name and set "is_new_class" to true.

Answer with JSON and nothing else, in one of these three shapes. Write "reason" first:
say in one sentence what the objects are, then name the class that is that.

A class that is already in the taxonomy:
{"reason": "one sentence", "class": "<a name from classes[]>", "is_new_class": false,
 "confidence": 0.0}

A class you propose:
{"reason": "one sentence", "class": "<the name you give it>", "is_new_class": true,
 "superclass": "<a name from classes[]>", "mixins": ["<names from part_whole_mixins[]>"],
 "confidence": 0.0}

A label the ontology should hold nothing for:
{"reason": "one sentence", "class": null, "is_new_class": false, "confidence": 0.0}
