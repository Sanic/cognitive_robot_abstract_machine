You name the segmentation labels of a scanned room in a robot's ontology.

{% if shows_pictures -%}
You are given one label, pictures of one object carrying it, the ontology as JSON, and
what every other label of the room was read as. An earlier question found no class of the
ontology that this label means. Judge from the pictures what the objects are.
{%- else -%}
You are given one label, what the objects carrying it were measured to meet, the ontology
as JSON, and what every other label of the room was read as. An earlier question found no
class of the ontology that this label means. You are not shown the objects, so judge from
the label and what it was measured beside.
{%- endif %}

First look again for a class of the ontology that *is* what the objects are: the class a
label means is often named differently from the label. If one is, answer with it. Where
none is -- because the objects are only *a kind of* a class, as a stool is a kind of
furniture, or because nothing in the ontology is like them -- propose a new class: name the
superclass it derives from and any of the ontology's part_whole_mixins to compose it from.
Choose those by what the object can hold, because they decide it: a class composed with
HasDrawers can hold drawers as its parts and one without it cannot. The labels measured to
share {% if shows_pictures %}the pictured object's{% else %}an object's{% endif %} faces are the candidates for its parts, so a class that cannot hold
them will never hold them.

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
- Answer with "class": null only if the label names nothing the ontology should hold at
  all.

Answer with JSON and nothing else, in one of these three shapes. Write "reason" first: say
in one sentence which classes come closest and why none of them is what the objects are,
or which one is after all.

A class that is already in the taxonomy after all:
{"reason": "one sentence", "class": "<a name from classes[]>", "is_new_class": false,
 "confidence": 0.0}

A class you propose:
{"reason": "one sentence", "class": "<the name you give it>", "is_new_class": true,
 "superclass": "<a name from classes[]>", "mixins": ["<names from part_whole_mixins[]>"],
 "confidence": 0.0}

A label the ontology should hold nothing for:
{"reason": "one sentence", "class": null, "is_new_class": false, "confidence": 0.0}
