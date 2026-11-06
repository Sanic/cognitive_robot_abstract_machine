You are naming objects in a scanned room, using an ontology of classes.

{% if shows_pictures -%}
Each picture shows the same room from a different side, with a few objects painted in
distinct colors and everything else left as it was scanned. Say what each painted object
is. You are told the label a previous step gave it and the class that label was mapped
to; both can be wrong about this particular object, which is what you are being asked.
{%- else -%}
You are given a list of the room's objects and nothing to look at. Say what each one is.
You are told the label a previous step gave it and the class that label was mapped to;
both can be wrong about this particular object, which is what you are being asked. You
cannot see the objects, so judge from what you are told and say so in your reason where
that leaves the answer uncertain.
{%- endif %}

Rules:
- Name every object you are given, once, by the name it is listed under.
{% if proposes_classes %}
- "class" is a name from the ontology's classes, or a name you propose.
- If you propose one, "is_new_class" is true and "superclass" is a class of the ontology.
- A class marked "abstract" is a category, never an answer: it says what kind of thing
  something is, not what it is. Where the object is a kind of one -- a stool is a kind of
  Furniture, an ornament a kind of Decor, a television a kind of ElectricalDevice -- name
  the kind, set "is_new_class" true, and give the category as the superclass.
- A class marked "mixin" is a base to build with and is never an answer either.
- Choose the superclass by what the thing is, not by which class is nearest to hand: a
  fire extinguisher is not a kind of furniture.
{% else %}
- "class" is a name from the ontology's classes, or null where none of them is what the
  object is. No class is proposed here.
- A class marked "abstract" is a category, never an answer: it says what kind of thing
  something is, not what it is.
- A class marked "mixin" is a base to build with and is never an answer either.
{% endif %}
{% if shows_pictures %}- Judge the object, not the paint: the colors mark what to look at, nothing more.
{% endif %}
{% if proposes_classes %}
- A class marked "abstract" cannot be given to an object. Name one of its subclasses,
  or propose a new class with it as the superclass.
{% else %}
- A class marked "abstract" cannot be given to an object. Name one of its subclasses.
{% endif %}

Answer with JSON and nothing else. For each object write "reason" before "class": say in
one sentence what the object is, then name the class that is that.
{% if proposes_classes %}
{"objects": [{"name": "drawer_19", "reason": "one sentence", "class": "Drawer",
              "is_new_class": false, "superclass": "Furniture", "confidence": 0.0}]}
{% else %}
{"objects": [{"name": "drawer_19", "reason": "one sentence", "class": "Drawer",
              "is_new_class": false, "confidence": 0.0}]}
{% endif %}
