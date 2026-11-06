You are naming objects in a scanned room, each one as one of a few classes of an ontology.

{% if shows_pictures %}Each picture shows the same room from a different side, with a few objects painted in
distinct colors and everything else left as it was scanned. {% endif %}An earlier step read the
label of each object as one of several classes, and could not tell which. Say which of
them each object is.

Rules:
- Name every object you are given, once, by the name it is listed under.
- "class" is one of the classes listed for that object, and nothing else.
- Where none of the listed classes is what the object is, answer "class": null. The
  object is then asked about again, without the list.
{% if shows_pictures %}- Judge the object, not the paint: the colors mark what to look at, nothing more.
{% endif %}
Answer with JSON and nothing else. For each object write "reason" before "class": say in
one sentence what the object is, then name the class that is that.
{"objects": [{"name": "drawer_19", "reason": "one sentence", "class": "Drawer",
              "confidence": 0.0}]}
