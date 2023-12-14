from pathlib import Path
import scispacy
import spacy
from scispacy.linking import EntityLinker
import json
from tqdm import tqdm

from umls_to_label_mapping import label_mapping

nlp = spacy.load("en_core_sci_md")
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "umls", "threshold": 0.5})

linker = nlp.get_pipe("scispacy_linker")



seg_entities = []
results = []

with open("umls_mapping.json") as f:
    for json_entity in f:
        seg_entities.append(json.loads(json_entity))


def link_to_seg_ent(text):

    for ent in seg_entities:
        for concept in ent["umls_concepts"]:
            if concept.lower() in text.lower():
                return ent["class"]
    return None

mapping_path = Path(f"mapping.json")
if mapping_path.exists():
    mapping_path.unlink()


for findings_path in tqdm(list(Path("MIMIC_reports/").glob("**/*.txt"))):
    s_id = findings_path.stem
    p_id = findings_path.parent.stem
    
    with open(findings_path) as f:
        text = f.read()


    doc = nlp(text)

    entry = {}


    entry["text"] = text
    entry["id"] = p_id + s_id
    entry["linked"] = []
    entry["unresolved"] = []
    
    entities = []
    #with open(log_path, "w") as log_f:
    # log_f.write("\n" + text)

    # log_f.write("\nFound:\n")
    for ent in doc.ents:
        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        if ent._.kb_ents:
            seg_ent = link_to_seg_ent(linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name)
            if seg_ent:
                # map to segmentation labels
                seg_labels = []
                if "mappings" in label_mapping[seg_ent]:
                    for keys, value in label_mapping[seg_ent]["mappings"]:
                        if all(k.lower() in str(ent).lower() for k in keys):
                            if isinstance(value, str):
                                value = [value]
                            seg_labels = value
                if not seg_labels:
                    seg_labels = [label_mapping[seg_ent]["default"]]
                entry["linked"].append({
                    "text": str(ent),
                    "seg_labels": seg_labels,
                    "class": seg_ent,
                    "UMLS_concept": linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "certainty": ent._.kb_ents[0][1],

                })
                
                # log_f.write(str(ent) + ", " + seg_ent + ", " + str(ent._.kb_ents[0][1]) + "\n")


    # log_f.write("\nMissed:\n")
    for ent in doc.ents:
        # Each entity is linked to UMLS with a score
        # (currently just char-3gram matching).
        if ent._.kb_ents:
            seg_ent = link_to_seg_ent(linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name)
            if not seg_ent:
                entry["unresolved"].append({
                    "text": str(ent),
                    "class": seg_ent,
                    "UMLS_concept": linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name,
                    "certainty": ent._.kb_ents[0][1]
                })
                # log_f.write(str(ent) + ", " + linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name + ", " + str(ent._.kb_ents[0][1]) + "\n")
    # log_f.write("\nAll:\n")
    # for ent in doc.ents:
    #     # Each entity is linked to UMLS with a score
    #     # (currently just char-3gram matching).
    #     if ent._.kb_ents:
    #         log_f.write(str(ent) + ", " + linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name + ", " + str(ent._.kb_ents[0][1]) + "\n")
    results.append(entry)
with open(mapping_path, "w") as mapping_f:
    json.dump(results, mapping_f)



