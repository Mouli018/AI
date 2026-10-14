"""Creates data/sample_mini.json: a tiny SYNTHETIC HotpotQA-format set used ONLY for offline smoke tests
(never for reported results)."""
import json, random
random.seed(0)
countries = ["Norland", "Estria", "Valdoria", "Karelia", "Montavia", "Zeldora"]
cities = [f"{p}{s}" for p in ["Bel", "Mar", "Tor", "Sil", "Ren", "Cor", "Dal", "Fen"] for s in ["burg", "ville"]]
people = [f"{a} {b}" for a in ["Anna", "Boris", "Clara", "Dmitri", "Elena", "Felix", "Greta", "Hugo", "Iris", "Jonas"]
          for b in ["Vale", "Stone", "Ford"]]
paras = []
qs = []
for i, p in enumerate(people[:30]):
    city = cities[i % len(cities)]
    country = countries[i % len(countries)]
    p1 = (p, [f"{p} is a novelist.", f"{p} was born in {city}.", f"{p} wrote several books."])
    p2 = (city, [f"{city} is a city.", f"{city} is located in {country}."])
    paras += [p1, p2]
    qs.append({"_id": f"mini{i}", "question": f"In which country was the novelist {p} born?", "answer": country,
               "type": "bridge", "level": "easy", "supporting_facts": [[p, 1], [city, 1]], "context": []})
distract = [(f"Distractor{j}", [f"Distractor{j} is about topic number {j}.", f"It has nothing to do with the question."]) for j in range(60)]
allp = paras + distract
for q in qs:
    gold = [t for t, _ in q["supporting_facts"]]
    ctx = [pp for pp in allp if pp[0] in gold] + random.sample(distract, 8)
    random.shuffle(ctx)
    q["context"] = [[t, s] for t, s in ctx]
json.dump(qs, open("data/sample_mini.json", "w"))
print("wrote data/sample_mini.json", len(qs))
