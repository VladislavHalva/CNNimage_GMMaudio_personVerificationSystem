Projekt do předmětu SUR - Verifikace osoby z obrázku a zvukové nahrávky
Vladislav Halva, Martin Dvořák
xhalva04, xdvora2l
25. duben 2020

#### Spuštění

Spustit evaluaci je možné zadáním příkazu

    python3 personVerification.py <data dir path>

v adresáři SRC, kde data dir path je cesta k adresáři s daty.

Pro spustění je také **nutné** stáhnout natrénované modely z [tohoto odkazu](https://l.facebook.com/l.php?u=https%3A%2F%2Fwww.stud.fit.vutbr.cz%2F~xdvora2l%2Fmodels.zip%3Ffbclid%3DIwAR26c3KOU5PqEVHT-A-dfyevoX0FX1aUv49NH9xIRVrD5qf7TCsyiRiX9Gw&h=AT3kW1YcqF-s-_TDrxqo7-oQh1p972YrZTia9kaRO_ISI98FBMMg6z358ygX1ys2QxdnbXH7dz0PYXH73KU09sZnVX0OdWE1GTyk4seoYvd2aZBnh8i4A5qveNe_LeJ8KCH96AVeHtQ) a umístit stažený adresář models do kořenového adresáře projektu (tj. na stejnou úroveň jako adresář SRC).

Program následně vytvoří v kořenovém adresáři tři soubory s výsledky pro jednotlivé varianty klasifikátoru tj. konvoluční neuronová síť pro obrázky, GMM pro zvukové nahrávky a kombinovaný model. Námi preferovaný výstup pro hodnocení je kombinovaný model.