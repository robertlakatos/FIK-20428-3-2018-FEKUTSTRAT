#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import click
import json
import pandas
import threading
import statistics


def add_new_record(buffer, group, index):
    for key in buffer.keys():
        buffer[key].append("")

    buffer["mrn"][-1] = group["mrn"].values[index]
    buffer["caseno"][-1] = group["caseno"].values[index]
    buffer["ordno"][-1] = group["ordno"].values[index]
    buffer["rectype"][-1] = group["rectype"].values[index]
    buffer["mrkeyword"][-1] = group["mrkeyword"].values[index]
    buffer["datatype"][-1] = group["datatype"].values[index]
    buffer["obsval"][-1] = group["obsval"].values[index]


def separte_and(group, index, key_1, key_2, buffer, count):
    val = group["obsval"].values[index]
    count["Total"] = count["Total"] + 1
    add_new_record(buffer, group, index)

    if type(val) != str:
        return

    tmp_data = val.split("&&")
    if len(tmp_data) != 2:
        return

    buffer[key_1][-1] = tmp_data[0]
    buffer[key_2][-1] = tmp_data[1]
    count["Found"] = count["Found"] + 1


def count_stat():
    result = {
        "Total": 0,
        "Found": 0
    }

    return result


def get_values(val):
    error = False
    result = ""
    for c in val:
        if c == ".":
            error = True
            break
        else:
            if c.isdigit() or c == "-":
                result = result + c
            else:
                continue

    if error:
        return ""
    else:
        return result


def get_imt_mean(val):
    # Mivan ha Több IMT érték is
    tmp_val = val.replace(". ", ",")
    tmp_val = tmp_val.replace(" ", "")
    tmp_val = tmp_val.split(",")
    tmp_val = tmp_val[0:10]
    end = 0
    try:
        end = tmp_val[-1].index("mm")
        tmp_val[-1] = tmp_val[-1][:end]
    except:
        pass

    tmp_val = [re.sub("[^0-9.]", "", item) for item in tmp_val]
    # tmp_val = val.replace("m", "")
    # tmp_val = tmp_val.replace(" ", "")

    tmp_val = [float(item) for item in tmp_val]

    return statistics.mean(tmp_val)


def filtering_from_templates(group, index, templates, buffer, stat):
    add_new_record(buffer, group, index)
    val = group["obsval"].values[index]
    find = False
    for item in templates:
        keys = item.keys()

        if "Minta prefix" in keys and "Minta prefix hossza" in keys:
            try:
                start = val.index(item["Minta prefix"])
                start = start + item["Minta prefix hossza"]

                if item["Sablon az."] == "IMT":
                    tmp_val = get_imt_mean(val[start:])
                else:
                    end = val.index(item["Minta postfix"])
                    tmp_val = get_values(val[start:end])

                if tmp_val != "":
                    buffer[item["Sablon az."]][-1] = tmp_val
                    stat[item["Sablon az."]
                         ]["Found"] = stat[item["Sablon az."]]["Found"] + 1
                    stat[item["Sablon az."]
                         ]["Total"] = stat[item["Sablon az."]]["Total"] + 1
                    find = True
            except:
                pass

    if find == False:
        stat["NOTFOUND"]["Found"] = stat["NOTFOUND"]["Found"] + 1
        stat["NOTFOUND"]["Total"] = stat["NOTFOUND"]["Total"] + 1


def collect_data(data, templates):
    print("COLLECT DATA: START")

    buffer = {
        "mrn": [],
        "caseno": [],
        "ordno": [],
        "reqno": [],
        "rectype": [],
        "mrkeyword": [],
        "datatype": [],
        "obsval": [],
        "IDYARTBA_1": [],
        "IDYARTBA_2": [],
        "IDYBALCA_1": [],
        "IDYBALCA_2": [],
        "IDYBALVE_1": [],
        "IDYBALVE_2": [],
        "IDYDOPIR_1": [],
        "IDYDOPIR_2": [],
        "IDYJOBBC_1": [],
        "IDYJOBBC_2": [],
        "IDYJOBBV_1": [],
        "IDYJOBBV_2": [],
        "IDYVIGE2": [],
        "IDYVIGEN": [],
        "IDY50MM_1": [],
        "IDY50MM_2": [],
        "IDY55MM_1": [],
        "IDY55MM_2": [],
        "IDY45MM_1": [],
        "IDY45MM_2": [],
        "IDYJASUB_1": [],
        "IDYJASUB_2": [],
        "IDYBASUB_1": [],
        "IDYBASUB_2": [],
        "KERESKEP": [],
        "IDY1ACCO": [],
        "IDY2ACIN": [],
        "IDY3CARO": [],
        "IDY3EXT1": [],
        "IDY6JAVA": [],
        "IDY7BAVA": [],
        "IDY8ARBA": [],
        "IMT": []
    }
    # add_new_record(buffer)

    stat = {
        "IDYARTBA": count_stat(),
        "IDYBALCA": count_stat(),
        "IDYBALVE": count_stat(),
        "IDYDOPIR": count_stat(),
        "IDYJOBBC": count_stat(),
        "IDYJOBBV": count_stat(),
        "IDYVIGE2": count_stat(),
        "IDYVIGEN": count_stat(),
        "IDY50MM": count_stat(),
        "IDY55MM": count_stat(),
        "IDY45MM": count_stat(),
        "IDYJASUB": count_stat(),
        "IDYBASUB": count_stat(),
        "KERESKEP": count_stat(),
        "IDY1ACCO": count_stat(),
        "IDY2ACIN": count_stat(),
        "IDY3CARO": count_stat(),
        "IDY3EXT1": count_stat(),
        "IDY6JAVA": count_stat(),
        "IDY7BAVA": count_stat(),
        "IDY8ARBA": count_stat(),
        "IMT": count_stat(),
        "IDYVKOP": count_stat(),
        "IDYCIMK2": count_stat(),
        "IDYCIMK3": count_stat(),
        "IDYVELEM": count_stat(),
        "NOTFOUND": count_stat()
    }

    lenv = len(data)
    with click.progressbar(length=lenv, label="COLLECT DATA: ", fill_char=click.style('=', fg='white')) as bar:
        for name, group in data:
            leng = len(group["mrkeyword"].values)
            for i in range(0, leng):
                if group["mrkeyword"].values[i] == "IDYARTBA" or \
                        group["mrkeyword"].values[i] == "IDYBALCA" or \
                        group["mrkeyword"].values[i] == "IDYBALVE" or \
                        group["mrkeyword"].values[i] == "IDYDOPIR" or \
                        group["mrkeyword"].values[i] == "IDYJOBBC" or \
                        group["mrkeyword"].values[i] == "IDYJOBBV" or \
                        group["mrkeyword"].values[i] == "IDY50MM" or \
                        group["mrkeyword"].values[i] == "IDY55MM" or \
                        group["mrkeyword"].values[i] == "IDY45MM" or \
                        group["mrkeyword"].values[i] == "IDYJASUB" or \
                        group["mrkeyword"].values[i] == "IDYBASUB":
                    separte_and(group,
                                i,
                                group["mrkeyword"].values[i] + "_1",
                                group["mrkeyword"].values[i] + "_2",
                                buffer,
                                stat[group["mrkeyword"].values[i]])
                elif group["mrkeyword"].values[i] == "IDYVIGEN" or \
                        group["mrkeyword"].values[i] == "IDYVIGE2" or \
                        group["mrkeyword"].values[i] == "KERESKEP":
                    add_new_record(buffer, group, i)
                    tmp_keyowrd = group["mrkeyword"].values[i]
                    buffer[tmp_keyowrd][-1] = group["obsval"].values[i]
                    stat[tmp_keyowrd]["Total"] = stat[tmp_keyowrd]["Total"] + 1
                    stat[tmp_keyowrd]["Found"] = stat[tmp_keyowrd]["Found"] + 1
                elif group["mrkeyword"].values[i] == "IDYCIMK2" or \
                        group["mrkeyword"].values[i] == "IDYCIMK3" or \
                        group["mrkeyword"].values[i] == "IDYVELEM":
                    filtering_from_templates(group, i, templates, buffer, stat)
                    tmp_keyowrd = group["mrkeyword"].values[i]
                    stat[tmp_keyowrd]["Total"] = stat[tmp_keyowrd]["Total"] + 1
                    stat[tmp_keyowrd]["Found"] = stat[tmp_keyowrd]["Found"] + 1
                elif group["mrkeyword"].values[i] == "IDYVKOP":
                    add_new_record(buffer, group, i)
                    stat["IDYVKOP"]["Total"] = stat["IDYVKOP"]["Total"] + 1
                    stat["IDYVKOP"]["Found"] = stat["IDYVKOP"]["Found"] + 1
                else:
                    add_new_record(buffer, group, i)
                    stat["NOTFOUND"]["Found"] = stat["NOTFOUND"]["Found"] + 1
                    stat["NOTFOUND"]["Total"] = stat["NOTFOUND"]["Total"] + 1
                    continue

            bar.update(1)

    data = pandas.DataFrame()
    for key in buffer.keys():
        data[key] = buffer[key]
    print("COLLECT DATA: END")

    print("STAT:")
    print(stat)

    return (data, stat)


def read_data(filename):
    print("READ EXCEL: " + filename)
    data = pandas.DataFrame()

    if os.path.isfile(filename) == False:
        return data

    if(filename.split(".")[-1] != "xlsx"):
        return data

    xl = pandas.ExcelFile(filename)
    parse = xl.parse(xl.sheet_names)

    data["mrn"] = parse[xl.sheet_names[0]]["mrn"]
    data["caseno"] = parse[xl.sheet_names[0]]["caseno"]
    data["ordno"] = parse[xl.sheet_names[0]]["ordno"]
    data["reqno"] = parse[xl.sheet_names[0]]["reqno"]
    data["rectype"] = parse[xl.sheet_names[0]]["rectype"]
    data["mrkeyword"] = parse[xl.sheet_names[0]]["mrkeyword"]
    data["datatype"] = parse[xl.sheet_names[0]]["datatype"]
    data["obsval"] = parse[xl.sheet_names[0]]["obsval"]

    print("READ EXCEL: COMPLETE")
    print("COUNT OF RECORDS OF DATA: %d" % (len(data)))

    print("GROUP DATA: START")
    data = data.groupby(data['caseno'])
    print("GROUP DATA: END")

    return data


def write_stat(out_dir, data, filename):
    out = out_dir + "/" + filename
    print("WRITE STAT: " + out)
    f = open(out, "w")
    f.write(";Total;Found\n")
    for key in data.keys():
        f.write(key + ";" + str(data[key]["Total"]) +
                ";" + str(data[key]["Found"]) + "\n")

    f.close()
    print("WRITE STAT: COMPLETE")


def write_date(out_dir, data, filename):
    out = out_dir + "/" + filename
    print("WRITE DATA: " + out)
    with pandas.ExcelWriter(out, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False)
    print("WRITE DATA: COMPLETE")


def run(config, templates, filename):
    # Read Data
    data = read_data(config["source"] + "/" + filename)

    # Collect Data
    (data, stat) = collect_data(data, templates)

    # Write Stat
    write_stat(config["output"], stat, filename.split(".")[0] + "_stat.csv")

    # Write Data
    write_date(config["output"], data, filename)

    return 0


def main():
    # Read config
    with open("configs/config_c01.json", "r") as config_file:
        config = json.load(config_file)
    print("READ CONFIGURATIONS: COMPLETE")

    # Read templates
    with open(config["templates"], "r", encoding='utf-8') as template_file:
        templates = json.load(template_file)
    print("READ TEMPLATES: COMPLETE")

    files = os.listdir(config["source"])

    index = 0
    count_of_files = len(files)
    while index < count_of_files:
        threads = []
        for job in range(0, config["n_jobs"]):
            if (index + job) < count_of_files:
                threads.append(myThread(config, templates, files[index + job]))
            else:
                break

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        index = index + config["n_jobs"]

        break  # majd ki kell szedni, hogy ne csak az első állományt dolgozzuk fel


class myThread (threading.Thread):
    def __init__(self, config, templates, filename):
        threading.Thread.__init__(self)
        self.config = config
        self.templates = templates
        self.filename = filename
        self.output = 0

    def run(self):
        self.output = run(self.config,
                          self.templates,
                          self.filename)


if __name__ == "__main__":
    main()
