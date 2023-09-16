from utils import updater,explain_printer

def cleaner(sample_file):
    attribute_list=['mean radius','mean texture','mean perimeter','mean area','mean smoothness','mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error','texture error','perimeter error','area error','smoothness error','compactness error','concavity error','concave points error','symmetry error','fractal dimension error','worst radius','worst texture','worst perimeter','worst area','worst smoothness','worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension']
    options=["min","max"]
    ex_vals=[float("inf"),float('-inf')]
    final_explanation={}
    for i in attribute_list:
        for j in [0,1]:
            final_explanation[(i,options[j])]=ex_vals[j]
    with open(sample_file,"r") as output_file:
        for line in output_file:
            if(len(line.split())!=0):
                if "Going"==line.split()[0]:
                    final_explanation=updater(line,final_explanation)
        result=explain_printer(final_explanation)
        result+=line
        # print(line)
    return result

if __name__=="__main__":
    result=cleaner("sample.txt")
    print(result)