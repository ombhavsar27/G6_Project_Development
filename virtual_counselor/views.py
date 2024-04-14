from django.shortcuts import render
from .utils.college_predictor import predict_colleges  # Import your Python code here

def index(request):
    return render(request, 'counselor/index.html')

def contact(request):
    return render(request, 'counselor/contact.html')

def about(request):
    return render(request, 'counselor/about.html')

def recom(request):
    if request.method == 'POST':
        percentile = float(request.POST.get('Percentile'))
        course = request.POST.get('Course')
        caste = request.POST.get('Caste')
        
        
        # Call your Python function to predict colleges
        recommended_colleges = predict_colleges(percentile, caste, course)  # Adjust parameters as needed
        
        # # Join the list of colleges into a single string with a separator
        # recommended_colleges_str = "\n".join(recommended_colleges)

# Prepare context to pass to template
        context = {
            'recommended_colleges': recommended_colleges  # Pass the joined string to the template
}
        print("Recommended Colleges:", recommended_colleges) 

        return render(request, 'counselor/output.html', context)
    
    return render(request, 'counselor/recom.html')

# View function
from django.shortcuts import render

# View function
from django.shortcuts import render

# View function
def output(request):
    if request.method == 'POST':
        # Retrieve data from the form submission
        percentile_str = request.POST.get('percentile')
        caste = request.POST.get('caste')
        course = request.POST.get('course')
        
        # Check if percentile_str is not empty
        if percentile_str:
            try:
                percentile = float(percentile_str)
            except (TypeError, ValueError):
                # Handle the case where the input is not a valid float
                # You might want to add appropriate error handling or redirect logic here
                percentile = None
        else:
            percentile = None
        
        # Call the predict_colleges function with the retrieved data
        recommended_colleges = predict_colleges(percentile, caste, course)
        
        # Prepare the context to pass to the template
        context = {'recommended_colleges': recommended_colleges}
        
        # Render the template with the context and return the HttpResponse
        return render(request, 'counselor/output.html', context)
    else:
        # Handle the case when the request method is not POST
        # You might want to add some error handling or redirect logic here
        pass
