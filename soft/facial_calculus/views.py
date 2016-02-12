from django.shortcuts import render
from django.http import HttpResponse
import os
from django.shortcuts import render_to_response, redirect
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from facial_calculus.models import Document, DocumentOutput
from facial_calculus.forms import DocumentForm
from facial_calculus.functions import *

def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'],filename = request.FILES['docfile'].name[:-4])
            newdoc.save()
			

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('facial_calculus.views.list'))
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render_to_response(
        'facial_calculus/list.html',
        {'documents': documents, 'form': form},
        context_instance=RequestContext(request)
    )

def index(request):
	return HttpResponse("Hello world")

def delete(request, pic_id):
	obj = Document.objects.get(pk=pic_id)
	exists = DocumentOutput.objects.filter(filename = obj.filename+"_output").exists()
	obj2 = None
	if exists == True:
		obj2 = DocumentOutput.objects.get( filename = obj.filename+"_output")
		print "postoji i 2"
	

	if os.path.isfile(obj.docfile.url[1:]):
		os.remove(obj.docfile.url[1:])
		obj.delete()
	if exists == True and os.path.isfile(obj2.docfile.url[1:]):
		os.remove(obj2.docfile.url[1:])
		obj2.delete()
	print "My work here is done, deleted!"
	return redirect('list')
	
def redirect_to_analyze(request, pic_id):
	#return HttpResponseRedirect(reverse('analyze.views.calculate'))
	obj = Document.objects.get(pk=pic_id)

	output = ""
	newdoc, created = DocumentOutput.objects.get_or_create(docfile = "output/"+obj.filename + "_output.jpg" ,filename = obj.filename + "_output")
	print "Da li je kreirao? ", created
	if created == True:
		putanja = obj.docfile.url
		putanja = putanja.replace('/','//')[2:]
	
		im1, landmarks1 = read_im_and_landmarks(putanja )
		result_dic = dict()
		calculate_face_width(landmarks1, result_dic )
		p = calculate_proportion(landmarks1)
		result_dic2 = calculate_percentage(result_dic, p)
	
		#image = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
		image = annotate_landmarks(im1,landmarks1)
		pp = "media//output//" + obj.filename + "_output.jpg"
		print obj.filename
		slika = ""
		if save_image_fun(pp, image) == True:
			slika = pp.replace('//','/')
		for key in result_dic2:
			output += key + ": " + str(result_dic2[key]) + "\n"
		newdoc.textfield = output
		newdoc.opis = make_description(result_dic2)
		newdoc.save()
	
	return render_to_response(
        'facial_calculus/analisys.html',
        {'output': newdoc.textfield, 'slika': newdoc, 'opis': newdoc.opis},
        context_instance=RequestContext(request)
    )
	