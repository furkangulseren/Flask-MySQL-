import base64
import io
import os
from base64 import b64encode, b64decode
from flask import Flask, render_template, url_for, flash, redirect, session, request, logging, make_response
from forms import RegistrationForm, LoginForm
from PIL import Image
from flask_mysqldb import MySQL
from wtforms import Form, StringField, SelectField, TextAreaField, PasswordField, validators, FileField, SubmitField
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request

import pickle
import os
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

# from flask_wtf import Form
# from wtforms import StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from datetime import datetime


app = Flask(__name__)
from database import Articles

#UPLOAD_FOLDER = 'templates/image'
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

Articles = Articles()
# app.config['SECRET_KEY'] = ''

# SECRET_KEY = os.urandom(32)
# app.config['SECRET_KEY'] = 'secret123'

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'books'
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MySQL
mysql = MySQL(app)

app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))


posts = [

    {
        'id': 1,
        'author': 'Furkan Gulseren',
        'title': 'Deep Learning',
        'content': 'ince kapak',
        'date_posted': 'March 10, 2019'
    },
    {
        'id': 2,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 3,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 4,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 5,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 6,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 7,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    },
    {
        'id': 8,
        'author': 'Ahmed Groshar',
        'title': 'Machine Learning',
        'content': 'ince kapak',
        'date_posted': 'March 12, 2019'
    }
]


class FilterForm(Form):
    book_category = SelectField('Category', choices=[(1, "Novel"), (2, "Technology"), (3, "Fantasy Fiction"), (4, "Arts&Literature"), (5, "Exam")], coerce=int)
    book_sale_type = SelectField('Sale Type', choices=[(1, "For Sale"), (2, "For Rent")], coerce=int)
    book_price_from = StringField('', [validators.length(min=0, max=4)], render_kw={"placeholder": str(0) + u" \u20BA"})
    book_price_to = StringField('', [validators.length(min=0, max=4)], render_kw={"placeholder": str(1000) + u" \u20BA"})


# Home
@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home():
    form = FilterForm(request.form)
    # return render_template('home.html', posts=posts)
    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get all books
        # name = session['username']
        cur.execute("SELECT * FROM books WHERE books.active = 1 ORDER BY id DESC")

        # Get stored hash
        data = cur.fetchall()
        #bookImageDB = b64decode(data[0]['image'])
        bookImage = data[0]['image']
        #print(bookImageDB)

        #bookImageDB = Image.frombytes((100, 50), data[0]['image'])
        #bookImage = Image.frombytes('RGB', (10, 15), bookImageDB)
        #bookImage.save("foo.png")

        #with open("foo.png", "wb") as f:
        #    f.write(b64decode(bookImageDB))

        #for i in data:
            #print(i)
        cur.close()
        return render_template('home.html', posts=data, form=form, bookImage=bookImage)

    if request.method == 'POST' and form.validate():
        book_category = form.book_category.data
        book_category_id = form.book_category.choices
        book_price_from = form.book_price_from.data
        book_price_to = form.book_price_to.data
        if book_price_to == "":
            book_price_to = 1000
        if book_price_from == "":
            book_price_from = 0

        try:
            for i in range(0, len(book_category_id)):
                if book_category == i:
                    book_category = book_category_id[i - 1][1]

            book_sale_type = form.book_sale_type.data
            book_sale_id = form.book_sale_type.choices
            if book_sale_type == 2:
                book_sale_type = str(book_sale_id[1][1])
            elif book_sale_type == 1:
                book_sale_type = str(book_sale_id[0][1])

            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM books WHERE books.active = 1 AND books.book_category =  %s  AND books.book_sale_type = %s AND books.book_price BETWEEN %s AND %s", (book_category, book_sale_type, int(book_price_from), int(book_price_to)))

            data = cur.fetchall()

            cur.close()
            return render_template('home.html', form=form, posts=data)
        except ValueError:
            flash('please enter a valid number', 'danger')
            #return render_template('home.html', form=form, posts=data)
            return render_template('home.html', form=form)

    return render_template('home.html')


porter = PorterStemmer()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


def comment_summary(comment):
    vect = CountVectorizer(max_features=10, stop_words='english')

    X = vect.fit_transform([comment])

    lda = LatentDirichletAllocation(learning_method="batch", max_iter=25, random_state=0)

    document_topics = lda.fit_transform(X)

    sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
    # get the feature names from the vectorizer:
    feature_names = np.array(vect.get_feature_names())

    summary_of_comment = []
    for i in range(10):
        summary_of_comment.append(feature_names[sorting[range(1), i]][0])

    return summary_of_comment


# Single Book
@app.route('/bookDetail/<string:id>/', methods=['GET', 'POST'])
def bookDetail(id):
    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM books WHERE books.id="+str(id))

        # Get stored hash
        data = cur.fetchone()

        cur.close()

        #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM books WHERE books.book_category=%s LIMIT 3", [data['book_category']])

        # Get stored hash
        same_categories = cur.fetchall()

        cur.close()

        #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM book_review WHERE book_review.book_id=%s", [str(id)])

        # Get stored hash
        book_comments = cur.fetchall()

        cur.close()

        #####

        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT visit_count FROM books WHERE books.id=%s", [str(id)])

        # Get stored hash
        visit_count = cur.fetchone()
        visit_count['visit_count'] = visit_count['visit_count'] + 1
        #print(visit_count)
        cur.execute("UPDATE books SET books.visit_count = %s WHERE books.id=%s", [int(visit_count['visit_count']), str(id)])
        # Commit to DB
        mysql.connection.commit()
        cur.close()

        # Comment Prediction Start #
        clf = pickle.load(open(os.path.join('pkl_objects', 'classifier.pkl'), 'rb'))
        label = {0: 'negative', 1: 'positive'}
        comment_prediction = []
        for i in book_comments:
            X = [str(i['review'])]
            prediction = label[clf.predict(X)[0]]
            #print('Prediction: %s\nProbability: %s' % (label[clf.predict(X)[0]], clf.predict_proba(X)))
            #print(round(max(clf.predict_proba(X)[0])*100, 2))
            probability = round(max(clf.predict_proba(X)[0]) * 100, 2)
            #print(str(i['review']))
            summary = comment_summary(str(i))
            reviewLength = len(str(i['review']))
            #print(reviewLength)
            comment_prediction.append({'prediction': prediction, 'probability': probability, 'summary': summary, 'reviewLength': reviewLength})
        # print(comment_prediction)
        # Comment Prediction End #
        # comment_prediction = comment_prediction[::-1]
        if 'username' not in session:
            pass
        else:
            checked_books = session['checked_books']
            if id not in checked_books:
                checked_books.append(id)
                session['checked_books'] = checked_books
                #print(checked_books)
            else:
                pass

        return render_template('bookDetail.html', comment_prediction=comment_prediction, posts=data, same_categories=same_categories, book_comments=book_comments)

    return render_template('bookDetail.html')


class CommentForm(Form):
    username = StringField('Username', [validators.Length(min=4, max=50)])
    book_name = StringField('Book Name', [validators.Length(min=0, max=500)])
    comment = StringField('Comment', [validators.Length(min=1, max=500)])
    rating = StringField('Rating', [validators.Length(min=0, max=2)])


@app.route('/comment/<string:id>/', methods=['GET', 'POST'])
def makeComment(id):
    if request.method == 'POST':
        form = CommentForm(request.form)
        #book_name = str(form.book_name.data)
        comment = form.comment.data
        rating = form.rating.data
        username = session['username']
        null = 0

        cur1 = mysql.connection.cursor()

        # Get user by username
        cur1.execute("SELECT books.book_name FROM books LEFT JOIN book_review ON books.id = book_review.book_id WHERE books.id= " + str(id))

        # Commit to DB
        book_name = cur1.fetchone()
        #print(book_name)

        cur1.close()


        if comment != '' and rating != '' and 6 > int(rating) >= 0:

            # Create cursor
            cur = mysql.connection.cursor()

            # Get user by username
            cur.execute("INSERT INTO book_review VALUES(%s,%s,%s,%s,%s,%s,%s)", (null, id, book_name['book_name'], username, rating, comment, datetime.now()))

            # Commit to DB
            mysql.connection.commit()

            cur.close()

            cur1 = mysql.connection.cursor()
            cur1.execute("SELECT books.review_number FROM books WHERE books.id= " + str(id))
            book_review_number = cur1.fetchone()
            cur1.close()

            #print(book_review_number[])
            new_review_number = int(book_review_number['review_number']) + 1
            cur2 = mysql.connection.cursor()
            cur2.execute("UPDATE books SET books.review_number = %s WHERE books.id=%s",
                        [int(new_review_number), str(id)])
            mysql.connection.commit()
            cur2.close()


        ##### ------------- #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM books WHERE books.id=" + str(id))

        # Get stored hash
        data = cur.fetchone()

        cur.close()

        #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM books WHERE books.book_category=%s", [data['book_category']])

        # Get stored hash
        same_categories = cur.fetchall()

        cur.close()

        #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM book_review WHERE book_review.book_id="+str(id))

        # Get stored hash
        book_comments = cur.fetchall()

        cur.close()

        #####
        return render_template('bookDetail.html', posts=data, same_categories=same_categories, book_comments=book_comments)
    return render_template('home.html')


# About
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/favori')
def favori():
    return render_template('favori.html')


########### BOOK REC. ##########
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    return Sxy / np.sqrt(Sxx * Syy)


def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        if similarity_score <= 0:
            continue

        filtered_list = [x for x in dataset[user] if x not in \
                         dataset[input_user] or dataset[input_user][x] == 0]

        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Generate movie ranks by normalization
    movie_scores = np.array([[score / similarity_scores[item], item]
                             for item, score in overall_scores.items()])

    # Sort in decreasing order
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    # Extract the movie recommendations
    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations


def get_recommendations_new(dataset, n=10):
    """Returns """

    book_count_dict = {}
    for i in dataset:
        for j in dataset[i]:
            if j not in book_count_dict:
                book_count_dict[j] = [1, dataset[i][j]]
                #print(book_count_dict[j])
            else:
                book_count_dict[j][0] += 1
                book_count_dict[j][1] += dataset[i][j]

    book_counts = list(book_count_dict.items())
    book_counts.sort(key=lambda x: x[1])
    top_n_list = []
    #print(book_counts[len(book_counts)-10:])
    for i in (book_counts[len(book_counts)-10:]):
        top_n_list.append((i[0], i[1][1]/i[1][0]))

    top_n_list.sort(key=lambda x: x[1])
    book_names = []
    for i in range(len(top_n_list)-1,-1,-1):
        book_names.append(top_n_list[i][0])
    return book_names


@app.route('/tavsiye')
def tavsiye():
    # Create cursor
    cur = mysql.connection.cursor()

    # Get user by username
    username = session['username']
    cur.execute("SELECT * FROM book_review ")

    # Get stored hash
    data = cur.fetchall()

    cur.close()

    newDict1 = {}
    newDict2 = {}
    for i in data:
         newDict1[i['book_name']] = i['reviewerRatings']
         newDict2[i['reviewerName']] = newDict1
    #print(newDict2)
    #newDict2= pd.DataFrame(newDict2.items(), columns=['book_name', 'reviewerRatings'])
    books = get_recommendations(newDict2, username)
    if books[0] == 'No recommendations possible':
        books = get_recommendations_new(newDict2, 10)
    k = 10
    #print(books)
    #for i, book in enumerate(books[:10]):
    #    print(str(i + 1) + '. ' + book)
    #print(books)

    return render_template('tavsiye.html', recommendations=books)


# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Password do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    form = RegistrationForm()
    if form.validate_on_submit():
        flash('Account created for {{ form.username.data }}', 'success')
        return redirect(url_for('home'))
    return render_template('register2.html', title ='Register', form=form)
    """
    form = RegisterForm(request.form)
    # form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM users WHERE username = %s OR email = %s", [username, email])

        if result >= 1:
            cur.close()
            error = 'Already registered User'
            return render_template('register2.html', form=form, error=error)

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)", (name, email,
                                                                                                  username, password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('You are now registered and can login', 'success')

        return redirect(url_for('login'))
        # return render_template('register2.html')
    return render_template('register2.html', form=form)


# User login
@app.route("/login", methods=['GET', 'POST'])
def login():
    """
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', title='Login', form=form)
    """
    if request.method == 'POST':
        # Get form fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                app.logger.info("PASSWORD MATCHED")
                # return render_template('home.html')
                # return redirect(url_for('home'))
                # Passed
                session['logged_in'] = True
                session['username'] = username
                session['checked_books'] = []

                flash('You are now logged in', 'success')
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid Login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


# Register Form Class
class BookSaleForm(Form):
    book_name = StringField('Book Name', [validators.Length(min=1, max=100)])
    #book_category = StringField('Book Category', [validators.Length(min=4, max=40)])
    book_category = SelectField('Book Category', choices=[(1, "Novel"), (2, "Technology"), (3, "Fantasy Fiction"), (4, "Arts&Literature"), (5, "Exam")], coerce=int)
    book_author = StringField('Book Author', [validators.Length(min=4, max=1000)])
    book_page_number = StringField('Book Page Number', [validators.Length(min=0, max=4)])
    #book_sale_type = StringField('Book Sale Type', [validators.Length(min=6, max=40)])
    book_sale_type = SelectField('Book Sale Type', choices=[(1, "For Sale"), (2, "For Rent")], coerce=int)
    book_price = StringField('Book Price', [validators.Length(min=0, max=6)])
    definition = TextAreaField('Book Definition', [validators.Length(min=4, max=1000)])


@app.route('/bookSale', methods=['GET', 'POST'])
def bookSale():
    form = BookSaleForm(request.form)
    book_category_id = form.book_category.choices
    book_category = form.book_category.data
    #file = request.files['inputImage']
    #print(file)
    #return render_template('makeOffer.html')
    # form = RegistrationForm(request.form)

    if request.method == 'POST' and form.validate():
        book_name = form.book_name.data
        book_author = form.book_author.data
        book_page_number = form.book_page_number.data
        book_category = form.book_category.data
        book_category_id = form.book_category.choices
        for i in range(0, len(book_category_id)):
            if book_category == i:
                book_category = book_category_id[i-1][1]

        book_sale_type = form.book_sale_type.data
        book_sale_id = form.book_sale_type.choices
        if book_sale_type == 2:
            book_sale_type = str(book_sale_id[1][1])
        elif book_sale_type == 1:
            book_sale_type = str(book_sale_id[0][1])

        book_price = form.book_price.data
        # confirm_password = form.confirm_password.data
        definition = form.definition.data

        target = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/image/')
        file = request.files['inputImage']

        filename = secure_filename(file.filename)
        destination = "/".join([target, filename])
        file.save(destination)
        # C:\Users\Muhammed Fatih\Desktop\flask_project\templates\image
        # image = file.read()
        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute(
            "INSERT INTO books(book_name, book_author, book_page_number, book_category, book_sale_type, book_price, definition, image, image_name, saler_user, active) "
            "VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (book_name, book_author, book_page_number, book_category, book_sale_type, book_price,
                                                   definition, '', filename, session['username'], 1))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('Your book is successfully added', 'success')

        #return redirect(url_for('bookSale'))
        return render_template('bookSale.html', form=form)
        # return render_template('register2.html')
    return render_template('bookSale.html', form=form)


@app.route('/allBooks')
def allBooks():
    if request.method == 'POST':
        # Get form fields
        # username = request.form['username']
        # password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM books")

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            # password = data['password']
            return render_template('home.html', posts=data)
            cur.close()
        else:
            error = 'Book ad not found'
            return render_template('home.html', error=error)
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        name = session['username']
        cur.execute("SELECT * FROM users WHERE users.username=%s", [name])

        # Get stored hash
        data = cur.fetchone()

        cur.close()

        #####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        name = session['username']
        cur.execute("SELECT * FROM books WHERE books.saler_user=%s ORDER BY books.active DESC", [name])

        # Get stored hash
        saler_data = cur.fetchall()

        cur.close()


        ########
        checked_books = session['checked_books']
        booksList = []
        for i in checked_books:
            cur = mysql.connection.cursor()

            # Get user by username
            cur.execute("SELECT * FROM books WHERE books.id=%s", [str(i)])

            # Get stored hash
            checked = cur.fetchall()
            booksList.append(checked)
            cur.close()
        return render_template('dashboard.html', posts=data, saler_posts=saler_data, checked=booksList)

    return render_template('dashboard.html')


# Search Form Class
class BookSearchForm(Form):
    search = StringField('Search', [validators.Length(min=1, max=100)])


@app.route('/bookSearch', methods=['POST', 'GET'])
def bookSearch():
    form = BookSearchForm(request.form)
    search = form.search.data

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM books WHERE books.active = 1 AND (books.book_name LIKE '%" + search + "%' OR books.definition LIKE '%" + search + "%')")

    data = cur.fetchall()
    cur.close()
    length = len(data)

    if length == 0:
        message = 'No result can be found.'
        flash(message, 'danger')
    else:
        message = str(length)+' results have found.'
        flash(message, 'success')

    return render_template('bookSearch.html', posts=data)





@app.route('/bookSearchName/<string:search>', methods=['POST', 'GET'])
def bookSearchName(search):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM books WHERE books.active = 1 AND (books.book_name LIKE '%" + search + "%' OR books.definition LIKE '%" + search + "%')")

    data = cur.fetchall()
    cur.close()
    length = len(data)

    if length == 0:
        message = 'No result can be found.'
        flash(message, 'danger')
    else:
        message = str(length)+' results have found.'
        flash(message, 'success')

    return render_template('bookSearch.html', posts=data)






@app.route('/makeOffer/<string:id>/', methods=['POST', 'GET'])
def makeOffer(id):
    return render_template('makeOffer.html')


@app.route('/upload/', methods=['POST', 'GET'])
def upload():
    file = request.files['image']
    print(file.filename)
    return render_template('makeOffer.html')


class AddNeedForm(Form):
    username = StringField('username', [validators.Length(min=4, max=50)])
    title = StringField('title', [validators.Length(min=1, max=200)])
    description = TextAreaField('description', [validators.Length(min=0, max=500)])


@app.route('/bookNeed/', methods=['POST', 'GET'])
def bookNeed():
    form = AddNeedForm(request.form)

    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get all needs
        # name = session['username']
        cur.execute("SELECT * FROM book_need_title WHERE book_need_title.active = 1 ORDER BY id DESC")

        # Get stored hash
        data = cur.fetchall()

        cur.close()
        return render_template('bookNeed.html', posts=data, form=form)

    if request.method == 'POST':
        title = form.title.data
        description = form.description.data

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO book_need_title(username, title, description, active) VALUES(%s, %s, %s, %s)", (session['username'], title, description, 1))

        mysql.connection.commit()

        cur.close()

        # Create cursor
        cur = mysql.connection.cursor()

        # Get all needs
        # name = session['username']
        cur.execute("SELECT * FROM book_need_title WHERE book_need_title.active = 1 ORDER BY id DESC")

        # Get stored hash
        data = cur.fetchall()

        cur.close()
        return render_template('bookNeed.html', form=form, posts=data)

    return render_template('bookNeed.html')


class AddNeedDetailForm(Form):
    username = StringField('username', [validators.Length(min=4, max=50)])
    userTitle = StringField('title', [validators.Length(min=1, max=200)])
    userMessage = TextAreaField('user_message', [validators.Length(min=0, max=500)])


@app.route('/bookNeedDetail/<string:id>', methods=['POST', 'GET'])
def bookNeedDetail(id):
    form = AddNeedDetailForm(request.form)

    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get all needs
        cur.execute("SELECT * FROM book_need_title WHERE book_need_title.active = 1 AND book_need_title.id = " + str(id))

        # Get stored hash
        data = cur.fetchone()



        # Get stored hash
        #visit_count = cur.fetchone()
        #print(data)
        data['visit_count'] = data['visit_count'] + 1
        # print(visit_count)
        cur.execute("UPDATE book_need_title SET book_need_title.visit_count = %s WHERE book_need_title.id=%s",
                    [int(data['visit_count']), str(id)])
        # Commit to DB
        mysql.connection.commit()
        cur.close()

        #### Comments ####

        # Create cursor
        cur = mysql.connection.cursor()

        # Get all needs
        # name = session['username']
        cur.execute(
            "SELECT * FROM book_need WHERE book_need.receiver_message_id = " + str(id))

        # Get stored hash
        needMessages = cur.fetchall()

        cur.close()

        return render_template('bookNeedDetail.html', posts=data, form=form, needMessages=needMessages)

    if request.method == 'POST':
        userTitle = form.userTitle.data
        userMessage = form.userMessage.data

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO book_need(sender_user_name, title, user_message, receiver_message_id) VALUES(%s, %s, %s, %s)", (session['username'], userTitle, userMessage, id))

        mysql.connection.commit()

        cur.close()

        # Create cursor
        cur = mysql.connection.cursor()

        cur.execute("SELECT * FROM book_need_title WHERE book_need_title.active = 1 AND book_need_title.id = " + str(id))

        # Get stored hash
        data = cur.fetchone()

        cur.close()

        #### Comments ####

        # Create cursor
        cur = mysql.connection.cursor()

        cur.execute(
            "SELECT * FROM book_need WHERE book_need.receiver_message_id = " + str(id) + " ORDER BY id ASC")

        needMessages = cur.fetchall()

        cur.close()
        return render_template('bookNeedDetail.html', form=form, posts=data, needMessages=needMessages)

    return render_template('bookNeedDetail.html')




@app.route('/messages/', methods=['POST', 'GET'])
def messages():
    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        username = session['username']
        cur.execute("SELECT * FROM user_messages WHERE user_messages.receiver_user_name=%s OR user_messages.sender_user_name=%s ORDER BY id DESC", ([username], [username]))

        data = cur.fetchall()

        usersList = set()
        for i in data:
            if i['receiver_user_name'] == session['username']:
                usersList.add(i['sender_user_name'])
            elif i['sender_user_name'] == session['username']:
                usersList.add(i['receiver_user_name'])

        cur.close()

        cur2 = mysql.connection.cursor()

        username = session['username']
        cur2.execute("SELECT COUNT(*) FROM user_messages WHERE user_messages.receiver_user_name = %s AND user_messages.is_read=0",
                    ([username]))

        data2 = cur.fetchall()

        cur2.close()
        #print(len(data))
        if len(data) == 0:
            flash('No message', 'danger')
            #return render_template('messages.html', posts=data)

        return render_template('messages.html', posts=usersList)

    return render_template('messages.html')


class SendMessageForm(Form):
    username = StringField('username', [validators.Length(min=4, max=50)])
    userTitle = StringField('title', [validators.Length(min=1, max=200)])
    userMessage = TextAreaField('user_message', [validators.Length(min=0, max=500)])


@app.route('/messageDetail/<string:user>', methods=['POST', 'GET'])
def messageDetail(user):
    form = SendMessageForm(request.form)

    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        username = session['username']
        cur.execute("SELECT * FROM user_messages WHERE user_messages.receiver_user_name = %s AND user_messages.sender_user_name= %s OR user_messages.receiver_user_name = %s AND user_messages.sender_user_name= %s ORDER BY id DESC", ([username], user, user, [username]))

        data = cur.fetchall()

        cur.close()

        #print(data)
        if len(data) == 0:
            flash('No message', 'danger')
            #return render_template('messages.html', posts=data)

        return render_template('messageDetail.html', posts=data, receiver=user)

    if request.method == 'POST':
        userTitle = form.userTitle.data
        userMessage = form.userMessage.data

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO user_messages(sender_user_name, receiver_user_name, message_title, message, is_read) VALUES(%s, %s, %s, %s, %s)",
            (session['username'], user, userTitle, userMessage, '0'))
        mysql.connection.commit()
        cur.close()

        cur = mysql.connection.cursor()
        username = session['username']
        cur.execute(
            "SELECT * FROM user_messages WHERE user_messages.receiver_user_name = %s AND user_messages.sender_user_name= %s OR user_messages.receiver_user_name = %s AND user_messages.sender_user_name= %s ORDER BY id DESC",
            ([username], user, user, [username]))
        data = cur.fetchall()
        cur.close()

        return render_template('messageDetail.html', posts=data, receiver=user)

    return render_template('messageDetail.html')


@app.route('/sameCategory/<string:category>')
def sameCategory(category):
    if request.method == 'GET':
        # Create cursor
        cur = mysql.connection.cursor()

        # Get all books
        # name = session['username']
        cur.execute("SELECT * FROM books WHERE books.active = 1 AND books.book_category= %s ORDER BY id DESC",([category]))

        # Get stored hash
        data = cur.fetchall()

        cur.close()
        return render_template('sameCategory.html', posts=data)


@app.route('/soldBook/<string:id>')
def soldBook(id):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE books SET books.active = 0 WHERE books.id=%s", [str(id)])
    mysql.connection.commit()
    cur.close()

    # Create cursor
    cur = mysql.connection.cursor()

    # Get user by username
    name = session['username']
    cur.execute("SELECT * FROM users WHERE users.username=%s", [name])

    # Get stored hash
    data = cur.fetchone()

    cur.close()

    #####

    # Create cursor
    cur = mysql.connection.cursor()

    # Get user by username
    name = session['username']
    cur.execute("SELECT * FROM books WHERE books.saler_user=%s", [name])

    # Get stored hash
    saler_data = cur.fetchall()

    cur.close()

    ########
    checked_books = session['checked_books']
    booksList = []
    for i in checked_books:
        cur = mysql.connection.cursor()

        # Get user by username
        cur.execute("SELECT * FROM books WHERE books.id=%s", [str(i)])

        # Get stored hash
        checked = cur.fetchall()
        booksList.append(checked)
        cur.close()
    return render_template('dashboard.html', posts=data, saler_posts=saler_data, checked=booksList)

    # return render_template('dashboard.html')


@app.route('/imageDetail/<string:image_name>')
def imageDetail(image_name):
    return render_template('imageDetail.html', image_name=image_name)

"""
# Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')
"""









if __name__ == '__main__':
    # app.secret_key = 'secret123'
    app.run(debug=True)
