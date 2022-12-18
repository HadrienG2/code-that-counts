# Code that counts

With a title like this, you might expect this little book to be a philosophical
or sociological exploration of how to write programs that are meaningful to
yourself, others, or humanity as a whole.

However, while that would certainly be an interesting topic to write about, and
I encourage you to do so and send me back a link if you think you can give it a
good shot, that's actually not the kind of subject that I personally feel
qualified or compelled to cover. You see, I work in academia, and it is a
well-known fact that in academia, our main area of expertise is writing
meaningless code.

So instead, here we're going to explore CPU performance micro-optimization
through the meaningless exercise of counting from zero to a certain limit N as
quickly as possible.

I'm going to go step by step from what would be my first "intuitive" answer to
the problem, to what I believe to be the optimal answer on current-generation
x86 hardware (as of 2022), and hopefully you'll learn an interesting thing or
two along the way.
