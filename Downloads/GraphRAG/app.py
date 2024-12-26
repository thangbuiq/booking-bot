import chainlit as cl 


@cl.on_chat_start
async def  on_chat_start():
    await cl.Message(
        "Hello! I'm your hotel booking assistant. How can I help you today?"
        ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Find a hotel for my trip",
            message="Can you help me find a hotel for my upcoming trip? I need recommendations based on my budget and preferences.",
            icon="/public/hotel.svg",
        ),
        cl.Starter(
            label="Check prices for a room",
            message="I would like check prices . Can you guide me through the process?",
            icon="/public/booking.svg",
        )
        
    ]

@cl.on_message
async def on_message(message: cl.Message):
    response = f"Hello, you just sent: {message.content}!"
    await cl.Message(response).send()
    
@cl.on_stop
async def on_stop():
    await cl.Message("Thank you for using the hotel booking assistant."
                      "If you need help in the future, "
                      "feel free to reach out!"
                      ).send()
    
@cl.on_chat_end
async def on_chat_end():
    await cl.Message("Goodbye!").send()




    


    