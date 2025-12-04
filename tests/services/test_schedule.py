import schedule


def test_schedule():
    schedule.every().day.at("13:16").do(print, "Hello, world!")

if __name__ == "__main__":
    test_schedule()
