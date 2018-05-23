import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-options', '-o')
    parser.add_argument('bot')
    args = parser.parse_args()

    bot = badukai.bots.load_bot(args.bot)
    if args.bot_options:
        options = badukai.options.parse(args.bot_options)
        for k, v in options.items():
            bot.set_option(k, v)

    runner = badukai.gtp.BotRunner(bot)
    runner.run_forever()


if __name__ == '__main__':
    main()
