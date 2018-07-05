import argparse

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot-options', '-o')
    parser.add_argument('--max-threads', '-t', type=int)
    parser.add_argument('bot')
    args = parser.parse_args()

    if args.max_threads:
        badukai.kerasutil.set_tf_options(
            inter_op_parallelism_threads=args.max_threads,
            intra_op_parallelism_threads=args.max_threads)

    bot = badukai.bots.load_bot(args.bot)
    if args.bot_options:
        options = badukai.options.parse(args.bot_options)
        for k, v in options.items():
            bot.set_option(k, v)

    runner = badukai.gtp.Runner(badukai.gtp.BotHandler(bot))
    runner.run_forever()


if __name__ == '__main__':
    main()
