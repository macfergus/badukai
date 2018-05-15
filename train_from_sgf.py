import argparse
import os

import badukai


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bot', required=True)
    parser.add_argument('--bot-out', required=True)
    parser.add_argument('index_file')
    parser.add_argument('checkpoint')
    args = parser.parse_args()

    with badukai.io.get_input(args.bot) as bot_file:
        bot = badukai.bots.load_bot(bot_file)

    corpus = badukai.corpora.open_index(open(args.index_file))
    if os.path.exists(args.checkpoint):
        chunk = badukai.corpora.load_checkpoint(open(args.checkpoint))
    else:
        chunk = corpus.start()

    while True:
        print('Loading chunk {}...'.format(chunk))
        game_records = corpus.get_game_records(chunk)
        print('Chunk {} has {} usable game records'.format(
            chunk, len(game_records)))
        bot.train_from_human(game_records, batch_size=512)

        with badukai.io.open_output_filename(args.bot_out) as output_filename:
            badukai.bots.save_bot(bot, output_filename)

        chunk = corpus.next(chunk)
        with open(args.checkpoint, 'w') as checkpoint_outf:
            chunk.save(checkpoint_outf)


if __name__ == '__main__':
    main()
