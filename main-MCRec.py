from src.MCRec import train
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='music', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
    # parser.add_argument('--predict_factor_dim', type=int, default=32, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='book', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
    # parser.add_argument('--predict_factor_dim', type=int, default=32, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    # parser.add_argument('--dataset', type=str, default='ml', help='dataset')
    # parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    # parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    # parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    # parser.add_argument('--epochs', type=int, default=20, help='epochs')
    # parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
    # parser.add_argument('--predict_factor_dim', type=int, default=32, help='embedding size')
    # parser.add_argument('--p', type=int, default=5, help='the number of paths')
    # parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    parser.add_argument('--dataset', type=str, default='yelp', help='dataset')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--l2', type=float, default=1e-4, help='L2')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--embedding_dim', type=int, default=128, help='embedding size')
    parser.add_argument('--predict_factor_dim', type=int, default=32, help='embedding size')
    parser.add_argument('--p', type=int, default=5, help='the number of paths')
    parser.add_argument('--ratio', type=float, default=1, help='The proportion of training set used')

    args = parser.parse_args()

    train(args, True)

'''
music	train_auc: 0.904 	 train_acc: 0.827 	 eval_auc: 0.827 	 eval_acc: 0.765 	 test_auc: 0.820 	 test_acc: 0.759 		[0.13, 0.28, 0.4, 0.47, 0.47, 0.51, 0.54, 0.55]
book	train_auc: 0.855 	 train_acc: 0.784 	 eval_auc: 0.744 	 eval_acc: 0.692 	 test_auc: 0.742 	 test_acc: 0.689 		[0.12, 0.16, 0.31, 0.33, 0.33, 0.38, 0.39, 0.42]
ml	train_auc: 0.928 	 train_acc: 0.849 	 eval_auc: 0.899 	 eval_acc: 0.820 	 test_auc: 0.901 	 test_acc: 0.822 		[0.21, 0.3, 0.43, 0.51, 0.51, 0.59, 0.62, 0.64]
yelp	train_auc: 0.914 	 train_acc: 0.835 	 eval_auc: 0.868 	 eval_acc: 0.795 	 test_auc: 0.867 	 test_acc: 0.792 		[0.16, 0.21, 0.4, 0.42, 0.42, 0.5, 0.53, 0.56]
'''