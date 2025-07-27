
import org.apache.mahout.cf.taste.eval.DataModelBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class ProductRecommender {

    public static void main(String[] args) {
        try {
            // Load data from a sample CSV file (userId, itemId, preference)
            DataModel model = new FileDataModel(new File("data.csv"));

            // Compute similarity between users
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Define neighborhood of N similar users
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // Create recommender engine
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // Generate recommendations for all users
            for (LongPrimitiveIterator users = model.getUserIDs(); users.hasNext(); ) {
                long userId = users.nextLong();
                List<RecommendedItem> recommendations = recommender.recommend(userId, 3);

                System.out.println("\n📦 Recommendations for user " + userId + ":");
                for (RecommendedItem recommendation : recommendations) {
                    System.out.println("👉 Item ID: " + recommendation.getItemID() + " (Score: " + recommendation.getValue() + ")");
                }
            }

        } catch (Exception e) {
            System.out.println("❌ Error: " + e.getMessage());
        }
    }
}
