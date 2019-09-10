import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

public class Jaccard {
	// http://bytes4u.blogspot.com/2013/03/jaccard-coeffecient-and-implementation.html
	public static double jaccard_coeffecient(String s1, String s2) {

		double j_coeffecient;
		ArrayList<String> j1 = new ArrayList<String>();
		ArrayList<String> j2 = new ArrayList<String>();
		HashSet<String> set1 = new HashSet<String>();
		HashSet<String> set2 = new HashSet<String>();

		s1 = "$" + s1 + "$";
		s2 = "$" + s2 + "$";
		int j = 0;
		int i = 3;

		while (i <= s1.length()) {
			j1.add(s1.substring(j, i));
			j++;
			i++;
		}
		j = 0;
		i = 3;
		while (i <= s2.length()) {
			j2.add(s2.substring(j, i));
			j++;
			i++;
		}

		Iterator<String> itr1 = j1.iterator();
		while (itr1.hasNext()) {
			String element = itr1.next();
			//System.out.print(element + " ");
		}
		//System.out.println();
		Iterator<String> itr2 = j2.iterator();
		while (itr2.hasNext()) {
			String element = itr2.next();
			//System.out.print(element + " ");
		}
		//System.out.println();

		set2.addAll(j2);
		set2.addAll(j1);
		set1.addAll(j1);
		set1.retainAll(j2);

		//System.out.println("Union=" + set2.size());
		//System.out.println("Intersection=" + set1.size());

		j_coeffecient = ((double) set1.size()) / ((double) set2.size());
		//System.out.println("Jaccard coeffecient=" + j_coeffecient);
		return j_coeffecient;
	}
}