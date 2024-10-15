import Mathlib

namespace NUMINAMATH_CALUDE_elevator_problem_l1831_183109

theorem elevator_problem (initial_avg : ℝ) (new_avg : ℝ) (new_person_weight : ℝ) :
  initial_avg = 152 →
  new_avg = 151 →
  new_person_weight = 145 →
  ∃ n : ℕ, n > 0 ∧ 
    n * initial_avg + new_person_weight = (n + 1) * new_avg ∧
    n = 6 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l1831_183109


namespace NUMINAMATH_CALUDE_fifth_term_is_14_l1831_183168

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The fifth term of the arithmetic sequence equals 14 -/
theorem fifth_term_is_14 (seq : ArithmeticSequence) : seq.a 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_14_l1831_183168


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1831_183155

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1831_183155


namespace NUMINAMATH_CALUDE_correct_factorization_l1831_183125

theorem correct_factorization (x : ℝ) : 2 * x^2 + 3 * x + 1 = (2 * x + 1) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l1831_183125


namespace NUMINAMATH_CALUDE_sports_club_total_members_l1831_183187

/-- The number of members in a sports club -/
def sports_club_members (badminton tennis both neither : ℕ) : ℕ :=
  badminton + tennis - both + neither

/-- Theorem: The sports club has 42 members -/
theorem sports_club_total_members :
  sports_club_members 20 23 7 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_total_members_l1831_183187


namespace NUMINAMATH_CALUDE_segment_point_relation_l1831_183163

/-- Represents a line segment with a midpoint -/
structure Segment where
  length : ℝ
  midpoint : ℝ

/-- Represents a point on a segment -/
structure PointOnSegment where
  segment : Segment
  distanceFromMidpoint : ℝ

/-- The theorem statement -/
theorem segment_point_relation 
  (ab : Segment)
  (a_prime_b_prime : Segment)
  (p : PointOnSegment)
  (p_prime : PointOnSegment)
  (h1 : ab.length = 10)
  (h2 : a_prime_b_prime.length = 18)
  (h3 : p.segment = ab)
  (h4 : p_prime.segment = a_prime_b_prime)
  (h5 : 3 * p.distanceFromMidpoint - 2 * p_prime.distanceFromMidpoint = 6)
  : p.distanceFromMidpoint + p_prime.distanceFromMidpoint = 12 := by
  sorry

end NUMINAMATH_CALUDE_segment_point_relation_l1831_183163


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1831_183183

theorem sqrt_equation_solution (x : ℚ) : 
  (Real.sqrt (4 * x + 7) / Real.sqrt (8 * x + 10) = Real.sqrt 7 / 4) → x = -21/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1831_183183


namespace NUMINAMATH_CALUDE_probability_point_in_circle_l1831_183167

/-- The probability of a point randomly selected from a square with side length 4
    being within a circle of radius 2 centered at the origin is π/4. -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) :
  square_side = 4 →
  circle_radius = 2 →
  (π * circle_radius^2) / (square_side^2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_point_in_circle_l1831_183167


namespace NUMINAMATH_CALUDE_stock_price_change_l1831_183153

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let day1_price := initial_price * (1 - 0.15)
  let day2_price := day1_price * (1 + 0.25)
  (day2_price - initial_price) / initial_price = 0.0625 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1831_183153


namespace NUMINAMATH_CALUDE_remainder_7n_mod_5_l1831_183166

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 5 = 3) : (7 * n) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_5_l1831_183166


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1831_183150

theorem no_solutions_for_equation : ¬∃ (x y : ℕ+), x^12 = 26*y^3 + 2023 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1831_183150


namespace NUMINAMATH_CALUDE_log_8_x_equals_3_75_l1831_183131

theorem log_8_x_equals_3_75 (x : ℝ) :
  Real.log x / Real.log 8 = 3.75 → x = 1024 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_x_equals_3_75_l1831_183131


namespace NUMINAMATH_CALUDE_unique_score_100_l1831_183107

/-- Represents a competition score -/
structure CompetitionScore where
  total : Nat
  correct : Nat
  wrong : Nat
  score : Nat
  h1 : total = 25
  h2 : score = 25 + 5 * correct - wrong
  h3 : total = correct + wrong

/-- Checks if a given score uniquely determines the number of correct and wrong answers -/
def isUniquelyDetermined (s : Nat) : Prop :=
  ∃! cs : CompetitionScore, cs.score = s

theorem unique_score_100 :
  isUniquelyDetermined 100 ∧
  ∀ s, 95 < s ∧ s < 100 → ¬isUniquelyDetermined s :=
sorry

end NUMINAMATH_CALUDE_unique_score_100_l1831_183107


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l1831_183171

theorem complex_subtraction_simplification :
  (-5 + 3 * Complex.I) - (2 - 7 * Complex.I) = -7 + 10 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l1831_183171


namespace NUMINAMATH_CALUDE_unselected_probability_l1831_183186

/-- The type representing a selection of five consecutive integers from a circle of 10 numbers -/
def Selection := Fin 10

/-- The type representing the choices of four people -/
def Choices := Fin 4 → Selection

/-- The probability that there exists a number not selected by any of the four people -/
def probability_unselected (choices : Choices) : ℚ :=
  sorry

/-- The main theorem stating the probability of an unselected number -/
theorem unselected_probability :
  ∃ (p : ℚ), (∀ (choices : Choices), probability_unselected choices = p) ∧ 10000 * p = 3690 :=
sorry

end NUMINAMATH_CALUDE_unselected_probability_l1831_183186


namespace NUMINAMATH_CALUDE_nick_babysitting_charge_l1831_183147

/-- Nick's babysitting charge calculation -/
theorem nick_babysitting_charge (y : ℝ) : 
  let travel_cost : ℝ := 7
  let hourly_rate : ℝ := 10
  let total_charge := hourly_rate * y + travel_cost
  total_charge = 10 * y + 7 := by sorry

end NUMINAMATH_CALUDE_nick_babysitting_charge_l1831_183147


namespace NUMINAMATH_CALUDE_merchant_articles_count_l1831_183177

theorem merchant_articles_count (N : ℕ) (CP SP : ℝ) : 
  N > 0 → 
  CP > 0 →
  N * CP = 15 * SP → 
  SP = CP * (1 + 33.33 / 100) → 
  N = 20 := by
sorry

end NUMINAMATH_CALUDE_merchant_articles_count_l1831_183177


namespace NUMINAMATH_CALUDE_medal_award_ways_l1831_183115

def total_sprinters : ℕ := 8
def american_sprinters : ℕ := 3
def medals : ℕ := 3

def ways_to_award_medals (total : ℕ) (americans : ℕ) (medals : ℕ) : ℕ :=
  -- Number of ways to award medals with at most one American getting a medal
  sorry

theorem medal_award_ways :
  ways_to_award_medals total_sprinters american_sprinters medals = 240 :=
sorry

end NUMINAMATH_CALUDE_medal_award_ways_l1831_183115


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1831_183154

theorem greatest_two_digit_multiple_of_17 : ∃ (n : ℕ), n = 85 ∧ 
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ 17 ∣ m → m ≤ n) ∧ 
  10 ≤ n ∧ n ≤ 99 ∧ 17 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l1831_183154


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l1831_183142

theorem certain_amount_calculation (x A : ℝ) (h1 : x = 230) (h2 : 0.65 * x = 0.20 * A) : A = 747.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l1831_183142


namespace NUMINAMATH_CALUDE_sequence_properties_l1831_183140

/-- The sum of the first n terms of sequence a_n -/
def S (n : ℕ) : ℝ := sorry

/-- The nth term of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- The nth term of arithmetic sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- The sum of the first n terms of sequence b_n -/
def T (n : ℕ) : ℝ := sorry

theorem sequence_properties :
  (∀ n, 2 * S n = 3 * a n - 3) ∧
  (b 1 = a 1) ∧
  (b 7 = b 1 * b 2) ∧
  (∀ n m, b (n + m) - b n = m * (b 2 - b 1)) →
  (∀ n, a n = 3^n) ∧
  (∀ n, T n = n^2 + 2*n) := by
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1831_183140


namespace NUMINAMATH_CALUDE_dans_work_time_l1831_183169

/-- Dan's work rate in job completion per hour -/
def dans_rate : ℚ := 1 / 15

/-- Annie's work rate in job completion per hour -/
def annies_rate : ℚ := 1 / 10

/-- The time Annie works to complete the job after Dan stops -/
def annies_time : ℚ := 6

theorem dans_work_time (x : ℚ) : 
  x * dans_rate + annies_time * annies_rate = 1 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_dans_work_time_l1831_183169


namespace NUMINAMATH_CALUDE_angle_b_measure_l1831_183191

theorem angle_b_measure (A B C : ℝ) (h1 : A + B + C = 180) 
  (h2 : C = 3 * A) (h3 : B = 2 * A) : B = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_measure_l1831_183191


namespace NUMINAMATH_CALUDE_first_number_in_set_l1831_183122

theorem first_number_in_set (x : ℝ) : 
  let set1 := [10, 70, 28]
  let set2 := [x, 40, 60]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 4 →
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_first_number_in_set_l1831_183122


namespace NUMINAMATH_CALUDE_identify_counterfeit_coins_l1831_183182

/-- Represents a coin which can be either real or counterfeit -/
inductive Coin
| Real
| CounterfeitLight
| CounterfeitHeavy

/-- Represents the result of a weighing -/
inductive WeighingResult
| LeftHeavier
| RightHeavier
| Equal

/-- Represents a set of five coins -/
def CoinSet := Fin 5 → Coin

/-- Represents a weighing operation on the balance scale -/
def Weighing := List Nat → List Nat → WeighingResult

/-- The main theorem stating that it's possible to identify counterfeit coins in three weighings -/
theorem identify_counterfeit_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 5), i ≠ j ∧ coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) 
  (h2 : ∀ (i : Fin 5), coins i ≠ Coin.CounterfeitLight → coins i ≠ Coin.CounterfeitHeavy → coins i = Coin.Real) :
  ∃ (w1 w2 w3 : Weighing), 
    ∀ (i j : Fin 5), 
      (coins i = Coin.CounterfeitLight ∧ coins j = Coin.CounterfeitHeavy) → 
      ∃ (f : Weighing → Weighing → Weighing → Fin 5 × Fin 5), 
        f w1 w2 w3 = (i, j) :=
sorry

end NUMINAMATH_CALUDE_identify_counterfeit_coins_l1831_183182


namespace NUMINAMATH_CALUDE_function_composition_l1831_183152

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_composition (h : ∀ x, f (3*x + 2) = 9*x + 8) : 
  ∀ x, f x = 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l1831_183152


namespace NUMINAMATH_CALUDE_password_decryption_probability_l1831_183102

theorem password_decryption_probability :
  let p_A : ℚ := 1/5  -- Probability of A decrypting the password
  let p_B : ℚ := 1/4  -- Probability of B decrypting the password
  let p_either : ℚ := 1 - (1 - p_A) * (1 - p_B)  -- Probability of either A or B (or both) decrypting the password
  p_either = 2/5 := by sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l1831_183102


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1831_183114

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B : Set ℝ := {x | 2*x - 3 > 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = Set.Ici (-1) := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1831_183114


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l1831_183145

theorem largest_integer_inequality : 
  (∀ x : ℤ, x ≤ 3 → (x : ℚ) / 5 + 6 / 7 < 8 / 5) ∧ 
  (4 : ℚ) / 5 + 6 / 7 ≥ 8 / 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l1831_183145


namespace NUMINAMATH_CALUDE_triangle_area_l1831_183103

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (ha : a = 15) (hb : b = 36) (hc : c = 39) :
  (1 / 2 : ℝ) * a * b = 270 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1831_183103


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1831_183170

-- Define the hyperbola
def Hyperbola (a b : ℝ) := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}

-- Define the foci
def Foci (a b : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

-- Define a point on the hyperbola
def PointOnHyperbola (a b : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def Distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the eccentricity
def Eccentricity (a b : ℝ) : ℝ := sorry

theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (f1x, f1y, f2x, f2y) := Foci a b
  let p := PointOnHyperbola a b
  let d1 := Distance p (f1x, f1y)
  let d2 := Distance p (f2x, f2y)
  d1 = 3 * d2 →
  let e := Eccentricity a b
  1 < e ∧ e ≤ 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l1831_183170


namespace NUMINAMATH_CALUDE_product_equals_32_over_9_l1831_183108

/-- The repeating decimal 0.4444... --/
def repeating_four : ℚ := 4/9

/-- The product of the repeating decimal 0.4444... and 8 --/
def product : ℚ := repeating_four * 8

theorem product_equals_32_over_9 : product = 32/9 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_32_over_9_l1831_183108


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1831_183124

-- Define the number in billions
def number_in_billions : ℝ := 8.36

-- Define the scientific notation
def scientific_notation : ℝ := 8.36 * (10 ^ 9)

-- Theorem statement
theorem billion_to_scientific_notation :
  (number_in_billions * 10^9) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1831_183124


namespace NUMINAMATH_CALUDE_equation_proof_l1831_183165

theorem equation_proof : (8/3 + 3/2) / (15/4) - 0.4 = 32/45 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1831_183165


namespace NUMINAMATH_CALUDE_initial_pineapples_l1831_183141

/-- Proves that the initial number of pineapples in the store was 86 -/
theorem initial_pineapples (sold : ℕ) (rotten : ℕ) (fresh : ℕ) 
  (h1 : sold = 48) 
  (h2 : rotten = 9) 
  (h3 : fresh = 29) : 
  sold + rotten + fresh = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_pineapples_l1831_183141


namespace NUMINAMATH_CALUDE_exactly_two_referees_match_l1831_183121

-- Define the number of referees/seats
def n : ℕ := 5

-- Define the number of referees that should match their seat number
def k : ℕ := 2

-- Define the function to calculate the number of permutations
-- where exactly k out of n elements are in their original positions
def permutations_with_k_fixed (n k : ℕ) : ℕ :=
  (n.choose k) * 2

-- State the theorem
theorem exactly_two_referees_match : permutations_with_k_fixed n k = 20 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_referees_match_l1831_183121


namespace NUMINAMATH_CALUDE_infinite_pairs_exist_l1831_183106

theorem infinite_pairs_exist (m : ℕ+) :
  ∃ f : ℕ → ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ,
      let (x, y) := f n
      Nat.gcd x y = 1 ∧
      y ∣ (x^2 + m) ∧
      x ∣ (y^2 + m) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_pairs_exist_l1831_183106


namespace NUMINAMATH_CALUDE_flower_pots_total_cost_l1831_183101

/-- The number of flower pots -/
def num_pots : ℕ := 6

/-- The price difference between consecutive pots -/
def price_diff : ℚ := 1/10

/-- The price of the largest pot -/
def largest_pot_price : ℚ := 13/8

/-- The total cost of all flower pots -/
def total_cost : ℚ := 33/4

theorem flower_pots_total_cost :
  let prices := List.range num_pots |>.map (fun i => largest_pot_price - i * price_diff)
  prices.sum = total_cost := by sorry

end NUMINAMATH_CALUDE_flower_pots_total_cost_l1831_183101


namespace NUMINAMATH_CALUDE_cubic_difference_division_l1831_183196

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_division_l1831_183196


namespace NUMINAMATH_CALUDE_puppy_adoption_cost_puppy_adoption_cost_proof_l1831_183156

/-- The cost to get each puppy ready for adoption, given:
  * Cost for cats is $50 per cat
  * Cost for adult dogs is $100 per dog
  * 2 cats, 3 adult dogs, and 2 puppies were adopted
  * Total cost for all animals is $700
-/
theorem puppy_adoption_cost : ℝ :=
  let cat_cost : ℝ := 50
  let dog_cost : ℝ := 100
  let num_cats : ℕ := 2
  let num_dogs : ℕ := 3
  let num_puppies : ℕ := 2
  let total_cost : ℝ := 700
  150

theorem puppy_adoption_cost_proof (cat_cost dog_cost total_cost : ℝ) (num_cats num_dogs num_puppies : ℕ) 
  (h_cat_cost : cat_cost = 50)
  (h_dog_cost : dog_cost = 100)
  (h_num_cats : num_cats = 2)
  (h_num_dogs : num_dogs = 3)
  (h_num_puppies : num_puppies = 2)
  (h_total_cost : total_cost = 700)
  : puppy_adoption_cost = (total_cost - (↑num_cats * cat_cost + ↑num_dogs * dog_cost)) / ↑num_puppies :=
by sorry

end NUMINAMATH_CALUDE_puppy_adoption_cost_puppy_adoption_cost_proof_l1831_183156


namespace NUMINAMATH_CALUDE_league_games_l1831_183174

theorem league_games (n : ℕ) (h1 : n = 8) : (n.choose 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_league_games_l1831_183174


namespace NUMINAMATH_CALUDE_symmetry_sum_l1831_183129

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetry_sum (a b : ℝ) : 
  symmetric_wrt_origin (a, 2) (4, b) → a + b = -6 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l1831_183129


namespace NUMINAMATH_CALUDE_total_games_in_league_l1831_183134

theorem total_games_in_league (n : ℕ) (h : n = 35) : 
  (n * (n - 1)) / 2 = 595 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_league_l1831_183134


namespace NUMINAMATH_CALUDE_card_difference_l1831_183189

/-- Given a total of 500 cards divided in the ratio of 11:9, prove that the difference between the larger share and the smaller share is 50 cards. -/
theorem card_difference (total : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (h1 : total = 500) (h2 : ratio_a = 11) (h3 : ratio_b = 9) : 
  (total * ratio_a) / (ratio_a + ratio_b) - (total * ratio_b) / (ratio_a + ratio_b) = 50 := by
sorry

end NUMINAMATH_CALUDE_card_difference_l1831_183189


namespace NUMINAMATH_CALUDE_average_age_of_fourteen_students_l1831_183151

theorem average_age_of_fourteen_students
  (total_students : Nat)
  (total_average_age : ℚ)
  (ten_students_average : ℚ)
  (twenty_fifth_student_age : ℚ)
  (h1 : total_students = 25)
  (h2 : total_average_age = 25)
  (h3 : ten_students_average = 22)
  (h4 : twenty_fifth_student_age = 13) :
  (total_students * total_average_age - 10 * ten_students_average - twenty_fifth_student_age) / 14 = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_fourteen_students_l1831_183151


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l1831_183128

theorem arithmetic_square_root_of_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_81_l1831_183128


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1831_183117

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = (x^4 + x^2 + 3) * (2*x^5 + x^3 + 7)) ∧ 
    (f 0 = 21) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1831_183117


namespace NUMINAMATH_CALUDE_remainder_proof_l1831_183138

theorem remainder_proof (n : ℕ) (h1 : n = 88) (h2 : (3815 - 31) % n = 0) (h3 : ∃ r, (4521 - r) % n = 0) :
  4521 % n = 33 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l1831_183138


namespace NUMINAMATH_CALUDE_parabola_intersection_l1831_183104

/-- First parabola equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 5

/-- Second parabola equation -/
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1

/-- Theorem stating that (-0.5, 0.25) and (6, 49) are the only intersection points -/
theorem parabola_intersection :
  (∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = -0.5 ∧ y = 0.25) ∨ (x = 6 ∧ y = 49)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l1831_183104


namespace NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l1831_183185

-- Define the cubic polynomial whose roots are a, b, c
def cubic (x : ℝ) := x^3 + 4*x^2 + 6*x + 8

-- Define the properties of P
def P_properties (P : ℝ → ℝ) (a b c : ℝ) : Prop :=
  cubic a = 0 ∧ cubic b = 0 ∧ cubic c = 0 ∧
  P a = b + c ∧ P b = a + c ∧ P c = a + b ∧
  P (a + b + c) = -20

-- Define the specific polynomial we want to prove is equal to P
def target_poly (x : ℝ) := 2*x^3 + 7*x^2 + 11*x + 12

-- The main theorem
theorem cubic_polynomial_uniqueness :
  ∀ (P : ℝ → ℝ) (a b c : ℝ),
  P_properties P a b c →
  (∀ x, P x = target_poly x) :=
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_uniqueness_l1831_183185


namespace NUMINAMATH_CALUDE_hollow_block_length_l1831_183118

/-- Represents a hollow rectangular block made of small cubes -/
structure HollowBlock where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the number of small cubes used in a hollow rectangular block -/
def cubesUsed (block : HollowBlock) : ℕ :=
  2 * (block.length * block.width + block.width * block.depth + block.length * block.depth) -
  4 * (block.length + block.width + block.depth) + 8 -
  ((block.length - 2) * (block.width - 2) * (block.depth - 2))

/-- Theorem stating that a hollow block with given dimensions uses 114 cubes and has a length of 10 -/
theorem hollow_block_length :
  ∃ (block : HollowBlock), block.width = 9 ∧ block.depth = 5 ∧ cubesUsed block = 114 ∧ block.length = 10 :=
by sorry

end NUMINAMATH_CALUDE_hollow_block_length_l1831_183118


namespace NUMINAMATH_CALUDE_parallel_line_length_l1831_183158

/-- Given a triangle with base 20 inches and a parallel line dividing it into two parts
    where the upper part has 3/4 of the total area, the length of this parallel line is 10 inches. -/
theorem parallel_line_length (base : ℝ) (parallel_line : ℝ) : 
  base = 20 →
  (parallel_line / base) ^ 2 = 1 / 4 →
  parallel_line = 10 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l1831_183158


namespace NUMINAMATH_CALUDE_future_value_proof_l1831_183113

/-- Calculates the future value of an investment with compound interest. -/
def future_value (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that given the specified conditions, the future value is $3600. -/
theorem future_value_proof :
  let principal : ℝ := 2500
  let rate : ℝ := 0.20
  let time : ℕ := 2
  future_value principal rate time = 3600 := by
sorry

#eval future_value 2500 0.20 2

end NUMINAMATH_CALUDE_future_value_proof_l1831_183113


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1831_183110

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- Theorem: Minimum distance from a point on the parabola to the line y = x + 3 -/
theorem min_distance_parabola_to_line (para : Parabola) (P : Point) :
  para.p = 4 →  -- Derived from directrix x = -2
  P.y^2 = 2 * para.p * P.x →  -- Point P is on the parabola
  ∃ (d : ℝ), d = |P.x - P.y + 3| / Real.sqrt 2 ∧  -- Distance formula
  d ≥ Real.sqrt 2 / 2 ∧  -- Minimum distance
  (∃ (Q : Point), Q.y^2 = 2 * para.p * Q.x ∧  -- Another point on parabola
    |Q.x - Q.y + 3| / Real.sqrt 2 = Real.sqrt 2 / 2) :=  -- Achieving minimum distance
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1831_183110


namespace NUMINAMATH_CALUDE_metal_disc_weight_expectation_l1831_183178

/-- The nominal radius of a metal disc in meters -/
def nominal_radius : ℝ := 0.5

/-- The standard deviation of the radius in meters -/
def radius_std_dev : ℝ := 0.01

/-- The weight of a disc with exactly 1 m diameter in kilograms -/
def nominal_weight : ℝ := 100

/-- The number of discs in the stack -/
def num_discs : ℕ := 100

/-- The expected weight of the stack of discs in kilograms -/
def expected_stack_weight : ℝ := 10004

theorem metal_disc_weight_expectation :
  let expected_area := π * (nominal_radius^2 + radius_std_dev^2)
  let expected_single_weight := nominal_weight * expected_area / (π * nominal_radius^2)
  expected_single_weight * num_discs = expected_stack_weight :=
sorry

end NUMINAMATH_CALUDE_metal_disc_weight_expectation_l1831_183178


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1831_183162

-- Define the concept of opposite (additive inverse)
def opposite (a : ℤ) : ℤ := -a

-- State the theorem
theorem opposite_of_2023 : opposite 2023 = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1831_183162


namespace NUMINAMATH_CALUDE_max_round_value_l1831_183149

/-- Represents a digit assignment for the letter puzzle --/
structure DigitAssignment where
  H : Fin 10
  M : Fin 10
  T : Fin 10
  G : Fin 10
  U : Fin 10
  S : Fin 10
  R : Fin 10
  O : Fin 10
  N : Fin 10
  D : Fin 10

/-- Checks if all digits in the assignment are distinct --/
def allDistinct (a : DigitAssignment) : Prop :=
  a.H ≠ a.M ∧ a.H ≠ a.T ∧ a.H ≠ a.G ∧ a.H ≠ a.U ∧ a.H ≠ a.S ∧ a.H ≠ a.R ∧ a.H ≠ a.O ∧ a.H ≠ a.N ∧ a.H ≠ a.D ∧
  a.M ≠ a.T ∧ a.M ≠ a.G ∧ a.M ≠ a.U ∧ a.M ≠ a.S ∧ a.M ≠ a.R ∧ a.M ≠ a.O ∧ a.M ≠ a.N ∧ a.M ≠ a.D ∧
  a.T ≠ a.G ∧ a.T ≠ a.U ∧ a.T ≠ a.S ∧ a.T ≠ a.R ∧ a.T ≠ a.O ∧ a.T ≠ a.N ∧ a.T ≠ a.D ∧
  a.G ≠ a.U ∧ a.G ≠ a.S ∧ a.G ≠ a.R ∧ a.G ≠ a.O ∧ a.G ≠ a.N ∧ a.G ≠ a.D ∧
  a.U ≠ a.S ∧ a.U ≠ a.R ∧ a.U ≠ a.O ∧ a.U ≠ a.N ∧ a.U ≠ a.D ∧
  a.S ≠ a.R ∧ a.S ≠ a.O ∧ a.S ≠ a.N ∧ a.S ≠ a.D ∧
  a.R ≠ a.O ∧ a.R ≠ a.N ∧ a.R ≠ a.D ∧
  a.O ≠ a.N ∧ a.O ≠ a.D ∧
  a.N ≠ a.D

/-- Checks if the equation HMMT + GUTS = ROUND is satisfied --/
def equationSatisfied (a : DigitAssignment) : Prop :=
  1000 * a.H.val + 100 * a.M.val + 10 * a.M.val + a.T.val +
  1000 * a.G.val + 100 * a.U.val + 10 * a.T.val + a.S.val =
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- Checks if there are no leading zeroes --/
def noLeadingZeroes (a : DigitAssignment) : Prop :=
  a.H ≠ 0 ∧ a.G ≠ 0 ∧ a.R ≠ 0

/-- The value of ROUND for a given digit assignment --/
def roundValue (a : DigitAssignment) : ℕ :=
  10000 * a.R.val + 1000 * a.O.val + 100 * a.U.val + 10 * a.N.val + a.D.val

/-- The main theorem statement --/
theorem max_round_value :
  ∀ a : DigitAssignment,
    allDistinct a →
    equationSatisfied a →
    noLeadingZeroes a →
    roundValue a ≤ 16352 :=
sorry

end NUMINAMATH_CALUDE_max_round_value_l1831_183149


namespace NUMINAMATH_CALUDE_perimeter_of_specific_cut_pentagon_l1831_183199

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℝ

/-- Represents a pentagon formed by cutting a smaller equilateral triangle from a larger one -/
structure CutPentagon where
  originalTriangle : EquilateralTriangle
  cutTriangle : EquilateralTriangle

/-- Calculates the perimeter of the pentagon formed by cutting a smaller equilateral triangle
    from a corner of a larger equilateral triangle -/
def perimeterOfCutPentagon (p : CutPentagon) : ℝ :=
  p.originalTriangle.sideLength + p.originalTriangle.sideLength +
  (p.originalTriangle.sideLength - p.cutTriangle.sideLength) +
  p.cutTriangle.sideLength + p.cutTriangle.sideLength

/-- Theorem stating that the perimeter of the specific cut pentagon is 14 units -/
theorem perimeter_of_specific_cut_pentagon :
  let largeTriangle : EquilateralTriangle := { sideLength := 5 }
  let smallTriangle : EquilateralTriangle := { sideLength := 2 }
  let cutPentagon : CutPentagon := { originalTriangle := largeTriangle, cutTriangle := smallTriangle }
  perimeterOfCutPentagon cutPentagon = 14 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_specific_cut_pentagon_l1831_183199


namespace NUMINAMATH_CALUDE_one_more_green_than_red_peaches_l1831_183190

/-- Given a basket of peaches with specified quantities of red, yellow, and green peaches,
    prove that there is one more green peach than red peaches. -/
theorem one_more_green_than_red_peaches 
  (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_red : red = 7)
  (h_yellow : yellow = 71)
  (h_green : green = 8) :
  green - red = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_green_than_red_peaches_l1831_183190


namespace NUMINAMATH_CALUDE_sector_area_l1831_183188

theorem sector_area (α : Real) (perimeter : Real) (h1 : α = 1/3) (h2 : perimeter = 7) :
  let r := perimeter / (2 + α)
  (1/2) * α * r^2 = 3/2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l1831_183188


namespace NUMINAMATH_CALUDE_bookstore_shipment_calculation_bookstore_shipment_proof_l1831_183136

/-- Calculates the number of books received in a shipment given initial inventory, sales data, and final inventory. -/
theorem bookstore_shipment_calculation 
  (initial_inventory : ℕ) 
  (saturday_in_store : ℕ) 
  (saturday_online : ℕ) 
  (sunday_in_store_multiplier : ℕ) 
  (sunday_online_increase : ℕ) 
  (final_inventory : ℕ) : ℕ :=
  let total_saturday_sales := saturday_in_store + saturday_online
  let sunday_in_store := sunday_in_store_multiplier * saturday_in_store
  let sunday_online := saturday_online + sunday_online_increase
  let total_sunday_sales := sunday_in_store + sunday_online
  let total_sales := total_saturday_sales + total_sunday_sales
  let inventory_after_sales := initial_inventory - total_sales
  final_inventory - inventory_after_sales

/-- Proves that the bookstore received 160 books in the shipment. -/
theorem bookstore_shipment_proof : 
  bookstore_shipment_calculation 743 37 128 2 34 502 = 160 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_calculation_bookstore_shipment_proof_l1831_183136


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1831_183139

theorem inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, |3*x + 2*a| + |2 - 3*x| - |a + 1| > 2) ↔ (a < -1/3 ∨ a > 5) := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1831_183139


namespace NUMINAMATH_CALUDE_scientific_notation_of_error_l1831_183160

theorem scientific_notation_of_error : ∃ (a : ℝ) (n : ℤ), 
  0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_error_l1831_183160


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l1831_183127

theorem smallest_dual_base_representation : ∃ (a b : ℕ), 
  a > 3 ∧ b > 3 ∧ 
  13 = a + 3 ∧ 
  13 = 3 * b + 1 ∧
  (∀ (x y : ℕ), x > 3 → y > 3 → x + 3 = 3 * y + 1 → x + 3 ≥ 13) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l1831_183127


namespace NUMINAMATH_CALUDE_constant_d_value_l1831_183175

variables (a d : ℝ)

theorem constant_d_value (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d*x + 12) : d = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_d_value_l1831_183175


namespace NUMINAMATH_CALUDE_chord_length_implies_a_value_l1831_183112

theorem chord_length_implies_a_value (a : ℝ) :
  (∃ (x y : ℝ), (a * x + y + 1 = 0) ∧ (x^2 + y^2 - 2*a*x + a = 0)) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (a * x₁ + y₁ + 1 = 0) ∧ (x₁^2 + y₁^2 - 2*a*x₁ + a = 0) ∧
    (a * x₂ + y₂ + 1 = 0) ∧ (x₂^2 + y₂^2 - 2*a*x₂ + a = 0) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  a = -2 :=
sorry


end NUMINAMATH_CALUDE_chord_length_implies_a_value_l1831_183112


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1831_183180

theorem smaller_number_problem (x y : ℕ) : 
  x * y = 56 → x + y = 15 → x ≤ y → x ∣ 28 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1831_183180


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1831_183198

theorem inequality_system_solution_range (k : ℝ) : 
  (∀ x : ℤ, (x^2 - x - 2 > 0 ∧ 2*x^2 + (2*k+5)*x + 5*k < 0) ↔ x = -2) →
  -3 ≤ k ∧ k < 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1831_183198


namespace NUMINAMATH_CALUDE_input_is_only_input_statement_l1831_183148

-- Define the possible statement types
inductive StatementType
| Output
| Input
| Conditional
| Termination

-- Define the statements
def PRINT : StatementType := StatementType.Output
def INPUT : StatementType := StatementType.Input
def IF : StatementType := StatementType.Conditional
def END : StatementType := StatementType.Termination

-- Theorem: INPUT is the only input statement among the given options
theorem input_is_only_input_statement :
  (PRINT = StatementType.Input → False) ∧
  (INPUT = StatementType.Input) ∧
  (IF = StatementType.Input → False) ∧
  (END = StatementType.Input → False) :=
by sorry

end NUMINAMATH_CALUDE_input_is_only_input_statement_l1831_183148


namespace NUMINAMATH_CALUDE_sum_of_ages_l1831_183176

def father_age : ℕ := 48
def son_age : ℕ := 27

theorem sum_of_ages : father_age + son_age = 75 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1831_183176


namespace NUMINAMATH_CALUDE_set_cardinality_lower_bound_l1831_183144

theorem set_cardinality_lower_bound
  (m : ℕ)
  (A : Finset ℤ)
  (B : Fin m → Finset ℤ)
  (h_m : m ≥ 2)
  (h_subset : ∀ k : Fin m, B k ⊆ A)
  (h_sum : ∀ k : Fin m, (B k).sum id = m ^ (k : ℕ).succ) :
  A.card ≥ m / 2 :=
sorry

end NUMINAMATH_CALUDE_set_cardinality_lower_bound_l1831_183144


namespace NUMINAMATH_CALUDE_normal_distribution_symmetry_l1831_183172

/-- A random variable following a normal distribution with mean μ and standard deviation σ -/
structure NormalRV (μ σ : ℝ) where
  X : ℝ → ℝ  -- The random variable as a function

/-- The probability that a random variable X is greater than a given value -/
noncomputable def prob_gt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable X is less than a given value -/
noncomputable def prob_lt (X : ℝ → ℝ) (x : ℝ) : ℝ := sorry

/-- Theorem: For a normal distribution N(3,1), if P(X > 2c-1) = P(X < c+3), then c = 4/3 -/
theorem normal_distribution_symmetry (c : ℝ) (X : NormalRV 3 1) :
  prob_gt X.X (2*c - 1) = prob_lt X.X (c + 3) → c = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_symmetry_l1831_183172


namespace NUMINAMATH_CALUDE_number_with_given_division_properties_l1831_183126

theorem number_with_given_division_properties : ∃ n : ℕ, n / 9 = 80 ∧ n % 9 = 4 ∧ n = 724 := by
  sorry

end NUMINAMATH_CALUDE_number_with_given_division_properties_l1831_183126


namespace NUMINAMATH_CALUDE_min_subsets_to_guess_l1831_183161

/-- The set of possible choices for player A -/
def S : Set Nat := Finset.range 1001

/-- The condition that ensures B can always guess correctly -/
def can_guess (k₁ k₂ k₃ : Nat) : Prop :=
  (k₁ + 1) * (k₂ + 1) * (k₃ + 1) ≥ 1001

/-- The sum of subsets chosen by B -/
def total_subsets (k₁ k₂ k₃ : Nat) : Nat :=
  k₁ + k₂ + k₃

/-- The theorem stating that 28 is the minimum value -/
theorem min_subsets_to_guess :
  ∃ k₁ k₂ k₃ : Nat,
    can_guess k₁ k₂ k₃ ∧
    total_subsets k₁ k₂ k₃ = 28 ∧
    ∀ k₁' k₂' k₃' : Nat,
      can_guess k₁' k₂' k₃' →
      total_subsets k₁' k₂' k₃' ≥ 28 :=
sorry

end NUMINAMATH_CALUDE_min_subsets_to_guess_l1831_183161


namespace NUMINAMATH_CALUDE_product_one_to_five_l1831_183130

theorem product_one_to_five : (List.range 5).foldl (·*·) 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_one_to_five_l1831_183130


namespace NUMINAMATH_CALUDE_ultra_marathon_average_time_l1831_183197

/-- Calculates the average time per mile given the total distance and time -/
def averageTimePerMile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  let totalMinutes : ℕ := hours * 60 + minutes
  (totalMinutes : ℚ) / distance

theorem ultra_marathon_average_time :
  averageTimePerMile 32 4 52 = 9.125 := by
  sorry

end NUMINAMATH_CALUDE_ultra_marathon_average_time_l1831_183197


namespace NUMINAMATH_CALUDE_scalar_projection_implies_k_l1831_183119

/-- Given vectors a and b in ℝ², prove that if the scalar projection of b onto a is 1, then the first component of b is 3. -/
theorem scalar_projection_implies_k (a b : ℝ × ℝ) :
  a = (3, 4) →
  b.2 = -1 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = 1 →
  b.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_scalar_projection_implies_k_l1831_183119


namespace NUMINAMATH_CALUDE_store_paid_twenty_six_l1831_183135

/-- The price the store paid for a pair of pants, given the selling price and the difference between the selling price and the store's cost. -/
def store_paid_price (selling_price : ℕ) (price_difference : ℕ) : ℕ :=
  selling_price - price_difference

/-- Theorem stating that if the selling price is $34 and the store paid $8 less, then the store paid $26. -/
theorem store_paid_twenty_six :
  store_paid_price 34 8 = 26 := by
  sorry

end NUMINAMATH_CALUDE_store_paid_twenty_six_l1831_183135


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l1831_183192

/-- A line parallel to y = -3x + 6 passing through (3, -2) has y-intercept 7 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b y = -3 * x + b 0) →  -- b is a line with slope -3
  b (-2) = 3 * (-3) + b 0 →     -- b passes through (3, -2)
  b 0 = 7 :=                    -- y-intercept of b is 7
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l1831_183192


namespace NUMINAMATH_CALUDE_function_property_l1831_183143

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

/-- The main theorem to be proved -/
theorem function_property (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
by sorry

end NUMINAMATH_CALUDE_function_property_l1831_183143


namespace NUMINAMATH_CALUDE_min_area_rectangle_l1831_183132

theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 60) → 
  (l * w ≥ 29) :=
sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l1831_183132


namespace NUMINAMATH_CALUDE_novel_pages_prove_novel_pages_l1831_183179

theorem novel_pages : ℕ → Prop :=
  fun total_pages =>
    let day1_read := total_pages / 6 + 10
    let day1_remaining := total_pages - day1_read
    let day2_read := day1_remaining / 5 + 20
    let day2_remaining := day1_remaining - day2_read
    let day3_read := day2_remaining / 4 + 25
    let day3_remaining := day2_remaining - day3_read
    day3_remaining = 130 → total_pages = 352

theorem prove_novel_pages : novel_pages 352 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_prove_novel_pages_l1831_183179


namespace NUMINAMATH_CALUDE_triangle_max_area_l1831_183105

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is (3√3)/4 when b = √3 and (2a-c)cos B = √3 cos C -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  (2 * a - c) * Real.cos B = Real.sqrt 3 * Real.cos C →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧ 
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ S) →
  (3 * Real.sqrt 3) / 4 = (1/2) * b * c * Real.sin A :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1831_183105


namespace NUMINAMATH_CALUDE_divisor_problem_l1831_183137

theorem divisor_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 166 → quotient = 9 → remainder = 4 → 
  dividend = divisor * quotient + remainder →
  divisor = 18 := by sorry

end NUMINAMATH_CALUDE_divisor_problem_l1831_183137


namespace NUMINAMATH_CALUDE_go_and_chess_problem_l1831_183159

theorem go_and_chess_problem (x y z : ℝ) : 
  (3 * x + 5 * y = 98) →
  (8 * x + 3 * y = 158) →
  (z + (40 - z) = 40) →
  (16 * z + 10 * (40 - z) ≤ 550) →
  (x = 16 ∧ y = 10 ∧ z ≤ 25) := by
  sorry

end NUMINAMATH_CALUDE_go_and_chess_problem_l1831_183159


namespace NUMINAMATH_CALUDE_expected_value_of_sum_l1831_183116

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).sum (fun t => t.sum id)

def number_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).card

theorem expected_value_of_sum (s : Finset ℕ) :
  s = marbles →
  (sum_of_pairs s : ℚ) / (number_of_pairs s : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_l1831_183116


namespace NUMINAMATH_CALUDE_martha_collected_90_cans_l1831_183193

/-- The number of cans Martha collected -/
def martha_cans : ℕ := sorry

/-- The number of cans Diego collected -/
def diego_cans (m : ℕ) : ℕ := m / 2 + 10

/-- The total number of cans collected -/
def total_cans : ℕ := 145

theorem martha_collected_90_cans :
  martha_cans = 90 ∧ 
  diego_cans martha_cans = martha_cans / 2 + 10 ∧
  martha_cans + diego_cans martha_cans = total_cans :=
sorry

end NUMINAMATH_CALUDE_martha_collected_90_cans_l1831_183193


namespace NUMINAMATH_CALUDE_roots_transformation_l1831_183100

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + r₁ + 6 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + r₂ + 6 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + r₃ + 6 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 9*(3*r₁) + 162 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 9*(3*r₂) + 162 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 9*(3*r₃) + 162 = 0) := by
  sorry

end NUMINAMATH_CALUDE_roots_transformation_l1831_183100


namespace NUMINAMATH_CALUDE_train_speed_fraction_l1831_183184

theorem train_speed_fraction (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 2 → delay = 1/3 → 
  (2 : ℝ) / (2 + delay) = 6/7 := by sorry

end NUMINAMATH_CALUDE_train_speed_fraction_l1831_183184


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_9000_satisfies_conditions_l1831_183157

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_9 n ∧ n % 3 = 0) →
    n ≥ 9000 :=
by sorry

theorem n_9000_satisfies_conditions :
  is_terminating_decimal 9000 ∧ contains_digit_9 9000 ∧ 9000 % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_9000_satisfies_conditions_l1831_183157


namespace NUMINAMATH_CALUDE_company_j_salary_difference_l1831_183194

/-- Represents the company J with its payroll information -/
structure CompanyJ where
  factory_workers : ℕ
  office_workers : ℕ
  factory_payroll : ℕ
  office_payroll : ℕ

/-- Calculates the difference between average monthly salaries of office and factory workers -/
def salary_difference (company : CompanyJ) : ℚ :=
  (company.office_payroll / company.office_workers) - (company.factory_payroll / company.factory_workers)

/-- Theorem stating the salary difference in Company J -/
theorem company_j_salary_difference :
  ∃ (company : CompanyJ),
    company.factory_workers = 15 ∧
    company.office_workers = 30 ∧
    company.factory_payroll = 30000 ∧
    company.office_payroll = 75000 ∧
    salary_difference company = 500 := by
  sorry

end NUMINAMATH_CALUDE_company_j_salary_difference_l1831_183194


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1831_183164

theorem necessary_not_sufficient_condition (a b : ℝ) (h : a > b) :
  (∃ c : ℝ, c ≥ 0 ∧ ¬(a * c > b * c)) ∧
  (∀ c : ℝ, a * c > b * c → c ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1831_183164


namespace NUMINAMATH_CALUDE_cellphone_survey_rate_increase_is_30_percent_l1831_183133

/-- Calculates the percentage increase in pay rate for cellphone surveys -/
def cellphone_survey_rate_increase (regular_rate : ℚ) (total_surveys : ℕ) 
  (cellphone_surveys : ℕ) (total_earnings : ℚ) : ℚ :=
  let regular_earnings := regular_rate * total_surveys
  let additional_earnings := total_earnings - regular_earnings
  let additional_rate := additional_earnings / cellphone_surveys
  let cellphone_rate := regular_rate + additional_rate
  (cellphone_rate - regular_rate) / regular_rate * 100

/-- Theorem stating the percentage increase in pay rate for cellphone surveys -/
theorem cellphone_survey_rate_increase_is_30_percent :
  cellphone_survey_rate_increase 10 100 60 1180 = 30 := by
  sorry

end NUMINAMATH_CALUDE_cellphone_survey_rate_increase_is_30_percent_l1831_183133


namespace NUMINAMATH_CALUDE_sin_cos_tan_order_l1831_183146

theorem sin_cos_tan_order :
  ∃ (a b c : ℝ), a = Real.sin 2 ∧ b = Real.cos 2 ∧ c = Real.tan 2 ∧ c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_tan_order_l1831_183146


namespace NUMINAMATH_CALUDE_finite_solutions_factorial_cube_plus_eight_l1831_183173

theorem finite_solutions_factorial_cube_plus_eight :
  {p : ℕ × ℕ | (p.1.factorial = p.2^3 + 8)}.Finite := by
  sorry

end NUMINAMATH_CALUDE_finite_solutions_factorial_cube_plus_eight_l1831_183173


namespace NUMINAMATH_CALUDE_yujeong_drank_most_l1831_183123

/-- Represents the amount of water drunk by each person in liters -/
structure WaterConsumption where
  yujeong : ℚ
  eunji : ℚ
  yuna : ℚ

/-- Determines who drank the most water -/
def drankMost (consumption : WaterConsumption) : String :=
  if consumption.yujeong > consumption.eunji ∧ consumption.yujeong > consumption.yuna then
    "Yujeong"
  else if consumption.eunji > consumption.yujeong ∧ consumption.eunji > consumption.yuna then
    "Eunji"
  else
    "Yuna"

theorem yujeong_drank_most (consumption : WaterConsumption) 
  (h1 : consumption.yujeong = 7/10)
  (h2 : consumption.eunji = 1/2)
  (h3 : consumption.yuna = 6/10) :
  drankMost consumption = "Yujeong" :=
by
  sorry

#eval drankMost ⟨7/10, 1/2, 6/10⟩

end NUMINAMATH_CALUDE_yujeong_drank_most_l1831_183123


namespace NUMINAMATH_CALUDE_triangle_problem_l1831_183181

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C opposite to them respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 3)
  (h2 : t.b - t.c = 2)
  (h3 : Real.cos t.B = -1/2) :
  t.b = 7 ∧ t.c = 5 ∧ Real.sin (t.B - t.C) = (4 * Real.sqrt 3) / 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1831_183181


namespace NUMINAMATH_CALUDE_function_inequality_l1831_183111

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = f x) 
  (h2 : ∀ x ≥ 1, f x = Real.log x) : 
  f (1/2) < f 2 ∧ f 2 < f (1/3) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1831_183111


namespace NUMINAMATH_CALUDE_principal_square_root_nine_sixteenths_l1831_183195

theorem principal_square_root_nine_sixteenths (x : ℝ) : x = Real.sqrt (9 / 16) → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_principal_square_root_nine_sixteenths_l1831_183195


namespace NUMINAMATH_CALUDE_horner_method_v3_l1831_183120

def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

def horner_v3 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((a₆ * x + a₅) * x + a₄) * x + a₃

theorem horner_method_v3 :
  horner_v3 2 5 6 23 (-8) 10 (-3) 2 = 71 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l1831_183120
