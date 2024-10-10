import Mathlib

namespace minimum_chocolates_l4121_412131

theorem minimum_chocolates (n : ℕ) : n ≥ 118 →
  (n % 6 = 4 ∧ n % 8 = 6 ∧ n % 10 = 8) →
  ∃ (m : ℕ), m < n → ¬(m % 6 = 4 ∧ m % 8 = 6 ∧ m % 10 = 8) :=
by sorry

end minimum_chocolates_l4121_412131


namespace smallest_k_for_inequality_l4121_412181

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 11 ∧ 
  (∀ m : ℕ, m < k → (128 : ℝ)^m ≤ 8^25 + 1000) ∧
  (128 : ℝ)^k > 8^25 + 1000 := by
  sorry

end smallest_k_for_inequality_l4121_412181


namespace lukas_average_points_l4121_412163

/-- Lukas's average points per game in basketball -/
def average_points (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: Lukas averages 12 points per game -/
theorem lukas_average_points :
  average_points 60 5 = 12 := by
  sorry

end lukas_average_points_l4121_412163


namespace y_over_z_equals_negative_five_l4121_412151

theorem y_over_z_equals_negative_five (x y z : ℝ) 
  (eq1 : x + y = 2 * x + z)
  (eq2 : x - 2 * y = 4 * z)
  (eq3 : x + y + z = 21) :
  y / z = -5 := by
sorry

end y_over_z_equals_negative_five_l4121_412151


namespace train_length_calculation_l4121_412133

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length. -/
theorem train_length_calculation (train_speed : ℝ) (time_to_pass : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) → 
  time_to_pass = 44 →
  bridge_length = 140 →
  train_speed * time_to_pass - bridge_length = 410 :=
by
  sorry

#check train_length_calculation

end train_length_calculation_l4121_412133


namespace geometric_sequence_sum_constant_l4121_412137

/-- Given a geometric sequence {aₙ} where the sum of the first n terms
    is Sₙ = 3ⁿ + r, prove that r = -1 -/
theorem geometric_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ n : ℕ, S n = 3^n + r) →
  (∀ n : ℕ, n ≥ 2 → a n = S n - S (n-1)) →
  (a 1 = S 1) →
  (∀ n : ℕ, n ≥ 2 → a (n+1) = 3 * a n) →
  r = -1 := by
  sorry

end geometric_sequence_sum_constant_l4121_412137


namespace polygon_sides_l4121_412171

theorem polygon_sides (n : ℕ) (n_pos : n > 0) :
  (((n - 2) * 180) / n = 108) → n = 5 := by
  sorry

end polygon_sides_l4121_412171


namespace geometric_mean_of_4_and_9_l4121_412123

theorem geometric_mean_of_4_and_9 :
  ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := by
  sorry

end geometric_mean_of_4_and_9_l4121_412123


namespace prob_dime_is_25_143_l4121_412136

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 900
  | Coin.Dime => 500
  | Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Dime + coin_count Coin.Penny

/-- The probability of picking a dime from the jar -/
def prob_dime : ℚ := coin_count Coin.Dime / total_coins

theorem prob_dime_is_25_143 : prob_dime = 25 / 143 := by
  sorry


end prob_dime_is_25_143_l4121_412136


namespace largest_palindrome_multiple_of_6_l4121_412165

def is_palindrome (n : ℕ) : Prop :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 = n % 10)

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_palindrome_multiple_of_6 :
  ∀ n : ℕ, is_palindrome n → n % 6 = 0 → n ≤ 888 ∧
  (∃ m : ℕ, is_palindrome m ∧ m % 6 = 0 ∧ m = 888) ∧
  sum_of_digits 888 = 24 :=
sorry

end largest_palindrome_multiple_of_6_l4121_412165


namespace circle_equation_correct_l4121_412182

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-2, 3)
def radius : ℝ := 2

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x + 2)^2 + (y - 3)^2 = 4

-- Theorem stating that the given equation represents the circle with the specified center and radius
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_equation_correct_l4121_412182


namespace triangle_properties_l4121_412108

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.A + Real.sin t.B = 5/4 * Real.sin t.C)
  (h2 : t.a + t.b + t.c = 9)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sin t.C) :
  t.C = 4 ∧ Real.cos t.C = -1/4 := by
  sorry

end triangle_properties_l4121_412108


namespace square_plus_integer_equality_find_integer_l4121_412149

theorem square_plus_integer_equality (y : ℝ) : ∃ k : ℤ, y^2 + 12*y + 40 = (y + 6)^2 + k := by
  sorry

theorem find_integer : ∃ k : ℤ, ∀ y : ℝ, y^2 + 12*y + 40 = (y + 6)^2 + k ∧ k = 4 := by
  sorry

end square_plus_integer_equality_find_integer_l4121_412149


namespace definite_integral_equality_l4121_412128

theorem definite_integral_equality : ∫ x in (1 : ℝ)..3, (2 * x - 1 / x^2) = 22 / 3 := by sorry

end definite_integral_equality_l4121_412128


namespace exponent_division_l4121_412121

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^4 / a^3 = a := by
  sorry

end exponent_division_l4121_412121


namespace plan_b_rate_l4121_412142

/-- Represents the cost of a call under Plan A -/
def costPlanA (minutes : ℕ) : ℚ :=
  if minutes ≤ 6 then 60/100
  else 60/100 + (minutes - 6) * (6/100)

/-- Represents the cost of a call under Plan B -/
def costPlanB (rate : ℚ) (minutes : ℕ) : ℚ :=
  rate * minutes

/-- The duration at which both plans charge the same amount -/
def equalDuration : ℕ := 12

theorem plan_b_rate : ∃ (rate : ℚ), 
  costPlanA equalDuration = costPlanB rate equalDuration ∧ rate = 8/100 := by
  sorry

end plan_b_rate_l4121_412142


namespace prob_heads_11th_toss_l4121_412147

/-- A fair coin is a coin with equal probability of heads and tails -/
def fair_coin (p : ℝ) : Prop := p = 1/2

/-- The probability of getting heads on a single toss of a fair coin -/
def prob_heads (p : ℝ) : ℝ := p

/-- The number of tosses -/
def num_tosses : ℕ := 10

/-- The number of heads observed -/
def heads_observed : ℕ := 7

/-- Theorem: The probability of getting heads on the 11th toss of a fair coin is 0.5,
    given that the coin was tossed 10 times with 7 heads as the result -/
theorem prob_heads_11th_toss (p : ℝ) (h : fair_coin p) :
  prob_heads p = 1/2 :=
by sorry

end prob_heads_11th_toss_l4121_412147


namespace f_composition_range_l4121_412164

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 else 2^x

theorem f_composition_range : 
  {a : ℝ | f (f a) = 2^(f a)} = {a : ℝ | a ≥ 2/3} := by sorry

end f_composition_range_l4121_412164


namespace max_homes_first_neighborhood_l4121_412112

def revenue_first (n : ℕ) : ℕ := 4 * n

def revenue_second : ℕ := 50

theorem max_homes_first_neighborhood :
  ∀ n : ℕ, revenue_first n ≤ revenue_second → n ≤ 12 :=
by
  sorry

end max_homes_first_neighborhood_l4121_412112


namespace batsman_average_l4121_412196

/-- 
Given a batsman who has played 16 innings, prove that if he scores 87 runs 
in the 17th inning and this increases his average by 4 runs, 
then his new average after the 17th inning is 23 runs.
-/
theorem batsman_average (prev_average : ℝ) : 
  (16 * prev_average + 87) / 17 = prev_average + 4 → 
  prev_average + 4 = 23 := by sorry

end batsman_average_l4121_412196


namespace comics_after_reassembly_l4121_412191

/-- The number of comics in the box after reassembly -/
def total_comics (pages_per_comic : ℕ) (extra_pages : ℕ) (total_pages : ℕ) (untorn_comics : ℕ) : ℕ :=
  untorn_comics + (total_pages - extra_pages) / pages_per_comic

/-- Theorem stating the total number of comics after reassembly -/
theorem comics_after_reassembly :
  total_comics 47 3 3256 20 = 89 := by
  sorry

end comics_after_reassembly_l4121_412191


namespace min_third_side_length_l4121_412175

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- Checks if the given sides satisfy the triangle inequality -/
def satisfies_triangle_inequality (a b c : ℕ+) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem: The minimum length of the third side in a triangle with two sides
    being multiples of 42 and 72 respectively is 7 -/
theorem min_third_side_length (t : Triangle) 
    (h1 : ∃ (k : ℕ+), t.a = 42 * k ∨ t.b = 42 * k ∨ t.c = 42 * k)
    (h2 : ∃ (m : ℕ+), t.a = 72 * m ∨ t.b = 72 * m ∨ t.c = 72 * m)
    (h3 : satisfies_triangle_inequality t.a t.b t.c) :
    min t.a (min t.b t.c) ≥ 7 := by
  sorry

end min_third_side_length_l4121_412175


namespace apple_distribution_l4121_412122

/-- Represents the number of apples each person has -/
structure Apples where
  greg : ℕ
  sarah : ℕ
  susan : ℕ
  mark : ℕ

/-- The ratio of Susan's apples to Greg's apples -/
def apple_ratio (a : Apples) : ℚ :=
  a.susan / a.greg

theorem apple_distribution (a : Apples) :
  a.greg = a.sarah ∧
  a.greg + a.sarah = 18 ∧
  a.mark = a.susan - 5 ∧
  a.greg + a.sarah + a.susan + a.mark = 49 →
  apple_ratio a = 2 := by
sorry

end apple_distribution_l4121_412122


namespace intersection_nonempty_implies_a_value_l4121_412180

theorem intersection_nonempty_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  P ∩ Q ≠ ∅ →
  a = 1 ∨ a = 2 := by
sorry

end intersection_nonempty_implies_a_value_l4121_412180


namespace jessie_weight_loss_l4121_412183

/-- Calculates the final weight after a two-week diet plan -/
def final_weight (initial_weight : ℝ) (first_week_loss : ℝ) (second_week_rate : ℝ) : ℝ :=
  initial_weight - (first_week_loss + second_week_rate * first_week_loss)

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 92
  let first_week_loss : ℝ := 5
  let second_week_rate : ℝ := 1.3
  final_weight initial_weight first_week_loss second_week_rate = 80.5 := by
  sorry

#eval final_weight 92 5 1.3

end jessie_weight_loss_l4121_412183


namespace quadratic_root_relation_l4121_412145

theorem quadratic_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c := by
sorry

end quadratic_root_relation_l4121_412145


namespace symmetry_of_even_functions_l4121_412127

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define symmetry about a point
def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

theorem symmetry_of_even_functions :
  (∀ f : ℝ → ℝ, IsEven f → IsSymmetricAbout (fun x ↦ f (x + 2)) (-2)) ∧
  (∀ f : ℝ → ℝ, IsEven (fun x ↦ f (x + 2)) → IsSymmetricAbout f 2) := by
  sorry


end symmetry_of_even_functions_l4121_412127


namespace choir_average_age_l4121_412169

theorem choir_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12)
  (h2 : num_males = 13)
  (h3 : avg_age_females = 32)
  (h4 : avg_age_males = 33)
  (h5 : num_females + num_males = 25) :
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  let total_members := num_females + num_males
  total_age / total_members = 32.52 := by
sorry

end choir_average_age_l4121_412169


namespace complex_number_in_second_quadrant_l4121_412166

theorem complex_number_in_second_quadrant :
  let z : ℂ := Complex.I / (1 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_number_in_second_quadrant_l4121_412166


namespace max_true_statements_four_true_statements_possible_l4121_412118

theorem max_true_statements (a b : ℝ) : 
  ¬(1/a < 1/b ∧ a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0) :=
by sorry

theorem four_true_statements_possible (a b : ℝ) : 
  ∃ (a b : ℝ), a^3 < b^3 ∧ a < b ∧ a < 0 ∧ b < 0 :=
by sorry

end max_true_statements_four_true_statements_possible_l4121_412118


namespace weight_after_jogging_first_week_l4121_412129

/-- Calculates the weight after one week of jogging given the initial weight and weight loss. -/
def weight_after_one_week (initial_weight weight_loss : ℕ) : ℕ :=
  initial_weight - weight_loss

/-- Proves that given an initial weight of 92 kg and a weight loss of 56 kg in the first week,
    the weight after the first week is equal to 36 kg. -/
theorem weight_after_jogging_first_week :
  weight_after_one_week 92 56 = 36 := by
  sorry

#eval weight_after_one_week 92 56

end weight_after_jogging_first_week_l4121_412129


namespace evaluate_expression_l4121_412100

theorem evaluate_expression : ((-2)^3)^(1/3) - (-1)^0 = -3 := by
  sorry

end evaluate_expression_l4121_412100


namespace sequence_a_l4121_412152

theorem sequence_a (a : ℕ → ℕ) (h : ∀ n, a (n + 1) = a n + n) :
  a 0 = 19 → a 1 = 20 ∧ a 2 = 22 := by sorry

end sequence_a_l4121_412152


namespace weight_difference_proof_l4121_412177

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is 7 kg, given the conditions of the problem. -/
theorem weight_difference_proof
  (initial_avg : ℝ)
  (joe_weight : ℝ)
  (new_avg : ℝ)
  (final_avg : ℝ)
  (h_initial_avg : initial_avg = 30)
  (h_joe_weight : joe_weight = 44)
  (h_new_avg : new_avg = initial_avg + 1)
  (h_final_avg : final_avg = initial_avg)
  : ∃ (n : ℕ) (departing_avg : ℝ),
    (n : ℝ) * initial_avg + joe_weight = (n + 1 : ℝ) * new_avg ∧
    (n + 1 : ℝ) * new_avg - departing_avg * 2 = (n - 1 : ℝ) * final_avg ∧
    joe_weight - departing_avg = 7 :=
by sorry

end weight_difference_proof_l4121_412177


namespace loss_percentage_calculation_l4121_412168

-- Define the cost price and selling price
def cost_price : ℚ := 1500
def selling_price : ℚ := 1200

-- Define the loss percentage calculation
def loss_percentage (cp sp : ℚ) : ℚ := (cp - sp) / cp * 100

-- Theorem statement
theorem loss_percentage_calculation :
  loss_percentage cost_price selling_price = 20 := by
  sorry

end loss_percentage_calculation_l4121_412168


namespace triangle_value_l4121_412150

theorem triangle_value (triangle q p : ℤ) 
  (eq1 : triangle + q = 73)
  (eq2 : 2 * (triangle + q) + p = 172)
  (eq3 : p = 26) : 
  triangle = 12 := by
sorry

end triangle_value_l4121_412150


namespace birds_nest_building_distance_l4121_412114

/-- Calculates the total distance covered by birds making round trips to collect nest materials. -/
def total_distance_covered (num_birds : ℕ) (num_trips : ℕ) (distance_to_materials : ℕ) : ℕ :=
  num_birds * num_trips * (2 * distance_to_materials)

/-- Theorem stating that two birds making 10 round trips each to collect materials 200 miles away cover a total distance of 8000 miles. -/
theorem birds_nest_building_distance :
  total_distance_covered 2 10 200 = 8000 := by
  sorry

end birds_nest_building_distance_l4121_412114


namespace z_in_first_quadrant_l4121_412197

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for z
def z_condition (z : ℂ) : Prop := z * (1 + i) = 2 * i + 1

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : z_condition z) : 
  Complex.re z > 0 ∧ Complex.im z > 0 := by
  sorry

end z_in_first_quadrant_l4121_412197


namespace inequality_counterexample_l4121_412179

theorem inequality_counterexample : 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b ≥ 2 * Real.sqrt (a * b)) → 
  ¬(∀ x : ℝ, x + 1/x ≥ 2) :=
by sorry

end inequality_counterexample_l4121_412179


namespace jam_distribution_l4121_412104

/-- The jam distribution problem -/
theorem jam_distribution (total_jam : ℝ) (ponchik_hypothetical_days : ℝ) (syrupchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syrupchik : syrupchik_hypothetical_days = 20) :
  ∃ (ponchik_jam syrupchik_jam ponchik_rate syrupchik_rate : ℝ),
    ponchik_jam + syrupchik_jam = total_jam ∧
    ponchik_jam = 40 ∧
    syrupchik_jam = 60 ∧
    ponchik_rate = 4/3 ∧
    syrupchik_rate = 2 ∧
    ponchik_jam / ponchik_rate = syrupchik_jam / syrupchik_rate ∧
    syrupchik_jam / ponchik_hypothetical_days = ponchik_rate ∧
    ponchik_jam / syrupchik_hypothetical_days = syrupchik_rate :=
by sorry

end jam_distribution_l4121_412104


namespace sine_function_vertical_shift_l4121_412193

/-- Given a sine function y = a * sin(b * x + c) + d that oscillates between 4 and -2,
    prove that the vertical shift d equals 1. -/
theorem sine_function_vertical_shift
  (a b c d : ℝ)
  (positive_constants : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (oscillation : ∀ x : ℝ, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 := by
  sorry

end sine_function_vertical_shift_l4121_412193


namespace telephone_number_increase_l4121_412157

/-- The number of possible n-digit telephone numbers with a non-zero first digit -/
def telephone_numbers (n : ℕ) : ℕ := 9 * 10^(n - 1)

/-- The increase in telephone numbers when moving from 6 to 7 digits -/
def increase_in_numbers : ℕ := telephone_numbers 7 - telephone_numbers 6

theorem telephone_number_increase :
  increase_in_numbers = 81 * 10^5 := by
  sorry

end telephone_number_increase_l4121_412157


namespace complex_product_magnitude_l4121_412138

theorem complex_product_magnitude : 
  Complex.abs ((3 - 4 * Complex.I) * (5 + 12 * Complex.I) * (2 - 7 * Complex.I)) = 65 * Real.sqrt 53 := by
  sorry

end complex_product_magnitude_l4121_412138


namespace count_two_digit_numbers_unit_gte_tens_is_45_l4121_412194

/-- The count of two-digit numbers where the unit digit is not less than the tens digit -/
def count_two_digit_numbers_unit_gte_tens : ℕ := 45

/-- Proof that the count of two-digit numbers where the unit digit is not less than the tens digit is 45 -/
theorem count_two_digit_numbers_unit_gte_tens_is_45 :
  count_two_digit_numbers_unit_gte_tens = 45 := by
  sorry

end count_two_digit_numbers_unit_gte_tens_is_45_l4121_412194


namespace dice_sum_probability_l4121_412117

-- Define the number of dice
def num_dice : ℕ := 8

-- Define the target sum
def target_sum : ℕ := 11

-- Define the function to calculate the number of ways to achieve the target sum
def num_ways_to_achieve_sum (n d s : ℕ) : ℕ :=
  Nat.choose (s - n + d - 1) (d - 1)

-- Theorem statement
theorem dice_sum_probability :
  num_ways_to_achieve_sum num_dice num_dice target_sum = 120 := by
  sorry

end dice_sum_probability_l4121_412117


namespace min_value_expression_min_value_attained_l4121_412184

theorem min_value_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x - 2*y + 3*z = 0) :
  y^2 / (x*z) ≥ 3 := by
  sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x - 2*y + 3*z = 0 ∧ y^2 / (x*z) < 3 + ε := by
  sorry

end min_value_expression_min_value_attained_l4121_412184


namespace ice_cube_distribution_l4121_412176

theorem ice_cube_distribution (total_ice_cubes : ℕ) (ice_cubes_per_cup : ℕ) (h1 : total_ice_cubes = 30) (h2 : ice_cubes_per_cup = 5) :
  total_ice_cubes / ice_cubes_per_cup = 6 := by
  sorry

end ice_cube_distribution_l4121_412176


namespace letter_lock_max_letters_l4121_412174

theorem letter_lock_max_letters (n : ℕ) : 
  (n ^ 3 - 1 ≤ 215) ∧ (∀ m : ℕ, m > n → m ^ 3 - 1 > 215) → n = 6 := by
  sorry

end letter_lock_max_letters_l4121_412174


namespace vector_sum_l4121_412102

-- Define the vectors
def a : ℝ × ℝ := (-1, 2)
def b : ℝ → ℝ × ℝ := λ x ↦ (2, x)
def c : ℝ → ℝ × ℝ := λ m ↦ (m, -3)

-- Define the parallel and perpendicular conditions
def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1
def perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

-- State the theorem
theorem vector_sum (x m : ℝ) 
  (h1 : parallel a (b x)) 
  (h2 : perpendicular (b x) (c m)) : 
  x + m = -10 := by sorry

end vector_sum_l4121_412102


namespace reading_pattern_l4121_412186

theorem reading_pattern (x y : ℝ) : 
  (∀ (days_xiaoming days_xiaoying : ℕ), 
    days_xiaoming = 3 ∧ days_xiaoying = 5 → 
    days_xiaoming * x + 6 = days_xiaoying * y) ∧
  (y = x - 10) →
  3 * x = 5 * y - 6 ∧ y = 2 * x - 10 := by
sorry

end reading_pattern_l4121_412186


namespace inequality_proof_l4121_412134

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end inequality_proof_l4121_412134


namespace inequality_proof_l4121_412198

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (sum : a + b + c = Real.sqrt 2) :
  1 / Real.sqrt (1 + a^2) + 1 / Real.sqrt (1 + b^2) + 1 / Real.sqrt (1 + c^2) ≥ 2 + 1 / Real.sqrt 3 := by
  sorry

end inequality_proof_l4121_412198


namespace no_real_roots_l4121_412109

theorem no_real_roots : ¬∃ x : ℝ, |2*x - 5| + |3*x - 7| + |5*x - 11| = 2015/2016 := by
  sorry

end no_real_roots_l4121_412109


namespace part_one_part_two_l4121_412126

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x| ≥ 2}
def B (a : ℝ) : Set ℝ := {x | (x - 2*a)*(x + 3) < 0}

-- Part I
theorem part_one : 
  A 3 ∩ B 3 = {x | -3 < x ∧ x ≤ -2 ∨ 2 ≤ x ∧ x < 6} := by sorry

-- Part II
theorem part_two (a : ℝ) (h : a > 0) :
  A a ∪ B a = Set.univ → a ≥ 1 := by sorry

end part_one_part_two_l4121_412126


namespace shaded_area_percentage_l4121_412103

/-- Given two congruent squares with side length 20 that overlap to form a 20 by 30 rectangle,
    the percentage of the area of the rectangle that is shaded is 100/3%. -/
theorem shaded_area_percentage (square_side : ℝ) (rect_width rect_length : ℝ) : 
  square_side = 20 →
  rect_width = 20 →
  rect_length = 30 →
  (((2 * square_side - rect_length) * square_side) / (rect_width * rect_length)) * 100 = 100 / 3 := by
  sorry

end shaded_area_percentage_l4121_412103


namespace assignment_statement_properties_l4121_412161

-- Define what an assignment statement is
def AssignmentStatement : Type := Unit

-- Define the properties of assignment statements
def can_provide_initial_values (a : AssignmentStatement) : Prop := sorry
def assigns_expression_value (a : AssignmentStatement) : Prop := sorry
def can_assign_multiple_times (a : AssignmentStatement) : Prop := sorry

-- Theorem stating the properties of assignment statements
theorem assignment_statement_properties (a : AssignmentStatement) :
  can_provide_initial_values a ∧
  assigns_expression_value a ∧
  can_assign_multiple_times a := by sorry

end assignment_statement_properties_l4121_412161


namespace function_value_proof_l4121_412120

theorem function_value_proof (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x - 1)
  (h2 : f 2 = 3) :
  f 3 = 5 := by
sorry

end function_value_proof_l4121_412120


namespace line_through_first_third_quadrants_l4121_412125

/-- A line y = kx passes through the first and third quadrants if and only if k > 0 -/
theorem line_through_first_third_quadrants (k : ℝ) (h1 : k ≠ 0) :
  (∀ x y : ℝ, y = k * x → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))) ↔ k > 0 := by
  sorry

end line_through_first_third_quadrants_l4121_412125


namespace car_speed_change_l4121_412107

theorem car_speed_change (V : ℝ) (x : ℝ) (h_V : V > 0) (h_x : x > 0) : 
  V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100) → x = 20 := by
  sorry

end car_speed_change_l4121_412107


namespace log_problem_l4121_412144

theorem log_problem : Real.log (648 * Real.rpow 6 (1/3)) / Real.log (Real.rpow 6 (1/3)) = 11.5 := by
  sorry

end log_problem_l4121_412144


namespace angle_complement_supplement_l4121_412159

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = (1/3) * (180 - x) ↔ x = 45 := by sorry

end angle_complement_supplement_l4121_412159


namespace percentage_of_students_liking_donuts_l4121_412115

theorem percentage_of_students_liking_donuts : 
  ∀ (total_donuts : ℕ) (total_students : ℕ) (donuts_per_student : ℕ),
    total_donuts = 4 * 12 →
    total_students = 30 →
    donuts_per_student = 2 →
    (((total_donuts / donuts_per_student) / total_students) * 100 : ℚ) = 80 := by
  sorry

end percentage_of_students_liking_donuts_l4121_412115


namespace f_eight_eq_twelve_f_two_f_odd_l4121_412178

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is not identically zero
axiom f_not_zero : ∃ x, f x ≠ 0

-- Define the functional equation
axiom f_eq (x y : ℝ) : f (x * y) = x * f y + y * f x

-- Theorem 1: f(8) = 12f(2)
theorem f_eight_eq_twelve_f_two : f 8 = 12 * f 2 := by sorry

-- Theorem 2: f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

end f_eight_eq_twelve_f_two_f_odd_l4121_412178


namespace negative_y_implies_m_gt_2_smallest_m_solution_l4121_412146

-- Define the equation
def equation (y m : ℝ) : Prop := 4 * y + 2 * m + 1 = 2 * y + 5

-- Define the inequality
def inequality (x m : ℝ) : Prop := x - 1 > (m * x + 1) / 2

theorem negative_y_implies_m_gt_2 :
  (∃ y, y < 0 ∧ equation y m) → m > 2 :=
sorry

theorem smallest_m_solution :
  m = 3 → (∀ x, inequality x m ↔ x < -3) :=
sorry

end negative_y_implies_m_gt_2_smallest_m_solution_l4121_412146


namespace square_area_ratio_l4121_412199

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 48) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 16 / 25 := by
  sorry

end square_area_ratio_l4121_412199


namespace peter_wins_iff_n_odd_l4121_412156

/-- Represents the state of a cup (empty or filled) -/
inductive CupState
| Empty : CupState
| Filled : CupState

/-- Represents a player in the game -/
inductive Player
| Peter : Player
| Vasya : Player

/-- The game state on a 2n-gon -/
structure GameState (n : ℕ) where
  cups : Fin (2 * n) → CupState
  currentPlayer : Player

/-- Checks if two positions are symmetric with respect to the center of the 2n-gon -/
def isSymmetric (n : ℕ) (i j : Fin (2 * n)) : Prop :=
  (i.val + j.val) % (2 * n) = 0

/-- A valid move in the game -/
inductive Move (n : ℕ)
| Single : Fin (2 * n) → Move n
| Double : (i j : Fin (2 * n)) → isSymmetric n i j → Move n

/-- Applies a move to the game state -/
def applyMove (n : ℕ) (state : GameState n) (move : Move n) : GameState n :=
  sorry

/-- Checks if a player has a winning strategy -/
def hasWinningStrategy (n : ℕ) (player : Player) : Prop :=
  sorry

/-- The main theorem: Peter has a winning strategy if and only if n is odd -/
theorem peter_wins_iff_n_odd (n : ℕ) :
  hasWinningStrategy n Player.Peter ↔ Odd n :=
sorry

end peter_wins_iff_n_odd_l4121_412156


namespace verbal_equals_algebraic_l4121_412187

/-- The verbal description of the algebraic expression "5-4a" -/
def verbal_description : String := "the difference of 5 and 4 times a"

/-- The algebraic expression -/
def algebraic_expression (a : ℝ) : ℝ := 5 - 4 * a

theorem verbal_equals_algebraic :
  ∀ a : ℝ, verbal_description = "the difference of 5 and 4 times a" ↔ 
  algebraic_expression a = 5 - 4 * a :=
by sorry

end verbal_equals_algebraic_l4121_412187


namespace isosceles_non_equilateral_distinct_lines_l4121_412119

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is isosceles --/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a triangle is equilateral --/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Function to count distinct lines representing altitudes, medians, and interior angle bisectors --/
def countDistinctLines (t : Triangle) : ℕ := sorry

/-- Theorem stating that an isosceles non-equilateral triangle has 5 distinct lines --/
theorem isosceles_non_equilateral_distinct_lines (t : Triangle) :
  isIsosceles t ∧ ¬isEquilateral t → countDistinctLines t = 5 := by
  sorry

end isosceles_non_equilateral_distinct_lines_l4121_412119


namespace system_of_equations_solutions_l4121_412158

theorem system_of_equations_solutions :
  -- System (1)
  (∃ x y : ℝ, 3 * y - 4 * x = 0 ∧ 4 * x + y = 8 ∧ x = 3/2 ∧ y = 2) ∧
  -- System (2)
  (∃ x y : ℝ, x + y = 3 ∧ (x - 1)/4 + y/2 = 3/4 ∧ x = 2 ∧ y = 1) :=
by
  sorry

end system_of_equations_solutions_l4121_412158


namespace angle_bisector_ratio_not_determine_shape_l4121_412195

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of an angle bisector to its corresponding side length --/
def angleBisectorToSideRatio (t : Triangle) : ℝ := sorry

/-- Two triangles are similar if they have the same shape --/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Theorem: The ratio of an angle bisector to the corresponding side's length
    does not uniquely determine the shape of a triangle --/
theorem angle_bisector_ratio_not_determine_shape :
  ∃ (t1 t2 : Triangle), angleBisectorToSideRatio t1 = angleBisectorToSideRatio t2 ∧ ¬ areSimilar t1 t2 := by
  sorry

end angle_bisector_ratio_not_determine_shape_l4121_412195


namespace complement_of_supplement_35_l4121_412160

/-- The supplement of an angle in degrees -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- The complement of an angle in degrees -/
def complement (x : ℝ) : ℝ := 90 - x

/-- Theorem: The degree measure of the complement of the supplement of a 35-degree angle is -55 degrees -/
theorem complement_of_supplement_35 : complement (supplement 35) = -55 := by
  sorry

end complement_of_supplement_35_l4121_412160


namespace roots_of_quadratic_equation_l4121_412110

theorem roots_of_quadratic_equation :
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end roots_of_quadratic_equation_l4121_412110


namespace sum_m_n_range_l4121_412140

/-- A quadratic function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- The theorem stating that given the conditions, m + n is in [-4, 0] -/
theorem sum_m_n_range (m n : ℝ) (h1 : m ≤ n) (h2 : ∀ x ∈ Set.Icc m n, -1 ≤ f x ∧ f x ≤ 3) :
  -4 ≤ m + n ∧ m + n ≤ 0 := by sorry

end sum_m_n_range_l4121_412140


namespace chocolate_boxes_sold_l4121_412167

/-- The number of chocolate biscuit boxes sold by Kaylee -/
def chocolate_boxes : ℕ :=
  let total_boxes : ℕ := 33
  let lemon_boxes : ℕ := 12
  let oatmeal_boxes : ℕ := 4
  let remaining_boxes : ℕ := 12
  total_boxes - (lemon_boxes + oatmeal_boxes + remaining_boxes)

theorem chocolate_boxes_sold :
  chocolate_boxes = 5 := by
  sorry

end chocolate_boxes_sold_l4121_412167


namespace matthews_friends_l4121_412170

theorem matthews_friends (initial_crackers initial_cakes cakes_per_person : ℕ) 
  (h1 : initial_crackers = 10)
  (h2 : initial_cakes = 8)
  (h3 : cakes_per_person = 2)
  (h4 : initial_cakes % cakes_per_person = 0) :
  initial_cakes / cakes_per_person = 4 := by
  sorry

end matthews_friends_l4121_412170


namespace side_xy_length_l4121_412113

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ) := X + Y + Z = 180

-- Define the right angle
def RightAngle (Z : ℝ) := Z = 90

-- Define the area of the triangle
def TriangleArea (A : ℝ) := A = 36

-- Define the angles of the triangle
def AngleX (X : ℝ) := X = 30
def AngleY (Y : ℝ) := Y = 60

-- Theorem statement
theorem side_xy_length 
  (X Y Z A : ℝ) 
  (tri : Triangle X Y Z) 
  (right : RightAngle Z) 
  (area : TriangleArea A) 
  (angleX : AngleX X) 
  (angleY : AngleY Y) : 
  ∃ (XY : ℝ), XY = Real.sqrt (36 / Real.sqrt 3) :=
sorry

end side_xy_length_l4121_412113


namespace xy_value_l4121_412143

theorem xy_value (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 16)
  (h2 : (27:ℝ)^(x+y) / (9:ℝ)^(5*y) = 729) : 
  x * y = 96 := by
sorry

end xy_value_l4121_412143


namespace smallest_five_digit_divisible_by_first_five_primes_l4121_412173

theorem smallest_five_digit_divisible_by_first_five_primes :
  (∀ n : ℕ, n ≥ 10000 ∧ n < 11550 → ¬(2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n)) ∧
  (11550 ≥ 10000) ∧
  (2 ∣ 11550) ∧ (3 ∣ 11550) ∧ (5 ∣ 11550) ∧ (7 ∣ 11550) ∧ (11 ∣ 11550) :=
by sorry

end smallest_five_digit_divisible_by_first_five_primes_l4121_412173


namespace min_angular_frequency_l4121_412189

/-- Given a cosine function with specific properties, prove that the minimum angular frequency is 2 -/
theorem min_angular_frequency (ω φ : ℝ) : 
  ω > 0 → 
  (∃ k : ℤ, ω * (π / 3) + φ = k * π) →
  1/2 * Real.cos (ω * (π / 12) + φ) + 1 = 1 →
  (∀ ω' > 0, 
    (∃ k : ℤ, ω' * (π / 3) + φ = k * π) →
    1/2 * Real.cos (ω' * (π / 12) + φ) + 1 = 1 →
    ω' ≥ ω) →
  ω = 2 :=
by sorry

end min_angular_frequency_l4121_412189


namespace cubic_sum_plus_eight_l4121_412101

theorem cubic_sum_plus_eight (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 8 = 978 := by
  sorry

end cubic_sum_plus_eight_l4121_412101


namespace probability_of_selection_l4121_412192

def total_students : ℕ := 10
def students_per_teacher : ℕ := 4

theorem probability_of_selection (total_students : ℕ) (students_per_teacher : ℕ) :
  total_students = 10 → students_per_teacher = 4 →
  (1 : ℚ) - (1 - students_per_teacher / total_students) ^ 2 = 16 / 25 := by
  sorry

end probability_of_selection_l4121_412192


namespace number_equation_l4121_412124

theorem number_equation (x : ℝ) : x - 2 + 4 = 9 ↔ x = 7 := by sorry

end number_equation_l4121_412124


namespace school_population_l4121_412116

/-- Given a school with boys and girls, prove the total number of students is 900 -/
theorem school_population (total boys girls : ℕ) : 
  total = boys + girls →
  boys = 90 →
  girls = (90 * total) / 100 →
  total = 900 := by
  sorry

end school_population_l4121_412116


namespace min_value_expression_min_value_achieved_l4121_412130

theorem min_value_expression (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 2)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 :=
by sorry

theorem min_value_achieved (a b c : ℝ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  ∃ (a₀ b₀ c₀ : ℝ), 2 ≤ a₀ ∧ a₀ ≤ b₀ ∧ b₀ ≤ c₀ ∧ c₀ ≤ 5 ∧
    (a₀ - 2)^2 + (b₀/a₀ - 1)^2 + (c₀/b₀ - 1)^2 + (5/c₀ - 1)^2 = 4 * (5^(1/4) - 1)^2 :=
by sorry

end min_value_expression_min_value_achieved_l4121_412130


namespace right_triangle_hypotenuse_squared_l4121_412188

/-- Given complex numbers p, q, and r that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 360, then the square of the hypotenuse of the
    triangle is 540. -/
theorem right_triangle_hypotenuse_squared 
  (p q r : ℂ) 
  (h_zeros : ∃ (s t u : ℂ), p^3 + s*p^2 + t*p + u = 0 ∧ 
                             q^3 + s*q^2 + t*q + u = 0 ∧ 
                             r^3 + s*r^2 + t*r + u = 0)
  (h_right_triangle : ∃ (k : ℝ), (Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
                                 (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
                                 (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2)
  (h_sum_squares : Complex.abs p^2 + Complex.abs q^2 + Complex.abs r^2 = 360) :
  ∃ (k : ℝ), k^2 = 540 ∧ 
    ((Complex.abs (p - q))^2 + (Complex.abs (q - r))^2 = k^2 ∨
     (Complex.abs (q - r))^2 + (Complex.abs (r - p))^2 = k^2 ∨
     (Complex.abs (r - p))^2 + (Complex.abs (p - q))^2 = k^2) :=
by sorry

end right_triangle_hypotenuse_squared_l4121_412188


namespace other_candidate_votes_l4121_412154

theorem other_candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ)
  (h_total : total_votes = 8500)
  (h_invalid : invalid_percent = 25 / 100)
  (h_winner : winner_percent = 60 / 100) :
  ⌊(1 - winner_percent) * ((1 - invalid_percent) * total_votes)⌋ = 2550 := by
  sorry

end other_candidate_votes_l4121_412154


namespace coin_denomination_problem_l4121_412139

theorem coin_denomination_problem (total_coins : ℕ) (unknown_coins : ℕ) (known_coins : ℕ) 
  (known_coin_value : ℕ) (total_value : ℕ) (x : ℕ) :
  total_coins = 324 →
  unknown_coins = 220 →
  known_coins = total_coins - unknown_coins →
  known_coin_value = 25 →
  total_value = 7000 →
  unknown_coins * x + known_coins * known_coin_value = total_value →
  x = 20 := by
  sorry

end coin_denomination_problem_l4121_412139


namespace greatest_of_three_consecutive_integers_l4121_412185

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := by
  sorry

end greatest_of_three_consecutive_integers_l4121_412185


namespace equation_root_l4121_412162

theorem equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 + 5*x + m = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + 5*y + m = 0 ∧ y = -4) :=
sorry

end equation_root_l4121_412162


namespace kah_to_zah_conversion_l4121_412132

/-- Conversion rate between zahs and tols -/
def zah_to_tol : ℚ := 24 / 15

/-- Conversion rate between tols and kahs -/
def tol_to_kah : ℚ := 15 / 9

/-- The number of kahs we want to convert -/
def kahs_to_convert : ℕ := 2000

/-- The expected number of zahs after conversion -/
def expected_zahs : ℕ := 750

theorem kah_to_zah_conversion :
  (kahs_to_convert : ℚ) / (zah_to_tol * tol_to_kah) = expected_zahs := by
  sorry

end kah_to_zah_conversion_l4121_412132


namespace contradiction_proof_l4121_412105

theorem contradiction_proof (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (product_inequality : a * c + b * d > 1) 
  (all_nonnegative : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) : 
  False := by
sorry

end contradiction_proof_l4121_412105


namespace perfect_square_condition_l4121_412135

theorem perfect_square_condition (y m : ℝ) : 
  (∃ k : ℝ, y^2 - 8*y + m = k^2) → m = 16 := by
  sorry

end perfect_square_condition_l4121_412135


namespace vector_angle_proof_l4121_412106

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_angle_proof (a b : ℝ × ℝ) 
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 4) 
  (h3 : (a + b) • a = 0) : 
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end vector_angle_proof_l4121_412106


namespace unique_solution_k_l4121_412148

theorem unique_solution_k (k : ℝ) : 
  (∃! x : ℝ, (1 / (3 * x) = (k - x) / 8)) ↔ k = 8/3 := by
  sorry

end unique_solution_k_l4121_412148


namespace product_inspection_problem_l4121_412172

def total_products : ℕ := 100
def defective_products : ℕ := 3
def drawn_products : ℕ := 4
def defective_in_sample : ℕ := 2

theorem product_inspection_problem :
  (Nat.choose defective_products defective_in_sample) *
  (Nat.choose (total_products - defective_products) (drawn_products - defective_in_sample)) = 13968 := by
  sorry

end product_inspection_problem_l4121_412172


namespace solution_set_part1_range_of_a_part2_l4121_412141

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1/2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Ioo 0 1, f a x > x) → a ∈ Set.Ioc 0 2 := by sorry

end solution_set_part1_range_of_a_part2_l4121_412141


namespace emily_calculation_l4121_412155

theorem emily_calculation (n : ℕ) (h : n = 50) : n^2 - 99 = (n - 1)^2 := by
  sorry

end emily_calculation_l4121_412155


namespace chess_players_lost_to_ai_l4121_412111

theorem chess_players_lost_to_ai (total_players : ℕ) (never_lost_fraction : ℚ) : 
  total_players = 40 → never_lost_fraction = 1/4 → 
  (total_players : ℚ) * (1 - never_lost_fraction) = 30 := by
  sorry

end chess_players_lost_to_ai_l4121_412111


namespace count_students_in_line_l4121_412153

/-- The number of students in a line formation -/
def students_in_line (between : ℕ) : ℕ :=
  between + 2

/-- Theorem: Given 14 people between Yoojung and Eunji, there are 16 students in line -/
theorem count_students_in_line :
  students_in_line 14 = 16 := by
  sorry

end count_students_in_line_l4121_412153


namespace not_perfect_square_for_prime_l4121_412190

theorem not_perfect_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) :
  ¬∃ (a : ℤ), a^2 = 7 * p + 3^p - 4 := by
  sorry

end not_perfect_square_for_prime_l4121_412190
