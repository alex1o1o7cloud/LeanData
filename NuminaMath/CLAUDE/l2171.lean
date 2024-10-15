import Mathlib

namespace NUMINAMATH_CALUDE_potato_bag_weight_l2171_217141

/-- The weight of each bag of potatoes -/
def bag_weight (total_potatoes damaged_potatoes : ℕ) (price_per_bag total_revenue : ℚ) : ℚ :=
  (total_potatoes - damaged_potatoes) * price_per_bag / total_revenue

/-- Theorem stating the weight of each bag of potatoes -/
theorem potato_bag_weight :
  bag_weight 6500 150 72 9144 = 50 := by
  sorry

end NUMINAMATH_CALUDE_potato_bag_weight_l2171_217141


namespace NUMINAMATH_CALUDE_unique_stamp_solution_l2171_217120

/-- Given a positive integer n, returns true if 120 cents is the greatest
    postage that cannot be formed using stamps of 9, n, and n+2 cents -/
def is_valid_stamp_set (n : ℕ+) : Prop :=
  (∀ k : ℕ, k ≤ 120 → ¬∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k) ∧
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, 9*a + n*b + (n+2)*c = k)

/-- The only positive integer n that satisfies the stamp condition is 17 -/
theorem unique_stamp_solution :
  ∃! n : ℕ+, is_valid_stamp_set n ∧ n = 17 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_solution_l2171_217120


namespace NUMINAMATH_CALUDE_element_selection_theorem_l2171_217161

variable {α : Type*} [DecidableEq α]

def SubsetProperty (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) : Prop :=
  (S.card = n) ∧ 
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → S_i i ⊆ S ∧ (S_i i).card = 2) ∧
  (∀ e ∈ S, (Finset.filter (fun i => e ∈ S_i i) (Finset.range (k * n))).card = 2 * k)

theorem element_selection_theorem (S : Finset α) (n k : ℕ) (S_i : ℕ → Finset α) 
  (h : SubsetProperty S n k S_i) :
  ∃ f : ℕ → α, 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ k * n → f i ∈ S_i i) ∧ 
    (∀ e ∈ S, (Finset.filter (fun i => f i = e) (Finset.range (k * n))).card = k) :=
sorry

end NUMINAMATH_CALUDE_element_selection_theorem_l2171_217161


namespace NUMINAMATH_CALUDE_pennsylvania_quarters_l2171_217176

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  (total : ℚ) * state_fraction * penn_fraction = 7 := by
sorry

end NUMINAMATH_CALUDE_pennsylvania_quarters_l2171_217176


namespace NUMINAMATH_CALUDE_sheep_count_l2171_217115

/-- Represents the number of animals on a boat and their fate after capsizing -/
structure BoatAnimals where
  sheep : ℕ
  cows : ℕ
  dogs : ℕ
  drownedSheep : ℕ
  drownedCows : ℕ
  survivedAnimals : ℕ

/-- Theorem stating the number of sheep on the boat given the conditions -/
theorem sheep_count (b : BoatAnimals) : b.sheep = 20 :=
  by
  have h1 : b.cows = 10 := sorry
  have h2 : b.dogs = 14 := sorry
  have h3 : b.drownedSheep = 3 := sorry
  have h4 : b.drownedCows = 2 * b.drownedSheep := sorry
  have h5 : b.survivedAnimals = 35 := sorry
  have h6 : b.survivedAnimals = b.sheep - b.drownedSheep + b.cows - b.drownedCows + b.dogs := sorry
  sorry

#check sheep_count

end NUMINAMATH_CALUDE_sheep_count_l2171_217115


namespace NUMINAMATH_CALUDE_inequality_proof_l2171_217170

/-- For any non-integer real number x > 1, the following inequality holds -/
theorem inequality_proof (x : ℝ) (h1 : x > 1) (h2 : ¬ ∃ n : ℤ, x = n) :
  let fx := x - ⌊x⌋
  ((x + fx) / ⌊x⌋ - ⌊x⌋ / (x + fx)) + ((x + ⌊x⌋) / fx - fx / (x + ⌊x⌋)) > 9/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2171_217170


namespace NUMINAMATH_CALUDE_math_competition_probability_l2171_217191

/-- The number of students in the math competition team -/
def num_students : ℕ := 4

/-- The number of comprehensive questions -/
def num_questions : ℕ := 4

/-- The probability that each student solves a different question -/
def prob_different_questions : ℚ := 3/32

theorem math_competition_probability :
  (num_students.factorial : ℚ) / (num_students ^ num_students : ℕ) = prob_different_questions :=
sorry

end NUMINAMATH_CALUDE_math_competition_probability_l2171_217191


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2171_217148

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (11 - 7 * Complex.I) / (1 - 2 * Complex.I) → a + b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2171_217148


namespace NUMINAMATH_CALUDE_prob_two_or_fewer_white_eq_23_28_l2171_217153

/-- The number of white balls in the bag -/
def white_balls : Nat := 5

/-- The number of red balls in the bag -/
def red_balls : Nat := 3

/-- The total number of balls in the bag -/
def total_balls : Nat := white_balls + red_balls

/-- The probability of drawing 2 or fewer white balls before a red ball -/
def prob_two_or_fewer_white : Rat :=
  (red_balls : Rat) / total_balls +
  (white_balls * red_balls : Rat) / (total_balls * (total_balls - 1)) +
  (white_balls * (white_balls - 1) * red_balls : Rat) / (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_two_or_fewer_white_eq_23_28 : prob_two_or_fewer_white = 23 / 28 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_or_fewer_white_eq_23_28_l2171_217153


namespace NUMINAMATH_CALUDE_parabola_directrix_l2171_217106

/-- The directrix of a parabola y = -x^2 --/
theorem parabola_directrix : ∃ (d : ℝ), ∀ (x y : ℝ),
  y = -x^2 → (∃ (p : ℝ × ℝ), (x - p.1)^2 + (y - p.2)^2 = (y - d)^2 ∧ y ≤ d) → d = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2171_217106


namespace NUMINAMATH_CALUDE_circus_ticket_sales_l2171_217145

/-- Calculates the total number of tickets sold at a circus given the prices, revenue, and number of lower seat tickets sold. -/
def total_tickets (lower_price upper_price : ℕ) (total_revenue : ℕ) (lower_tickets : ℕ) : ℕ :=
  lower_tickets + (total_revenue - lower_price * lower_tickets) / upper_price

/-- Theorem stating that given the specific conditions of the circus problem, the total number of tickets sold is 80. -/
theorem circus_ticket_sales :
  total_tickets 30 20 2100 50 = 80 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_sales_l2171_217145


namespace NUMINAMATH_CALUDE_min_value_of_f_l2171_217100

def f (x : ℝ) (m : ℝ) := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 2) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m ∧ f x m = -6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2171_217100


namespace NUMINAMATH_CALUDE_polynomial_powered_function_is_polynomial_l2171_217178

/-- A function f: ℝ → ℝ such that (f(x))^n is a polynomial for every integer n ≥ 2 -/
def PolynomialPoweredFunction (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ p : Polynomial ℝ, ∀ x : ℝ, (f x)^n = p.eval x

/-- If f: ℝ → ℝ is a function such that (f(x))^n is a polynomial for every integer n ≥ 2,
    then f is a polynomial -/
theorem polynomial_powered_function_is_polynomial (f : ℝ → ℝ) 
  (h : PolynomialPoweredFunction f) : ∃ p : Polynomial ℝ, ∀ x : ℝ, f x = p.eval x :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_powered_function_is_polynomial_l2171_217178


namespace NUMINAMATH_CALUDE_card_collection_difference_l2171_217172

theorem card_collection_difference (total : ℕ) (baseball : ℕ) (football : ℕ) 
  (h1 : total = 125)
  (h2 : baseball = 95)
  (h3 : total = baseball + football)
  (h4 : ∃ k : ℕ, baseball = 3 * football + k) :
  baseball - 3 * football = 5 := by
  sorry

end NUMINAMATH_CALUDE_card_collection_difference_l2171_217172


namespace NUMINAMATH_CALUDE_winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l2171_217108

/-- Represents a game state in Winners Make Zeros --/
structure GameState where
  m : ℕ
  n : ℕ

/-- Determines if a given game state is a winning position --/
def is_winning_position (state : GameState) : Prop :=
  sorry

/-- The largest valid choice for c that results in a winning position --/
def largest_winning_c : ℕ :=
  999

theorem winners_make_zeros_largest_c :
  ∀ c : ℕ,
    c > largest_winning_c →
    c > 0 ∧
    2007777 - c * 2007 ≥ 0 →
    ¬is_winning_position ⟨2007777 - c * 2007, 2007⟩ :=
by sorry

theorem winners_make_zeros_optimal_c :
  largest_winning_c > 0 ∧
  2007777 - largest_winning_c * 2007 ≥ 0 ∧
  is_winning_position ⟨2007777 - largest_winning_c * 2007, 2007⟩ :=
by sorry

end NUMINAMATH_CALUDE_winners_make_zeros_largest_c_winners_make_zeros_optimal_c_l2171_217108


namespace NUMINAMATH_CALUDE_power_product_equality_l2171_217128

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 * 11 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2171_217128


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l2171_217125

theorem min_value_fraction_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ a₀ + b₀ + c₀ = 6 ∧ (9 / a₀ + 4 / b₀ + 25 / c₀ = 50 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l2171_217125


namespace NUMINAMATH_CALUDE_lcm_6_15_l2171_217127

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l2171_217127


namespace NUMINAMATH_CALUDE_balloon_ratio_l2171_217181

theorem balloon_ratio : 
  let dan_balloons : ℝ := 29.0
  let tim_balloons : ℝ := 4.142857143
  dan_balloons / tim_balloons = 7 := by
sorry

end NUMINAMATH_CALUDE_balloon_ratio_l2171_217181


namespace NUMINAMATH_CALUDE_daniels_animals_legs_l2171_217174

/-- Calculates the total number of legs for Daniel's animals -/
def totalAnimalLegs (horses dogs cats turtles goats : ℕ) : ℕ :=
  4 * (horses + dogs + cats + turtles + goats)

/-- Theorem: Daniel's animals have 72 legs in total -/
theorem daniels_animals_legs :
  totalAnimalLegs 2 5 7 3 1 = 72 := by
  sorry

end NUMINAMATH_CALUDE_daniels_animals_legs_l2171_217174


namespace NUMINAMATH_CALUDE_intersection_complement_problem_l2171_217104

def I : Set ℤ := {x | -3 < x ∧ x < 3}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem intersection_complement_problem :
  A ∩ (I \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_problem_l2171_217104


namespace NUMINAMATH_CALUDE_laundry_charge_per_shirt_l2171_217164

theorem laundry_charge_per_shirt 
  (total_trousers : ℕ) 
  (cost_per_trouser : ℚ) 
  (total_bill : ℚ) 
  (total_shirts : ℕ) : 
  (total_bill - total_trousers * cost_per_trouser) / total_shirts = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_laundry_charge_per_shirt_l2171_217164


namespace NUMINAMATH_CALUDE_gcd_and_binary_conversion_l2171_217133

theorem gcd_and_binary_conversion :
  (Nat.gcd 153 119 = 17) ∧
  (ToString.toString (Nat.toDigits 2 89) = "1011001") := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_binary_conversion_l2171_217133


namespace NUMINAMATH_CALUDE_complex_equation_sum_l2171_217116

theorem complex_equation_sum (a b : ℝ) : 
  (a + 2 * Complex.I) / Complex.I = b + Complex.I → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l2171_217116


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2171_217166

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (ha : a > 1)
  (f : ℝ → ℝ) (hf : ∀ x, f x = a^(x^2 + 2*x)) :
  ∀ x, -1 < x ∧ x < 0 → f x < 1 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l2171_217166


namespace NUMINAMATH_CALUDE_walking_distance_ratio_l2171_217107

/-- The ratio of walking distances given different walking speeds and times -/
theorem walking_distance_ratio 
  (your_speed : ℝ) 
  (harris_speed : ℝ) 
  (harris_time : ℝ) 
  (your_time : ℝ) 
  (h1 : your_speed = 2 * harris_speed) 
  (h2 : harris_time = 2) 
  (h3 : your_time = 3) : 
  (your_speed * your_time) / (harris_speed * harris_time) = 3 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_ratio_l2171_217107


namespace NUMINAMATH_CALUDE_typistSalary_l2171_217192

/-- Calculates the final salary after a raise and a reduction -/
def finalSalary (originalSalary : ℚ) (raisePercentage : ℚ) (reductionPercentage : ℚ) : ℚ :=
  let salaryAfterRaise := originalSalary * (1 + raisePercentage / 100)
  salaryAfterRaise * (1 - reductionPercentage / 100)

/-- Theorem stating that the typist's final salary is 6270 Rs -/
theorem typistSalary :
  finalSalary 6000 10 5 = 6270 := by
  sorry

#eval finalSalary 6000 10 5

end NUMINAMATH_CALUDE_typistSalary_l2171_217192


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2171_217143

/-- Time taken for a train to cross a bridge -/
theorem train_bridge_crossing_time
  (bridge_length : ℝ)
  (train_length : ℝ)
  (lamp_post_time : ℝ)
  (h1 : bridge_length = 200)
  (h2 : train_length = 200)
  (h3 : lamp_post_time = 5)
  : ℝ :=
by
  -- The time taken for the train to cross the bridge is 10 seconds
  sorry

#check train_bridge_crossing_time

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2171_217143


namespace NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2171_217177

/-- The expected number of boy-girl adjacencies in a row of 6 boys and 14 girls -/
theorem expected_boy_girl_adjacencies :
  let num_boys : ℕ := 6
  let num_girls : ℕ := 14
  let total_people : ℕ := num_boys + num_girls
  let num_adjacencies : ℕ := total_people - 1
  let prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
  let expected_adjacencies : ℚ := 2 * prob_boy_girl * num_adjacencies
  expected_adjacencies = 798 / 95 := by
  sorry

end NUMINAMATH_CALUDE_expected_boy_girl_adjacencies_l2171_217177


namespace NUMINAMATH_CALUDE_m_range_l2171_217157

theorem m_range (m : ℝ) (h1 : m < 0) (h2 : ∀ x : ℝ, x^2 + m*x + 1 > 0) : -2 < m ∧ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2171_217157


namespace NUMINAMATH_CALUDE_mowing_earnings_l2171_217179

/-- Calculates the earnings for a single hour of mowing based on the hour number within a cycle -/
def hourly_pay (hour : ℕ) : ℕ :=
  5 * (hour % 6 + 1)

/-- Calculates the total earnings for a given number of hours of mowing -/
def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_pay |>.sum

/-- Theorem stating that mowing for 24 hours results in earnings of 420 dollars -/
theorem mowing_earnings : total_earnings 24 = 420 := by
  sorry

end NUMINAMATH_CALUDE_mowing_earnings_l2171_217179


namespace NUMINAMATH_CALUDE_smaller_package_size_l2171_217188

/-- The number of notebooks in a large package -/
def large_package : ℕ := 7

/-- The total number of notebooks Wilson bought -/
def total_notebooks : ℕ := 69

/-- The number of large packages Wilson bought -/
def large_packages_bought : ℕ := 7

/-- The number of notebooks in the smaller package -/
def small_package : ℕ := 5

/-- Theorem stating that the smaller package contains 5 notebooks -/
theorem smaller_package_size :
  ∃ (n : ℕ), 
    n * small_package + large_packages_bought * large_package = total_notebooks ∧
    n > 0 ∧
    small_package < large_package ∧
    small_package ∣ (total_notebooks - large_packages_bought * large_package) :=
by sorry

end NUMINAMATH_CALUDE_smaller_package_size_l2171_217188


namespace NUMINAMATH_CALUDE_min_questions_to_identify_apartment_l2171_217158

theorem min_questions_to_identify_apartment (n : ℕ) (h : n = 80) : 
  (∀ m : ℕ, 2^m < n → m < 7) ∧ (2^7 ≥ n) := by
  sorry

end NUMINAMATH_CALUDE_min_questions_to_identify_apartment_l2171_217158


namespace NUMINAMATH_CALUDE_octal_to_binary_conversion_l2171_217184

/-- Converts an octal number to decimal -/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary -/
def decimal_to_binary (decimal : ℕ) : ℕ := sorry

/-- The octal representation of the number -/
def octal_num : ℕ := 135

/-- The binary representation of the number -/
def binary_num : ℕ := 1011101

theorem octal_to_binary_conversion :
  decimal_to_binary (octal_to_decimal octal_num) = binary_num := by sorry

end NUMINAMATH_CALUDE_octal_to_binary_conversion_l2171_217184


namespace NUMINAMATH_CALUDE_min_value_expression_l2171_217198

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a + 2/b) * (a + 2/b - 100) + (b + 2/a) * (b + 2/a - 100) ≥ -2500 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2171_217198


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2171_217130

def U : Set ℕ := {x | x > 0 ∧ x < 9}

def M : Set ℕ := {1, 2, 3}

def N : Set ℕ := {3, 4, 5, 6}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {4, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2171_217130


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l2171_217135

/-- 
Given two vectors a and b in R^2, where a = (1, m) and b = (-1, 2m+1),
prove that if a and b are parallel, then m = -1/3.
-/
theorem parallel_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, m]
  let b : Fin 2 → ℝ := ![-1, 2*m+1]
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → m = -1/3 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l2171_217135


namespace NUMINAMATH_CALUDE_box_removal_proof_l2171_217151

theorem box_removal_proof (total_boxes : Nat) (boxes_10lb boxes_20lb boxes_30lb boxes_40lb : Nat)
  (initial_avg_weight : Nat) (target_avg_weight : Nat) 
  (h1 : total_boxes = 30)
  (h2 : boxes_10lb = 10)
  (h3 : boxes_20lb = 10)
  (h4 : boxes_30lb = 5)
  (h5 : boxes_40lb = 5)
  (h6 : initial_avg_weight = 20)
  (h7 : target_avg_weight = 17) :
  let total_weight := boxes_10lb * 10 + boxes_20lb * 20 + boxes_30lb * 30 + boxes_40lb * 40
  let remaining_boxes := total_boxes - 6
  let remaining_weight := total_weight - (5 * 20 + 1 * 40)
  remaining_weight / remaining_boxes = target_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_box_removal_proof_l2171_217151


namespace NUMINAMATH_CALUDE_rate_of_profit_l2171_217187

/-- Calculate the rate of profit given the cost price and selling price -/
theorem rate_of_profit (cost_price selling_price : ℕ) : 
  cost_price = 50 → selling_price = 60 → 
  (selling_price - cost_price) * 100 / cost_price = 20 := by
  sorry

end NUMINAMATH_CALUDE_rate_of_profit_l2171_217187


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2171_217162

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  (g 1 = 2) ∧ 
  (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y)

/-- The main theorem stating that the function g satisfying the functional equation
    is equal to 2(4^x - 3^x) for all real x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = 2 * (4^x - 3^x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2171_217162


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l2171_217180

theorem integer_ratio_problem (a b c : ℕ) : 
  a < b → b < c → 
  a = 0 → b ≠ a + 1 → 
  (a + b + c : ℚ) / 3 = 4 * b → 
  c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l2171_217180


namespace NUMINAMATH_CALUDE_a_range_l2171_217147

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 2 * x - 3 * y + a = 0

-- Define the condition for points being on opposite sides of the line
def opposite_sides (a : ℝ) : Prop :=
  (2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0

-- Theorem statement
theorem a_range (a : ℝ) :
  (∀ x y, line_equation x y a) →
  opposite_sides a →
  -1 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_a_range_l2171_217147


namespace NUMINAMATH_CALUDE_geometric_series_sum_first_six_terms_l2171_217194

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_first_six_terms :
  let a : ℚ := 3
  let r : ℚ := 1/3
  let n : ℕ := 6
  geometric_series_sum a r n = 364/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_first_six_terms_l2171_217194


namespace NUMINAMATH_CALUDE_equation_proof_l2171_217150

theorem equation_proof (a b : ℚ) (h : 3 * a = 2 * b) : (a + b) / b = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2171_217150


namespace NUMINAMATH_CALUDE_survey_support_l2171_217185

theorem survey_support (N A B N_o : ℕ) (h1 : N = 198) (h2 : A = 149) (h3 : B = 119) (h4 : N_o = 29) :
  A + B - (N - N_o) = 99 :=
by sorry

end NUMINAMATH_CALUDE_survey_support_l2171_217185


namespace NUMINAMATH_CALUDE_fractional_equation_root_l2171_217144

/-- If the equation (3 / (x - 4)) + ((x + m) / (4 - x)) = 1 has a root, then m = -1 -/
theorem fractional_equation_root (x m : ℚ) : 
  (∃ x, (3 / (x - 4)) + ((x + m) / (4 - x)) = 1) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_root_l2171_217144


namespace NUMINAMATH_CALUDE_num_bounces_correct_l2171_217167

/-- The initial height of the ball in meters -/
def initial_height : ℝ := 500

/-- The ratio of the bounce height to the previous height -/
def bounce_ratio : ℝ := 0.6

/-- The height threshold for counting bounces, in meters -/
def bounce_threshold : ℝ := 5

/-- The height at which the ball stops bouncing, in meters -/
def stop_threshold : ℝ := 0.1

/-- The height of the ball after k bounces -/
def height_after_bounces (k : ℕ) : ℝ := initial_height * bounce_ratio ^ k

/-- The number of bounces after which the ball first reaches a maximum height less than the bounce threshold -/
def num_bounces : ℕ := sorry

theorem num_bounces_correct :
  (∀ k < num_bounces, height_after_bounces k ≥ bounce_threshold) ∧
  height_after_bounces num_bounces < bounce_threshold ∧
  (∀ n : ℕ, height_after_bounces n ≥ stop_threshold → n ≤ num_bounces) ∧
  num_bounces = 10 := by sorry

end NUMINAMATH_CALUDE_num_bounces_correct_l2171_217167


namespace NUMINAMATH_CALUDE_divide_algebraic_expression_l2171_217138

theorem divide_algebraic_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  12 * a^4 * b^3 * c / (-4 * a^3 * b^2) = -3 * a * b * c :=
by sorry

end NUMINAMATH_CALUDE_divide_algebraic_expression_l2171_217138


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2171_217195

theorem missing_fraction_sum (x : ℚ) : 
  (1 / 3 : ℚ) + (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 4 : ℚ) + (-9 / 20 : ℚ) + (-5 / 6 : ℚ) + x = 0.8333333333333334 
  → x = 0.8333333333333334 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2171_217195


namespace NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l2171_217146

/-- A pentagonal prism is a three-dimensional geometric shape with a pentagonal base
    and rectangular lateral faces. -/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle between a lateral edge and the base of a pentagonal prism. -/
def lateral_angle (p : PentagonalPrism) : ℝ := sorry

/-- Theorem: The angle between any lateral edge and the base of a pentagonal prism is 90°. -/
theorem pentagonal_prism_lateral_angle (p : PentagonalPrism) :
  lateral_angle p = Real.pi / 2 := by sorry

end NUMINAMATH_CALUDE_pentagonal_prism_lateral_angle_l2171_217146


namespace NUMINAMATH_CALUDE_total_cost_pencils_erasers_l2171_217114

/-- Given the price of a pencil (a) and an eraser (b) in yuan, 
    prove that the total cost of 3 pencils and 7 erasers is 3a + 7b yuan. -/
theorem total_cost_pencils_erasers (a b : ℝ) : 3 * a + 7 * b = 3 * a + 7 * b := by
  sorry

end NUMINAMATH_CALUDE_total_cost_pencils_erasers_l2171_217114


namespace NUMINAMATH_CALUDE_comparison_theorem_l2171_217152

theorem comparison_theorem (a b c : ℝ) 
  (ha : a = Real.log 1.01)
  (hb : b = 1 / 101)
  (hc : c = Real.sin 0.01) :
  a > b ∧ c > a := by sorry

end NUMINAMATH_CALUDE_comparison_theorem_l2171_217152


namespace NUMINAMATH_CALUDE_time_against_walkway_l2171_217132

/-- The time it takes to walk against a moving walkway given specific conditions -/
theorem time_against_walkway 
  (walkway_length : ℝ) 
  (time_with_walkway : ℝ) 
  (time_without_movement : ℝ) 
  (h1 : walkway_length = 100) 
  (h2 : time_with_walkway = 25) 
  (h3 : time_without_movement = 42.857142857142854) :
  let person_speed := walkway_length / time_without_movement
  let walkway_speed := walkway_length / time_with_walkway - person_speed
  walkway_length / (person_speed - walkway_speed) = 150 := by
  sorry

end NUMINAMATH_CALUDE_time_against_walkway_l2171_217132


namespace NUMINAMATH_CALUDE_partition_remainder_l2171_217154

theorem partition_remainder (S : Finset ℕ) : 
  S.card = 15 → 
  (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000 = 406 := by
  sorry

#eval (4^15 - 3 * 3^15 + 3 * 2^15 - 1) % 1000

end NUMINAMATH_CALUDE_partition_remainder_l2171_217154


namespace NUMINAMATH_CALUDE_bottles_sold_wed_to_sun_is_250_l2171_217168

/-- Represents the inventory and sales of hand sanitizer bottles at Danivan Drugstore --/
structure DrugstoreInventory where
  initial_inventory : ℕ
  monday_sales : ℕ
  tuesday_sales : ℕ
  saturday_delivery : ℕ
  final_inventory : ℕ

/-- Calculates the number of bottles sold from Wednesday to Sunday --/
def bottles_sold_wed_to_sun (d : DrugstoreInventory) : ℕ :=
  d.initial_inventory - d.monday_sales - d.tuesday_sales + d.saturday_delivery - d.final_inventory

/-- Theorem stating that the number of bottles sold from Wednesday to Sunday is 250 --/
theorem bottles_sold_wed_to_sun_is_250 (d : DrugstoreInventory) 
    (h1 : d.initial_inventory = 4500)
    (h2 : d.monday_sales = 2445)
    (h3 : d.tuesday_sales = 900)
    (h4 : d.saturday_delivery = 650)
    (h5 : d.final_inventory = 1555) :
    bottles_sold_wed_to_sun d = 250 := by
  sorry

end NUMINAMATH_CALUDE_bottles_sold_wed_to_sun_is_250_l2171_217168


namespace NUMINAMATH_CALUDE_binomial_square_difference_specific_case_l2171_217126

theorem binomial_square_difference (a b : ℕ) : (a + b)^2 - (a^2 + b^2) = 2 * a * b := by sorry

theorem specific_case : (45 + 15)^2 - (45^2 + 15^2) = 1350 := by sorry

end NUMINAMATH_CALUDE_binomial_square_difference_specific_case_l2171_217126


namespace NUMINAMATH_CALUDE_car_speed_problem_l2171_217119

theorem car_speed_problem (D : ℝ) (h : D > 0) :
  let t1 := D / 3 / 80
  let t2 := D / 3 / 30
  let t3 := D / 3 / 48
  45 = D / (t1 + t2 + t3) :=
by
  sorry

#check car_speed_problem

end NUMINAMATH_CALUDE_car_speed_problem_l2171_217119


namespace NUMINAMATH_CALUDE_jills_salary_solution_l2171_217149

def jills_salary_problem (net_salary : ℝ) : Prop :=
  let discretionary_income := net_salary / 5
  let vacation_fund := 0.30 * discretionary_income
  let savings := 0.20 * discretionary_income
  let eating_out := 0.35 * discretionary_income
  let fitness_classes := 0.05 * discretionary_income
  let gifts_and_charity := 99
  vacation_fund + savings + eating_out + fitness_classes + gifts_and_charity = discretionary_income ∧
  net_salary = 4950

theorem jills_salary_solution :
  ∃ (net_salary : ℝ), jills_salary_problem net_salary :=
sorry

end NUMINAMATH_CALUDE_jills_salary_solution_l2171_217149


namespace NUMINAMATH_CALUDE_apple_pear_cost_l2171_217101

theorem apple_pear_cost (x y : ℝ) 
  (eq1 : x + 2*y = 194) 
  (eq2 : 2*x + 5*y = 458) : 
  x = 54 ∧ y = 70 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_cost_l2171_217101


namespace NUMINAMATH_CALUDE_equation_solution_l2171_217156

theorem equation_solution (x : ℝ) : 
  1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2171_217156


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2171_217112

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Theorem for part (1)
theorem solution_set_part1 (a : ℝ) (h : a = 1) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} :=
by sorry

-- Theorem for part (2)
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2171_217112


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2171_217121

-- Equation 1
theorem equation_one_solution :
  ∃! x : ℚ, (x / (2 * x - 1)) + (2 / (1 - 2 * x)) = 3 :=
by sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ x : ℚ, (4 / (x^2 - 4)) - (1 / (x - 2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l2171_217121


namespace NUMINAMATH_CALUDE_sam_eats_280_apples_in_week_l2171_217118

/-- Calculates the number of apples Sam eats in a week -/
def apples_eaten_in_week (apples_per_sandwich : ℕ) (sandwiches_per_day : ℕ) (days_in_week : ℕ) : ℕ :=
  apples_per_sandwich * sandwiches_per_day * days_in_week

/-- Proves that Sam eats 280 apples in a week -/
theorem sam_eats_280_apples_in_week :
  apples_eaten_in_week 4 10 7 = 280 := by
  sorry

end NUMINAMATH_CALUDE_sam_eats_280_apples_in_week_l2171_217118


namespace NUMINAMATH_CALUDE_system_solution_expression_simplification_l2171_217111

-- Part 1: System of equations
theorem system_solution :
  ∃ (x y : ℝ), 2 * x + y = 3 ∧ 3 * x + y = 5 ∧ x = 2 ∧ y = -1 := by sorry

-- Part 2: Expression calculation
theorem expression_simplification (a : ℝ) (h : a ≠ 1) :
  (a^2 / (a^2 - 2*a + 1)) * ((a - 1) / a) - (1 / (a - 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_expression_simplification_l2171_217111


namespace NUMINAMATH_CALUDE_circle_condition_l2171_217137

theorem circle_condition (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0 → 
    ∃ r : ℝ, r > 0 ∧ ∃ a b : ℝ, (x - a)^2 + (y - b)^2 = r^2) → 
  m < 1 := by
sorry

end NUMINAMATH_CALUDE_circle_condition_l2171_217137


namespace NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l2171_217122

theorem probability_failed_math_given_failed_chinese 
  (failed_math : ℝ) 
  (failed_chinese : ℝ) 
  (failed_both : ℝ) 
  (h1 : failed_math = 0.16)
  (h2 : failed_chinese = 0.07)
  (h3 : failed_both = 0.04) :
  failed_both / failed_chinese = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_failed_math_given_failed_chinese_l2171_217122


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l2171_217163

theorem sum_of_cubes_product : ∃ x y : ℤ, x^3 + y^3 = 35 ∧ x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l2171_217163


namespace NUMINAMATH_CALUDE_cost_of_tax_free_items_l2171_217160

/-- Calculates the cost of tax-free items given total amount spent, sales tax paid, and tax rate -/
theorem cost_of_tax_free_items
  (total_spent : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h_total : total_spent = 25)
  (h_tax : sales_tax = 0.30)
  (h_rate : tax_rate = 0.06)
  : ∃ (cost_tax_free : ℝ), cost_tax_free = 20 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tax_free_items_l2171_217160


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2171_217175

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 36 = 0) → m = 12 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2171_217175


namespace NUMINAMATH_CALUDE_mikaela_tiled_walls_l2171_217190

/-- Calculates the number of walls tiled instead of painted given the initial number of paint containers, 
    total number of walls, containers used for the ceiling, and containers left over. -/
def walls_tiled (initial_containers : ℕ) (total_walls : ℕ) (ceiling_containers : ℕ) (leftover_containers : ℕ) : ℕ :=
  total_walls - (initial_containers - ceiling_containers - leftover_containers) / (initial_containers / total_walls)

/-- Proves that given 16 containers of paint initially, 4 equally-sized walls, 1 container used for the ceiling, 
    and 3 containers left over, the number of walls tiled instead of painted is 1. -/
theorem mikaela_tiled_walls :
  walls_tiled 16 4 1 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_tiled_walls_l2171_217190


namespace NUMINAMATH_CALUDE_property_length_proof_l2171_217103

/-- Given a rectangular property and a garden within it, prove the length of the property. -/
theorem property_length_proof (property_width : ℝ) (garden_area : ℝ) : 
  property_width = 1000 →
  garden_area = 28125 →
  ∃ (property_length : ℝ),
    property_length = 2250 ∧
    garden_area = (property_width / 8) * (property_length / 10) :=
by sorry

end NUMINAMATH_CALUDE_property_length_proof_l2171_217103


namespace NUMINAMATH_CALUDE_marie_task_completion_time_l2171_217102

-- Define the start time of the first task
def start_time : Nat := 7 * 60  -- 7:00 AM in minutes since midnight

-- Define the end time of the second task
def end_second_task : Nat := 9 * 60 + 20  -- 9:20 AM in minutes since midnight

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem statement
theorem marie_task_completion_time :
  let total_time_two_tasks := end_second_task - start_time
  let task_duration := total_time_two_tasks / 2
  let completion_time := end_second_task + 2 * task_duration
  completion_time = 11 * 60 + 40  -- 11:40 AM in minutes since midnight
:= by sorry

end NUMINAMATH_CALUDE_marie_task_completion_time_l2171_217102


namespace NUMINAMATH_CALUDE_speed_increase_percentage_l2171_217155

theorem speed_increase_percentage (distance : ℝ) (current_speed : ℝ) (speed_reduction : ℝ) (time_difference : ℝ) :
  distance = 96 →
  current_speed = 8 →
  speed_reduction = 4 →
  time_difference = 16 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 50 ∧
    distance / (current_speed * (1 + increase_percentage / 100)) = 
    distance / (current_speed - speed_reduction) - time_difference :=
by sorry

end NUMINAMATH_CALUDE_speed_increase_percentage_l2171_217155


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2171_217142

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {y | ∃ x, y = Real.exp x + 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2171_217142


namespace NUMINAMATH_CALUDE_lowest_price_breaks_even_l2171_217134

/-- Calculates the lowest price per component to break even --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_costs : ℚ) (num_components : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_components)

theorem lowest_price_breaks_even 
  (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_components : ℕ) :
  let price := lowest_price_per_component production_cost shipping_cost fixed_costs num_components
  (price * num_components : ℚ) = (production_cost + shipping_cost) * num_components + fixed_costs :=
by sorry

#eval lowest_price_per_component 80 5 16500 150

end NUMINAMATH_CALUDE_lowest_price_breaks_even_l2171_217134


namespace NUMINAMATH_CALUDE_greene_nursery_flower_count_l2171_217169

theorem greene_nursery_flower_count : 
  let red_roses : ℕ := 1491
  let yellow_carnations : ℕ := 3025
  let white_roses : ℕ := 1768
  let purple_tulips : ℕ := 2150
  let pink_daisies : ℕ := 3500
  let blue_irises : ℕ := 2973
  let orange_marigolds : ℕ := 4234
  let lavender_orchids : ℕ := 350
  let orchid_pots : ℕ := 5
  let sunflower_boxes : ℕ := 7
  let sunflowers_per_box : ℕ := 120
  let sunflowers_last_box : ℕ := 95
  let violet_lily_pairs : ℕ := 13

  red_roses + yellow_carnations + white_roses + purple_tulips + 
  pink_daisies + blue_irises + orange_marigolds + lavender_orchids + 
  (sunflower_boxes - 1) * sunflowers_per_box + sunflowers_last_box + 
  2 * violet_lily_pairs = 21332 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_flower_count_l2171_217169


namespace NUMINAMATH_CALUDE_three_means_sum_of_squares_l2171_217199

theorem three_means_sum_of_squares 
  (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 5)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 712.5 := by
sorry

end NUMINAMATH_CALUDE_three_means_sum_of_squares_l2171_217199


namespace NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l2171_217123

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relationships
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1: Two different planes perpendicular to the same line are parallel
theorem planes_perpendicular_to_line_are_parallel
  (l : Line) (p1 p2 : Plane) (h1 : p1 ≠ p2)
  (h2 : perpendicular_plane_line p1 l) (h3 : perpendicular_plane_line p2 l) :
  parallel_planes p1 p2 :=
sorry

-- Theorem 2: Two different lines perpendicular to the same plane are parallel
theorem lines_perpendicular_to_plane_are_parallel
  (p : Plane) (l1 l2 : Line) (h1 : l1 ≠ l2)
  (h2 : perpendicular_line_plane l1 p) (h3 : perpendicular_line_plane l2 p) :
  parallel_lines l1 l2 :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_to_line_are_parallel_lines_perpendicular_to_plane_are_parallel_l2171_217123


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l2171_217171

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel α β → perpendicular m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l2171_217171


namespace NUMINAMATH_CALUDE_smallest_relatively_prime_to_180_l2171_217196

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_relatively_prime_to_180 :
  ∃ (x : ℕ), x > 1 ∧ is_relatively_prime x 180 ∧
  ∀ (y : ℕ), y > 1 ∧ y < x → ¬(is_relatively_prime y 180) :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_smallest_relatively_prime_to_180_l2171_217196


namespace NUMINAMATH_CALUDE_hcf_of_210_and_605_l2171_217113

theorem hcf_of_210_and_605 :
  let a := 210
  let b := 605
  let lcm_ab := 2310
  lcm a b = lcm_ab →
  Nat.gcd a b = 55 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_210_and_605_l2171_217113


namespace NUMINAMATH_CALUDE_one_third_between_one_eighth_and_one_third_l2171_217189

def one_third_between (a b : ℚ) : ℚ :=
  (1 - 1/3) * a + 1/3 * b

theorem one_third_between_one_eighth_and_one_third :
  one_third_between (1/8) (1/3) = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_one_third_between_one_eighth_and_one_third_l2171_217189


namespace NUMINAMATH_CALUDE_triangle_inequality_l2171_217117

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c ∧
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) = Real.sqrt a + Real.sqrt b + Real.sqrt c ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2171_217117


namespace NUMINAMATH_CALUDE_corrected_mean_specific_corrected_mean_l2171_217140

/-- Given a set of observations with an incorrect entry, calculate the corrected mean -/
theorem corrected_mean (n : ℕ) (original_mean : ℝ) (incorrect_value correct_value : ℝ) :
  n > 0 →
  let total_sum := n * original_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = (n * original_mean - incorrect_value + correct_value) / n :=
by sorry

/-- The specific problem instance -/
theorem specific_corrected_mean :
  let n : ℕ := 40
  let original_mean : ℝ := 100
  let incorrect_value : ℝ := 75
  let correct_value : ℝ := 50
  (n * original_mean - incorrect_value + correct_value) / n = 99.375 :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_specific_corrected_mean_l2171_217140


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2171_217186

theorem max_value_of_expression (x : ℝ) (h : x > 0) :
  1 - x - 16 / x ≤ -7 ∧ ∃ y > 0, 1 - y - 16 / y = -7 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2171_217186


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2171_217129

theorem sum_of_solutions (x : ℝ) : 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b) → 
  (∃ a b : ℝ, (4*x + 6) * (3*x - 8) = 0 ∧ x = a ∨ x = b ∧ a + b = 7/6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2171_217129


namespace NUMINAMATH_CALUDE_square_units_tens_digits_l2171_217173

theorem square_units_tens_digits (x : ℤ) (h : x^2 % 100 = 9) : 
  x^2 % 200 = 0 ∨ x^2 % 200 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_units_tens_digits_l2171_217173


namespace NUMINAMATH_CALUDE_distinct_collections_count_l2171_217165

/-- Represents the count of each letter in ALGEBRAICS --/
structure LetterCount where
  a : Nat
  b : Nat
  c : Nat
  e : Nat
  g : Nat
  i : Nat
  l : Nat
  r : Nat
  s : Nat

/-- The initial count of letters in ALGEBRAICS --/
def initialCount : LetterCount :=
  { a := 2, b := 1, c := 1, e := 1, g := 1, i := 1, l := 1, r := 1, s := 1 }

/-- Counts the number of distinct collections of two vowels and two consonants --/
def countDistinctCollections (count : LetterCount) : Nat :=
  sorry

theorem distinct_collections_count :
  countDistinctCollections initialCount = 68 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_count_l2171_217165


namespace NUMINAMATH_CALUDE_initial_cards_eq_sum_l2171_217197

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := 242

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := 136

/-- The number of cards Nell has left -/
def cards_left : ℕ := 106

/-- Theorem stating that the initial number of cards is equal to the sum of cards given and cards left -/
theorem initial_cards_eq_sum : initial_cards = cards_given + cards_left := by
  sorry

end NUMINAMATH_CALUDE_initial_cards_eq_sum_l2171_217197


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2171_217131

/-- The shaded area of a square with six inscribed circles --/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_diameter : ℝ) :
  square_side = 24 →
  circle_diameter = 8 →
  let square_area := square_side ^ 2
  let circle_area := π * (circle_diameter / 2) ^ 2
  let total_circles_area := 6 * circle_area
  let shaded_area := square_area - total_circles_area
  shaded_area = 576 - 96 * π := by
  sorry

#check shaded_area_square_with_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l2171_217131


namespace NUMINAMATH_CALUDE_incorrect_statement_l2171_217139

def U : Finset Nat := {1, 2, 3, 4}
def M : Finset Nat := {1, 2}
def N : Finset Nat := {2, 4}

theorem incorrect_statement : M ∩ (U \ N) ≠ {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_incorrect_statement_l2171_217139


namespace NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_for_ln_l2171_217183

theorem sqrt_necessary_not_sufficient_for_ln :
  (∀ x y, x > 0 ∧ y > 0 → (Real.log x > Real.log y → Real.sqrt x > Real.sqrt y)) ∧
  (∃ x y, Real.sqrt x > Real.sqrt y ∧ ¬(Real.log x > Real.log y)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_necessary_not_sufficient_for_ln_l2171_217183


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l2171_217182

theorem solution_set_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l2171_217182


namespace NUMINAMATH_CALUDE_square_of_negative_square_l2171_217136

theorem square_of_negative_square (a : ℝ) : (-a^2)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_square_l2171_217136


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l2171_217110

/-- Given a triangle ABC, prove that (sin A + sin B + sin C) / (sin A * sin B * sin C) ≥ 4,
    with equality if and only if the triangle is equilateral -/
theorem triangle_sine_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :
  (Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) ≥ 4 ∧
  ((Real.sin A + Real.sin B + Real.sin C) / (Real.sin A * Real.sin B * Real.sin C) = 4 ↔ A = B ∧ B = C) :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l2171_217110


namespace NUMINAMATH_CALUDE_inverse_function_range_l2171_217105

/-- Given a function f and its inverse, prove the range of a -/
theorem inverse_function_range (a : ℝ) (f : ℝ → ℝ) (f_inv : ℝ → ℝ) : 
  (∀ x, f x = a^(x+1) - 2) →
  (a > 1) →
  (∀ x, f_inv (f x) = x) →
  (∀ x, x ≤ 0 → f_inv x ≤ 0) →
  a ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_range_l2171_217105


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_l2171_217193

theorem binomial_coefficient_1000 : 
  (Nat.choose 1000 1000 = 1) ∧ (Nat.choose 1000 999 = 1000) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_l2171_217193


namespace NUMINAMATH_CALUDE_max_d_value_l2171_217159

def is_prime (n : ℕ) : Prop := sorry

def max_list (l : List ℕ) : ℕ := sorry

def min_list (l : List ℕ) : ℕ := sorry

theorem max_d_value (a b c : ℕ) : 
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  is_prime (a + b - c) ∧ is_prime (a + c - b) ∧ is_prime (b + c - a) ∧ is_prime (a + b + c) ∧
  (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
  b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
  c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
  (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
  (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
  (b + c - a) ≠ (a + b + c) →
  max_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] - 
  min_list [a, b, c, a + b - c, a + c - b, b + c - a, a + b + c] ≤ 
  max_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] - 
  min_list [3, 797, c, 800 - c, 3 + c, 797 + c, 800 + c] := by
sorry

end NUMINAMATH_CALUDE_max_d_value_l2171_217159


namespace NUMINAMATH_CALUDE_alloy_mixture_l2171_217109

theorem alloy_mixture (first_alloy_chromium_percent : Real)
                      (second_alloy_chromium_percent : Real)
                      (first_alloy_weight : Real)
                      (new_alloy_chromium_percent : Real)
                      (h1 : first_alloy_chromium_percent = 0.15)
                      (h2 : second_alloy_chromium_percent = 0.08)
                      (h3 : first_alloy_weight = 15)
                      (h4 : new_alloy_chromium_percent = 0.101) :
  ∃ (second_alloy_weight : Real),
    second_alloy_weight = 35 ∧
    new_alloy_chromium_percent * (first_alloy_weight + second_alloy_weight) =
    first_alloy_chromium_percent * first_alloy_weight +
    second_alloy_chromium_percent * second_alloy_weight :=
by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_l2171_217109


namespace NUMINAMATH_CALUDE_custard_combinations_l2171_217124

theorem custard_combinations (flavors : ℕ) (toppings : ℕ) 
  (h1 : flavors = 5) (h2 : toppings = 7) :
  flavors * (toppings.choose 2) = 105 := by
  sorry

end NUMINAMATH_CALUDE_custard_combinations_l2171_217124
