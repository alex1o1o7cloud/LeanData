import Mathlib

namespace probability_six_consecutive_heads_l1402_140297

/-- The number of ways to get at least 6 consecutive heads in 9 coin flips -/
def consecutiveHeadsCount : ℕ := 49

/-- The total number of possible outcomes when flipping a coin 9 times -/
def totalOutcomes : ℕ := 512

/-- A fair coin is flipped 9 times. This theorem states that the probability
    of getting at least 6 consecutive heads is 49/512. -/
theorem probability_six_consecutive_heads :
  (consecutiveHeadsCount : ℚ) / totalOutcomes = 49 / 512 := by
  sorry

end probability_six_consecutive_heads_l1402_140297


namespace principal_calculation_l1402_140215

def interest_rates : List ℚ := [6/100, 75/1000, 8/100, 85/1000, 9/100]

theorem principal_calculation (total_interest : ℚ) (rates : List ℚ) :
  total_interest = 6016.75 ∧ rates = interest_rates →
  (total_interest / rates.sum) = 15430 := by
  sorry

end principal_calculation_l1402_140215


namespace quadratic_inequality_range_l1402_140261

theorem quadratic_inequality_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end quadratic_inequality_range_l1402_140261


namespace egg_price_per_dozen_l1402_140257

/-- Calculates the price per dozen eggs given the number of hens, eggs laid per hen per week,
    number of weeks, and total revenue. -/
theorem egg_price_per_dozen 
  (num_hens : ℕ) 
  (eggs_per_hen_per_week : ℕ) 
  (num_weeks : ℕ) 
  (total_revenue : ℚ) : 
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  num_weeks = 4 →
  total_revenue = 120 →
  (total_revenue / (↑(num_hens * eggs_per_hen_per_week * num_weeks) / 12)) = 3 :=
by sorry

end egg_price_per_dozen_l1402_140257


namespace square_hexagon_side_ratio_l1402_140225

theorem square_hexagon_side_ratio :
  ∀ (s_s s_h : ℝ),
  s_s > 0 → s_h > 0 →
  s_s^2 = (3 * s_h^2 * Real.sqrt 3) / 2 →
  s_s / s_h = Real.sqrt ((3 * Real.sqrt 3) / 2) :=
by
  sorry

end square_hexagon_side_ratio_l1402_140225


namespace marco_coins_l1402_140247

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five_cent : ℕ
  ten_cent : ℕ
  twenty_cent : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def different_values (coins : CoinCounts) : ℕ :=
  59 - 3 * coins.five_cent - 2 * coins.ten_cent

theorem marco_coins :
  ∀ (coins : CoinCounts),
    coins.five_cent + coins.ten_cent + coins.twenty_cent = 15 →
    different_values coins = 28 →
    coins.twenty_cent = 4 := by
  sorry

end marco_coins_l1402_140247


namespace symmetric_points_sum_power_l1402_140270

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- The problem statement -/
theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b →
  (a + b)^2012 = 1 := by
  sorry

end symmetric_points_sum_power_l1402_140270


namespace brendan_tax_payment_l1402_140246

/-- Calculates the weekly tax payment for a waiter named Brendan --/
def brendan_weekly_tax (hourly_wage : ℝ) (shift_hours : List ℝ) (tip_per_hour : ℝ) (tax_rate : ℝ) (tip_report_ratio : ℝ) : ℝ :=
  let total_hours := shift_hours.sum
  let wage_income := hourly_wage * total_hours
  let total_tips := tip_per_hour * total_hours
  let reported_tips := total_tips * tip_report_ratio
  let reported_income := wage_income + reported_tips
  reported_income * tax_rate

/-- Theorem stating that Brendan's weekly tax payment is $56 --/
theorem brendan_tax_payment :
  brendan_weekly_tax 6 [8, 8, 12] 12 0.2 (1/3) = 56 := by
  sorry

end brendan_tax_payment_l1402_140246


namespace union_equals_universal_l1402_140279

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def M : Set ℕ := {3, 4, 5, 7}
def N : Set ℕ := {2, 4, 5, 6}

theorem union_equals_universal : M ∪ N = U := by
  sorry

end union_equals_universal_l1402_140279


namespace total_apples_picked_l1402_140214

/-- Given that Benny picked 2 apples and Dan picked 9 apples, 
    prove that the total number of apples picked is 11. -/
theorem total_apples_picked (benny_apples dan_apples : ℕ) 
  (h1 : benny_apples = 2) (h2 : dan_apples = 9) : 
  benny_apples + dan_apples = 11 := by
  sorry

end total_apples_picked_l1402_140214


namespace simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l1402_140237

-- Part I
theorem simplify_trigonometric_expression_I :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by sorry

-- Part II
theorem simplify_trigonometric_expression_II (α : Real) (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by sorry

end simplify_trigonometric_expression_I_simplify_trigonometric_expression_II_l1402_140237


namespace inequality_implies_a_nonpositive_l1402_140271

theorem inequality_implies_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, x ∈ [1, 2] → 4^x - 2^(x+1) - a ≥ 0) →
  a ≤ 0 := by
  sorry

end inequality_implies_a_nonpositive_l1402_140271


namespace only_set2_forms_triangle_l1402_140244

-- Define a structure for a set of three line segments
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle inequality check
def satisfiesTriangleInequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

-- Define the given sets of line segments
def set1 : LineSegmentSet := ⟨1, 2, 3⟩
def set2 : LineSegmentSet := ⟨2, 3, 4⟩
def set3 : LineSegmentSet := ⟨4, 4, 8⟩
def set4 : LineSegmentSet := ⟨5, 6, 12⟩

-- Theorem stating that only set2 satisfies the triangle inequality
theorem only_set2_forms_triangle :
  ¬(satisfiesTriangleInequality set1) ∧
  (satisfiesTriangleInequality set2) ∧
  ¬(satisfiesTriangleInequality set3) ∧
  ¬(satisfiesTriangleInequality set4) :=
sorry

end only_set2_forms_triangle_l1402_140244


namespace distinct_weights_count_l1402_140264

def weights : List ℕ := [1, 2, 3, 4]

def possible_combinations (weights : List ℕ) : List (List ℕ) :=
  sorry

def distinct_weights (combinations : List (List ℕ)) : List ℕ :=
  sorry

theorem distinct_weights_count :
  weights.length = 4 →
  (distinct_weights (possible_combinations weights)).length = 10 :=
by sorry

end distinct_weights_count_l1402_140264


namespace book_sale_problem_l1402_140254

theorem book_sale_problem (total_cost book1_cost book2_cost selling_price : ℚ) :
  total_cost = 300 ∧
  book1_cost + book2_cost = total_cost ∧
  selling_price = book1_cost * (1 - 15/100) ∧
  selling_price = book2_cost * (1 + 19/100) →
  book1_cost = 175 := by
  sorry

end book_sale_problem_l1402_140254


namespace unique_solution_lcm_gcd_equation_l1402_140262

theorem unique_solution_lcm_gcd_equation :
  ∃! (n : ℕ), n > 0 ∧ Nat.lcm n 60 = 2 * Nat.gcd n 60 + 300 :=
by sorry

end unique_solution_lcm_gcd_equation_l1402_140262


namespace hidden_face_sum_l1402_140243

/-- Represents a standard six-sided die -/
def Die := Fin 6

/-- The sum of all numbers on a standard die -/
def dieTotalSum : ℕ := 21

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [2, 2, 3, 3, 4, 5, 6]

/-- The number of dice stacked -/
def numberOfDice : ℕ := 3

theorem hidden_face_sum :
  (numberOfDice * dieTotalSum) - visibleNumbers.sum = 38 := by
  sorry

end hidden_face_sum_l1402_140243


namespace election_vote_count_l1402_140235

theorem election_vote_count 
  (total_votes : ℕ) 
  (candidate_a_votes : ℕ) 
  (candidate_b_votes : ℕ) :
  (candidate_a_votes = candidate_b_votes + (15 * total_votes) / 100) →
  (candidate_b_votes = 3159) →
  ((80 * total_votes) / 100 = candidate_a_votes + candidate_b_votes) →
  (total_votes = 9720) :=
by sorry

end election_vote_count_l1402_140235


namespace f_composition_value_l1402_140228

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

-- State the theorem
theorem f_composition_value : f (f (1/3)) = 1/2 := by sorry

end f_composition_value_l1402_140228


namespace product_selection_l1402_140229

/-- Given 12 products with 10 genuine and 2 defective, prove the following:
    (1) The number of ways to select 3 products is 220.
    (2) The number of ways to select exactly 1 defective product out of 3 is 90.
    (3) The number of ways to select at least 1 defective product out of 3 is 100. -/
theorem product_selection (total : Nat) (genuine : Nat) (defective : Nat) (select : Nat)
    (h1 : total = 12)
    (h2 : genuine = 10)
    (h3 : defective = 2)
    (h4 : select = 3)
    (h5 : total = genuine + defective) :
    (Nat.choose total select = 220) ∧
    (Nat.choose defective 1 * Nat.choose genuine 2 = 90) ∧
    (Nat.choose total select - Nat.choose genuine select = 100) :=
  sorry

end product_selection_l1402_140229


namespace angelina_speed_to_library_l1402_140217

-- Define the distances
def home_to_grocery : ℝ := 150
def grocery_to_gym : ℝ := 200
def gym_to_park : ℝ := 250
def park_to_library : ℝ := 300

-- Define Angelina's initial speed
def v : ℝ := 5

-- Define the theorem
theorem angelina_speed_to_library :
  let time_home_to_grocery := home_to_grocery / v
  let time_grocery_to_gym := grocery_to_gym / (2 * v)
  let time_gym_to_park := gym_to_park / (v / 2)
  let time_park_to_library := park_to_library / (6 * v)
  time_grocery_to_gym = time_home_to_grocery - 10 ∧
  time_gym_to_park = time_park_to_library + 20 →
  6 * v = 30 := by
  sorry

end angelina_speed_to_library_l1402_140217


namespace translated_quadratic_increases_l1402_140272

/-- Original quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- Translated quadratic function -/
def g (x : ℝ) : ℝ := f (x - 2)

/-- Theorem stating that the translated function increases for x < 2 -/
theorem translated_quadratic_increases (x1 x2 : ℝ) 
  (h1 : x1 < 2) (h2 : x2 < 2) (h3 : x1 < x2) : 
  g x1 < g x2 := by
  sorry

end translated_quadratic_increases_l1402_140272


namespace project_completion_time_project_completion_time_solution_l1402_140299

/-- Represents the project completion time problem -/
theorem project_completion_time 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (additional_workers : ℕ) 
  (efficiency_improvement : ℚ) : ℕ :=
  let total_work := initial_workers * initial_days
  let new_workers := initial_workers + additional_workers
  let new_efficiency := 1 + efficiency_improvement
  let new_daily_work := new_workers * new_efficiency
  ⌊(total_work / new_daily_work : ℚ)⌋₊
    
/-- The solution to the specific problem instance -/
theorem project_completion_time_solution :
  project_completion_time 10 20 5 (1/10) = 12 := by
  sorry

end project_completion_time_project_completion_time_solution_l1402_140299


namespace rectangle_ratio_l1402_140253

theorem rectangle_ratio (w : ℝ) : 
  w > 0 ∧ 2 * (w + 10) = 30 → w / 10 = 1 / 2 := by
  sorry

end rectangle_ratio_l1402_140253


namespace middle_number_proof_l1402_140268

theorem middle_number_proof (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
    (h3 : a + b = 15) (h4 : a + c = 18) (h5 : b + c = 21) : b = 9 := by
  sorry

end middle_number_proof_l1402_140268


namespace min_value_quadratic_l1402_140296

theorem min_value_quadratic (x y : ℝ) (h : x + y = 5) :
  x^2 - x*y + y^2 ≥ 25/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ x₀^2 - x₀*y₀ + y₀^2 = 25/4 :=
by sorry

end min_value_quadratic_l1402_140296


namespace max_garden_area_l1402_140226

/-- Represents a rectangular garden with one side bounded by a house. -/
structure Garden where
  width : ℝ
  length : ℝ

/-- The total fencing available -/
def total_fencing : ℝ := 500

/-- Calculates the area of the garden -/
def garden_area (g : Garden) : ℝ := g.width * g.length

/-- Calculates the amount of fencing used for three sides of the garden -/
def fencing_used (g : Garden) : ℝ := g.length + 2 * g.width

/-- Theorem stating the maximum area of the garden -/
theorem max_garden_area :
  ∃ (g : Garden), fencing_used g = total_fencing ∧
    ∀ (h : Garden), fencing_used h = total_fencing → garden_area h ≤ garden_area g ∧
    garden_area g = 31250 := by
  sorry

end max_garden_area_l1402_140226


namespace glorys_favorite_number_l1402_140242

theorem glorys_favorite_number (glory misty : ℕ) : 
  misty = glory / 3 →
  misty + glory = 600 →
  glory = 450 := by sorry

end glorys_favorite_number_l1402_140242


namespace solution_pairs_l1402_140213

theorem solution_pairs : 
  {(x, y) : ℝ × ℝ | (x^2 + y + 1) * (y^2 + x + 1) = 4 ∧ (x^2 + y)^2 + (y^2 + x)^2 = 2} = 
  {(0, 1), (1, 0), ((Real.sqrt 5 - 1) / 2, (Real.sqrt 5 - 1) / 2), 
   (-(Real.sqrt 5 + 1) / 2, -(Real.sqrt 5 + 1) / 2)} := by
  sorry

end solution_pairs_l1402_140213


namespace students_in_diligence_before_transfer_l1402_140245

theorem students_in_diligence_before_transfer 
  (total_students : ℕ) 
  (transferred_students : ℕ) 
  (h1 : total_students = 50)
  (h2 : transferred_students = 2)
  (h3 : ∃ (x : ℕ), x + transferred_students = total_students - x) :
  ∃ (initial_diligence : ℕ), initial_diligence = (total_students / 2) - transferred_students :=
sorry

end students_in_diligence_before_transfer_l1402_140245


namespace triangle_side_length_l1402_140201

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that when a = 2, b = 3, and angle C = 60°, the length of side c (AB) is √7. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b = 3 →
  C = Real.pi / 3 →  -- 60° in radians
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt 7 := by
  sorry

end triangle_side_length_l1402_140201


namespace exists_permutation_satisfying_average_condition_l1402_140285

/-- A permutation of the first n natural numbers. -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- Predicate to check if a permutation satisfies the average condition. -/
def SatisfiesAverageCondition (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i j k : Fin n, i < j → j < k →
    (p i).val + (p k).val ≠ 2 * (p j).val

/-- Theorem stating that for any n, there exists a permutation satisfying the average condition. -/
theorem exists_permutation_satisfying_average_condition (n : ℕ) :
  ∃ p : Permutation n, SatisfiesAverageCondition n p :=
sorry

end exists_permutation_satisfying_average_condition_l1402_140285


namespace initial_bushes_count_l1402_140283

/-- The number of orchid bushes planted today -/
def bushes_planted_today : ℕ := 37

/-- The number of orchid bushes planted tomorrow -/
def bushes_planted_tomorrow : ℕ := 25

/-- The total number of orchid bushes after planting -/
def total_bushes_after_planting : ℕ := 109

/-- The number of workers who finished the planting -/
def number_of_workers : ℕ := 35

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := total_bushes_after_planting - (bushes_planted_today + bushes_planted_tomorrow)

theorem initial_bushes_count : initial_bushes = 47 := by
  sorry

end initial_bushes_count_l1402_140283


namespace simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l1402_140233

-- 1. Simplify complex fraction
theorem simplify_complex_fraction (x y : ℝ) (h : y ≠ x) :
  (1 + x / y) / (1 - x / y) = (y + x) / (y - x) := by sorry

-- 2. Simplify resistance formula
theorem simplify_resistance_formula (R R₁ R₂ : ℝ) (h₁ : R₁ > 0) (h₂ : R₂ > 0) :
  (1 / R = 1 / R₁ + 1 / R₂) → R = (R₁ * R₂) / (R₁ + R₂) := by sorry

-- 3. Compare time taken
theorem compare_time_taken (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  x / (1 / (1 / y + 1 / z)) = (x * y + x * z) / (y * z) := by sorry

end simplify_complex_fraction_simplify_resistance_formula_compare_time_taken_l1402_140233


namespace triangle_area_l1402_140208

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  (1/2 : ℝ) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_area_l1402_140208


namespace equation_solution_l1402_140258

theorem equation_solution : 
  ∃ x : ℝ, (((1 - x) / (x - 4)) + (1 / (4 - x)) = 1) ∧ (x = 2) := by
  sorry

end equation_solution_l1402_140258


namespace distance_between_first_and_last_trees_l1402_140224

/-- Given 30 trees along a straight road with 3 meters between each adjacent pair of trees,
    the distance between the first and last trees is 87 meters. -/
theorem distance_between_first_and_last_trees (num_trees : ℕ) (distance_between_trees : ℝ) :
  num_trees = 30 →
  distance_between_trees = 3 →
  (num_trees - 1) * distance_between_trees = 87 := by
  sorry

end distance_between_first_and_last_trees_l1402_140224


namespace classroom_notebooks_l1402_140277

theorem classroom_notebooks :
  ∀ (x : ℕ),
    (28 : ℕ) / 2 * x + (28 : ℕ) / 2 * 3 = 112 →
    x = 5 := by
  sorry

end classroom_notebooks_l1402_140277


namespace sixth_quiz_score_for_target_mean_l1402_140273

def quiz_scores : List ℕ := [92, 96, 87, 89, 100]
def target_mean : ℕ := 94
def num_quizzes : ℕ := 6

theorem sixth_quiz_score_for_target_mean :
  ∃ (x : ℕ), (quiz_scores.sum + x) / num_quizzes = target_mean ∧ x = 100 := by
sorry

end sixth_quiz_score_for_target_mean_l1402_140273


namespace inequality_proof_l1402_140248

theorem inequality_proof (a b c : ℝ) 
  (h1 : 4 * a * c - b^2 ≥ 0) 
  (h2 : a > 0) : 
  a + c - Real.sqrt ((a - c)^2 + b^2) ≤ (4 * a * c - b^2) / (2 * a) ∧ 
  (a + c - Real.sqrt ((a - c)^2 + b^2) = (4 * a * c - b^2) / (2 * a) ↔ 
    (b = 0 ∧ a ≥ c) ∨ 4 * a * c = b^2) :=
by sorry

end inequality_proof_l1402_140248


namespace nn_plus_one_prime_l1402_140211

theorem nn_plus_one_prime (n : ℕ) : n ∈ Finset.range 16 \ {0} →
  Nat.Prime (n^n + 1) ↔ n = 1 ∨ n = 2 ∨ n = 4 := by
  sorry

end nn_plus_one_prime_l1402_140211


namespace max_stores_visited_is_four_l1402_140293

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ
  total_visits : ℕ

/-- The maximum number of stores visited by any individual -/
def max_stores_visited (s : ShoppingScenario) : ℕ :=
  let remaining_visits := s.total_visits - 2 * s.two_store_visitors
  let remaining_shoppers := s.num_shoppers - s.two_store_visitors
  let extra_visits := remaining_visits - remaining_shoppers
  1 + extra_visits

/-- Theorem stating the maximum number of stores visited by any individual in the given scenario -/
theorem max_stores_visited_is_four (s : ShoppingScenario) :
  s.num_stores = 8 ∧ 
  s.num_shoppers = 12 ∧ 
  s.two_store_visitors = 8 ∧ 
  s.total_visits = 23 →
  max_stores_visited s = 4 :=
by
  sorry

#eval max_stores_visited {num_stores := 8, num_shoppers := 12, two_store_visitors := 8, total_visits := 23}

end max_stores_visited_is_four_l1402_140293


namespace no_solution_for_seven_power_plus_cube_divisible_by_nine_l1402_140295

theorem no_solution_for_seven_power_plus_cube_divisible_by_nine :
  ∀ n : ℕ, n ≥ 1 → ¬(9 ∣ 7^n + n^3) :=
by sorry

end no_solution_for_seven_power_plus_cube_divisible_by_nine_l1402_140295


namespace division_of_fractions_l1402_140222

theorem division_of_fractions : 
  (-1/24) / ((1/3) - (1/6) + (3/8)) = -1/13 := by sorry

end division_of_fractions_l1402_140222


namespace trapezoid_de_length_l1402_140232

/-- Represents a trapezoid ABCD formed by a rectangle ABCE and a right triangle EDF -/
structure Trapezoid where
  /-- Length of side AB of the rectangle -/
  ab : ℝ
  /-- Length of side BC of the rectangle -/
  bc : ℝ
  /-- Length of side DE of the trapezoid -/
  de : ℝ
  /-- Length of side EF of the triangle -/
  ef : ℝ
  /-- Condition that AB = 7 -/
  ab_eq : ab = 7
  /-- Condition that BC = 8 -/
  bc_eq : bc = 8
  /-- Condition that DE is twice EF -/
  de_twice_ef : de = 2 * ef
  /-- Condition that the areas of the rectangle and triangle are equal -/
  areas_equal : ab * bc = (1 / 2) * de * ef

/-- Theorem stating that the length of DE in the described trapezoid is 4√14 -/
theorem trapezoid_de_length (t : Trapezoid) : t.de = 4 * Real.sqrt 14 := by
  sorry

end trapezoid_de_length_l1402_140232


namespace rectangle_area_l1402_140227

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let perimeter := 2 * l + 2 * w
  perimeter = 4 → w * l = 8 / 9 := by
sorry

end rectangle_area_l1402_140227


namespace geometric_sequence_sixth_term_l1402_140259

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_2 : a 2 = 5)
  (h_4 : a 4 = 20) :
  a 6 = 80 := by
sorry

end geometric_sequence_sixth_term_l1402_140259


namespace simplify_fraction_l1402_140276

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  x / (x - 1)^2 - 1 / (x - 1)^2 = 1 / (x - 1) := by sorry

end simplify_fraction_l1402_140276


namespace binomial_coefficient_relation_l1402_140280

theorem binomial_coefficient_relation (n : ℕ) : 
  (Nat.choose n 3 = 7 * Nat.choose n 1) → n = 8 := by
  sorry

end binomial_coefficient_relation_l1402_140280


namespace smallest_x_abs_equation_l1402_140231

theorem smallest_x_abs_equation : ∃ x : ℝ, x = -7 ∧ 
  (∀ y : ℝ, |4*y + 8| = 20 → y ≥ x) ∧ |4*x + 8| = 20 := by
  sorry

end smallest_x_abs_equation_l1402_140231


namespace proposition_p_true_q_false_l1402_140275

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define a triangle
structure Triangle :=
(A B C : ℝ)
(angle_sum : A + B + C = π)
(positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

theorem proposition_p_true_q_false :
  (∀ x : ℝ, 0 < x → x < 1 → lg (x * (1 - x) + 1) > 0) ∧
  (∃ t : Triangle, t.A > t.B ∧ Real.cos (t.A / 2)^2 ≥ Real.cos (t.B / 2)^2) :=
by sorry

end proposition_p_true_q_false_l1402_140275


namespace min_sum_squares_roots_l1402_140286

/-- For a quadratic equation x^2 + (2-p)x - p - 3 = 0, 
    the sum of the squares of its roots is minimized when p = 1 -/
theorem min_sum_squares_roots (p : ℝ) : 
  let f : ℝ → ℝ := λ p => p^2 - 2*p + 10
  ∀ q : ℝ, f p ≥ f 1 := by
  sorry

end min_sum_squares_roots_l1402_140286


namespace min_value_of_expression_l1402_140281

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 9 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def Circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def Circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have three shared tangents -/
axiom three_shared_tangents (a b : ℝ) : ∃ (t1 t2 t3 : ℝ × ℝ → ℝ), 
  (∀ x y, Circle1 a x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t3 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t3 (x, y) = 0)

/-- The theorem to be proved -/
theorem min_value_of_expression (a b : ℝ) (h : a * b ≠ 0) : 
  (∃ (x y : ℝ), Circle1 a x y) → 
  (∃ (x y : ℝ), Circle2 b x y) → 
  (∀ c, c ≥ 1 → 4 / a^2 + 1 / b^2 ≥ c) ∧ 
  (∃ a0 b0, a0 * b0 ≠ 0 ∧ 4 / a0^2 + 1 / b0^2 = 1) :=
sorry

end min_value_of_expression_l1402_140281


namespace incircle_radius_of_special_triangle_l1402_140205

-- Define the triangle DEF
structure Triangle :=
  (D E F : ℝ × ℝ)

-- Define properties of the triangle
def is_right_triangle (t : Triangle) : Prop :=
  -- Right angle at F (we don't need to specify this explicitly in Lean)
  true

def angle_D_is_60_degrees (t : Triangle) : Prop :=
  -- Angle D is 60 degrees
  true

def DF_length (t : Triangle) : ℝ :=
  12

-- Define the incircle radius function
noncomputable def incircle_radius (t : Triangle) : ℝ :=
  sorry

-- Theorem statement
theorem incircle_radius_of_special_triangle (t : Triangle) 
  (h1 : is_right_triangle t)
  (h2 : angle_D_is_60_degrees t)
  (h3 : DF_length t = 12) :
  incircle_radius t = 6 * (Real.sqrt 3 - 1) := by
  sorry

end incircle_radius_of_special_triangle_l1402_140205


namespace go_relay_match_sequences_l1402_140204

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the maximum number of matches possible -/
def max_matches : ℕ := 2 * team_size - 1

/-- Represents the number of matches the winning team must win -/
def required_wins : ℕ := team_size

/-- The number of possible match sequences in a Go relay match -/
def match_sequences : ℕ := 2 * (Nat.choose max_matches required_wins)

theorem go_relay_match_sequences :
  match_sequences = 3432 :=
sorry

end go_relay_match_sequences_l1402_140204


namespace difficult_vs_easy_problems_l1402_140263

/-- Represents the number of problems solved by different combinations of students -/
structure ProblemDistribution where
  x₁ : ℕ  -- problems solved only by student 1
  x₂ : ℕ  -- problems solved only by student 2
  x₃ : ℕ  -- problems solved only by student 3
  y₁₂ : ℕ -- problems solved only by students 1 and 2
  y₁₃ : ℕ -- problems solved only by students 1 and 3
  y₂₃ : ℕ -- problems solved only by students 2 and 3
  z : ℕ   -- problems solved by all three students

/-- The main theorem stating the relationship between difficult and easy problems -/
theorem difficult_vs_easy_problems (d : ProblemDistribution) :
  d.x₁ + d.x₂ + d.x₃ + d.y₁₂ + d.y₁₃ + d.y₂₃ + d.z = 100 →
  d.x₁ + d.y₁₂ + d.y₁₃ + d.z = 60 →
  d.x₂ + d.y₁₂ + d.y₂₃ + d.z = 60 →
  d.x₃ + d.y₁₃ + d.y₂₃ + d.z = 60 →
  d.x₁ + d.x₂ + d.x₃ - d.z = 20 :=
by sorry

end difficult_vs_easy_problems_l1402_140263


namespace prob_one_black_in_three_draws_l1402_140218

-- Define the number of balls
def total_balls : ℕ := 6
def black_balls : ℕ := 2
def white_balls : ℕ := 4

-- Define the number of draws
def num_draws : ℕ := 3

-- Define the probability of drawing a black ball
def prob_black : ℚ := black_balls / total_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Define the number of ways to choose 1 draw out of 3
def ways_to_choose : ℕ := 3

-- Theorem to prove
theorem prob_one_black_in_three_draws : 
  ways_to_choose * prob_black * prob_white^2 = 4/9 := by
  sorry

end prob_one_black_in_three_draws_l1402_140218


namespace supplier_A_better_performance_l1402_140294

def supplier_A : List ℕ := [10, 9, 10, 10, 11, 11, 9, 11, 10, 10]
def supplier_B : List ℕ := [8, 10, 14, 7, 10, 11, 10, 8, 15, 12]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (fun x => ((x : ℚ) - m) ^ 2)).sum / l.length

theorem supplier_A_better_performance (A : List ℕ) (B : List ℕ)
  (hA : A = supplier_A) (hB : B = supplier_B) :
  mean A < mean B ∧ variance A < variance B := by
  sorry

end supplier_A_better_performance_l1402_140294


namespace inequality_system_integer_solutions_l1402_140212

theorem inequality_system_integer_solutions :
  ∀ x : ℤ, (x - 3 * (x - 2) ≤ 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ (x = 1 ∨ x = 2 ∨ x = 3) :=
by sorry

end inequality_system_integer_solutions_l1402_140212


namespace cube_surface_area_l1402_140256

-- Define the volume of the cube
def cube_volume : ℝ := 4913

-- Define the surface area we want to prove
def target_surface_area : ℝ := 1734

-- Theorem statement
theorem cube_surface_area :
  let side := (cube_volume ^ (1/3 : ℝ))
  6 * side^2 = target_surface_area := by
  sorry

end cube_surface_area_l1402_140256


namespace initial_typists_count_l1402_140282

/-- The number of typists in the initial group -/
def initial_typists : ℕ := 20

/-- The number of letters typed by the initial group in 20 minutes -/
def letters_20min : ℕ := 40

/-- The number of typists in the second group -/
def second_typists : ℕ := 30

/-- The number of letters typed by the second group in 1 hour -/
def letters_1hour : ℕ := 180

/-- The rate of typing (letters per hour per typist) is consistent between groups -/
axiom typing_rate_consistent : 
  (letters_20min : ℚ) / initial_typists * 3 = (letters_1hour : ℚ) / second_typists

theorem initial_typists_count : initial_typists = 20 := by
  sorry

end initial_typists_count_l1402_140282


namespace exam_score_problem_l1402_140234

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 80)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 120) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * (correct_answers : ℤ) + wrong_score * ((total_questions - correct_answers) : ℤ) = total_score ∧
    correct_answers = 40 := by
  sorry

end exam_score_problem_l1402_140234


namespace max_profit_theorem_additional_cost_range_l1402_140267

/-- Represents the monthly sales and profit model for a product. -/
structure SalesModel where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  price_sensitivity : ℝ
  max_price : ℝ

/-- Calculates the monthly sales volume given a price increase. -/
def sales_volume (model : SalesModel) (price_increase : ℝ) : ℝ :=
  model.initial_sales - model.price_sensitivity * price_increase

/-- Calculates the monthly profit given a price increase. -/
def monthly_profit (model : SalesModel) (price_increase : ℝ) : ℝ :=
  (sales_volume model price_increase) * (model.initial_price + price_increase - model.cost_price)

/-- Theorem stating the maximum monthly profit and optimal selling price. -/
theorem max_profit_theorem (model : SalesModel) 
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  ∃ (x : ℝ), x ∈ Set.Icc 5 6 ∧ 
  ∀ (y : ℝ), y > 0 ∧ y ≤ 15 → monthly_profit model x ≥ monthly_profit model y ∧
  monthly_profit model x = 2400 := by sorry

/-- Theorem stating the range of additional costs. -/
theorem additional_cost_range (model : SalesModel) (a : ℝ)
  (h_cost : model.cost_price = 40)
  (h_initial_price : model.initial_price = 50)
  (h_initial_sales : model.initial_sales = 210)
  (h_price_sensitivity : model.price_sensitivity = 10)
  (h_max_price : model.max_price = 65) :
  (∀ (x y : ℝ), 8 ≤ x ∧ x < y ∧ y ≤ 15 → 
    monthly_profit model x - (sales_volume model x * a) > 
    monthly_profit model y - (sales_volume model y * a)) 
  ↔ 0 < a ∧ a < 6 := by sorry

end max_profit_theorem_additional_cost_range_l1402_140267


namespace parallel_vectors_x_value_l1402_140287

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (-2, 4)
  are_parallel a b → x = -1 := by
sorry

end parallel_vectors_x_value_l1402_140287


namespace four_digit_multiples_of_seven_l1402_140238

theorem four_digit_multiples_of_seven (n : ℕ) : 
  (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 7 = 0) ↔ 
  (n ∈ Finset.range 1286 ∧ ∃ k : ℕ, n = 7 * k + 1001) :=
sorry

end four_digit_multiples_of_seven_l1402_140238


namespace distance_equation_l1402_140265

/-- The distance between the boy's house and school -/
def D : ℝ := sorry

/-- The speed from house to library (km/hr) -/
def speed_to_library : ℝ := 3

/-- The speed from library to school (km/hr) -/
def speed_library_to_school : ℝ := 2.5

/-- The speed from school to house (km/hr) -/
def speed_return : ℝ := 2

/-- The time spent at the library (hours) -/
def library_time : ℝ := 0.5

/-- The total trip time (hours) -/
def total_time : ℝ := 5.5

theorem distance_equation : 
  (D / 2) / speed_to_library + library_time + 
  (D / 2) / speed_library_to_school + 
  D / speed_return = total_time := by sorry

end distance_equation_l1402_140265


namespace sequence_sum_l1402_140266

/-- Given a sequence {a_n} with a₁ = 1 and S_{n+1} = ((n+1)a_n)/n + S_n, 
    prove that S_n = n(n+1)/2 for all positive integers n. -/
theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : 
  a 1 = 1 → 
  (∀ n : ℕ, n > 0 → S (n + 1) = ((n + 1) * a n) / n + S n) → 
  (∀ n : ℕ, n > 0 → S n = n * (n + 1) / 2) := by
  sorry

end sequence_sum_l1402_140266


namespace inequality_equivalence_l1402_140239

theorem inequality_equivalence (x y : ℝ) :
  y - x > Real.sqrt (x^2 + 9) ↔ y > x + Real.sqrt (x^2 + 9) := by
  sorry

end inequality_equivalence_l1402_140239


namespace coin_difference_l1402_140284

/-- Represents the denominations of coins available --/
inductive Coin
  | Five : Coin
  | Ten : Coin
  | Twenty : Coin

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Five => 5
  | Coin.Ten => 10
  | Coin.Twenty => 20

/-- The target amount to be paid in cents --/
def targetAmount : Nat := 40

/-- A function that calculates the minimum number of coins needed --/
def minCoins : Nat := sorry

/-- A function that calculates the maximum number of coins needed --/
def maxCoins : Nat := sorry

/-- Theorem stating the difference between max and min number of coins --/
theorem coin_difference : maxCoins - minCoins = 6 := by sorry

end coin_difference_l1402_140284


namespace pure_imaginary_implies_a_eq_neg_two_l1402_140251

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero. -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The complex number z defined in terms of real number a. -/
def z (a : ℝ) : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)

/-- If z(a) is a pure imaginary number, then a = -2. -/
theorem pure_imaginary_implies_a_eq_neg_two :
  ∀ a : ℝ, is_pure_imaginary (z a) → a = -2 := by sorry

end pure_imaginary_implies_a_eq_neg_two_l1402_140251


namespace three_different_days_probability_three_different_days_probability_value_l1402_140255

/-- The probability of three group members working on exactly three different days in a week -/
theorem three_different_days_probability : ℝ :=
  let total_outcomes := 7^3
  let favorable_outcomes := 7 * 6 * 5
  favorable_outcomes / total_outcomes

/-- The probability of three group members working on exactly three different days in a week is 30/49 -/
theorem three_different_days_probability_value : three_different_days_probability = 30 / 49 := by
  sorry

end three_different_days_probability_three_different_days_probability_value_l1402_140255


namespace books_on_shelf_initial_books_count_l1402_140241

/-- The number of books on the shelf before Marta added more -/
def initial_books : ℕ := sorry

/-- The number of books Marta added to the shelf -/
def books_added : ℕ := 10

/-- The total number of books on the shelf after Marta added more -/
def total_books : ℕ := 48

/-- Theorem stating that the initial number of books plus the added books equals the total books -/
theorem books_on_shelf : initial_books + books_added = total_books := by sorry

/-- Theorem proving that the initial number of books is 38 -/
theorem initial_books_count : initial_books = 38 := by sorry

end books_on_shelf_initial_books_count_l1402_140241


namespace brick_height_l1402_140206

/-- Represents a rectangular solid brick made of unit cubes -/
structure RectangularBrick where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular brick in unit cubes -/
def RectangularBrick.volume (brick : RectangularBrick) : ℕ :=
  brick.length * brick.width * brick.height

/-- The perimeter of the base of a rectangular brick -/
def RectangularBrick.basePerimeter (brick : RectangularBrick) : ℕ :=
  2 * (brick.length + brick.width)

theorem brick_height (brick : RectangularBrick) :
  brick.volume = 42 ∧
  brick.basePerimeter = 18 →
  brick.height = 3 := by
  sorry

end brick_height_l1402_140206


namespace abs_negative_2010_l1402_140200

theorem abs_negative_2010 : |(-2010 : ℤ)| = 2010 := by
  sorry

end abs_negative_2010_l1402_140200


namespace mira_sticker_arrangement_l1402_140219

/-- The number of stickers Mira currently has -/
def current_stickers : ℕ := 31

/-- The number of stickers required in each row -/
def stickers_per_row : ℕ := 7

/-- The function to calculate the number of additional stickers needed -/
def additional_stickers_needed (current : ℕ) (per_row : ℕ) : ℕ :=
  (per_row - (current % per_row)) % per_row

theorem mira_sticker_arrangement :
  additional_stickers_needed current_stickers stickers_per_row = 4 :=
by sorry

end mira_sticker_arrangement_l1402_140219


namespace sin_15_cos_15_l1402_140230

theorem sin_15_cos_15 : Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 4 := by
  sorry

end sin_15_cos_15_l1402_140230


namespace piglet_growth_period_l1402_140278

/-- Represents the problem of determining the growth period for piglets --/
theorem piglet_growth_period (num_piglets : ℕ) (sale_price : ℕ) (feed_cost : ℕ) 
  (num_sold_early : ℕ) (num_sold_late : ℕ) (late_sale_months : ℕ) (total_profit : ℕ) :
  num_piglets = 6 →
  sale_price = 300 →
  feed_cost = 10 →
  num_sold_early = 3 →
  num_sold_late = 3 →
  late_sale_months = 16 →
  total_profit = 960 →
  ∃ x : ℕ, 
    x = 12 ∧
    (num_sold_early * sale_price + num_sold_late * sale_price) - 
    (num_sold_early * feed_cost * x + num_sold_late * feed_cost * late_sale_months) = total_profit :=
by sorry

end piglet_growth_period_l1402_140278


namespace airplane_average_speed_l1402_140220

/-- The average speed of an airplane -/
theorem airplane_average_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 1584) 
  (h2 : time = 24) 
  (h3 : speed = distance / time) : speed = 66 := by
  sorry

end airplane_average_speed_l1402_140220


namespace prime_divisors_of_50_factorial_l1402_140252

theorem prime_divisors_of_50_factorial (n : ℕ) :
  (n = 50) →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end prime_divisors_of_50_factorial_l1402_140252


namespace bag_volume_proof_l1402_140210

/-- The volume of a cuboid-shaped bag -/
def bag_volume (width length height : ℝ) : ℝ := width * length * height

/-- Theorem: The volume of a cuboid-shaped bag with width 9 cm, length 4 cm, and height 7 cm is 252 cm³ -/
theorem bag_volume_proof : bag_volume 9 4 7 = 252 := by
  sorry

end bag_volume_proof_l1402_140210


namespace fraction_equality_l1402_140223

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 8)
  (h2 : r / q = 5)
  (h3 : r / s = 3 / 4) :
  s / p = 5 / 6 := by
  sorry

end fraction_equality_l1402_140223


namespace expansion_coefficient_equation_l1402_140207

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 in the expansion of (√x + a)^6
def coefficient_x_squared (a : ℝ) : ℝ := binomial 6 2 * a^2

-- State the theorem
theorem expansion_coefficient_equation :
  ∃ a : ℝ, coefficient_x_squared a = 60 ∧ (a = 2 ∨ a = -2) :=
sorry

end expansion_coefficient_equation_l1402_140207


namespace bicycle_trip_time_l1402_140250

theorem bicycle_trip_time (adam_speed simon_speed separation_distance : ℝ) 
  (adam_speed_pos : adam_speed > 0)
  (simon_speed_pos : simon_speed > 0)
  (separation_distance_pos : separation_distance > 0)
  (h_adam : adam_speed = 12)
  (h_simon : simon_speed = 9)
  (h_separation : separation_distance = 90) : 
  ∃ t : ℝ, t > 0 ∧ t * (adam_speed ^ 2 + simon_speed ^ 2) ^ (1/2 : ℝ) = separation_distance ∧ t = 6 :=
sorry

end bicycle_trip_time_l1402_140250


namespace min_daily_pages_for_given_plan_l1402_140202

/-- Represents a reading plan for a book -/
structure ReadingPlan where
  total_pages : ℕ
  total_days : ℕ
  initial_days : ℕ
  initial_pages : ℕ

/-- Calculates the minimum pages to read daily for the remaining days -/
def min_daily_pages (plan : ReadingPlan) : ℕ :=
  ((plan.total_pages - plan.initial_pages) + (plan.total_days - plan.initial_days - 1)) / (plan.total_days - plan.initial_days)

/-- Theorem stating the minimum daily pages for the given reading plan -/
theorem min_daily_pages_for_given_plan :
  let plan := ReadingPlan.mk 400 10 5 100
  min_daily_pages plan = 60 := by
  sorry

end min_daily_pages_for_given_plan_l1402_140202


namespace base_9_digits_of_2500_l1402_140203

/-- The number of digits in the base-9 representation of a positive integer -/
def num_digits_base_9 (n : ℕ+) : ℕ :=
  Nat.log 9 n.val + 1

/-- Theorem: The number of digits in the base-9 representation of 2500 is 4 -/
theorem base_9_digits_of_2500 : num_digits_base_9 2500 = 4 := by
  sorry

end base_9_digits_of_2500_l1402_140203


namespace isosceles_triangle_sides_l1402_140260

theorem isosceles_triangle_sides (p a : ℝ) (h1 : p = 14) (h2 : a = 4) :
  (∃ b c : ℝ, (a + b + c = p ∧ (b = c ∨ a = b ∨ a = c)) →
    ((b = 5 ∧ c = 5) ∨ (b = 4 ∧ c = 6) ∨ (b = 6 ∧ c = 4))) :=
by sorry

end isosceles_triangle_sides_l1402_140260


namespace tetrakis_hexahedron_colorings_l1402_140288

/-- The number of faces in a regular tetrakis hexahedron -/
def num_faces : ℕ := 16

/-- The number of available colors -/
def num_colors : ℕ := 12

/-- The order of the rotational symmetry group of a tetrakis hexahedron -/
def symmetry_order : ℕ := 24

/-- Calculates the number of permutations of k items chosen from n items -/
def permutations (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

/-- The number of distinguishable colorings of a tetrakis hexahedron -/
def distinguishable_colorings : ℕ :=
  permutations num_colors (num_faces - 1) / symmetry_order

theorem tetrakis_hexahedron_colorings :
  distinguishable_colorings = 479001600 := by
  sorry

end tetrakis_hexahedron_colorings_l1402_140288


namespace problem_1_problem_2_problem_3_l1402_140216

-- Problem 1
theorem problem_1 : (1 * (-12)) - 5 + (-14) - (-39) = 8 := by sorry

-- Problem 2
theorem problem_2 : (1 : ℚ) / 3 + (-3 / 4) + (-1 / 3) + (-1 / 4) + 18 / 19 = -1 / 19 := by sorry

-- Problem 3
theorem problem_3 : (10 + 1 / 3) + (-11.5) + (-(10 + 1 / 3)) - 4.5 = -16 := by sorry

end problem_1_problem_2_problem_3_l1402_140216


namespace concave_number_probability_l1402_140249

/-- A five-digit natural number formed by digits 0, 1, 2, 3, and 4 -/
def FiveDigitNumber := Fin 5 → Fin 5

/-- Predicate for a "concave number" -/
def IsConcave (n : FiveDigitNumber) : Prop :=
  n 0 > n 1 ∧ n 1 > n 2 ∧ n 2 < n 3 ∧ n 3 < n 4

/-- The set of all possible five-digit numbers -/
def AllNumbers : Finset FiveDigitNumber := sorry

/-- The set of all concave numbers -/
def ConcaveNumbers : Finset FiveDigitNumber := sorry

theorem concave_number_probability :
  (Finset.card ConcaveNumbers : ℚ) / (Finset.card AllNumbers : ℚ) = 23 / 1250 := by sorry

end concave_number_probability_l1402_140249


namespace jamie_grape_juice_theorem_l1402_140291

/-- The amount of grape juice Jamie had at recess -/
def grape_juice_amount (max_liquid bathroom_threshold planned_water milk_amount : ℕ) : ℕ :=
  max_liquid - bathroom_threshold - planned_water - milk_amount

theorem jamie_grape_juice_theorem :
  grape_juice_amount 32 0 8 8 = 16 := by
  sorry

end jamie_grape_juice_theorem_l1402_140291


namespace line_intercepts_l1402_140240

/-- Given a line with equation x - 2y - 2 = 0, prove that its x-intercept is 2 and y-intercept is -1 -/
theorem line_intercepts :
  let line := {(x, y) : ℝ × ℝ | x - 2*y - 2 = 0}
  let x_intercept := {x : ℝ | ∃ y, (x, y) ∈ line ∧ y = 0}
  let y_intercept := {y : ℝ | ∃ x, (x, y) ∈ line ∧ x = 0}
  x_intercept = {2} ∧ y_intercept = {-1} := by
  sorry

end line_intercepts_l1402_140240


namespace parallel_planes_normal_vectors_l1402_140292

/-- Given two planes α and β with normal vectors n1 and n2 respectively,
    prove that if the planes are parallel, then k = 4. -/
theorem parallel_planes_normal_vectors (n1 n2 : ℝ × ℝ × ℝ) (k : ℝ) : 
  n1 = (1, 2, -2) → n2 = (-2, -4, k) → (∃ (c : ℝ), n1 = c • n2) → k = 4 := by
  sorry

end parallel_planes_normal_vectors_l1402_140292


namespace interest_rate_problem_l1402_140290

/-- Prove that for given conditions, the interest rate is 4% --/
theorem interest_rate_problem (P t : ℝ) (diff : ℝ) (h1 : P = 2000) (h2 : t = 2) (h3 : diff = 3.20) :
  ∃ r : ℝ, r = 4 ∧ 
    P * ((1 + r / 100) ^ t - 1) - (P * r * t / 100) = diff :=
by sorry

end interest_rate_problem_l1402_140290


namespace left_seats_count_l1402_140269

/-- Represents the seating configuration of a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusConfig (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeat = 8 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 89 ∧
  bus.leftSeats * bus.seatCapacity + bus.rightSeats * bus.seatCapacity + bus.backSeat = bus.totalCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeats) (h : validBusConfig bus) : bus.leftSeats = 15 := by
  sorry

end left_seats_count_l1402_140269


namespace derivative_at_two_l1402_140221

theorem derivative_at_two (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h_eq : ∀ x, f x = 2 * f (2 - x) - x^2 + 8*x - 8) : 
  deriv f 2 = 4 := by
sorry

end derivative_at_two_l1402_140221


namespace circle_through_AB_with_center_on_line_l1402_140289

-- Define the points A and B
def A : ℝ × ℝ := (2, 4)
def B : ℝ × ℝ := (1, -3)

-- Define the line on which the center lies
def centerLine (x y : ℝ) : Prop := y = x + 3

-- Define the standard form of a circle
def isCircle (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- State the theorem
theorem circle_through_AB_with_center_on_line :
  ∃ (h k r : ℝ), 
    centerLine h k ∧
    isCircle h k r A.1 A.2 ∧
    isCircle h k r B.1 B.2 ∧
    h = -2 ∧ k = 1 ∧ r = 5 := by
  sorry

end circle_through_AB_with_center_on_line_l1402_140289


namespace complex_quotient_plus_modulus_l1402_140209

theorem complex_quotient_plus_modulus :
  let z₁ : ℂ := 2 - I
  let z₂ : ℂ := -I
  z₁ / z₂ + Complex.abs z₂ = 2 + 2 * I := by sorry

end complex_quotient_plus_modulus_l1402_140209


namespace container_capacity_l1402_140274

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 18 →
  final_fill = 0.75 →
  ∃ capacity : Real, 
    capacity * final_fill - capacity * initial_fill = added_water ∧
    capacity = 40 := by
  sorry

end container_capacity_l1402_140274


namespace graces_age_fraction_l1402_140236

theorem graces_age_fraction (mother_age : ℕ) (grace_age : ℕ) :
  mother_age = 80 →
  grace_age = 60 →
  (grace_age : ℚ) / ((2 * mother_age) : ℚ) = 3 / 8 := by
  sorry

end graces_age_fraction_l1402_140236


namespace square_in_M_l1402_140298

/-- The set of functions f: ℝ → ℝ with the property that there exist real numbers a and k (k ≠ 0)
    such that f(a+x) = kf(a-x) for all x ∈ ℝ -/
def M : Set (ℝ → ℝ) :=
  {f | ∃ (a k : ℝ), k ≠ 0 ∧ ∀ x, f (a + x) = k * f (a - x)}

/-- The square function -/
def square : ℝ → ℝ := fun x ↦ x^2

/-- Theorem: The square function belongs to set M -/
theorem square_in_M : square ∈ M := by sorry

end square_in_M_l1402_140298
