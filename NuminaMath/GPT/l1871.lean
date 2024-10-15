import Mathlib

namespace NUMINAMATH_GPT_negation_of_existential_statement_l1871_187115

theorem negation_of_existential_statement : 
  (¬∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_existential_statement_l1871_187115


namespace NUMINAMATH_GPT_sheep_daddy_input_l1871_187179

-- Conditions for black box transformations
def black_box (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k + 1 else k / 2

-- The transformation chain with three black boxes
def black_box_chain (k : ℕ) : ℕ :=
  black_box (black_box (black_box k))

-- Theorem statement capturing the problem:
-- Final output m is 2, and the largest input leading to this is 64.
theorem sheep_daddy_input : ∃ k : ℕ, ∀ (k1 k2 k3 k4 : ℕ), 
  black_box_chain k1 = 2 ∧ 
  black_box_chain k2 = 2 ∧ 
  black_box_chain k3 = 2 ∧ 
  black_box_chain k4 = 2 ∧ 
  k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1 ∧ 
  k = max k1 (max k2 (max k3 k4)) → k = 64 :=
sorry  -- Proof is not required

end NUMINAMATH_GPT_sheep_daddy_input_l1871_187179


namespace NUMINAMATH_GPT_problem_statement_l1871_187180

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem problem_statement :
  ¬ is_pythagorean_triple 2 3 4 ∧ 
  is_pythagorean_triple 3 4 5 ∧ 
  is_pythagorean_triple 6 8 10 ∧ 
  is_pythagorean_triple 5 12 13 :=
by 
  constructor
  sorry
  constructor
  sorry
  constructor
  sorry
  sorry

end NUMINAMATH_GPT_problem_statement_l1871_187180


namespace NUMINAMATH_GPT_number_of_red_balls_l1871_187123

-- Definitions and conditions
def ratio_white_red (w : ℕ) (r : ℕ) : Prop := (w : ℤ) * 3 = 5 * (r : ℤ)
def white_balls : ℕ := 15

-- The theorem to prove
theorem number_of_red_balls (r : ℕ) (h : ratio_white_red white_balls r) : r = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l1871_187123


namespace NUMINAMATH_GPT_inequality_transitive_l1871_187141

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  a + c > b + d :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_transitive_l1871_187141


namespace NUMINAMATH_GPT_smallest_number_bob_l1871_187128

-- Define the conditions given in the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors (x : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ x }

-- The problem statement
theorem smallest_number_bob (b : ℕ) (h1 : prime_factors 30 = prime_factors b) : b = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_bob_l1871_187128


namespace NUMINAMATH_GPT_solve_system_a_l1871_187108

theorem solve_system_a (x y : ℝ) (h1 : x^2 - 3 * x * y - 4 * y^2 = 0) (h2 : x^3 + y^3 = 65) : 
    x = 4 ∧ y = 1 :=
sorry

end NUMINAMATH_GPT_solve_system_a_l1871_187108


namespace NUMINAMATH_GPT_evaluate_sets_are_equal_l1871_187184

theorem evaluate_sets_are_equal :
  (-3^5) = ((-3)^5) ∧
  ¬ ((-2^2) = ((-2)^2)) ∧
  ¬ ((-4 * 2^3) = (-4^2 * 3)) ∧
  ¬ ((- (-3)^2) = (- (-2)^3)) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_sets_are_equal_l1871_187184


namespace NUMINAMATH_GPT_abs_condition_iff_range_l1871_187195

theorem abs_condition_iff_range (x : ℝ) : 
  (|x-1| + |x+2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 2) := 
sorry

end NUMINAMATH_GPT_abs_condition_iff_range_l1871_187195


namespace NUMINAMATH_GPT_root_line_discriminant_curve_intersection_l1871_187151

theorem root_line_discriminant_curve_intersection (a p q : ℝ) :
  (4 * p^3 + 27 * q^2 = 0) ∧ (ap + q + a^3 = 0) →
  (a = 0 ∧ ∀ p q, 4 * p^3 + 27 * q^2 = 0 → ap + q + a^3 = 0 → (p = 0 ∧ q = 0)) ∨
  (a ≠ 0 ∧ (∃ p1 q1 p2 q2, 
             4 * p1^3 + 27 * q1^2 = 0 ∧ ap + q1 + a^3 = 0 ∧ 
             4 * p2^3 + 27 * q2^2 = 0 ∧ ap + q2 + a^3 = 0 ∧ 
             (p1, q1) ≠ (p2, q2))) := 
sorry

end NUMINAMATH_GPT_root_line_discriminant_curve_intersection_l1871_187151


namespace NUMINAMATH_GPT_value_of_x_plus_2y_l1871_187159

theorem value_of_x_plus_2y (x y : ℝ) (h1 : (x + y) / 3 = 1.6666666666666667) (h2 : 2 * x + y = 7) : x + 2 * y = 8 := by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_2y_l1871_187159


namespace NUMINAMATH_GPT_find_a_l1871_187187

noncomputable def a : ℚ := ((68^3 - 65^3) * (32^3 + 18^3)) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2))

theorem find_a : a = 150 := 
  sorry

end NUMINAMATH_GPT_find_a_l1871_187187


namespace NUMINAMATH_GPT_Laran_large_posters_daily_l1871_187124

/-
Problem statement:
Laran has started a poster business. She is selling 5 posters per day at school. Some posters per day are her large posters that sell for $10. The large posters cost her $5 to make. The remaining posters are small posters that sell for $6. They cost $3 to produce. Laran makes a profit of $95 per 5-day school week. How many large posters does Laran sell per day?
-/

/-
Mathematically equivalent proof problem:
Prove that the number of large posters Laran sells per day is 2, given the following conditions:
1) L + S = 5
2) 5L + 3S = 19
-/

variables (L S : ℕ)

-- Given conditions
def condition1 := L + S = 5
def condition2 := 5 * L + 3 * S = 19

-- Prove the desired statement
theorem Laran_large_posters_daily 
    (h1 : condition1 L S) 
    (h2 : condition2 L S) : 
    L = 2 := 
sorry

end NUMINAMATH_GPT_Laran_large_posters_daily_l1871_187124


namespace NUMINAMATH_GPT_average_distance_is_600_l1871_187146

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end NUMINAMATH_GPT_average_distance_is_600_l1871_187146


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1871_187131

-- First problem: Prove x = 4.2 given x + 2x = 12.6
theorem problem1 (x : ℝ) (h1 : x + 2 * x = 12.6) : x = 4.2 :=
  sorry

-- Second problem: Prove x = 2/5 given 1/4 * x + 1/2 = 3/5
theorem problem2 (x : ℚ) (h2 : (1 / 4) * x + 1 / 2 = 3 / 5) : x = 2 / 5 :=
  sorry

-- Third problem: Prove x = 20 given x + 130% * x = 46 (where 130% is 130/100)
theorem problem3 (x : ℝ) (h3 : x + (130 / 100) * x = 46) : x = 20 :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1871_187131


namespace NUMINAMATH_GPT_cost_per_item_l1871_187193

theorem cost_per_item (total_cost : ℝ) (num_items : ℕ) (cost_per_item : ℝ) 
                      (h1 : total_cost = 26) (h2 : num_items = 8) : 
                      cost_per_item = total_cost / num_items := 
by
  sorry

end NUMINAMATH_GPT_cost_per_item_l1871_187193


namespace NUMINAMATH_GPT_base6_add_sub_l1871_187170

theorem base6_add_sub (a b c : ℕ) (ha : a = 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 6 * 6^1 + 5 * 6^0) (hc : c = 1 * 6^1 + 1 * 6^0) :
  (a + b - c) = 1 * 6^3 + 0 * 6^2 + 5 * 6^1 + 3 * 6^0 :=
by
  -- We should translate the problem context into equivalence
  -- but this part of the actual proof is skipped with sorry.
  sorry

end NUMINAMATH_GPT_base6_add_sub_l1871_187170


namespace NUMINAMATH_GPT_scientific_notation_of_42000_l1871_187103

theorem scientific_notation_of_42000 : 42000 = 4.2 * 10^4 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_42000_l1871_187103


namespace NUMINAMATH_GPT_distance_from_home_to_high_school_l1871_187167

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end NUMINAMATH_GPT_distance_from_home_to_high_school_l1871_187167


namespace NUMINAMATH_GPT_totalTaxIsCorrect_l1871_187126

-- Define the different income sources
def dividends : ℝ := 50000
def couponIncomeOFZ : ℝ := 40000
def couponIncomeCorporate : ℝ := 30000
def capitalGain : ℝ := (100 * 200) - (100 * 150)

-- Define the tax rates
def taxRateDividends : ℝ := 0.13
def taxRateCorporateBond : ℝ := 0.13
def taxRateCapitalGain : ℝ := 0.13

-- Calculate the tax for each type of income
def taxOnDividends : ℝ := dividends * taxRateDividends
def taxOnCorporateCoupon : ℝ := couponIncomeCorporate * taxRateCorporateBond
def taxOnCapitalGain : ℝ := capitalGain * taxRateCapitalGain

-- Sum of all tax amounts
def totalTax : ℝ := taxOnDividends + taxOnCorporateCoupon + taxOnCapitalGain

-- Prove that total tax equals the calculated figure
theorem totalTaxIsCorrect : totalTax = 11050 := by
  sorry

end NUMINAMATH_GPT_totalTaxIsCorrect_l1871_187126


namespace NUMINAMATH_GPT_gcd_47_pow6_plus_1_l1871_187144

theorem gcd_47_pow6_plus_1 (h_prime : Prime 47) : 
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_47_pow6_plus_1_l1871_187144


namespace NUMINAMATH_GPT_line_through_point_inequality_l1871_187174

theorem line_through_point_inequality
  (a b θ : ℝ)
  (h : (b * Real.cos θ + a * Real.sin θ = a * b)) :
  1 / a^2 + 1 / b^2 ≥ 1 := 
  sorry

end NUMINAMATH_GPT_line_through_point_inequality_l1871_187174


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1871_187127

theorem problem1 (x : ℝ) : x * x^3 + x^2 * x^2 = 2 * x^4 := sorry
theorem problem2 (p q : ℝ) : (-p * q)^3 = -p^3 * q^3 := sorry
theorem problem3 (a : ℝ) : a^3 * a^4 * a + (a^2)^4 - (-2 * a^4)^2 = -2 * a^8 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1871_187127


namespace NUMINAMATH_GPT_max_constant_k_l1871_187172

theorem max_constant_k (x y : ℤ) : 4 * x^2 + y^2 + 1 ≥ 3 * x * (y + 1) :=
sorry

end NUMINAMATH_GPT_max_constant_k_l1871_187172


namespace NUMINAMATH_GPT_original_price_l1871_187185

theorem original_price (SP : ℝ) (gain_percent : ℝ) (P : ℝ) : SP = 1080 → gain_percent = 0.08 → SP = P * (1 + gain_percent) → P = 1000 :=
by
  intro hSP hGainPercent hEquation
  sorry

end NUMINAMATH_GPT_original_price_l1871_187185


namespace NUMINAMATH_GPT_phyllis_marbles_l1871_187157

theorem phyllis_marbles (num_groups : ℕ) (num_marbles_per_group : ℕ) (h1 : num_groups = 32) (h2 : num_marbles_per_group = 2) : 
  num_groups * num_marbles_per_group = 64 :=
by
  sorry

end NUMINAMATH_GPT_phyllis_marbles_l1871_187157


namespace NUMINAMATH_GPT_pipes_fill_tank_in_one_hour_l1871_187189

theorem pipes_fill_tank_in_one_hour (p q r s : ℝ) (hp : p = 1/2) (hq : q = 1/4) (hr : r = 1/12) (hs : s = 1/6) :
  1 / (p + q + r + s) = 1 :=
by
  sorry

end NUMINAMATH_GPT_pipes_fill_tank_in_one_hour_l1871_187189


namespace NUMINAMATH_GPT_right_triangle_candidate_l1871_187106

theorem right_triangle_candidate :
  (∃ a b c : ℕ, (a, b, c) = (1, 2, 3) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (2, 3, 4) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) ∨
  (∃ a b c : ℕ, (a, b, c) = (4, 5, 6) ∧ a^2 + b^2 = c^2) ↔
  (∃ a b c : ℕ, (a, b, c) = (3, 4, 5) ∧ a^2 + b^2 = c^2) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_candidate_l1871_187106


namespace NUMINAMATH_GPT_find_z_add_inv_y_l1871_187162

theorem find_z_add_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) : z + 1 / y = 5 / 27 := by
  sorry

end NUMINAMATH_GPT_find_z_add_inv_y_l1871_187162


namespace NUMINAMATH_GPT_cubic_intersection_2_points_l1871_187113

theorem cubic_intersection_2_points (c : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^3 - 3*x₁ + c = 0) ∧ (x₂^3 - 3*x₂ + c = 0)) 
  → (c = -2 ∨ c = 2) :=
sorry

end NUMINAMATH_GPT_cubic_intersection_2_points_l1871_187113


namespace NUMINAMATH_GPT_min_value_xy_l1871_187160

-- Defining the operation ⊗
def otimes (a b : ℝ) : ℝ := a * b - a - b

theorem min_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : otimes x y = 3) : 9 ≤ x * y := by
  sorry

end NUMINAMATH_GPT_min_value_xy_l1871_187160


namespace NUMINAMATH_GPT_find_x_l1871_187190

theorem find_x (k : ℝ) (x : ℝ) (h : x ≠ 4) :
  (x = (1 - k) / 2) ↔ ((x^2 - 3 * x - 4) / (x - 4) = 3 * x + k) :=
by sorry

end NUMINAMATH_GPT_find_x_l1871_187190


namespace NUMINAMATH_GPT_train_speed_l1871_187132

theorem train_speed 
  (length_train : ℕ) 
  (time_crossing : ℕ) 
  (speed_kmph : ℕ)
  (h_length : length_train = 120)
  (h_time : time_crossing = 9)
  (h_speed : speed_kmph = 48) : 
  length_train / time_crossing * 3600 / 1000 = speed_kmph := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_l1871_187132


namespace NUMINAMATH_GPT_mother_daughter_age_l1871_187130

theorem mother_daughter_age (x : ℕ) :
  let mother_age := 42
  let daughter_age := 8
  (mother_age + x = 3 * (daughter_age + x)) → x = 9 :=
by
  let mother_age := 42
  let daughter_age := 8
  intro h
  sorry

end NUMINAMATH_GPT_mother_daughter_age_l1871_187130


namespace NUMINAMATH_GPT_median_in_interval_65_69_l1871_187182

-- Definitions for student counts in each interval
def count_50_54 := 5
def count_55_59 := 7
def count_60_64 := 22
def count_65_69 := 19
def count_70_74 := 15
def count_75_79 := 10
def count_80_84 := 18
def count_85_89 := 5

-- Total number of students
def total_students := 101

-- Calculation of the position of the median
def median_position := (total_students + 1) / 2

-- Cumulative counts
def cumulative_up_to_59 := count_50_54 + count_55_59
def cumulative_up_to_64 := cumulative_up_to_59 + count_60_64
def cumulative_up_to_69 := cumulative_up_to_64 + count_65_69

-- Proof statement
theorem median_in_interval_65_69 :
  34 < median_position ∧ median_position ≤ cumulative_up_to_69 :=
by
  sorry

end NUMINAMATH_GPT_median_in_interval_65_69_l1871_187182


namespace NUMINAMATH_GPT_rice_mixed_grain_amount_l1871_187150

theorem rice_mixed_grain_amount (total_rice : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) (proportion : ℚ) 
    (h1 : total_rice = 1536) 
    (h2 : sample_size = 256)
    (h3 : mixed_in_sample = 18)
    (h4 : proportion = mixed_in_sample / sample_size) : 
    total_rice * proportion = 108 :=
  sorry

end NUMINAMATH_GPT_rice_mixed_grain_amount_l1871_187150


namespace NUMINAMATH_GPT_scaling_transformation_l1871_187105

theorem scaling_transformation:
  ∀ (x y x' y': ℝ), 
  (x^2 + y^2 = 1) ∧ (x' = 5 * x) ∧ (y' = 3 * y) → 
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by intros x y x' y'
   sorry

end NUMINAMATH_GPT_scaling_transformation_l1871_187105


namespace NUMINAMATH_GPT_amy_total_score_l1871_187153

theorem amy_total_score :
  let points_per_treasure := 4
  let treasures_first_level := 6
  let treasures_second_level := 2
  let score_first_level := treasures_first_level * points_per_treasure
  let score_second_level := treasures_second_level * points_per_treasure
  let total_score := score_first_level + score_second_level
  total_score = 32 := by
sorry

end NUMINAMATH_GPT_amy_total_score_l1871_187153


namespace NUMINAMATH_GPT_first_ship_rescued_boy_l1871_187173

noncomputable def river_speed : ℝ := 3 -- River speed is 3 km/h

-- Define the speeds of the ships
def ship1_speed_upstream : ℝ := 4 
def ship2_speed_upstream : ℝ := 6 
def ship3_speed_upstream : ℝ := 10 

-- Define the distance downstream where the boy was found
def boy_distance_from_bridge : ℝ := 6

-- Define the equation for the first ship
def first_ship_equation (c : ℝ) : Prop := (10 - c) / (4 + c) = 1 + 6 / c

-- The problem to prove:
theorem first_ship_rescued_boy : first_ship_equation river_speed :=
by sorry

end NUMINAMATH_GPT_first_ship_rescued_boy_l1871_187173


namespace NUMINAMATH_GPT_eval_expression_l1871_187198

def square_avg (a b : ℚ) : ℚ := (a^2 + b^2) / 2
def custom_avg (a b c : ℚ) : ℚ := (a + b + 2 * c) / 3

theorem eval_expression : 
  custom_avg (custom_avg 2 (-1) 1) (square_avg 2 3) 1 = 19 / 6 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1871_187198


namespace NUMINAMATH_GPT_range_for_a_l1871_187154

variable (a : ℝ)

theorem range_for_a (h : ∀ x : ℝ, x^2 + 2 * x + a > 0) : 1 < a := 
sorry

end NUMINAMATH_GPT_range_for_a_l1871_187154


namespace NUMINAMATH_GPT_fraction_of_bread_slices_eaten_l1871_187194

theorem fraction_of_bread_slices_eaten
    (total_slices : ℕ)
    (slices_used_for_sandwich : ℕ)
    (remaining_slices : ℕ)
    (slices_eaten_for_breakfast : ℕ)
    (h1 : total_slices = 12)
    (h2 : slices_used_for_sandwich = 2)
    (h3 : remaining_slices = 6)
    (h4 : total_slices - slices_used_for_sandwich - remaining_slices = slices_eaten_for_breakfast) :
    slices_eaten_for_breakfast / total_slices = 1 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_of_bread_slices_eaten_l1871_187194


namespace NUMINAMATH_GPT_all_xi_equal_l1871_187176

theorem all_xi_equal (P : Polynomial ℤ) (n : ℕ) (hn : n % 2 = 1) (x : Fin n → ℤ) 
  (hP : ∀ i : Fin n, P.eval (x i) = x ⟨i + 1, sorry⟩) : 
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end NUMINAMATH_GPT_all_xi_equal_l1871_187176


namespace NUMINAMATH_GPT_arrangement_meeting_ways_l1871_187175

-- For convenience, define the number of members per school and the combination function.
def num_members_per_school : ℕ := 6
def num_schools : ℕ :=  4
def combination (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_meeting_ways : 
  let host_ways := num_schools
  let host_reps_ways := combination num_members_per_school 2
  let non_host_schools := num_schools - 1
  let non_host_reps_ways := combination num_members_per_school 2
  let total_non_host_reps_ways := non_host_reps_ways ^ non_host_schools
  let total_ways := host_ways * host_reps_ways * total_non_host_reps_ways
  total_ways = 202500 :=
by 
  -- Definitions and computation is deferred to the steps,
  -- which are to be filled during the proof.
  sorry

end NUMINAMATH_GPT_arrangement_meeting_ways_l1871_187175


namespace NUMINAMATH_GPT_sqrt_14400_eq_120_l1871_187116

theorem sqrt_14400_eq_120 : Real.sqrt 14400 = 120 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_14400_eq_120_l1871_187116


namespace NUMINAMATH_GPT_distinct_real_roots_m_range_root_zero_other_root_l1871_187125

open Real

-- Definitions of the quadratic equation and the conditions
def quadratic_eq (m x : ℝ) := x^2 + 2 * (m - 1) * x + m^2 - 1

-- Problem (1)
theorem distinct_real_roots_m_range (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0) → m < 1 :=
by
  sorry

-- Problem (2)
theorem root_zero_other_root (m x : ℝ) :
  (quadratic_eq m 0 = 0 ∧ quadratic_eq m x = 0) → (m = 1 ∧ x = 0) ∨ (m = -1 ∧ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_m_range_root_zero_other_root_l1871_187125


namespace NUMINAMATH_GPT_natural_numbers_partition_l1871_187181

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end NUMINAMATH_GPT_natural_numbers_partition_l1871_187181


namespace NUMINAMATH_GPT_parabola_vertex_parabola_point_condition_l1871_187197

-- Define the parabola function 
def parabola (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- 1. Prove the vertex of the parabola
theorem parabola_vertex (m : ℝ) : ∃ x y, (∀ x m, parabola x m = (x - m)^2 - 1) ∧ (x = m ∧ y = -1) :=
by
  sorry

-- 2. Prove the range of values for m given the conditions on points A and B
theorem parabola_point_condition (m : ℝ) (y1 y2 : ℝ) :
  (y1 > y2) ∧ 
  (parabola (1 - 2*m) m = y1) ∧ 
  (parabola (m + 1) m = y2) → m < 0 ∨ m > 2/3 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_parabola_point_condition_l1871_187197


namespace NUMINAMATH_GPT_max_teams_in_chess_tournament_l1871_187138

theorem max_teams_in_chess_tournament :
  ∃ n : ℕ, n * (n - 1) ≤ 500 / 9 ∧ ∀ m : ℕ, m * (m - 1) ≤ 500 / 9 → m ≤ n :=
sorry

end NUMINAMATH_GPT_max_teams_in_chess_tournament_l1871_187138


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1871_187152

theorem solve_eq1 :
  ∀ x : ℝ, 6 * x - 7 = 4 * x - 5 ↔ x = 1 := by
  intro x
  sorry

theorem solve_eq2 :
  ∀ x : ℝ, 5 * (x + 8) - 5 = 6 * (2 * x - 7) ↔ x = 11 := by
  intro x
  sorry

theorem solve_eq3 :
  ∀ x : ℝ, x - (x - 1) / 2 = 2 - (x + 2) / 5 ↔ x = 11 / 7 := by
  intro x
  sorry

theorem solve_eq4 :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  intro x
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1871_187152


namespace NUMINAMATH_GPT_vertical_asymptote_x_value_l1871_187145

theorem vertical_asymptote_x_value (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_x_value_l1871_187145


namespace NUMINAMATH_GPT_sum_of_series_l1871_187188

noncomputable def infinite_series_sum : ℚ :=
∑' n : ℕ, (3 * (n + 1) - 2) / (((n + 1) : ℚ) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_of_series : infinite_series_sum = 11 / 24 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l1871_187188


namespace NUMINAMATH_GPT_cell_division_after_three_hours_l1871_187107

theorem cell_division_after_three_hours : (2 ^ 6) = 64 := by
  sorry

end NUMINAMATH_GPT_cell_division_after_three_hours_l1871_187107


namespace NUMINAMATH_GPT_find_certain_number_l1871_187140

theorem find_certain_number (h1 : 2994 / 14.5 = 173) (h2 : ∃ x, x / 1.45 = 17.3) : ∃ x, x = 25.085 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_certain_number_l1871_187140


namespace NUMINAMATH_GPT_stratified_sampling_yogurt_adult_milk_powder_sum_l1871_187142

theorem stratified_sampling_yogurt_adult_milk_powder_sum :
  let liquid_milk_brands := 40
  let yogurt_brands := 10
  let infant_formula_brands := 30
  let adult_milk_powder_brands := 20
  let total_brands := liquid_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sample_size := 20
  let yogurt_sample := sample_size * yogurt_brands / total_brands
  let adult_milk_powder_sample := sample_size * adult_milk_powder_brands / total_brands
  yogurt_sample + adult_milk_powder_sample = 6 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_yogurt_adult_milk_powder_sum_l1871_187142


namespace NUMINAMATH_GPT_weightlifting_winner_l1871_187121

theorem weightlifting_winner
  (A B C : ℝ)
  (h1 : A + B = 220)
  (h2 : A + C = 240)
  (h3 : B + C = 250) :
  max A (max B C) = 135 := 
sorry

end NUMINAMATH_GPT_weightlifting_winner_l1871_187121


namespace NUMINAMATH_GPT_machine_p_takes_longer_l1871_187117

variable (MachineP MachineQ MachineA : Type)
variable (s_prockets_per_hr : MachineA → ℝ)
variable (time_produce_s_prockets : MachineP → ℝ → ℝ)

noncomputable def machine_a_production : ℝ := 3
noncomputable def machine_q_production : ℝ := machine_a_production + 0.10 * machine_a_production

noncomputable def machine_q_time : ℝ := 330 / machine_q_production
noncomputable def additional_time : ℝ := sorry -- Since L is undefined

axiom machine_p_time : ℝ
axiom machine_p_time_eq_machine_q_time_plus_additional : machine_p_time = machine_q_time + additional_time

theorem machine_p_takes_longer : machine_p_time > machine_q_time := by
  rw [machine_p_time_eq_machine_q_time_plus_additional]
  exact lt_add_of_pos_right machine_q_time sorry  -- Need the exact L to conclude


end NUMINAMATH_GPT_machine_p_takes_longer_l1871_187117


namespace NUMINAMATH_GPT_benny_turnips_l1871_187169

theorem benny_turnips (M B : ℕ) (h1 : M = 139) (h2 : M = B + 26) : B = 113 := 
by 
  sorry

end NUMINAMATH_GPT_benny_turnips_l1871_187169


namespace NUMINAMATH_GPT_roots_pure_imaginary_if_negative_real_k_l1871_187147

theorem roots_pure_imaginary_if_negative_real_k (k : ℝ) (h_neg : k < 0) :
  (∃ (z : ℂ), 10 * z^2 - 3 * Complex.I * z - (k : ℂ) = 0 ∧ z.im ≠ 0 ∧ z.re = 0) :=
sorry

end NUMINAMATH_GPT_roots_pure_imaginary_if_negative_real_k_l1871_187147


namespace NUMINAMATH_GPT_probability_interval_contains_p_l1871_187118

theorem probability_interval_contains_p (P_A P_B p : ℝ) 
  (hA : P_A = 5 / 6) 
  (hB : P_B = 3 / 4) 
  (hp : p = P_A + P_B - 1) : 
  (5 / 12 ≤ p ∧ p ≤ 3 / 4) :=
by
  -- The proof is skipped by sorry as per the instructions.
  sorry

end NUMINAMATH_GPT_probability_interval_contains_p_l1871_187118


namespace NUMINAMATH_GPT_x_sq_plus_inv_sq_l1871_187111

theorem x_sq_plus_inv_sq (x : ℝ) (h : x + 1/x = 5) : x^2 + 1/x^2 = 23 :=
  sorry

end NUMINAMATH_GPT_x_sq_plus_inv_sq_l1871_187111


namespace NUMINAMATH_GPT_product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l1871_187158

theorem product_of_two_numbers_less_than_the_smaller_of_the_two_factors
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b < min a b := 
sorry

end NUMINAMATH_GPT_product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l1871_187158


namespace NUMINAMATH_GPT_fraction_sum_equals_mixed_number_l1871_187199

theorem fraction_sum_equals_mixed_number :
  (3 / 5 : ℚ) + (2 / 3) + (16 / 15) = (7 / 3) :=
by sorry

end NUMINAMATH_GPT_fraction_sum_equals_mixed_number_l1871_187199


namespace NUMINAMATH_GPT_probability_of_two_green_apples_l1871_187178

theorem probability_of_two_green_apples (total_apples green_apples choose_apples : ℕ)
  (h_total : total_apples = 8)
  (h_green : green_apples = 4)
  (h_choose : choose_apples = 2) 
: (Nat.choose green_apples choose_apples : ℚ) / (Nat.choose total_apples choose_apples) = 3 / 14 := 
by
  -- This part we would provide a proof, but for now we will use sorry
  sorry

end NUMINAMATH_GPT_probability_of_two_green_apples_l1871_187178


namespace NUMINAMATH_GPT_ethan_pages_left_l1871_187119

-- Definitions based on the conditions
def total_pages := 360
def pages_read_morning := 40
def pages_read_night := 10
def pages_read_saturday := pages_read_morning + pages_read_night
def pages_read_sunday := 2 * pages_read_saturday
def total_pages_read := pages_read_saturday + pages_read_sunday

-- Lean 4 statement for the proof problem
theorem ethan_pages_left : total_pages - total_pages_read = 210 := by
  sorry

end NUMINAMATH_GPT_ethan_pages_left_l1871_187119


namespace NUMINAMATH_GPT_find_m_containing_2015_l1871_187136

theorem find_m_containing_2015 : 
  ∃ n : ℕ, ∀ k, 0 ≤ k ∧ k < n → 2015 = n^3 → (1979 + 2*k < 2015 ∧ 2015 < 1979 + 2*k + 2*n) :=
by
  sorry

end NUMINAMATH_GPT_find_m_containing_2015_l1871_187136


namespace NUMINAMATH_GPT_find_length_of_room_l1871_187109

theorem find_length_of_room (width area_existing area_needed : ℕ) (h_width : width = 15) (h_area_existing : area_existing = 16) (h_area_needed : area_needed = 149) :
  (area_existing + area_needed) / width = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_length_of_room_l1871_187109


namespace NUMINAMATH_GPT_total_trail_length_l1871_187102

-- Definitions based on conditions
variables (a b c d e : ℕ)

-- Conditions
def condition1 : Prop := a + b + c = 36
def condition2 : Prop := b + c + d = 48
def condition3 : Prop := c + d + e = 45
def condition4 : Prop := a + d = 31

-- Theorem statement
theorem total_trail_length (h1 : condition1 a b c) (h2 : condition2 b c d) (h3 : condition3 c d e) (h4 : condition4 a d) : 
  a + b + c + d + e = 81 :=
by 
  sorry

end NUMINAMATH_GPT_total_trail_length_l1871_187102


namespace NUMINAMATH_GPT_more_geese_than_ducks_l1871_187149

def mallard_start := 25
def wood_start := 15
def geese_start := 2 * mallard_start - 10
def swan_start := 3 * wood_start + 8

def mallard_after_morning := mallard_start + 4
def wood_after_morning := wood_start + 8
def geese_after_morning := geese_start + 7
def swan_after_morning := swan_start

def mallard_after_noon := mallard_after_morning
def wood_after_noon := wood_after_morning - 6
def geese_after_noon := geese_after_morning - 5
def swan_after_noon := swan_after_morning - 9

def mallard_after_later := mallard_after_noon + 8
def wood_after_later := wood_after_noon + 10
def geese_after_later := geese_after_noon
def swan_after_later := swan_after_noon + 4

def mallard_after_evening := mallard_after_later + 5
def wood_after_evening := wood_after_later + 3
def geese_after_evening := geese_after_later + 15
def swan_after_evening := swan_after_later + 11

def mallard_final := 0
def wood_final := wood_after_evening - (3 / 4 : ℚ) * wood_after_evening
def geese_final := geese_after_evening - (1 / 5 : ℚ) * geese_after_evening
def swan_final := swan_after_evening - (1 / 2 : ℚ) * swan_after_evening

theorem more_geese_than_ducks :
  (geese_final - (mallard_final + wood_final)) = 38 :=
by sorry

end NUMINAMATH_GPT_more_geese_than_ducks_l1871_187149


namespace NUMINAMATH_GPT_correct_conclusions_l1871_187161

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem correct_conclusions :
  (∃ (a b : ℝ), a < b ∧ f a < f b ∧ ∀ x, a < x ∧ x < b → f x < f (x+1)) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = (x₁ - 2012) ∧ f x₂ = (x₂ - 2012)) :=
by
  sorry

end NUMINAMATH_GPT_correct_conclusions_l1871_187161


namespace NUMINAMATH_GPT_leonards_age_l1871_187163

variable (L N J : ℕ)

theorem leonards_age (h1 : L = N - 4) (h2 : N = J / 2) (h3 : L + N + J = 36) : L = 6 := 
by 
  sorry

end NUMINAMATH_GPT_leonards_age_l1871_187163


namespace NUMINAMATH_GPT_relationship_xyz_l1871_187186

theorem relationship_xyz (a b : ℝ) (x y z : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a > b) (hab_sum : a + b = 1) 
  (hx : x = Real.log b / Real.log a)
  (hy : y = Real.log (1 / b) / Real.log a)
  (hz : z = Real.log 3 / Real.log ((1 / a) + (1 / b))) : 
  y < z ∧ z < x := 
sorry

end NUMINAMATH_GPT_relationship_xyz_l1871_187186


namespace NUMINAMATH_GPT_count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l1871_187135

theorem count_of_numbers_less_than_100_divisible_by_2_but_not_by_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

theorem count_of_numbers_less_than_100_divisible_by_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∨ n % 3 = 0) (Finset.range 100)) = 66 :=
sorry

theorem count_of_numbers_less_than_100_not_divisible_by_either_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

end NUMINAMATH_GPT_count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l1871_187135


namespace NUMINAMATH_GPT_find_m_l1871_187112

def A (m : ℝ) : Set ℝ := {3, 4, m^2 - 3 * m - 1}
def B (m : ℝ) : Set ℝ := {2 * m, -3}
def C : Set ℝ := {-3}

theorem find_m (m : ℝ) : A m ∩ B m = C → m = 1 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_find_m_l1871_187112


namespace NUMINAMATH_GPT_manufacturing_section_degrees_l1871_187104

def circle_total_degrees : ℕ := 360
def percentage_to_degree (percentage : ℕ) : ℕ := (circle_total_degrees / 100) * percentage
def manufacturing_percentage : ℕ := 60

theorem manufacturing_section_degrees : percentage_to_degree manufacturing_percentage = 216 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_manufacturing_section_degrees_l1871_187104


namespace NUMINAMATH_GPT_student_B_speed_l1871_187139

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end NUMINAMATH_GPT_student_B_speed_l1871_187139


namespace NUMINAMATH_GPT_find_subtracted_value_l1871_187183

theorem find_subtracted_value (N V : ℕ) (h1 : N = 1376) (h2 : N / 8 - V = 12) : V = 160 :=
by
  sorry

end NUMINAMATH_GPT_find_subtracted_value_l1871_187183


namespace NUMINAMATH_GPT_cans_collected_by_first_group_l1871_187165

def class_total_students : ℕ := 30
def students_didnt_collect : ℕ := 2
def students_collected_4 : ℕ := 13
def total_cans_collected : ℕ := 232

theorem cans_collected_by_first_group :
  let remaining_students := class_total_students - (students_didnt_collect + students_collected_4)
  let cans_by_13_students := students_collected_4 * 4
  let cans_by_first_group := total_cans_collected - cans_by_13_students
  let cans_per_student := cans_by_first_group / remaining_students
  cans_per_student = 12 := by
  sorry

end NUMINAMATH_GPT_cans_collected_by_first_group_l1871_187165


namespace NUMINAMATH_GPT_equivalent_exponentiation_l1871_187168

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_exponentiation_l1871_187168


namespace NUMINAMATH_GPT_log_eq_one_l1871_187110

theorem log_eq_one (log : ℝ → ℝ) (h1 : ∀ a b, log (a ^ b) = b * log a) (h2 : ∀ a b, log (a * b) = log a + log b) :
  (log 5) ^ 2 + log 2 * log 50 = 1 :=
sorry

end NUMINAMATH_GPT_log_eq_one_l1871_187110


namespace NUMINAMATH_GPT_prove_fraction_identity_l1871_187137

-- Define the conditions and the entities involved
variables {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 * x + y / 3 ≠ 0)

-- Formulate the theorem statement
theorem prove_fraction_identity :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ :=
sorry

end NUMINAMATH_GPT_prove_fraction_identity_l1871_187137


namespace NUMINAMATH_GPT_prove_intersection_l1871_187171

-- Defining the set M
def M : Set ℝ := { x | x^2 - 2 * x < 0 }

-- Defining the set N
def N : Set ℝ := { x | x ≥ 1 }

-- Defining the complement of N in ℝ
def complement_N : Set ℝ := { x | x < 1 }

-- The intersection M ∩ complement_N
def intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The statement to be proven
theorem prove_intersection : M ∩ complement_N = intersection :=
by
  sorry

end NUMINAMATH_GPT_prove_intersection_l1871_187171


namespace NUMINAMATH_GPT_infinite_solutions_of_linear_eq_l1871_187196

theorem infinite_solutions_of_linear_eq (a b : ℝ) : 
  (∃ b : ℝ, ∃ a : ℝ, 5 * a - 11 * b = 21) := sorry

end NUMINAMATH_GPT_infinite_solutions_of_linear_eq_l1871_187196


namespace NUMINAMATH_GPT_sin_cos_sum_l1871_187133

/--
Given point P with coordinates (-3, 4) lies on the terminal side of angle α, prove that
sin α + cos α = 1/5.
-/
theorem sin_cos_sum (α : ℝ) (P : ℝ × ℝ) (hx : P = (-3, 4)) :
  Real.sin α + Real.cos α = 1/5 := sorry

end NUMINAMATH_GPT_sin_cos_sum_l1871_187133


namespace NUMINAMATH_GPT_possible_atomic_numbers_l1871_187156

/-
Given the following conditions:
1. An element X is from Group IIA and exhibits a +2 charge.
2. An element Y is from Group VIIA and exhibits a -1 charge.
Prove that the possible atomic numbers for elements X and Y that can form an ionic compound with the formula XY₂ are 12 for X and 9 for Y.
-/

structure Element :=
  (atomic_number : Nat)
  (group : Nat)
  (charge : Int)

def GroupIIACharge := 2
def GroupVIIACharge := -1

axiom X : Element
axiom Y : Element

theorem possible_atomic_numbers (X_group_IIA : X.group = 2)
                                (X_charge : X.charge = GroupIIACharge)
                                (Y_group_VIIA : Y.group = 7)
                                (Y_charge : Y.charge = GroupVIIACharge) :
  (X.atomic_number = 12) ∧ (Y.atomic_number = 9) :=
sorry

end NUMINAMATH_GPT_possible_atomic_numbers_l1871_187156


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l1871_187100

theorem speed_of_boat_in_still_water :
  ∀ (v : ℚ), (33 = (v + 3) * (44 / 60)) → v = 42 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l1871_187100


namespace NUMINAMATH_GPT_not_rain_probability_l1871_187166

-- Define the probability of rain tomorrow
def prob_rain : ℚ := 3 / 10

-- Define the complementary probability (probability that it will not rain tomorrow)
def prob_no_rain : ℚ := 1 - prob_rain

-- Statement to prove: probability that it will not rain tomorrow equals 7/10 
theorem not_rain_probability : prob_no_rain = 7 / 10 := 
by sorry

end NUMINAMATH_GPT_not_rain_probability_l1871_187166


namespace NUMINAMATH_GPT_find_multiple_l1871_187134

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_ineq : (m * n - 15) > 2 * n) : m = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_multiple_l1871_187134


namespace NUMINAMATH_GPT_probability_top_card_is_star_l1871_187148

theorem probability_top_card_is_star :
  let total_cards := 65
  let suits := 5
  let ranks_per_suit := 13
  let star_cards := 13
  (star_cards / total_cards) = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_top_card_is_star_l1871_187148


namespace NUMINAMATH_GPT_sum_of_digits_B_equals_4_l1871_187164

theorem sum_of_digits_B_equals_4 (A B : ℕ) (N : ℕ) (hN : N = 4444 ^ 4444)
    (hA : A = (N.digits 10).sum) (hB : B = (A.digits 10).sum) :
    (B.digits 10).sum = 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_B_equals_4_l1871_187164


namespace NUMINAMATH_GPT_num_ways_to_choose_officers_same_gender_l1871_187177

-- Definitions based on conditions
def num_members : Nat := 24
def num_boys : Nat := 12
def num_girls : Nat := 12
def num_officers : Nat := 3

-- Theorem statement using these definitions
theorem num_ways_to_choose_officers_same_gender :
  (num_boys * (num_boys-1) * (num_boys-2) * 2) = 2640 :=
by
  sorry

end NUMINAMATH_GPT_num_ways_to_choose_officers_same_gender_l1871_187177


namespace NUMINAMATH_GPT_find_nickels_l1871_187101

noncomputable def num_quarters1 := 25
noncomputable def num_dimes := 15
noncomputable def num_quarters2 := 15
noncomputable def value_quarter := 25
noncomputable def value_dime := 10
noncomputable def value_nickel := 5

theorem find_nickels (n : ℕ) :
  value_quarter * num_quarters1 + value_dime * num_dimes = value_quarter * num_quarters2 + value_nickel * n → 
  n = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_nickels_l1871_187101


namespace NUMINAMATH_GPT_blue_balls_taken_out_l1871_187114

theorem blue_balls_taken_out (x : ℕ) :
  ∀ (total_balls : ℕ) (initial_blue_balls : ℕ)
    (remaining_probability : ℚ),
    total_balls = 25 ∧ initial_blue_balls = 9 ∧ remaining_probability = 1/5 →
    (9 - x : ℚ) / (25 - x : ℚ) = 1/5 →
    x = 5 :=
by
  intros total_balls initial_blue_balls remaining_probability
  rintro ⟨h_total_balls, h_initial_blue_balls, h_remaining_probability⟩ h_eq
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_blue_balls_taken_out_l1871_187114


namespace NUMINAMATH_GPT_probability_of_sum_5_when_two_dice_rolled_l1871_187120

theorem probability_of_sum_5_when_two_dice_rolled :
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_possible_outcomes : ℝ) = (1 / 9 : ℝ) :=
by
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  have h : (favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ) = (1 / 9 : ℝ) := sorry
  exact h

end NUMINAMATH_GPT_probability_of_sum_5_when_two_dice_rolled_l1871_187120


namespace NUMINAMATH_GPT_number_of_members_l1871_187192

noncomputable def club_members (n O N : ℕ) : Prop :=
  (3 * n = O - N) ∧ (O - N = 15)

theorem number_of_members (n O N : ℕ) (h : club_members n O N) : n = 5 :=
  by
    sorry

end NUMINAMATH_GPT_number_of_members_l1871_187192


namespace NUMINAMATH_GPT_additional_miles_proof_l1871_187122

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end NUMINAMATH_GPT_additional_miles_proof_l1871_187122


namespace NUMINAMATH_GPT_correlation_highly_related_l1871_187143

-- Conditions:
-- Let corr be the linear correlation coefficient of product output and unit cost.
-- Let rel be the relationship between product output and unit cost.

def corr : ℝ := -0.87

-- Proof Goal:
-- If corr = -0.87, then the relationship is "highly related".

theorem correlation_highly_related (h : corr = -0.87) : rel = "highly related" := by
  sorry

end NUMINAMATH_GPT_correlation_highly_related_l1871_187143


namespace NUMINAMATH_GPT_distance_between_locations_l1871_187191

theorem distance_between_locations
  (d_AC d_BC : ℚ)
  (d : ℚ)
  (meet_C : d_AC + d_BC = d)
  (travel_A_B : 150 + 150 + 540 = 840)
  (distance_ratio : 840 / 540 = 14 / 9)
  (distance_ratios : d_AC / d_BC = 14 / 9)
  (C_D : 540 = 5 * d / 23) :
  d = 2484 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_locations_l1871_187191


namespace NUMINAMATH_GPT_mixture_milk_quantity_l1871_187129

variable (M W : ℕ)

theorem mixture_milk_quantity
  (h1 : M = 2 * W)
  (h2 : 6 * (W + 10) = 5 * M) :
  M = 30 := by
  sorry

end NUMINAMATH_GPT_mixture_milk_quantity_l1871_187129


namespace NUMINAMATH_GPT_fraction_meaningful_l1871_187155

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l1871_187155
