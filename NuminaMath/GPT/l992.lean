import Mathlib

namespace NUMINAMATH_GPT_range_of_x_for_sqrt_l992_99287

-- Define the condition under which the expression inside the square root is non-negative.
def sqrt_condition (x : ℝ) : Prop :=
  x - 7 ≥ 0

-- Main theorem to prove the range of values for x
theorem range_of_x_for_sqrt (x : ℝ) : sqrt_condition x ↔ x ≥ 7 :=
by
  -- Proof steps go here (omitted as per instructions)
  sorry

end NUMINAMATH_GPT_range_of_x_for_sqrt_l992_99287


namespace NUMINAMATH_GPT_find_power_y_l992_99217

theorem find_power_y 
  (y : ℕ) 
  (h : (12 : ℝ)^y * (6 : ℝ)^3 / (432 : ℝ) = 72) : 
  y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_power_y_l992_99217


namespace NUMINAMATH_GPT_intersect_x_axis_unique_l992_99293

theorem intersect_x_axis_unique (a : ℝ) : (∀ x, (ax^2 + (3 - a) * x + 1) = 0 → x = 0) ↔ (a = 0 ∨ a = 1 ∨ a = 9) := by
  sorry

end NUMINAMATH_GPT_intersect_x_axis_unique_l992_99293


namespace NUMINAMATH_GPT_net_progress_l992_99214

def lost_yards : Int := 5
def gained_yards : Int := 7

theorem net_progress : gained_yards - lost_yards = 2 := 
by
  sorry

end NUMINAMATH_GPT_net_progress_l992_99214


namespace NUMINAMATH_GPT_sets_are_equal_l992_99222

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end NUMINAMATH_GPT_sets_are_equal_l992_99222


namespace NUMINAMATH_GPT_estimated_percentage_negative_attitude_l992_99230

-- Define the conditions
def total_parents := 2500
def sample_size := 400
def negative_attitude := 360

-- Prove the estimated percentage of parents with a negative attitude is 90%
theorem estimated_percentage_negative_attitude : 
  (negative_attitude: ℝ) / (sample_size: ℝ) * 100 = 90 := by
  sorry

end NUMINAMATH_GPT_estimated_percentage_negative_attitude_l992_99230


namespace NUMINAMATH_GPT_budget_for_supplies_l992_99297

-- Conditions as definitions
def percentage_transportation := 20
def percentage_research_development := 9
def percentage_utilities := 5
def percentage_equipment := 4
def degrees_salaries := 216
def total_degrees := 360
def total_percentage := 100

-- Mathematical problem: Prove the percentage spent on supplies
theorem budget_for_supplies :
  (total_percentage - (percentage_transportation +
                       percentage_research_development +
                       percentage_utilities +
                       percentage_equipment) - 
   ((degrees_salaries * total_percentage) / total_degrees)) = 2 := by
  sorry

end NUMINAMATH_GPT_budget_for_supplies_l992_99297


namespace NUMINAMATH_GPT_solve_S20_minus_2S10_l992_99244

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → a n > 0) ∧
  (∀ n : ℕ, n ≥ 2 → S n = (n / (n - 1 : ℝ)) * (a n ^ 2 - a 1 ^ 2))

theorem solve_S20_minus_2S10 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    arithmetic_sequence a S →
    S 20 - 2 * S 10 = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_solve_S20_minus_2S10_l992_99244


namespace NUMINAMATH_GPT_smallest_possible_AC_l992_99245

theorem smallest_possible_AC 
    (AB AC CD : ℤ) 
    (BD_squared : ℕ) 
    (h_isosceles : AB = AC)
    (h_point_D : ∃ D : ℤ, D = CD)
    (h_perpendicular : BD_squared = 85) 
    (h_integers : ∃ x y : ℤ, AC = x ∧ CD = y) 
    : AC = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_AC_l992_99245


namespace NUMINAMATH_GPT_find_w_l992_99252

noncomputable def roots_cubic_eq (x : ℝ) : ℝ := x^3 + 2 * x^2 + 5 * x - 8

def p : ℝ := sorry -- one root of x^3 + 2x^2 + 5x - 8 = 0
def q : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0
def r : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0

theorem find_w 
  (h1 : roots_cubic_eq p = 0)
  (h2 : roots_cubic_eq q = 0)
  (h3 : roots_cubic_eq r = 0)
  (h4 : p + q + r = -2): 
  ∃ w : ℝ, w = 18 := 
sorry

end NUMINAMATH_GPT_find_w_l992_99252


namespace NUMINAMATH_GPT_joan_dimes_l992_99266

theorem joan_dimes (initial_dimes spent_dimes remaining_dimes : ℕ) 
    (h1 : initial_dimes = 5) (h2 : spent_dimes = 2) 
    (h3 : remaining_dimes = initial_dimes - spent_dimes) : 
    remaining_dimes = 3 := 
sorry

end NUMINAMATH_GPT_joan_dimes_l992_99266


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000003_l992_99251

theorem scientific_notation_of_0_0000003 :
  0.0000003 = 3 * 10^(-7) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000003_l992_99251


namespace NUMINAMATH_GPT_area_of_backyard_eq_400_l992_99292

-- Define the conditions
def length_condition (l : ℕ) : Prop := 25 * l = 1000
def perimeter_condition (l w : ℕ) : Prop := 20 * (l + w) = 1000

-- State the theorem
theorem area_of_backyard_eq_400 (l w : ℕ) (h_length : length_condition l) (h_perimeter : perimeter_condition l w) : l * w = 400 :=
  sorry

end NUMINAMATH_GPT_area_of_backyard_eq_400_l992_99292


namespace NUMINAMATH_GPT_time_difference_l992_99202

-- Definitions for the problem conditions
def Zoe_speed : ℕ := 9 -- Zoe's speed in minutes per mile
def Henry_speed : ℕ := 7 -- Henry's speed in minutes per mile
def Race_length : ℕ := 12 -- Race length in miles

-- Theorem to prove the time difference
theorem time_difference : (Race_length * Zoe_speed) - (Race_length * Henry_speed) = 24 :=
by
  sorry

end NUMINAMATH_GPT_time_difference_l992_99202


namespace NUMINAMATH_GPT_distance_between_closest_points_correct_l992_99296

noncomputable def circle_1_center : ℝ × ℝ := (3, 3)
noncomputable def circle_2_center : ℝ × ℝ := (20, 12)
noncomputable def circle_1_radius : ℝ := circle_1_center.2
noncomputable def circle_2_radius : ℝ := circle_2_center.2
noncomputable def distance_between_centers : ℝ := Real.sqrt ((20 - 3)^2 + (12 - 3)^2)
noncomputable def distance_between_closest_points : ℝ := distance_between_centers - (circle_1_radius + circle_2_radius)

theorem distance_between_closest_points_correct :
  distance_between_closest_points = Real.sqrt 370 - 15 :=
sorry

end NUMINAMATH_GPT_distance_between_closest_points_correct_l992_99296


namespace NUMINAMATH_GPT_colorings_without_two_corners_l992_99283

def valid_colorings (n: ℕ) (exclude_cells : Finset (Fin n × Fin n)) : ℕ := sorry

theorem colorings_without_two_corners :
  valid_colorings 5 ∅ = 120 →
  valid_colorings 5 {(0, 0)} = 96 →
  valid_colorings 5 {(0, 0), (4, 4)} = 78 :=
by {
  sorry
}

end NUMINAMATH_GPT_colorings_without_two_corners_l992_99283


namespace NUMINAMATH_GPT_error_percent_in_area_l992_99208

theorem error_percent_in_area 
    (L W : ℝ) 
    (measured_length : ℝ := 1.09 * L) 
    (measured_width : ℝ := 0.92 * W) 
    (correct_area : ℝ := L * W) 
    (incorrect_area : ℝ := measured_length * measured_width) :
    100 * (incorrect_area - correct_area) / correct_area = 0.28 :=
by
  sorry

end NUMINAMATH_GPT_error_percent_in_area_l992_99208


namespace NUMINAMATH_GPT_total_copies_produced_l992_99211

theorem total_copies_produced
  (rate_A : ℕ)
  (rate_B : ℕ)
  (rate_C : ℕ)
  (time_A : ℕ)
  (time_B : ℕ)
  (time_C : ℕ)
  (total_time : ℕ)
  (ha : rate_A = 10)
  (hb : rate_B = 10)
  (hc : rate_C = 10)
  (hA_time : time_A = 15)
  (hB_time : time_B = 20)
  (hC_time : time_C = 25)
  (h_total_time : total_time = 30) :
  rate_A * time_A + rate_B * time_B + rate_C * time_C = 600 :=
by 
  -- Machine A: 10 copies per minute * 15 minutes = 150 copies
  -- Machine B: 10 copies per minute * 20 minutes = 200 copies
  -- Machine C: 10 copies per minute * 25 minutes = 250 copies
  -- Hence, the total number of copies = 150 + 200 + 250 = 600
  sorry

end NUMINAMATH_GPT_total_copies_produced_l992_99211


namespace NUMINAMATH_GPT_original_equation_proof_l992_99257

theorem original_equation_proof :
  ∃ (A O H M J : ℕ),
  A ≠ O ∧ A ≠ H ∧ A ≠ M ∧ A ≠ J ∧
  O ≠ H ∧ O ≠ M ∧ O ≠ J ∧
  H ≠ M ∧ H ≠ J ∧
  M ≠ J ∧
  A + 8 * (10 * O + H) = 10 * M + J ∧
  (O = 1) ∧ (H = 2) ∧ (M = 9) ∧ (J = 6) ∧ (A = 0) :=
by
  sorry

end NUMINAMATH_GPT_original_equation_proof_l992_99257


namespace NUMINAMATH_GPT_decreasing_interval_of_even_function_l992_99200

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k*x^2 + (k - 1)*x + 2

theorem decreasing_interval_of_even_function (k : ℝ) (h : ∀ x : ℝ, f k x = f k (-x)) :
  ∃ k : ℝ, k = 1 ∧ ∀ x : ℝ, (x < 0 → f k x > f k (-x)) := 
sorry

end NUMINAMATH_GPT_decreasing_interval_of_even_function_l992_99200


namespace NUMINAMATH_GPT_mark_total_eggs_in_a_week_l992_99276

-- Define the given conditions
def first_store_eggs_per_day := 5 * 12 -- 5 dozen eggs per day
def second_store_eggs_per_day := 30
def third_store_eggs_per_odd_day := 25 * 12 -- 25 dozen eggs per odd day
def third_store_eggs_per_even_day := 15 * 12 -- 15 dozen eggs per even day
def days_per_week := 7
def odd_days_per_week := 4
def even_days_per_week := 3

-- Lean theorem statement to prove the total eggs supplied in a week
theorem mark_total_eggs_in_a_week : 
    first_store_eggs_per_day * days_per_week + 
    second_store_eggs_per_day * days_per_week + 
    third_store_eggs_per_odd_day * odd_days_per_week + 
    third_store_eggs_per_even_day * even_days_per_week =
    2370 := 
    sorry  -- Placeholder for the actual proof

end NUMINAMATH_GPT_mark_total_eggs_in_a_week_l992_99276


namespace NUMINAMATH_GPT_simplify_fraction_l992_99288

theorem simplify_fraction (a : ℝ) (h : a ≠ 2) : (3 - a) / (a - 2) + 1 = 1 / (a - 2) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_simplify_fraction_l992_99288


namespace NUMINAMATH_GPT_largest_prime_factor_1001_l992_99207

theorem largest_prime_factor_1001 : ∃ p : ℕ, p = 13 ∧ Prime p ∧ (∀ q : ℕ, Prime q ∧ q ∣ 1001 → q ≤ 13) := sorry

end NUMINAMATH_GPT_largest_prime_factor_1001_l992_99207


namespace NUMINAMATH_GPT_miracle_tree_fruit_count_l992_99269

theorem miracle_tree_fruit_count :
  ∃ (apples oranges pears : ℕ), 
  apples + oranges + pears = 30 ∧
  apples = 6 ∧ oranges = 9 ∧ pears = 15 := by
  sorry

end NUMINAMATH_GPT_miracle_tree_fruit_count_l992_99269


namespace NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l992_99235

theorem num_rectangles_in_5x5_grid : 
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  num_ways_choose_2 * num_ways_choose_2 = 100 :=
by
  -- Definitions based on conditions
  let n := 5
  let num_ways_choose_2 := (n * (n - 1)) / 2
  
  -- Required proof (just showing the statement here)
  show num_ways_choose_2 * num_ways_choose_2 = 100
  sorry

end NUMINAMATH_GPT_num_rectangles_in_5x5_grid_l992_99235


namespace NUMINAMATH_GPT_center_of_circle_l992_99206

theorem center_of_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos (θ - π / 4)) : (ρ, θ) = (1, π / 4) :=
sorry

end NUMINAMATH_GPT_center_of_circle_l992_99206


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l992_99250

/-
Problem:
Given an isosceles triangle with side lengths 5 and 6, prove that the perimeter of the triangle is either 16 or 17.
-/

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5 ∨ a = 6) (h₂ : b = 5 ∨ b = 6) (h₃ : a ≠ b) : 
  (a + a + b = 16 ∨ a + a + b = 17) ∧ (b + b + a = 16 ∨ b + b + a = 17) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l992_99250


namespace NUMINAMATH_GPT_max_bag_weight_l992_99238

-- Let's define the conditions first
def green_beans_weight := 4
def milk_weight := 6
def carrots_weight := 2 * green_beans_weight
def additional_capacity := 2

-- The total weight of groceries
def total_groceries_weight := green_beans_weight + milk_weight + carrots_weight

-- The maximum weight the bag can hold is the total weight of groceries plus the additional capacity
theorem max_bag_weight : (total_groceries_weight + additional_capacity) = 20 := by
  sorry

end NUMINAMATH_GPT_max_bag_weight_l992_99238


namespace NUMINAMATH_GPT_find_a_l992_99242

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {1, 3, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Theorem to be proved
theorem find_a (a : ℝ) 
  (h1 : A a ∪ B a = {1, 3, a}) : a = 0 ∨ a = 1 ∨ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l992_99242


namespace NUMINAMATH_GPT_beta_cannot_be_determined_l992_99231

variables (α β : ℝ)
def consecutive_interior_angles (α β : ℝ) : Prop := -- define what it means for angles to be consecutive interior angles
  α + β = 180  -- this is true for interior angles, for illustrative purposes.

theorem beta_cannot_be_determined
  (h1 : consecutive_interior_angles α β)
  (h2 : α = 55) :
  ¬(∃ β, β = α) :=
by
  sorry

end NUMINAMATH_GPT_beta_cannot_be_determined_l992_99231


namespace NUMINAMATH_GPT_percentage_problem_l992_99203

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end NUMINAMATH_GPT_percentage_problem_l992_99203


namespace NUMINAMATH_GPT_calc_nabla_example_l992_99227

-- Define the custom operation ∇
def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- State the proof problem
theorem calc_nabla_example : op_nabla (op_nabla 2 3) (op_nabla 4 5) = 49 / 56 := by
  sorry

end NUMINAMATH_GPT_calc_nabla_example_l992_99227


namespace NUMINAMATH_GPT_calvin_overall_score_l992_99286

theorem calvin_overall_score :
  let test1_pct := 0.6
  let test1_total := 15
  let test2_pct := 0.85
  let test2_total := 20
  let test3_pct := 0.75
  let test3_total := 40
  let total_problems := 75

  let correct_test1 := test1_pct * test1_total
  let correct_test2 := test2_pct * test2_total
  let correct_test3 := test3_pct * test3_total
  let total_correct := correct_test1 + correct_test2 + correct_test3

  let overall_percentage := (total_correct / total_problems) * 100
  overall_percentage.round = 75 :=
sorry

end NUMINAMATH_GPT_calvin_overall_score_l992_99286


namespace NUMINAMATH_GPT_largest_arithmetic_seq_3digit_l992_99237

theorem largest_arithmetic_seq_3digit : 
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ (∃ a b c : ℕ, n = 100*a + 10*b + c ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a = 9 ∧ ∃ d, b = a - d ∧ c = a - 2*d) ∧ n = 963 :=
by sorry

end NUMINAMATH_GPT_largest_arithmetic_seq_3digit_l992_99237


namespace NUMINAMATH_GPT_rearrange_distinct_sums_mod_4028_l992_99285

theorem rearrange_distinct_sums_mod_4028 
  (x : Fin 2014 → ℤ) (y : Fin 2014 → ℤ) 
  (hx : ∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Fin 2014 → Fin 2014, Function.Bijective σ ∧ 
  ∀ i j : Fin 2014, i ≠ j → ( x i + y (σ i) ) % 4028 ≠ ( x j + y (σ j) ) % 4028 
:= by
  sorry

end NUMINAMATH_GPT_rearrange_distinct_sums_mod_4028_l992_99285


namespace NUMINAMATH_GPT_arrangements_TOOTH_l992_99218
-- Import necessary libraries

-- Define the problem conditions
def word_length : Nat := 5
def count_T : Nat := 2
def count_O : Nat := 2

-- State the problem as a theorem
theorem arrangements_TOOTH : 
  (word_length.factorial / (count_T.factorial * count_O.factorial)) = 30 := by
  sorry

end NUMINAMATH_GPT_arrangements_TOOTH_l992_99218


namespace NUMINAMATH_GPT_base_of_second_fraction_l992_99261

theorem base_of_second_fraction (x k : ℝ) (h1 : (1/2)^18 * (1/x)^k = 1/18^18) (h2 : k = 9) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_base_of_second_fraction_l992_99261


namespace NUMINAMATH_GPT_card_area_after_shortening_l992_99256

theorem card_area_after_shortening 
  (length : ℕ) (width : ℕ) (area_after_shortening : ℕ) 
  (h_initial : length = 8) (h_initial_width : width = 3)
  (h_area_shortened_by_2 : area_after_shortening = 15) :
  (length - 2) * width = 8 :=
by
  -- Original dimensions
  let original_length := 8
  let original_width := 3
  -- Area after shortening one side by 2 inches
  let area_after_shortening_width := (original_length) * (original_width - 2)
  let area_after_shortening_length := (original_length - 2) * (original_width)
  sorry

end NUMINAMATH_GPT_card_area_after_shortening_l992_99256


namespace NUMINAMATH_GPT_max_value_T_n_l992_99254

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q^n

noncomputable def sum_of_first_n_terms (a₁ q : ℝ) (n : ℕ) :=
  a₁ * (1 - q^(n + 1)) / (1 - q)

noncomputable def T_n (a₁ q : ℝ) (n : ℕ) :=
  (9 * sum_of_first_n_terms a₁ q n - sum_of_first_n_terms a₁ q (2 * n)) /
  geometric_sequence a₁ q (n + 1)

theorem max_value_T_n
  (a₁ : ℝ) (n : ℕ) (h : n > 0) (q : ℝ) (hq : q = 2) :
  ∃ n₀ : ℕ, T_n a₁ q n₀ = 3 := sorry

end NUMINAMATH_GPT_max_value_T_n_l992_99254


namespace NUMINAMATH_GPT_rectangle_circle_area_ratio_l992_99243

theorem rectangle_circle_area_ratio (w r: ℝ) (h1: 3 * w = π * r) (h2: 2 * w = l) :
  (l * w) / (π * r ^ 2) = 2 * π / 9 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_circle_area_ratio_l992_99243


namespace NUMINAMATH_GPT_sum_mod_12_l992_99219

def remainder_sum_mod :=
  let nums := [10331, 10333, 10335, 10337, 10339, 10341, 10343]
  let sum_nums := nums.sum
  sum_nums % 12 = 7

theorem sum_mod_12 : remainder_sum_mod :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_12_l992_99219


namespace NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_over_2_l992_99289

theorem cos_210_eq_neg_sqrt3_over_2 :
  Real.cos (210 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_GPT_cos_210_eq_neg_sqrt3_over_2_l992_99289


namespace NUMINAMATH_GPT_no_intersection_l992_99246

-- Definitions of the sets M1 and M2 based on parameters A, B, C and integer x
def M1 (A B : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = x^2 + A * x + B}
def M2 (C : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = 2 * x^2 + 2 * x + C}

-- The statement of the theorem
theorem no_intersection (A B : ℤ) : ∃ C : ℤ, M1 A B ∩ M2 C = ∅ :=
sorry

end NUMINAMATH_GPT_no_intersection_l992_99246


namespace NUMINAMATH_GPT_find_g2_l992_99209

-- Given conditions:
variables (g : ℝ → ℝ) 
axiom cond1 : ∀ (x y : ℝ), x * g y = 2 * y * g x
axiom cond2 : g 10 = 5

-- Proof to show g(2) = 2
theorem find_g2 : g 2 = 2 := 
by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_find_g2_l992_99209


namespace NUMINAMATH_GPT_roots_of_polynomial_l992_99278

-- Define the polynomial
def P (x : ℝ) : ℝ := x^3 - 7 * x^2 + 14 * x - 8

-- Prove that the roots of P are {1, 2, 4}
theorem roots_of_polynomial :
  ∃ (S : Set ℝ), S = {1, 2, 4} ∧ ∀ x, P x = 0 ↔ x ∈ S :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l992_99278


namespace NUMINAMATH_GPT_sum_symmetry_l992_99264

def f (x : ℝ) : ℝ :=
  x^2 * (1 - x)^2

theorem sum_symmetry :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_symmetry_l992_99264


namespace NUMINAMATH_GPT_initial_amount_l992_99201

theorem initial_amount (P : ℝ) (h1 : ∀ x : ℝ, x * (9 / 8) * (9 / 8) = 81000) : P = 64000 :=
sorry

end NUMINAMATH_GPT_initial_amount_l992_99201


namespace NUMINAMATH_GPT_shortest_remaining_side_length_l992_99290

noncomputable def triangle_has_right_angle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem shortest_remaining_side_length {a b : ℝ} (ha : a = 5) (hb : b = 12) (h_right_angle : ∃ c, triangle_has_right_angle a b c) :
  ∃ c, c = 5 :=
by 
  sorry

end NUMINAMATH_GPT_shortest_remaining_side_length_l992_99290


namespace NUMINAMATH_GPT_volume_ratio_of_frustum_l992_99295

theorem volume_ratio_of_frustum
  (h_s h : ℝ)
  (A_s A : ℝ)
  (V_s V : ℝ)
  (ratio_lateral_area : ℝ)
  (ratio_height : ℝ)
  (ratio_base_area : ℝ)
  (H_lateral_area: ratio_lateral_area = 9 / 16)
  (H_height: ratio_height = 3 / 5)
  (H_base_area: ratio_base_area = 9 / 25)
  (H_volume_small: V_s = 1 / 3 * h_s * A_s)
  (H_volume_total: V = 1 / 3 * h * A - 1 / 3 * h_s * A_s) :
  V_s / V = 27 / 98 :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_frustum_l992_99295


namespace NUMINAMATH_GPT_speed_of_X_l992_99262

theorem speed_of_X (t1 t2 Vx : ℝ) (h1 : t2 - t1 = 3) 
  (h2 : 3 * Vx + Vx * t1 = 60 * t1 + 30)
  (h3 : 3 * Vx + Vx * t2 + 30 = 60 * t2) : Vx = 60 :=
by sorry

end NUMINAMATH_GPT_speed_of_X_l992_99262


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l992_99233

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Define the arithmetic sequence

theorem arithmetic_sequence_problem
  (a_4 : ℝ) (a_9 : ℝ)
  (h_a4 : a_4 = 5)
  (h_a9 : a_9 = 17)
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) = a_n n + (a_n 2 - a_n 1)) :
  a_n 14 = 29 :=
by
  -- the proof will utilize the property of arithmetic sequence and substitutions
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l992_99233


namespace NUMINAMATH_GPT_geometric_sequence_sum_l992_99241

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : ∀ n, a n > 0)
  (h1 : a 1 + a 3 = 3)
  (h2 : a 4 + a 6 = 6):
  a 1 * a 3 + a 2 * a 4 + a 3 * a 5 + a 4 * a 6 + a 5 * a 7 = 62 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l992_99241


namespace NUMINAMATH_GPT_reading_time_per_disc_l992_99260

theorem reading_time_per_disc (total_minutes : ℕ) (disc_capacity : ℕ) (d : ℕ) (reading_per_disc : ℕ) :
  total_minutes = 528 ∧ disc_capacity = 45 ∧ d = 12 ∧ total_minutes = d * reading_per_disc → reading_per_disc = 44 :=
by
  sorry

end NUMINAMATH_GPT_reading_time_per_disc_l992_99260


namespace NUMINAMATH_GPT_regular_polygon_sides_l992_99294

theorem regular_polygon_sides (n : ℕ) (h : 360 / n = 18) : n = 20 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l992_99294


namespace NUMINAMATH_GPT_problem1_problem2_l992_99239

-- Define the conditions
variable (a x : ℝ)
variable (h_gt_zero : x > 0) (a_gt_zero : a > 0)

-- Problem 1: Prove that 0 < x ≤ 300
theorem problem1 (h: 12 * (500 - x) * (1 + 0.005 * x) ≥ 12 * 500) : 0 < x ∧ x ≤ 300 := 
sorry

-- Problem 2: Prove that 0 < a ≤ 5.5 given the conditions
theorem problem2 (h1 : 12 * (a - 13 / 1000 * x) * x ≤ 12 * (500 - x) * (1 + 0.005 * x))
                (h2 : x = 250) : 0 < a ∧ a ≤ 5.5 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l992_99239


namespace NUMINAMATH_GPT_solve_eq_l992_99249

theorem solve_eq (x : ℝ) : (x - 2)^2 = 9 * x^2 ↔ x = -1 ∨ x = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_eq_l992_99249


namespace NUMINAMATH_GPT_ratio_squirrels_to_raccoons_l992_99263

def animals_total : ℕ := 84
def raccoons : ℕ := 12
def squirrels : ℕ := animals_total - raccoons

theorem ratio_squirrels_to_raccoons : (squirrels : ℚ) / raccoons = 6 :=
by
  sorry

end NUMINAMATH_GPT_ratio_squirrels_to_raccoons_l992_99263


namespace NUMINAMATH_GPT_milk_leftover_l992_99234

variable {v : ℕ} -- 'v' is the number of sets of milkshakes in the 2:1 ratio.
variables {milk vanilla_chocolate : ℕ} -- spoon amounts per milkshake types
variables {total_milk total_vanilla_ice_cream total_chocolate_ice_cream : ℕ} -- total amount constraints
variables {milk_left : ℕ} -- amount of milk left after

-- Definitions based on the conditions
def milk_per_vanilla := 4
def milk_per_chocolate := 5
def ice_vanilla_per_milkshake := 12
def ice_chocolate_per_milkshake := 10
def initial_milk := 72
def initial_vanilla_ice_cream := 96
def initial_chocolate_ice_cream := 96

-- Constraints
def max_milkshakes := 16
def milk_needed (v : ℕ) := (4 * 2 * v) + (5 * v)
def vanilla_needed (v : ℕ) := 12 * 2 * v
def chocolate_needed (v : ℕ) := 10 * v 

-- Inequalities
lemma milk_constraint (v : ℕ) : milk_needed v ≤ initial_milk := sorry

lemma vanilla_constraint (v : ℕ) : vanilla_needed v ≤ initial_vanilla_ice_cream := sorry

lemma chocolate_constraint (v : ℕ) : chocolate_needed v ≤ initial_chocolate_ice_cream := sorry

lemma total_milkshakes_constraint (v : ℕ) : 3 * v ≤ max_milkshakes := sorry

-- Conclusion
theorem milk_leftover : milk_left = initial_milk - milk_needed 5 := sorry

end NUMINAMATH_GPT_milk_leftover_l992_99234


namespace NUMINAMATH_GPT_james_fish_tanks_l992_99221

theorem james_fish_tanks (n t1 t2 t3 : ℕ) (h1 : t1 = 20) (h2 : t2 = 2 * t1) (h3 : t3 = 2 * t1) (h4 : t1 + t2 + t3 = 100) : n = 3 :=
sorry

end NUMINAMATH_GPT_james_fish_tanks_l992_99221


namespace NUMINAMATH_GPT_largest_value_in_interval_l992_99277

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x^3, 3*x, x^(1/3), 1/x} : Set ℝ), y ≤ 1/x) :=
sorry

end NUMINAMATH_GPT_largest_value_in_interval_l992_99277


namespace NUMINAMATH_GPT_product_of_last_two_digits_l992_99275

theorem product_of_last_two_digits (A B : ℕ) (hn1 : 10 * A + B ≡ 0 [MOD 5]) (hn2 : A + B = 16) : A * B = 30 :=
sorry

end NUMINAMATH_GPT_product_of_last_two_digits_l992_99275


namespace NUMINAMATH_GPT_fraction_eval_l992_99267

theorem fraction_eval :
  (3 / 7 + 5 / 8) / (5 / 12 + 1 / 4) = 177 / 112 :=
by
  sorry

end NUMINAMATH_GPT_fraction_eval_l992_99267


namespace NUMINAMATH_GPT_increase_in_average_l992_99226

theorem increase_in_average {a1 a2 a3 a4 : ℕ} 
                            (h1 : a1 = 92) 
                            (h2 : a2 = 89) 
                            (h3 : a3 = 91) 
                            (h4 : a4 = 93) : 
    ((a1 + a2 + a3 + a4 : ℚ) / 4) - ((a1 + a2 + a3 : ℚ) / 3) = 0.58 := 
by
  sorry

end NUMINAMATH_GPT_increase_in_average_l992_99226


namespace NUMINAMATH_GPT_word_count_in_language_l992_99279

theorem word_count_in_language :
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  num_words = 900 :=
by
  let vowels := 3
  let consonants := 5
  let num_syllables := (vowels * consonants) + (consonants * vowels)
  let num_words := num_syllables * num_syllables
  have : num_words = 900 := sorry
  exact this

end NUMINAMATH_GPT_word_count_in_language_l992_99279


namespace NUMINAMATH_GPT_gcd_product_eq_gcd_l992_99272

theorem gcd_product_eq_gcd {a b c : ℤ} (hab : Int.gcd a b = 1) : Int.gcd a (b * c) = Int.gcd a c := 
by 
  sorry

end NUMINAMATH_GPT_gcd_product_eq_gcd_l992_99272


namespace NUMINAMATH_GPT_largest_five_digit_number_divisible_by_5_l992_99270

theorem largest_five_digit_number_divisible_by_5 : 
  ∃ n, (n % 5 = 0) ∧ (99990 ≤ n) ∧ (n ≤ 99995) ∧ (∀ m, (m % 5 = 0) → (99990 ≤ m) → (m ≤ 99995) → m ≤ n) :=
by
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_largest_five_digit_number_divisible_by_5_l992_99270


namespace NUMINAMATH_GPT_hotel_charge_comparison_l992_99268

theorem hotel_charge_comparison (R G P : ℝ) 
  (h1 : P = R - 0.70 * R)
  (h2 : P = G - 0.10 * G) :
  ((R - G) / G) * 100 = 170 :=
by
  sorry

end NUMINAMATH_GPT_hotel_charge_comparison_l992_99268


namespace NUMINAMATH_GPT_cut_scene_length_l992_99229

theorem cut_scene_length
  (original_length final_length : ℕ)
  (h_original : original_length = 60)
  (h_final : final_length = 54) :
  original_length - final_length = 6 :=
by 
  sorry

end NUMINAMATH_GPT_cut_scene_length_l992_99229


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l992_99271

def A : Set ℝ := { x | x < 3 }
def B : Set ℝ := { x | Real.log (x - 1) / Real.log 3 > 0 }

theorem intersection_of_A_and_B :
  (A ∩ B) = { x | 2 < x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l992_99271


namespace NUMINAMATH_GPT_circle_equation_with_diameter_endpoints_l992_99282

theorem circle_equation_with_diameter_endpoints (A B : ℝ × ℝ) (x y : ℝ) :
  A = (1, 4) → B = (3, -2) → (x-2)^2 + (y-1)^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_with_diameter_endpoints_l992_99282


namespace NUMINAMATH_GPT_find_n_l992_99213

theorem find_n : ∃ n : ℤ, 100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180) ∧ n = 317 := 
by
  sorry

end NUMINAMATH_GPT_find_n_l992_99213


namespace NUMINAMATH_GPT_remy_used_25_gallons_l992_99291

noncomputable def RomanGallons : ℕ := 8

noncomputable def RemyGallons (R : ℕ) : ℕ := 3 * R + 1

theorem remy_used_25_gallons (R : ℕ) (h1 : RemyGallons R = 1 + 3 * R) (h2 : R + RemyGallons R = 33) : RemyGallons R = 25 := by
  sorry

end NUMINAMATH_GPT_remy_used_25_gallons_l992_99291


namespace NUMINAMATH_GPT_fraction_fliers_afternoon_l992_99298

theorem fraction_fliers_afternoon :
  ∀ (initial_fliers remaining_fliers next_day_fliers : ℕ),
    initial_fliers = 2500 →
    next_day_fliers = 1500 →
    remaining_fliers = initial_fliers - initial_fliers / 5 →
    (remaining_fliers - next_day_fliers) / remaining_fliers = 1 / 4 :=
by
  intros initial_fliers remaining_fliers next_day_fliers
  sorry

end NUMINAMATH_GPT_fraction_fliers_afternoon_l992_99298


namespace NUMINAMATH_GPT_game_no_loser_l992_99232

theorem game_no_loser (x : ℕ) (h_start : x = 2017) :
  ∀ y, (y = x ∨ ∀ n, (n = 2 * y ∨ n = y - 1000) → (n > 1000 ∧ n < 4000)) →
       (y > 1000 ∧ y < 4000) :=
sorry

end NUMINAMATH_GPT_game_no_loser_l992_99232


namespace NUMINAMATH_GPT_value_to_subtract_l992_99225

theorem value_to_subtract (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 34) / 10 = 2) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_value_to_subtract_l992_99225


namespace NUMINAMATH_GPT_percent_commute_l992_99265

variable (x : ℝ)

theorem percent_commute (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end NUMINAMATH_GPT_percent_commute_l992_99265


namespace NUMINAMATH_GPT_defective_pens_count_l992_99210

theorem defective_pens_count (total_pens : ℕ) (prob_not_defective : ℚ) (D : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : prob_not_defective = 0.5357142857142857) : 
  D = 2 := 
by
  sorry

end NUMINAMATH_GPT_defective_pens_count_l992_99210


namespace NUMINAMATH_GPT_range_of_a_l992_99216

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then a + |x - 2|
  else x^2 - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : -1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l992_99216


namespace NUMINAMATH_GPT_quadratic_nonnegative_quadratic_inv_nonnegative_l992_99236

-- Problem Definitions and Proof Statements

variables {R : Type*} [LinearOrderedField R]

def f (a b c x : R) : R := a * x^2 + 2 * b * x + c

theorem quadratic_nonnegative {a b c : R} (ha : a ≠ 0) (h : ∀ x : R, f a b c x ≥ 0) : 
  a ≥ 0 ∧ c ≥ 0 ∧ a * c - b^2 ≥ 0 :=
sorry

theorem quadratic_inv_nonnegative {a b c : R} (ha : a ≥ 0) (hc : c ≥ 0) (hac : a * c - b^2 ≥ 0) :
  ∀ x : R, f a b c x ≥ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_nonnegative_quadratic_inv_nonnegative_l992_99236


namespace NUMINAMATH_GPT_cube_div_identity_l992_99259

theorem cube_div_identity (a b : ℕ) (h₁ : a = 6) (h₂ : b = 3) :
  (a^3 + b^3) / (a^2 - a * b + b^2) = 9 := by
  sorry

end NUMINAMATH_GPT_cube_div_identity_l992_99259


namespace NUMINAMATH_GPT_smallest_number_with_2020_divisors_l992_99258

-- Given a natural number n expressed in terms of its prime factors
def divisor_count (n : ℕ) (f : ℕ → ℕ) : ℕ :=
  f 2 + 1 * f 3 + 1 * f 5 + 1

-- The smallest number with exactly 2020 distinct natural divisors
theorem smallest_number_with_2020_divisors :
  ∃ n : ℕ, divisor_count n = 2020 ∧ 
           n = 2 ^ 100 * 3 ^ 4 * 5 ^ 1 :=
sorry

end NUMINAMATH_GPT_smallest_number_with_2020_divisors_l992_99258


namespace NUMINAMATH_GPT_ferris_wheel_small_seat_capacity_l992_99223

def num_small_seats : Nat := 2
def capacity_per_small_seat : Nat := 14

theorem ferris_wheel_small_seat_capacity : num_small_seats * capacity_per_small_seat = 28 := by
  sorry

end NUMINAMATH_GPT_ferris_wheel_small_seat_capacity_l992_99223


namespace NUMINAMATH_GPT_perimeter_of_rectangular_field_l992_99280

theorem perimeter_of_rectangular_field (L B : ℝ) 
    (h1 : B = 0.60 * L) 
    (h2 : L * B = 37500) : 
    2 * L + 2 * B = 800 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangular_field_l992_99280


namespace NUMINAMATH_GPT_validate_operation_l992_99255

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end NUMINAMATH_GPT_validate_operation_l992_99255


namespace NUMINAMATH_GPT_ratio_of_b_to_a_l992_99220

theorem ratio_of_b_to_a (a b c : ℕ) (x y : ℕ) 
  (h1 : a > 0) 
  (h2 : x = 100 * a + 10 * b + c)
  (h3 : y = 100 * 9 + 10 * 9 + 9 - 241) 
  (h4 : x = y) :
  b = 5 → a = 7 → (b / a : ℚ) = 5 / 7 := 
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_ratio_of_b_to_a_l992_99220


namespace NUMINAMATH_GPT_max_parts_divided_by_three_planes_l992_99248

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_parts_divided_by_three_planes_l992_99248


namespace NUMINAMATH_GPT_hexagon_chord_length_valid_l992_99247

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end NUMINAMATH_GPT_hexagon_chord_length_valid_l992_99247


namespace NUMINAMATH_GPT_girls_from_clay_is_30_l992_99204

-- Definitions for the given conditions
def total_students : ℕ := 150
def total_boys : ℕ := 90
def total_girls : ℕ := 60
def students_jonas : ℕ := 50
def students_clay : ℕ := 70
def students_hart : ℕ := 30
def boys_jonas : ℕ := 25

-- Theorem to prove that the number of girls from Clay Middle School is 30
theorem girls_from_clay_is_30 
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : students_jonas = 50)
  (h5 : students_clay = 70)
  (h6 : students_hart = 30)
  (h7 : boys_jonas = 25) : 
  ∃ girls_clay : ℕ, girls_clay = 30 :=
by 
  sorry

end NUMINAMATH_GPT_girls_from_clay_is_30_l992_99204


namespace NUMINAMATH_GPT_initial_salt_percentage_is_10_l992_99205

-- Declarations for terminology
def initial_volume : ℕ := 72
def added_water : ℕ := 18
def final_volume : ℕ := initial_volume + added_water
def final_salt_percentage : ℝ := 0.08

-- Amount of salt in the initial solution
def initial_salt_amount (P : ℝ) := initial_volume * P

-- Amount of salt in the final solution
def final_salt_amount : ℝ := final_volume * final_salt_percentage

-- Proof that the initial percentage of salt was 10%
theorem initial_salt_percentage_is_10 :
  ∃ P : ℝ, initial_salt_amount P = final_salt_amount ∧ P = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_initial_salt_percentage_is_10_l992_99205


namespace NUMINAMATH_GPT_find_functional_f_l992_99224

-- Define the problem domain and functions
variable (f : ℕ → ℕ)
variable (ℕ_star : Set ℕ) -- ℕ_star is {1,2,3,...}

-- Conditions
axiom f_increasing (h1 : ℕ) (h2 : ℕ) (h1_lt_h2 : h1 < h2) : f h1 < f h2
axiom f_functional (x : ℕ) (y : ℕ) : f (y * f x) = x^2 * f (x * y)

-- The proof problem
theorem find_functional_f : (∀ x ∈ ℕ_star, f x = x^2) :=
sorry

end NUMINAMATH_GPT_find_functional_f_l992_99224


namespace NUMINAMATH_GPT_cannot_be_value_of_omega_l992_99274

theorem cannot_be_value_of_omega (ω : ℤ) (φ : ℝ) (k n : ℤ) 
  (h1 : 0 < ω) 
  (h2 : |φ| < π / 2)
  (h3 : ω * (π / 12) + φ = k * π + π / 2)
  (h4 : -ω * (π / 6) + φ = n * π) : 
  ∀ m : ℤ, ω ≠ 4 * m := 
sorry

end NUMINAMATH_GPT_cannot_be_value_of_omega_l992_99274


namespace NUMINAMATH_GPT_walmart_knives_eq_three_l992_99273

variable (k : ℕ)

-- Walmart multitool
def walmart_tools : ℕ := 1 + k + 2

-- Target multitool (with twice as many knives as Walmart)
def target_tools : ℕ := 1 + 2 * k + 3 + 1

-- The condition that Target multitool has 5 more tools compared to Walmart
theorem walmart_knives_eq_three (h : target_tools k = walmart_tools k + 5) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_walmart_knives_eq_three_l992_99273


namespace NUMINAMATH_GPT_intersects_x_axis_vertex_coordinates_l992_99215

-- Definition of the quadratic function and conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - a * x - 2 * a^2

-- Condition: a ≠ 0
axiom a_nonzero (a : ℝ) : a ≠ 0

-- Statement for the first part of the problem
theorem intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x₁ x₂ : ℝ, quadratic_function a x₁ = 0 ∧ quadratic_function a x₂ = 0 ∧ x₁ * x₂ < 0 :=
by 
  sorry

-- Statement for the second part of the problem
theorem vertex_coordinates (a : ℝ) (h : a ≠ 0) (hy_intercept : quadratic_function a 0 = -2) :
  ∃ x_vertex : ℝ, quadratic_function a x_vertex = (if a = 1 then (1/2)^2 - 9/4 else (1/2)^2 - 9/4) :=
by 
  sorry


end NUMINAMATH_GPT_intersects_x_axis_vertex_coordinates_l992_99215


namespace NUMINAMATH_GPT_medium_pizza_promotion_price_l992_99212

-- Define the conditions
def regular_price_medium_pizza : ℝ := 18
def total_savings : ℝ := 39
def number_of_medium_pizzas : ℝ := 3

-- Define the goal
theorem medium_pizza_promotion_price : 
  ∃ P : ℝ, 3 * regular_price_medium_pizza - 3 * P = total_savings ∧ P = 5 := 
by
  sorry

end NUMINAMATH_GPT_medium_pizza_promotion_price_l992_99212


namespace NUMINAMATH_GPT_probability_other_side_red_given_seen_red_l992_99299

-- Definition of conditions
def total_cards := 9
def black_black_cards := 5
def black_red_cards := 2
def red_red_cards := 2
def red_sides := (2 * red_red_cards) + black_red_cards -- Total number of red sides
def favorable_red_red_sides := 2 * red_red_cards      -- Number of red sides on fully red cards

-- The required probability
def probability_other_side_red_given_red : ℚ := sorry

-- The main statement to prove
theorem probability_other_side_red_given_seen_red :
  probability_other_side_red_given_red = 2/3 :=
sorry

end NUMINAMATH_GPT_probability_other_side_red_given_seen_red_l992_99299


namespace NUMINAMATH_GPT_standard_parabola_with_symmetry_axis_eq_1_l992_99253

-- Define the condition that the axis of symmetry is x = 1
def axis_of_symmetry_x_eq_one (x : ℝ) : Prop :=
  x = 1

-- Define the standard equation of the parabola y^2 = -4x
def standard_parabola_eq (y x : ℝ) : Prop :=
  y^2 = -4 * x

-- Theorem: Prove that given the axis of symmetry of the parabola is x = 1,
-- the standard equation of the parabola is y^2 = -4x.
theorem standard_parabola_with_symmetry_axis_eq_1 : ∀ (x y : ℝ),
  axis_of_symmetry_x_eq_one x → standard_parabola_eq y x :=
by
  intros
  sorry

end NUMINAMATH_GPT_standard_parabola_with_symmetry_axis_eq_1_l992_99253


namespace NUMINAMATH_GPT_white_trees_count_l992_99281

noncomputable def calculate_white_trees (total_trees pink_percent red_trees : ℕ) : ℕ :=
  total_trees - (total_trees * pink_percent / 100 + red_trees)

theorem white_trees_count 
  (h1 : total_trees = 42)
  (h2 : pink_percent = 100 / 3)
  (h3 : red_trees = 2) :
  calculate_white_trees total_trees pink_percent red_trees = 26 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_white_trees_count_l992_99281


namespace NUMINAMATH_GPT_count_perfect_cube_or_fourth_power_lt_1000_l992_99284

theorem count_perfect_cube_or_fourth_power_lt_1000 :
  ∃ n, n = 14 ∧ (∀ x, (0 < x ∧ x < 1000 ∧ (∃ k, x = k^3 ∨ x = k^4)) ↔ ∃ i, i < n) :=
by sorry

end NUMINAMATH_GPT_count_perfect_cube_or_fourth_power_lt_1000_l992_99284


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l992_99228

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l992_99228


namespace NUMINAMATH_GPT_geometric_sequence_eighth_term_l992_99240

variable (a r : ℕ)
variable (h1 : a = 3)
variable (h2 : a * r^6 = 2187)
variable (h3 : a = 3)

theorem geometric_sequence_eighth_term (a r : ℕ) (h1 : a = 3) (h2 : a * r^6 = 2187) (h3 : a = 3) :
  a * r^7 = 6561 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_eighth_term_l992_99240
