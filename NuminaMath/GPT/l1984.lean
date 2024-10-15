import Mathlib

namespace NUMINAMATH_GPT_total_cats_l1984_198449

def num_white_cats : Nat := 2
def num_black_cats : Nat := 10
def num_gray_cats : Nat := 3

theorem total_cats : (num_white_cats + num_black_cats + num_gray_cats) = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_cats_l1984_198449


namespace NUMINAMATH_GPT_third_bouquet_carnations_l1984_198479

/--
Trevor buys three bouquets of carnations. The first included 9 carnations, and the second included 14 carnations. If the average number of carnations in the bouquets is 12, then the third bouquet contains 13 carnations.
-/
theorem third_bouquet_carnations (n1 n2 n3 : ℕ)
  (h1 : n1 = 9)
  (h2 : n2 = 14)
  (h3 : (n1 + n2 + n3) / 3 = 12) :
  n3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_third_bouquet_carnations_l1984_198479


namespace NUMINAMATH_GPT_trapezoid_ratio_l1984_198446

theorem trapezoid_ratio (u v : ℝ) (h1 : u > v) (h2 : (u + v) * (14 / u + 6 / v) = 40) : u / v = 7 / 3 :=
sorry

end NUMINAMATH_GPT_trapezoid_ratio_l1984_198446


namespace NUMINAMATH_GPT_remainder_of_sum_mod_18_l1984_198488

theorem remainder_of_sum_mod_18 :
  let nums := [85, 86, 87, 88, 89, 90, 91, 92, 93]
  let sum_nums := nums.sum
  let product := 90 * sum_nums
  product % 18 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_18_l1984_198488


namespace NUMINAMATH_GPT_probability_contemporaries_correct_l1984_198400

def alice_lifespan : ℝ := 150
def bob_lifespan : ℝ := 150
def total_years : ℝ := 800

noncomputable def probability_contemporaries : ℝ :=
  let unshaded_tri_area := (650 * 150) / 2
  let unshaded_area := 2 * unshaded_tri_area
  let total_area := total_years * total_years
  let shaded_area := total_area - unshaded_area
  shaded_area / total_area

theorem probability_contemporaries_correct : 
  probability_contemporaries = 27125 / 32000 :=
by
  sorry

end NUMINAMATH_GPT_probability_contemporaries_correct_l1984_198400


namespace NUMINAMATH_GPT_find_savings_l1984_198426

theorem find_savings (I E : ℕ) (h1 : I = 21000) (h2 : I / E = 7 / 6) : I - E = 3000 := by
  sorry

end NUMINAMATH_GPT_find_savings_l1984_198426


namespace NUMINAMATH_GPT_point_above_line_l1984_198438

theorem point_above_line (a : ℝ) : 3 * (-3) - 2 * (-1) - a < 0 ↔ a > -7 :=
by sorry

end NUMINAMATH_GPT_point_above_line_l1984_198438


namespace NUMINAMATH_GPT_find_c_in_triangle_l1984_198475

theorem find_c_in_triangle
  (angle_B : ℝ)
  (a : ℝ)
  (S : ℝ)
  (h1 : angle_B = 45)
  (h2 : a = 4)
  (h3 : S = 16 * Real.sqrt 2) :
  ∃ c : ℝ, c = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_c_in_triangle_l1984_198475


namespace NUMINAMATH_GPT_train_probability_correct_l1984_198415

noncomputable def train_prob (a_train b_train a_john b_john wait : ℝ) : ℝ :=
  let total_time_frame := (b_train - a_train) * (b_john - a_john)
  let triangle_area := (1 / 2) * wait * wait
  let rectangle_area := wait * wait
  let total_overlap_area := triangle_area + rectangle_area
  total_overlap_area / total_time_frame

theorem train_probability_correct :
  train_prob 120 240 150 210 30 = 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_train_probability_correct_l1984_198415


namespace NUMINAMATH_GPT_expected_return_correct_l1984_198428

-- Define the probabilities
def p1 := 1/4
def p2 := 1/4
def p3 := 1/6
def p4 := 1/3

-- Define the payouts
def payout (n : ℕ) (previous_odd : Bool) : ℝ :=
  match n with
  | 1 => 2
  | 2 => if previous_odd then -3 else 0
  | 3 => 0
  | 4 => 5
  | _ => 0

-- Define the expected values of one throw
def E1 : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

def E2_odd : ℝ :=
  p1 * payout 1 true + p2 * payout 2 true + p3 * payout 3 true + p4 * payout 4 true

def E2_even : ℝ :=
  p1 * payout 1 false + p2 * payout 2 false + p3 * payout 3 false + p4 * payout 4 false

-- Define the probability of throwing an odd number first
def p_odd : ℝ := p1 + p3

-- Define the probability of not throwing an odd number first
def p_even : ℝ := 1 - p_odd

-- Define the total expected return
def total_expected_return : ℝ :=
  E1 + (p_odd * E2_odd + p_even * E2_even)


theorem expected_return_correct :
  total_expected_return = 4.18 :=
  by
    -- The proof is omitted
    sorry

end NUMINAMATH_GPT_expected_return_correct_l1984_198428


namespace NUMINAMATH_GPT_total_days_to_finish_tea_and_coffee_l1984_198441

-- Define the given conditions formally before expressing the theorem
def drinks_coffee_together (days : ℕ) : Prop := days = 10
def drinks_coffee_alone_A (days : ℕ) : Prop := days = 12
def drinks_tea_together (days : ℕ) : Prop := days = 12
def drinks_tea_alone_B (days : ℕ) : Prop := days = 20

-- The goal is to prove that A and B together finish a pound of tea and a can of coffee in 35 days
theorem total_days_to_finish_tea_and_coffee : 
  ∃ days : ℕ, 
    drinks_coffee_together 10 ∧ 
    drinks_coffee_alone_A 12 ∧ 
    drinks_tea_together 12 ∧ 
    drinks_tea_alone_B 20 ∧ 
    days = 35 :=
by
  sorry

end NUMINAMATH_GPT_total_days_to_finish_tea_and_coffee_l1984_198441


namespace NUMINAMATH_GPT_calculation_eq_minus_one_l1984_198454

noncomputable def calculation : ℝ :=
  (-1)^(53 : ℤ) + 3^((2^3 + 5^2 - 7^2) : ℤ)

theorem calculation_eq_minus_one : calculation = -1 := 
by 
  sorry

end NUMINAMATH_GPT_calculation_eq_minus_one_l1984_198454


namespace NUMINAMATH_GPT_carpenter_wood_split_l1984_198478

theorem carpenter_wood_split :
  let original_length : ℚ := 35 / 8
  let first_cut : ℚ := 5 / 3
  let second_cut : ℚ := 9 / 4
  let remaining_length := original_length - first_cut - second_cut
  let part_length := remaining_length / 3
  part_length = 11 / 72 :=
sorry

end NUMINAMATH_GPT_carpenter_wood_split_l1984_198478


namespace NUMINAMATH_GPT_find_general_term_l1984_198491

-- Definition of sequence sum condition
def seq_sum_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (2/3) * a n + 1/3

-- Statement of the proof problem
theorem find_general_term (a S : ℕ → ℝ) 
  (h : seq_sum_condition a S) : 
  ∀ n, a n = (-2)^(n-1) := 
by
  sorry

end NUMINAMATH_GPT_find_general_term_l1984_198491


namespace NUMINAMATH_GPT_range_of_m_l1984_198451

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 1) * x + 1

theorem range_of_m (m : ℝ) : (∀ x, x ≤ 1 → f m x ≥ f m 1) ↔ 0 ≤ m ∧ m ≤ 1 / 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1984_198451


namespace NUMINAMATH_GPT_compute_n_l1984_198433

theorem compute_n (n : ℕ) : 5^n = 5 * 25^(3/2) * 125^(5/3) → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_compute_n_l1984_198433


namespace NUMINAMATH_GPT_bicycle_speed_l1984_198484

theorem bicycle_speed (x : ℝ) (h : (2.4 / x) - (2.4 / (4 * x)) = 0.5) : 4 * x = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_speed_l1984_198484


namespace NUMINAMATH_GPT_picnic_problem_l1984_198410

variable (M W A C : ℕ)

theorem picnic_problem (h1 : M = 90)
  (h2 : M = W + 40)
  (h3 : M + W + C = 240) :
  A = M + W ∧ A - C = 40 := by
  sorry

end NUMINAMATH_GPT_picnic_problem_l1984_198410


namespace NUMINAMATH_GPT_no_solution_exists_l1984_198401

theorem no_solution_exists :
  ¬ ∃ m n : ℕ, 
    m + n = 2009 ∧ 
    (m * (m - 1) + n * (n - 1) = 2009 * 2008 / 2) := by
  sorry

end NUMINAMATH_GPT_no_solution_exists_l1984_198401


namespace NUMINAMATH_GPT_sum_of_powers_mod_l1984_198461

-- Define a function that calculates the nth power of a number modulo a given base
def power_mod (a n k : ℕ) : ℕ := (a^n) % k

-- The main theorem: prove that the sum of powers modulo 5 gives the remainder 0
theorem sum_of_powers_mod 
  : ((power_mod 1 2013 5) + (power_mod 2 2013 5) + (power_mod 3 2013 5) + (power_mod 4 2013 5) + (power_mod 5 2013 5)) % 5 = 0 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_powers_mod_l1984_198461


namespace NUMINAMATH_GPT_ellen_smoothies_total_cups_l1984_198437

structure SmoothieIngredients where
  strawberries : ℝ
  yogurt       : ℝ
  orange_juice : ℝ
  honey        : ℝ
  chia_seeds   : ℝ
  spinach      : ℝ

def ounces_to_cups (ounces : ℝ) : ℝ := ounces * 0.125
def tablespoons_to_cups (tablespoons : ℝ) : ℝ := tablespoons * 0.0625

noncomputable def total_cups (ing : SmoothieIngredients) : ℝ :=
  ing.strawberries +
  ing.yogurt +
  ing.orange_juice +
  ounces_to_cups (ing.honey) +
  tablespoons_to_cups (ing.chia_seeds) +
  ing.spinach

theorem ellen_smoothies_total_cups :
  total_cups {
    strawberries := 0.2,
    yogurt := 0.1,
    orange_juice := 0.2,
    honey := 1.0,
    chia_seeds := 2.0,
    spinach := 0.5
  } = 1.25 := by
  sorry

end NUMINAMATH_GPT_ellen_smoothies_total_cups_l1984_198437


namespace NUMINAMATH_GPT_specific_divisors_count_l1984_198421

-- Declare the value of n
def n : ℕ := (2^40) * (3^25) * (5^10)

-- Definition to count the number of positive divisors of a number less than n that don't divide n.
def count_specific_divisors (n : ℕ) : ℕ :=
sorry  -- This would be the function implementation

-- Lean statement to assert the number of such divisors
theorem specific_divisors_count : 
  count_specific_divisors n = 31514 :=
sorry

end NUMINAMATH_GPT_specific_divisors_count_l1984_198421


namespace NUMINAMATH_GPT_find_cos_beta_l1984_198445

noncomputable def cos_beta (α β : ℝ) : ℝ :=
  - (6 * Real.sqrt 2 + 4) / 15

theorem find_cos_beta (α β : ℝ)
  (h0 : α ∈ Set.Ioc 0 (Real.pi / 2))
  (h1 : β ∈ Set.Ioc (Real.pi / 2) Real.pi)
  (h2 : Real.cos α = 1 / 3)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.cos β = cos_beta α β :=
by
  sorry

end NUMINAMATH_GPT_find_cos_beta_l1984_198445


namespace NUMINAMATH_GPT_expression_simplification_l1984_198467

theorem expression_simplification (x y : ℝ) : x^2 + (y - x) * (y + x) = y^2 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1984_198467


namespace NUMINAMATH_GPT_gwen_average_speed_l1984_198482

def average_speed (distance1 distance2 speed1 speed2 : ℕ) : ℕ :=
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  total_distance / total_time

theorem gwen_average_speed :
  average_speed 40 40 15 30 = 20 :=
by
  sorry

end NUMINAMATH_GPT_gwen_average_speed_l1984_198482


namespace NUMINAMATH_GPT_total_time_is_12_years_l1984_198492

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_time_is_12_years_l1984_198492


namespace NUMINAMATH_GPT_milo_eggs_weight_l1984_198405

def weight_of_one_egg : ℚ := 1/16
def eggs_per_dozen : ℕ := 12
def dozens_needed : ℕ := 8

theorem milo_eggs_weight :
  (dozens_needed * eggs_per_dozen : ℚ) * weight_of_one_egg = 6 := by sorry

end NUMINAMATH_GPT_milo_eggs_weight_l1984_198405


namespace NUMINAMATH_GPT_trivia_game_points_per_question_l1984_198436

theorem trivia_game_points_per_question (correct_first_half correct_second_half total_score points_per_question : ℕ) 
  (h1 : correct_first_half = 5) 
  (h2 : correct_second_half = 5) 
  (h3 : total_score = 50) 
  (h4 : correct_first_half + correct_second_half = 10) : 
  points_per_question = 5 :=
by 
  sorry

end NUMINAMATH_GPT_trivia_game_points_per_question_l1984_198436


namespace NUMINAMATH_GPT_find_H2SO4_moles_l1984_198473

-- Let KOH, H2SO4, and KHSO4 represent the moles of each substance in the reaction.
variable (KOH H2SO4 KHSO4 : ℕ)

-- Conditions provided in the problem
def KOH_moles : ℕ := 2
def KHSO4_moles (H2SO4 : ℕ) : ℕ := H2SO4

-- Main statement, we need to prove that given the conditions,
-- 2 moles of KOH and 2 moles of KHSO4 imply 2 moles of H2SO4.
theorem find_H2SO4_moles (KOH_sufficient : KOH = KOH_moles) 
  (KHSO4_produced : KHSO4 = KOH) : KHSO4_moles H2SO4 = 2 := 
sorry

end NUMINAMATH_GPT_find_H2SO4_moles_l1984_198473


namespace NUMINAMATH_GPT_tangent_slope_l1984_198420

noncomputable def f (x : ℝ) : ℝ := x - 1 + 1 / Real.exp x

noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / Real.exp x

theorem tangent_slope (k : ℝ) (x₀ : ℝ) (y₀ : ℝ) 
  (h_tangent_point: (x₀ = -1) ∧ (y₀ = x₀ - 1 + 1 / Real.exp x₀))
  (h_tangent_line : ∀ x, y₀ = f x₀ + f' x₀ * (x - x₀)) :
  k = 1 - Real.exp 1 := 
sorry

end NUMINAMATH_GPT_tangent_slope_l1984_198420


namespace NUMINAMATH_GPT_find_value_l1984_198490

theorem find_value (x : ℝ) (h : x^2 - x - 1 = 0) : 2 * x^2 - 2 * x + 2021 = 2023 := 
by 
  sorry -- Proof needs to be provided

end NUMINAMATH_GPT_find_value_l1984_198490


namespace NUMINAMATH_GPT_hall_area_proof_l1984_198448

noncomputable def hall_length (L : ℕ) : ℕ := L
noncomputable def hall_width (L : ℕ) (W : ℕ) : ℕ := W
noncomputable def hall_area (L W : ℕ) : ℕ := L * W

theorem hall_area_proof (L W : ℕ) (h1 : W = 1 / 2 * L) (h2 : L - W = 15) :
  hall_area L W = 450 := by
  sorry

end NUMINAMATH_GPT_hall_area_proof_l1984_198448


namespace NUMINAMATH_GPT_angle_difference_proof_l1984_198425

-- Define the angles A and B
def angle_A : ℝ := 65
def angle_B : ℝ := 180 - angle_A

-- Define the difference
def angle_difference : ℝ := angle_B - angle_A

theorem angle_difference_proof : angle_difference = 50 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_angle_difference_proof_l1984_198425


namespace NUMINAMATH_GPT_road_building_equation_l1984_198455

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end NUMINAMATH_GPT_road_building_equation_l1984_198455


namespace NUMINAMATH_GPT_donald_oranges_l1984_198469

-- Define the initial number of oranges
def initial_oranges : ℕ := 4

-- Define the number of additional oranges found
def additional_oranges : ℕ := 5

-- Define the total number of oranges as the sum of initial and additional oranges
def total_oranges : ℕ := initial_oranges + additional_oranges

-- Theorem stating that the total number of oranges is 9
theorem donald_oranges : total_oranges = 9 := by
    -- Proof not provided, so we put sorry to indicate that this is a place for the proof.
    sorry

end NUMINAMATH_GPT_donald_oranges_l1984_198469


namespace NUMINAMATH_GPT_player1_points_after_13_rotations_l1984_198465

theorem player1_points_after_13_rotations :
  ∃ (player1_points : ℕ), 
    (∀ (i : ℕ),  (i = 5 → player1_points = 72) ∧ (i = 9 → player1_points = 84)) → 
    player1_points = 20 :=
by
  sorry

end NUMINAMATH_GPT_player1_points_after_13_rotations_l1984_198465


namespace NUMINAMATH_GPT_abs_quadratic_bound_l1984_198480

theorem abs_quadratic_bound (a b : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + a * x + b) :
  (|f 1| ≥ (1 / 2)) ∨ (|f 2| ≥ (1 / 2)) ∨ (|f 3| ≥ (1 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_abs_quadratic_bound_l1984_198480


namespace NUMINAMATH_GPT_division_by_fraction_l1984_198457

theorem division_by_fraction :
  (3 : ℚ) / (6 / 11) = 11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_division_by_fraction_l1984_198457


namespace NUMINAMATH_GPT_unknown_number_is_105_l1984_198477

theorem unknown_number_is_105 :
  ∃ x : ℝ, x^2 + 94^2 = 19872 ∧ x = 105 :=
by
  sorry

end NUMINAMATH_GPT_unknown_number_is_105_l1984_198477


namespace NUMINAMATH_GPT_find_x_l1984_198497

theorem find_x (x : ℤ) (h : (1 + 2 + 4 + 5 + 6 + 9 + 9 + 10 + 12 + x) / 10 = 7) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1984_198497


namespace NUMINAMATH_GPT_total_marks_math_physics_l1984_198485

variables (M P C : ℕ)
axiom condition1 : C = P + 20
axiom condition2 : (M + C) / 2 = 45

theorem total_marks_math_physics : M + P = 70 :=
by sorry

end NUMINAMATH_GPT_total_marks_math_physics_l1984_198485


namespace NUMINAMATH_GPT_rectangle_perimeters_l1984_198442

theorem rectangle_perimeters (a b : ℕ) (h1 : a ≠ b) (h2 : a * b = 3 * (2 * a + 2 * b)) : 
  2 * (a + b) = 36 ∨ 2 * (a + b) = 28 :=
by sorry

end NUMINAMATH_GPT_rectangle_perimeters_l1984_198442


namespace NUMINAMATH_GPT_sum_of_three_consecutive_eq_product_of_distinct_l1984_198404

theorem sum_of_three_consecutive_eq_product_of_distinct (n : ℕ) (h : 100 < n) :
  ∃ a b c, (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ a > 1 ∧ b > 1 ∧ c > 1 ∧
  ((n + (n+1) + (n+2) = a * b * c) ∨
   ((n+1) + (n+2) + (n+3) = a * b * c) ∨
   (n + (n+1) + (n+3) = a * b * c) ∨
   (n + (n+2) + (n+3) = a * b * c)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_eq_product_of_distinct_l1984_198404


namespace NUMINAMATH_GPT_Jana_new_walking_speed_l1984_198418

variable (minutes : ℕ) (distance1 distance2 : ℝ)

-- Given conditions
def minutes_taken_to_walk := 30
def current_distance := 2
def new_distance := 3
def time_in_hours := minutes / 60

-- Define outcomes
def current_speed_per_minute := current_distance / minutes
def current_speed_per_hour := current_speed_per_minute * 60
def required_speed_per_minute := new_distance / minutes
def required_speed_per_hour := required_speed_per_minute * 60

-- Final statement to prove
theorem Jana_new_walking_speed : required_speed_per_hour = 6 := by
  sorry

end NUMINAMATH_GPT_Jana_new_walking_speed_l1984_198418


namespace NUMINAMATH_GPT_set_intersection_l1984_198440

noncomputable def U : Set ℝ := Set.univ
noncomputable def M : Set ℝ := {x | x < 3}
noncomputable def N : Set ℝ := {y | y > 2}
noncomputable def CU_M : Set ℝ := {x | x ≥ 3}

theorem set_intersection :
  (CU_M ∩ N) = {x | x ≥ 3} := by
  sorry

end NUMINAMATH_GPT_set_intersection_l1984_198440


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1984_198496

theorem geometric_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1/2)
  (h2 : a 3 + a 4 = 1)
  (h_geom : ∀ n, a n + a (n+1) = (a 1 + a 2) * 2^(n-1)) :
  a 7 + a 8 + a 9 + a 10 = 12 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1984_198496


namespace NUMINAMATH_GPT_problem_l1984_198417

theorem problem (n : ℕ) (h : n = 8 ^ 2022) : n / 4 = 4 ^ 3032 := 
sorry

end NUMINAMATH_GPT_problem_l1984_198417


namespace NUMINAMATH_GPT_inequality_proof_l1984_198466

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256 * x * y * z :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inequality_proof_l1984_198466


namespace NUMINAMATH_GPT_angle_Q_measure_in_triangle_PQR_l1984_198460

theorem angle_Q_measure_in_triangle_PQR (angle_R angle_Q angle_P : ℝ) (h1 : angle_P = 3 * angle_R) (h2 : angle_Q = angle_R) (h3 : angle_R + angle_Q + angle_P = 180) : angle_Q = 36 :=
by {
  -- Placeholder for the proof, which is not required as per the instructions
  sorry
}

end NUMINAMATH_GPT_angle_Q_measure_in_triangle_PQR_l1984_198460


namespace NUMINAMATH_GPT_no_rational_solution_of_odd_quadratic_l1984_198481

theorem no_rational_solution_of_odd_quadratic (a b c : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ¬ ∃ x : ℚ, a * x^2 + b * x + c = 0 :=
sorry

end NUMINAMATH_GPT_no_rational_solution_of_odd_quadratic_l1984_198481


namespace NUMINAMATH_GPT_jonathan_daily_calories_l1984_198423

theorem jonathan_daily_calories (C : ℕ) (daily_burn weekly_deficit extra_calories total_burn : ℕ) 
  (h1 : daily_burn = 3000) 
  (h2 : weekly_deficit = 2500) 
  (h3 : extra_calories = 1000) 
  (h4 : total_burn = 7 * daily_burn) 
  (h5 : total_burn - weekly_deficit = 7 * C + extra_calories) :
  C = 2500 :=
by 
  sorry

end NUMINAMATH_GPT_jonathan_daily_calories_l1984_198423


namespace NUMINAMATH_GPT_compute_expression_l1984_198486

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 :=
by sorry

end NUMINAMATH_GPT_compute_expression_l1984_198486


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1984_198412

theorem rhombus_diagonal_length 
  (side_length : ℕ) (shorter_diagonal : ℕ) (longer_diagonal : ℕ)
  (h1 : side_length = 34) (h2 : shorter_diagonal = 32) :
  longer_diagonal = 60 :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1984_198412


namespace NUMINAMATH_GPT_first_train_speed_is_80_kmph_l1984_198407

noncomputable def speedOfFirstTrain
  (lenFirstTrain : ℝ)
  (lenSecondTrain : ℝ)
  (speedSecondTrain : ℝ)
  (clearTime : ℝ)
  (oppositeDirections : Bool) : ℝ :=
  if oppositeDirections then
    let totalDistance := (lenFirstTrain + lenSecondTrain) / 1000  -- convert meters to kilometers
    let timeHours := clearTime / 3600 -- convert seconds to hours
    let relativeSpeed := totalDistance / timeHours
    relativeSpeed - speedSecondTrain
  else
    0 -- This should not happen based on problem conditions

theorem first_train_speed_is_80_kmph :
  speedOfFirstTrain 151 165 65 7.844889650207294 true = 80 :=
by
  sorry

end NUMINAMATH_GPT_first_train_speed_is_80_kmph_l1984_198407


namespace NUMINAMATH_GPT_norma_cards_lost_l1984_198464

def initial_cards : ℕ := 88
def final_cards : ℕ := 18
def cards_lost : ℕ := initial_cards - final_cards

theorem norma_cards_lost : cards_lost = 70 :=
by
  sorry

end NUMINAMATH_GPT_norma_cards_lost_l1984_198464


namespace NUMINAMATH_GPT_solution_count_l1984_198409

theorem solution_count (a : ℝ) :
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) → 
  (∃ (num_solutions : ℕ), 
    (num_solutions = 3 ∧ a = 1 ∨ a = -1) ∨ 
    (num_solutions = 2 ∧ a = Real.sqrt 2 ∨ a = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_GPT_solution_count_l1984_198409


namespace NUMINAMATH_GPT_vertical_strips_count_l1984_198468

theorem vertical_strips_count (a b x y : ℕ)
  (h_outer : 2 * a + 2 * b = 50)
  (h_inner : 2 * x + 2 * y = 32)
  (h_strips : a + x = 20) :
  b + y = 21 :=
by
  have h1 : a + b = 25 := by
    linarith
  have h2 : x + y = 16 := by
    linarith
  linarith


end NUMINAMATH_GPT_vertical_strips_count_l1984_198468


namespace NUMINAMATH_GPT_marching_band_max_l1984_198495

-- Define the conditions
variables (m k n : ℕ)

-- Lean statement of the problem
theorem marching_band_max (H1 : m = k^2 + 9) (H2 : m = n * (n + 5)) : m = 234 :=
sorry

end NUMINAMATH_GPT_marching_band_max_l1984_198495


namespace NUMINAMATH_GPT_sum_of_numbers_l1984_198444

variable {R : Type*} [LinearOrderedField R]

theorem sum_of_numbers (x y : R) (h1 : x ≠ y) (h2 : x^2 - 2000*x = y^2 - 2000*y) : x + y = 2000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l1984_198444


namespace NUMINAMATH_GPT_heather_starts_24_minutes_after_stacy_l1984_198476

theorem heather_starts_24_minutes_after_stacy :
  ∀ (distance_between : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) (heather_distance : ℝ),
    distance_between = 10 →
    heather_speed = 5 →
    stacy_speed = heather_speed + 1 →
    heather_distance = 3.4545454545454546 →
    60 * ((heather_distance / heather_speed) - ((distance_between - heather_distance) / stacy_speed)) = -24 :=
by
  sorry

end NUMINAMATH_GPT_heather_starts_24_minutes_after_stacy_l1984_198476


namespace NUMINAMATH_GPT_lamp_pricing_problem_l1984_198462

theorem lamp_pricing_problem
  (purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (sales_decrease_rate : ℝ)
  (desired_profit : ℝ) :
  purchase_price = 30 →
  initial_selling_price = 40 →
  initial_sales_volume = 600 →
  sales_decrease_rate = 10 →
  desired_profit = 10000 →
  (∃ (selling_price : ℝ) (sales_volume : ℝ), selling_price = 50 ∧ sales_volume = 500) :=
by
  intros h_purchase h_initial_selling h_initial_sales h_sales_decrease h_desired_profit
  sorry

end NUMINAMATH_GPT_lamp_pricing_problem_l1984_198462


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1984_198439

def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}
def CU_A : Set ℤ := {x ∈ U | x ∉ A}

theorem complement_of_A_in_U :
  CU_A = {-2, 1, 5} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1984_198439


namespace NUMINAMATH_GPT_coat_price_reduction_l1984_198458

theorem coat_price_reduction 
  (original_price : ℝ) 
  (reduction_percentage : ℝ)
  (h_original_price : original_price = 500)
  (h_reduction_percentage : reduction_percentage = 60) :
  original_price * (reduction_percentage / 100) = 300 :=
by 
  sorry

end NUMINAMATH_GPT_coat_price_reduction_l1984_198458


namespace NUMINAMATH_GPT_gcd_b_n_b_n_plus_1_l1984_198427

-- Definitions based on the conditions in the problem
def b_n (n : ℕ) : ℕ := 150 + n^3

theorem gcd_b_n_b_n_plus_1 (n : ℕ) : gcd (b_n n) (b_n (n + 1)) = 1 := by
  -- We acknowledge that we need to skip the proof steps
  sorry

end NUMINAMATH_GPT_gcd_b_n_b_n_plus_1_l1984_198427


namespace NUMINAMATH_GPT_contrapositive_proposition_l1984_198499

theorem contrapositive_proposition
  (a b c d : ℝ) 
  (h : a + c ≠ b + d) : a ≠ b ∨ c ≠ d :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_l1984_198499


namespace NUMINAMATH_GPT_fifty_percent_greater_l1984_198432

theorem fifty_percent_greater (x : ℕ) (h : x = 88 + (88 / 2)) : x = 132 := 
by {
  sorry
}

end NUMINAMATH_GPT_fifty_percent_greater_l1984_198432


namespace NUMINAMATH_GPT_mingming_actual_height_l1984_198463

def mingming_height (h : ℝ) : Prop := 1.495 ≤ h ∧ h < 1.505

theorem mingming_actual_height : ∃ α : ℝ, mingming_height α :=
by
  use 1.50
  sorry

end NUMINAMATH_GPT_mingming_actual_height_l1984_198463


namespace NUMINAMATH_GPT_greatest_possible_b_l1984_198483

theorem greatest_possible_b (b : ℕ) (h : ∃ x : ℤ, x^2 + b * x = -21) : b ≤ 22 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_b_l1984_198483


namespace NUMINAMATH_GPT_complex_number_value_l1984_198430

-- Declare the imaginary unit 'i'
noncomputable def i : ℂ := Complex.I

-- Define the problem statement
theorem complex_number_value : (i / ((1 - i) ^ 2)) = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_complex_number_value_l1984_198430


namespace NUMINAMATH_GPT_find_arith_seq_common_diff_l1984_198403

-- Let a_n be the nth term of the arithmetic sequence and S_n be the sum of the first n terms
variable {a : ℕ → ℝ} -- arithmetic sequence
variable {S : ℕ → ℝ} -- Sum of first n terms of the sequence

-- Given conditions in the problem
axiom sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3)
axiom sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2)
axiom condition1 : ((S 4) / 12) - ((S 3) / 9) = 1

-- Prove that the common difference d is 6
theorem find_arith_seq_common_diff (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (sum_first_4_terms : S 4 = (4 / 2) * (2 * a 1 + 3))
  (sum_first_3_terms : S 3 = (3 / 2) * (2 * a 1 + 2))
  (condition1 : (S 4) / 12 - (S 3) / 9 = 1) : 
  d = 6 := 
sorry

end NUMINAMATH_GPT_find_arith_seq_common_diff_l1984_198403


namespace NUMINAMATH_GPT_opposite_of_neg_third_l1984_198413

theorem opposite_of_neg_third : (-(-1 / 3)) = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_third_l1984_198413


namespace NUMINAMATH_GPT_birds_flew_up_l1984_198470

-- Definitions based on conditions in the problem
def initial_birds : ℕ := 29
def new_total_birds : ℕ := 42

-- The statement to be proven
theorem birds_flew_up (x y z : ℕ) (h1 : x = initial_birds) (h2 : y = new_total_birds) (h3 : z = y - x) : z = 13 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_birds_flew_up_l1984_198470


namespace NUMINAMATH_GPT_joe_collected_cards_l1984_198489

theorem joe_collected_cards (boxes : ℕ) (cards_per_box : ℕ) (filled_boxes : boxes = 11) (max_cards_per_box : cards_per_box = 8) : boxes * cards_per_box = 88 := by
  sorry

end NUMINAMATH_GPT_joe_collected_cards_l1984_198489


namespace NUMINAMATH_GPT_bacteria_fill_sixteenth_of_dish_in_26_days_l1984_198429

theorem bacteria_fill_sixteenth_of_dish_in_26_days
  (days_to_fill_dish : ℕ)
  (doubling_rate : ℕ → ℕ)
  (H1 : days_to_fill_dish = 30)
  (H2 : ∀ n, doubling_rate (n + 1) = 2 * doubling_rate n) :
  doubling_rate 26 = doubling_rate 30 / 2^4 :=
sorry

end NUMINAMATH_GPT_bacteria_fill_sixteenth_of_dish_in_26_days_l1984_198429


namespace NUMINAMATH_GPT_pentagon_angles_l1984_198422

theorem pentagon_angles (M T H A S : ℝ) 
  (h1 : M = T) 
  (h2 : T = H) 
  (h3 : A + S = 180) 
  (h4 : M + A + T + H + S = 540) : 
  H = 120 := 
by 
  -- The proof would be inserted here.
  sorry

end NUMINAMATH_GPT_pentagon_angles_l1984_198422


namespace NUMINAMATH_GPT_inequality_proof_l1984_198447

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  (2 * a / (a^2 + b * c) + 2 * b / (b^2 + c * a) + 2 * c / (c^2 + a * b)) ≤ (a / (b * c) + b / (c * a) + c / (a * b)) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1984_198447


namespace NUMINAMATH_GPT_find_num_tables_l1984_198414

-- Definitions based on conditions
def num_students_in_class : ℕ := 47
def num_girls_bathroom : ℕ := 3
def num_students_canteen : ℕ := 3 * 3
def num_students_new_groups : ℕ := 2 * 4
def num_students_exchange : ℕ := 3 * 3 + 3 * 3 + 3 * 3

-- Calculation of the number of tables (corresponding to the answer)
def num_missing_students : ℕ := num_girls_bathroom + num_students_canteen + num_students_new_groups + num_students_exchange

def num_students_currently_in_class : ℕ := num_students_in_class - num_missing_students
def students_per_table : ℕ := 3

def num_tables : ℕ := num_students_currently_in_class / students_per_table

-- The theorem we want to prove
theorem find_num_tables : num_tables = 6 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_num_tables_l1984_198414


namespace NUMINAMATH_GPT_a_pow_11_b_pow_11_l1984_198471

-- Define the conditions a + b = 1, a^2 + b^2 = 3, a^3 + b^3 = 4, a^4 + b^4 = 7, and a^5 + b^5 = 11
def a : ℝ := sorry
def b : ℝ := sorry

axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Define the recursion pattern for n ≥ 3
axiom h6 (n : ℕ) (hn : n ≥ 3) : a^n + b^n = a^(n-1) + b^(n-1) + a^(n-2) + b^(n-2)

-- Prove that a^11 + b^11 = 199
theorem a_pow_11_b_pow_11 : a^11 + b^11 = 199 :=
by sorry

end NUMINAMATH_GPT_a_pow_11_b_pow_11_l1984_198471


namespace NUMINAMATH_GPT_lending_rate_l1984_198494

noncomputable def principal: ℝ := 5000
noncomputable def rate_borrowed: ℝ := 4
noncomputable def time_years: ℝ := 2
noncomputable def gain_per_year: ℝ := 100

theorem lending_rate :
  ∃ (rate_lent: ℝ), 
  (principal * rate_lent * time_years / 100) - (principal * rate_borrowed * time_years / 100) / time_years = gain_per_year ∧
  rate_lent = 6 :=
by
  sorry

end NUMINAMATH_GPT_lending_rate_l1984_198494


namespace NUMINAMATH_GPT_wang_hao_not_last_l1984_198452

-- Define the total number of ways to select and arrange 3 players out of 6
def ways_total : ℕ := Nat.factorial 6 / Nat.factorial (6 - 3)

-- Define the number of ways in which Wang Hao is the last player
def ways_wang_last : ℕ := Nat.factorial 5 / Nat.factorial (5 - 2)

-- Proof statement
theorem wang_hao_not_last : ways_total - ways_wang_last = 100 :=
by sorry

end NUMINAMATH_GPT_wang_hao_not_last_l1984_198452


namespace NUMINAMATH_GPT_total_students_l1984_198459

theorem total_students (S : ℕ) (H1 : S / 2 = S - 15) : S = 30 :=
sorry

end NUMINAMATH_GPT_total_students_l1984_198459


namespace NUMINAMATH_GPT_log_sequence_equality_l1984_198419

theorem log_sequence_equality (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 1) (h2: a 2 + a 4 + a 6 = 18) : 
  Real.logb 3 (a 5 + a 7 + a 9) = 3 := 
by
  sorry

end NUMINAMATH_GPT_log_sequence_equality_l1984_198419


namespace NUMINAMATH_GPT_remainder_when_3y_divided_by_9_l1984_198411

theorem remainder_when_3y_divided_by_9 (y : ℕ) (k : ℕ) (hy : y = 9 * k + 5) : (3 * y) % 9 = 6 :=
sorry

end NUMINAMATH_GPT_remainder_when_3y_divided_by_9_l1984_198411


namespace NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l1984_198498

theorem arithmetic_sequence_tenth_term (a d : ℤ) (h₁ : a + 3 * d = 23) (h₂ : a + 8 * d = 38) : a + 9 * d = 41 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_tenth_term_l1984_198498


namespace NUMINAMATH_GPT_walk_to_bus_stop_time_l1984_198443

/-- Walking with 4/5 of my usual speed, I arrive at the bus stop 7 minutes later than normal.
    How many minutes does it take to walk to the bus stop at my usual speed? -/
theorem walk_to_bus_stop_time (S T : ℝ) (h : T > 0) 
  (d_usual : S * T = (4/5) * S * (T + 7)) : 
  T = 28 :=
by
  sorry

end NUMINAMATH_GPT_walk_to_bus_stop_time_l1984_198443


namespace NUMINAMATH_GPT_tourist_journey_home_days_l1984_198453

theorem tourist_journey_home_days (x v : ℝ)
  (h1 : (x / 2 + 1) * v = 246)
  (h2 : x * (v + 15) = 276) :
  x + (x / 2 + 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_tourist_journey_home_days_l1984_198453


namespace NUMINAMATH_GPT_molecular_weight_of_compound_l1984_198474

-- Given atomic weights in g/mol
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O  : ℝ := 15.999
def atomic_weight_H  : ℝ := 1.008

-- Given number of atoms in the compound
def num_atoms_Ca : ℕ := 1
def num_atoms_O  : ℕ := 2
def num_atoms_H  : ℕ := 2

-- Definition of the molecular weight
def molecular_weight : ℝ :=
  (num_atoms_Ca * atomic_weight_Ca) +
  (num_atoms_O * atomic_weight_O) +
  (num_atoms_H * atomic_weight_H)

-- The theorem to prove
theorem molecular_weight_of_compound : molecular_weight = 74.094 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_compound_l1984_198474


namespace NUMINAMATH_GPT_shaded_area_of_logo_l1984_198408

theorem shaded_area_of_logo 
  (side_length_of_square : ℝ)
  (side_length_of_square_eq : side_length_of_square = 30)
  (radius_of_circle : ℝ)
  (radius_eq : radius_of_circle = side_length_of_square / 4)
  (number_of_circles : ℕ)
  (number_of_circles_eq : number_of_circles = 4)
  : (side_length_of_square^2) - (number_of_circles * Real.pi * (radius_of_circle^2)) = 900 - 225 * Real.pi := by
    sorry

end NUMINAMATH_GPT_shaded_area_of_logo_l1984_198408


namespace NUMINAMATH_GPT_octagon_mass_is_19kg_l1984_198402

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end NUMINAMATH_GPT_octagon_mass_is_19kg_l1984_198402


namespace NUMINAMATH_GPT_largest_valid_number_l1984_198416

/-
Problem: 
What is the largest number, all of whose digits are 3, 2, or 4 whose digits add up to 16?

We prove that 4432 is the largest such number.
-/

def digits := [3, 2, 4]

def sum_of_digits (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 2 ∨ d = 4

def generate_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc d => acc * 10 + d) 0

theorem largest_valid_number : 
  ∃ l : List ℕ, (∀ d ∈ l, is_valid_digit d) ∧ sum_of_digits l = 16 ∧ generate_number l = 4432 :=
  sorry

end NUMINAMATH_GPT_largest_valid_number_l1984_198416


namespace NUMINAMATH_GPT_calculate_star_operation_l1984_198493

def operation (a b : ℚ) : ℚ := 2 * a - b + 1

theorem calculate_star_operation :
  operation 1 (operation 3 (-2)) = -6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_star_operation_l1984_198493


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1984_198456

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : a 1 + a 7 = 22) 
  (h2 : a 4 + a 10 = 40) 
  (h_general_term : ∀ n : ℕ, a n = a 1 + (n - 1) * d) 
  : d = 3 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1984_198456


namespace NUMINAMATH_GPT_prob_at_least_seven_friends_stay_for_entire_game_l1984_198431

-- Definitions of conditions
def numFriends : ℕ := 8
def numUnsureFriends : ℕ := 5
def probabilityStay (p : ℚ) : ℚ := p
def sureFriends := 3

-- The probabilities
def prob_one_third : ℚ := 1 / 3
def prob_two_thirds : ℚ := 2 / 3

-- Variables to hold binomial coefficient and power calculation
noncomputable def C (n k : ℕ) : ℚ := (Nat.choose n k)
noncomputable def probability_at_least_seven_friends_stay : ℚ :=
  C numUnsureFriends 4 * (probabilityStay prob_one_third)^4 * (probabilityStay prob_two_thirds)^1 +
  (probabilityStay prob_one_third)^5

-- Theorem statement
theorem prob_at_least_seven_friends_stay_for_entire_game :
  probability_at_least_seven_friends_stay = 11 / 243 :=
  by sorry

end NUMINAMATH_GPT_prob_at_least_seven_friends_stay_for_entire_game_l1984_198431


namespace NUMINAMATH_GPT_sum_of_even_conditions_l1984_198472

theorem sum_of_even_conditions (m n : ℤ) :
  ((∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → ∃ p : ℤ, m + n = 2 * p) ∧
  (∃ q : ℤ, m + n = 2 * q → (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → False) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_even_conditions_l1984_198472


namespace NUMINAMATH_GPT_pages_read_over_weekend_l1984_198435

-- Define the given conditions
def total_pages : ℕ := 408
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

-- Define the calculated pages to be read over the remaining days
def pages_remaining := days_left * pages_per_day

-- Define the pages read over the weekend
def pages_over_weekend := total_pages - pages_remaining

-- Prove that Bekah read 113 pages over the weekend
theorem pages_read_over_weekend : pages_over_weekend = 113 :=
by {
  -- proof should be here, but we place sorry since proof is not required
  sorry
}

end NUMINAMATH_GPT_pages_read_over_weekend_l1984_198435


namespace NUMINAMATH_GPT_find_f_expression_l1984_198487

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = x^2 + 2 * x + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_f_expression_l1984_198487


namespace NUMINAMATH_GPT_intersection_complement_eq_l1984_198434

def M : Set ℝ := {-1, 1, 2, 4}
def N : Set ℝ := {x : ℝ | x^2 - 2 * x ≥ 3 }
def N_complement : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_complement_eq :
  M ∩ N_complement = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1984_198434


namespace NUMINAMATH_GPT_greatest_number_of_kits_l1984_198406

-- Given conditions
def bottles_of_water := 20
def cans_of_food := 12
def flashlights := 30
def blankets := 18

def no_more_than_10_items_per_kit (kits : ℕ) := 
  (bottles_of_water / kits ≤ 10) ∧ 
  (cans_of_food / kits ≤ 10) ∧ 
  (flashlights / kits ≤ 10) ∧ 
  (blankets / kits ≤ 10)

def greater_than_or_equal_to_5_kits (kits : ℕ) := kits ≥ 5

def all_items_distributed_equally (kits : ℕ) := 
  (bottles_of_water % kits = 0) ∧ 
  (cans_of_food % kits = 0) ∧ 
  (flashlights % kits = 0) ∧ 
  (blankets % kits = 0)

-- Proof goal
theorem greatest_number_of_kits : 
  ∃ kits : ℕ, 
    no_more_than_10_items_per_kit kits ∧ 
    greater_than_or_equal_to_5_kits kits ∧ 
    all_items_distributed_equally kits ∧ 
    kits = 6 := 
sorry

end NUMINAMATH_GPT_greatest_number_of_kits_l1984_198406


namespace NUMINAMATH_GPT_finished_in_6th_l1984_198424

variable (p : ℕ → Prop)
variable (Sana Max Omar Jonah Leila : ℕ)

-- Conditions
def condition1 : Prop := Omar = Jonah - 7
def condition2 : Prop := Sana = Max - 2
def condition3 : Prop := Leila = Jonah + 3
def condition4 : Prop := Max = Omar + 1
def condition5 : Prop := Sana = 4

-- Conclusion
theorem finished_in_6th (h1 : condition1 Omar Jonah)
                         (h2 : condition2 Sana Max)
                         (h3 : condition3 Leila Jonah)
                         (h4 : condition4 Max Omar)
                         (h5 : condition5 Sana) :
  Max = 6 := by
  sorry

end NUMINAMATH_GPT_finished_in_6th_l1984_198424


namespace NUMINAMATH_GPT_problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l1984_198450

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_part_1_solution_set_of_f_when_a_is_3 :
  {x : ℝ | 0 ≤ x ∧ x ≤ 3} = {x : ℝ | f x 3 ≤ 6} :=
by
  sorry

def g (x : ℝ) : ℝ := abs (2 * x - 3)

theorem problem_part_2_range_of_a :
  {a : ℝ | 4 ≤ a} = {a : ℝ | ∀ x : ℝ, f x a + g x ≥ 5} :=
by
  sorry

end NUMINAMATH_GPT_problem_part_1_solution_set_of_f_when_a_is_3_problem_part_2_range_of_a_l1984_198450
