import Mathlib

namespace NUMINAMATH_GPT_sum_of_a_for_repeated_root_l1794_179408

theorem sum_of_a_for_repeated_root :
  ∀ a : ℝ, (∀ x : ℝ, 2 * x^2 + a * x + 10 * x + 16 = 0 → 
               (a + 10 = 8 * Real.sqrt 2 ∨ a + 10 = -8 * Real.sqrt 2)) → 
               (a = -10 + 8 * Real.sqrt 2 ∨ a = -10 - 8 * Real.sqrt 2) → 
               ((-10 + 8 * Real.sqrt 2) + (-10 - 8 * Real.sqrt 2) = -20) := by
sorry

end NUMINAMATH_GPT_sum_of_a_for_repeated_root_l1794_179408


namespace NUMINAMATH_GPT_required_run_rate_is_correct_l1794_179461

-- Define the initial conditions
def run_rate_first_10_overs : ℝ := 3.2
def overs_first_10 : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 40

-- Given total runs in the first 10 overs
def total_runs_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_10
-- Given runs needed in the remaining 40 overs
def runs_needed_remaining_overs : ℝ := target_runs - total_runs_first_10_overs

-- Lean statement to prove the required run rate in the remaining 40 overs
theorem required_run_rate_is_correct (h1 : run_rate_first_10_overs = 3.2)
                                     (h2 : overs_first_10 = 10)
                                     (h3 : target_runs = 282)
                                     (h4 : remaining_overs = 40) :
  (runs_needed_remaining_overs / remaining_overs) = 6.25 :=
by sorry


end NUMINAMATH_GPT_required_run_rate_is_correct_l1794_179461


namespace NUMINAMATH_GPT_distinct_real_roots_max_abs_gt_2_l1794_179492

theorem distinct_real_roots_max_abs_gt_2 
  (r1 r2 r3 q : ℝ)
  (h_distinct : r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h_sum : r1 + r2 + r3 = -q)
  (h_product : r1 * r2 * r3 = -9)
  (h_sum_prod : r1 * r2 + r2 * r3 + r3 * r1 = 6)
  (h_nonzero_discriminant : q^2 * 6^2 - 4 * 6^3 - 4 * q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9) ≠ 0) :
  max (|r1|) (max (|r2|) (|r3|)) > 2 :=
sorry

end NUMINAMATH_GPT_distinct_real_roots_max_abs_gt_2_l1794_179492


namespace NUMINAMATH_GPT_chocoBites_mod_l1794_179405

theorem chocoBites_mod (m : ℕ) (hm : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end NUMINAMATH_GPT_chocoBites_mod_l1794_179405


namespace NUMINAMATH_GPT_range_of_m_l1794_179437

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x^2)/(9 - m) + (y^2)/(m - 5) = 1 → 
  (∃ m, (7 < m ∧ m < 9))) := 
sorry

end NUMINAMATH_GPT_range_of_m_l1794_179437


namespace NUMINAMATH_GPT_triplet_solution_l1794_179466

theorem triplet_solution (a b c : ℕ) (h1 : a^2 + b^2 + c^2 = 2005) (h2 : a ≤ b) (h3 : b ≤ c) :
  (a = 24 ∧ b = 30 ∧ c = 23) ∨ 
  (a = 12 ∧ b = 30 ∧ c = 31) ∨
  (a = 18 ∧ b = 40 ∧ c = 9) ∨
  (a = 15 ∧ b = 22 ∧ c = 36) ∨
  (a = 12 ∧ b = 30 ∧ c = 31) :=
sorry

end NUMINAMATH_GPT_triplet_solution_l1794_179466


namespace NUMINAMATH_GPT_infinitely_many_primes_congruent_3_mod_4_l1794_179478

def is_congruent_3_mod_4 (p : ℕ) : Prop :=
  p % 4 = 3

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def S (p : ℕ) : Prop :=
  is_prime p ∧ is_congruent_3_mod_4 p

theorem infinitely_many_primes_congruent_3_mod_4 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ S p :=
sorry

end NUMINAMATH_GPT_infinitely_many_primes_congruent_3_mod_4_l1794_179478


namespace NUMINAMATH_GPT_problem_1_problem_2_l1794_179422

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), Real.cos (2 * x))
noncomputable def vec_b : ℝ × ℝ := (1 / 2, Real.sqrt 3 / 2)
noncomputable def f (x : ℝ) : ℝ := (vec_a x).fst * vec_b.fst + (vec_a x).snd * vec_b.snd + 2

theorem problem_1 (x : ℝ) : x ∈ Set.Icc (k * Real.pi - (5 / 12) * Real.pi) (k * Real.pi + (1 / 12) * Real.pi) → ∃ k : ℤ, ∀ x : ℝ, f (x) = Real.sin (2 * x + (1 / 3) * Real.pi) + 2 :=
sorry

theorem problem_2 (x : ℝ) : x ∈ Set.Icc (π / 6) (2 * π / 3) → f (π / 6) = (Real.sqrt 3 / 2) + 2 ∧ f (7 * π / 12) = 1 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1794_179422


namespace NUMINAMATH_GPT_ratio_melina_alma_age_l1794_179485

theorem ratio_melina_alma_age
  (A M : ℕ)
  (alma_score : ℕ)
  (h1 : M = 60)
  (h2 : alma_score = 40)
  (h3 : A + M = 2 * alma_score)
  : M / A = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_melina_alma_age_l1794_179485


namespace NUMINAMATH_GPT_rational_cos_terms_l1794_179416

open Real

noncomputable def rational_sum (x : ℝ) (rS : ℚ) (rC : ℚ) :=
  let S := sin (64 * x) + sin (65 * x)
  let C := cos (64 * x) + cos (65 * x)
  S = rS ∧ C = rC

theorem rational_cos_terms (x : ℝ) (rS : ℚ) (rC : ℚ) :
  rational_sum x rS rC → (∃ q1 q2 : ℚ, cos (64 * x) = q1 ∧ cos (65 * x) = q2) :=
sorry

end NUMINAMATH_GPT_rational_cos_terms_l1794_179416


namespace NUMINAMATH_GPT_student_average_vs_true_average_l1794_179487

theorem student_average_vs_true_average (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2 * w + 2 * x + y + z) / 6 < (w + x + y + z) / 4 :=
by
  sorry

end NUMINAMATH_GPT_student_average_vs_true_average_l1794_179487


namespace NUMINAMATH_GPT_find_pairs_l1794_179410

open Nat

-- m and n are odd natural numbers greater than 2009
def is_odd_gt_2009 (x : ℕ) : Prop := (x % 2 = 1) ∧ (x > 2009)

-- condition: m divides n^2 + 8
def divides_m_n_squared_plus_8 (m n : ℕ) : Prop := m ∣ (n ^ 2 + 8)

-- condition: n divides m^2 + 8
def divides_n_m_squared_plus_8 (m n : ℕ) : Prop := n ∣ (m ^ 2 + 8)

-- Final statement
theorem find_pairs :
  ∃ m n : ℕ, is_odd_gt_2009 m ∧ is_odd_gt_2009 n ∧ divides_m_n_squared_plus_8 m n ∧ divides_n_m_squared_plus_8 m n ∧ ((m, n) = (881, 89) ∨ (m, n) = (3303, 567)) :=
sorry

end NUMINAMATH_GPT_find_pairs_l1794_179410


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1794_179400

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_of_a_l1794_179400


namespace NUMINAMATH_GPT_total_number_of_questions_l1794_179443

theorem total_number_of_questions (type_a_problems type_b_problems : ℕ) 
(time_spent_type_a time_spent_type_b : ℕ) 
(total_exam_time : ℕ) 
(h1 : type_a_problems = 50) 
(h2 : time_spent_type_a = 2 * time_spent_type_b) 
(h3 : time_spent_type_a * type_a_problems = 72) 
(h4 : total_exam_time = 180) :
type_a_problems + type_b_problems = 200 := 
by
  sorry

end NUMINAMATH_GPT_total_number_of_questions_l1794_179443


namespace NUMINAMATH_GPT_actual_cost_of_article_l1794_179479

theorem actual_cost_of_article (x : ℝ) (hx : 0.76 * x = 988) : x = 1300 :=
sorry

end NUMINAMATH_GPT_actual_cost_of_article_l1794_179479


namespace NUMINAMATH_GPT_nonnegative_integer_pairs_solution_l1794_179455

theorem nonnegative_integer_pairs_solution :
  ∀ (x y: ℕ), ((x * y + 2) ^ 2 = x^2 + y^2) ↔ (x = 0 ∧ y = 2) ∨ (x = 2 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_nonnegative_integer_pairs_solution_l1794_179455


namespace NUMINAMATH_GPT_ratio_circumscribed_circle_area_triangle_area_l1794_179414

open Real

theorem ratio_circumscribed_circle_area_triangle_area (h R : ℝ) (h_eq : R = h / 2) :
  let circle_area := π * R^2
  let triangle_area := (h^2) / 4
  (circle_area / triangle_area) = π :=
by
  sorry

end NUMINAMATH_GPT_ratio_circumscribed_circle_area_triangle_area_l1794_179414


namespace NUMINAMATH_GPT_white_tile_count_l1794_179486

theorem white_tile_count (total_tiles yellow_tiles blue_tiles purple_tiles white_tiles : ℕ)
  (h_total : total_tiles = 20)
  (h_yellow : yellow_tiles = 3)
  (h_blue : blue_tiles = yellow_tiles + 1)
  (h_purple : purple_tiles = 6)
  (h_sum : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  white_tiles = 7 :=
sorry

end NUMINAMATH_GPT_white_tile_count_l1794_179486


namespace NUMINAMATH_GPT_part1_part2_l1794_179496

noncomputable def f (x : ℝ) : ℝ := abs (x + 20) - abs (16 - x)

theorem part1 (x : ℝ) : f x ≥ 0 ↔ x ≥ -2 := 
by sorry

theorem part2 (m : ℝ) (x_exists : ∃ x : ℝ, f x ≥ m) : m ≤ 36 := 
by sorry

end NUMINAMATH_GPT_part1_part2_l1794_179496


namespace NUMINAMATH_GPT_billy_questions_third_hour_l1794_179483

variable (x : ℝ)
variable (questions_in_first_hour : ℝ := x)
variable (questions_in_second_hour : ℝ := 1.5 * x)
variable (questions_in_third_hour : ℝ := 3 * x)
variable (total_questions_solved : ℝ := 242)

theorem billy_questions_third_hour (h : questions_in_first_hour + questions_in_second_hour + questions_in_third_hour = total_questions_solved) :
  questions_in_third_hour = 132 :=
by
  sorry

end NUMINAMATH_GPT_billy_questions_third_hour_l1794_179483


namespace NUMINAMATH_GPT_speed_man_l1794_179427

noncomputable def speedOfMan : ℝ := 
  let d := 437.535 / 1000  -- distance in kilometers
  let t := 25 / 3600      -- time in hours
  d / t                    -- speed in kilometers per hour

theorem speed_man : speedOfMan = 63 := by
  sorry

end NUMINAMATH_GPT_speed_man_l1794_179427


namespace NUMINAMATH_GPT_points_for_win_l1794_179475

variable (W T : ℕ)

theorem points_for_win (W T : ℕ) (h1 : W * (T + 12) + T = 60) : W = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_points_for_win_l1794_179475


namespace NUMINAMATH_GPT_platform_length_l1794_179406

theorem platform_length (train_speed_kmph : ℕ) (train_time_man_seconds : ℕ) (train_time_platform_seconds : ℕ) (train_speed_mps : ℕ) : 
  train_speed_kmph = 54 →
  train_time_man_seconds = 20 →
  train_time_platform_seconds = 30 →
  train_speed_mps = (54 * 1000 / 3600) →
  (54 * 5 / 18) = 15 →
  ∃ (P : ℕ), (train_speed_mps * train_time_platform_seconds) = (train_speed_mps * train_time_man_seconds) + P ∧ P = 150 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_l1794_179406


namespace NUMINAMATH_GPT_solve_abs_eq_l1794_179413

theorem solve_abs_eq (x : ℝ) : 
    (3 * x + 9 = abs (-20 + 4 * x)) ↔ 
    (x = 29) ∨ (x = 11 / 7) := 
by sorry

end NUMINAMATH_GPT_solve_abs_eq_l1794_179413


namespace NUMINAMATH_GPT_kevin_total_cost_l1794_179440

theorem kevin_total_cost :
  let muffin_cost := 0.75
  let juice_cost := 1.45
  let total_muffins := 3
  let cost_muffins := total_muffins * muffin_cost
  let total_cost := cost_muffins + juice_cost
  total_cost = 3.70 :=
by
  sorry

end NUMINAMATH_GPT_kevin_total_cost_l1794_179440


namespace NUMINAMATH_GPT_binom_60_3_eq_34220_l1794_179470

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_GPT_binom_60_3_eq_34220_l1794_179470


namespace NUMINAMATH_GPT_tan_of_log_conditions_l1794_179491

theorem tan_of_log_conditions (x : ℝ) (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.log (Real.sin (2 * x)) - Real.log (Real.sin x) = Real.log (1 / 2)) :
  Real.tan x = Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_tan_of_log_conditions_l1794_179491


namespace NUMINAMATH_GPT_wall_length_l1794_179429

theorem wall_length (s : ℕ) (d : ℕ) (w : ℕ) (L : ℝ) 
  (hs : s = 18) 
  (hd : d = 20) 
  (hw : w = 32)
  (hcombined : (s ^ 2 + Real.pi * ((d / 2) ^ 2)) = (1 / 2) * (w * L)) :
  L = 39.88 := 
sorry

end NUMINAMATH_GPT_wall_length_l1794_179429


namespace NUMINAMATH_GPT_probability_of_rectangle_area_greater_than_32_l1794_179490

-- Definitions representing the problem conditions
def segment_length : ℝ := 12
def point_C (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ segment_length
def rectangle_area (x : ℝ) : ℝ := x * (segment_length - x)

-- The probability we need to prove. 
noncomputable def desired_probability : ℝ := 1 / 3

theorem probability_of_rectangle_area_greater_than_32 :
  (∀ x, point_C x → rectangle_area x > 32) → (desired_probability = 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rectangle_area_greater_than_32_l1794_179490


namespace NUMINAMATH_GPT_inequality_solution_l1794_179468

theorem inequality_solution (x : ℝ) :
  (x+3)/(x+4) > (4*x+5)/(3*x+10) ↔ x ∈ Set.Ioo (-4 : ℝ) (- (10 : ℝ) / 3) ∪ Set.Ioi 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1794_179468


namespace NUMINAMATH_GPT_intersection_of_sets_l1794_179418

open Set

theorem intersection_of_sets (M N : Set ℕ) (hM : M = {1, 2, 3}) (hN : N = {2, 3, 4}) :
  M ∩ N = {2, 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1794_179418


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1794_179434

theorem hyperbola_foci_distance :
  (∀ (x y : ℝ), (y = 2 * x + 3) ∨ (y = -2 * x + 7)) →
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ ((y = 2 * x + 3) ∨ (y = -2 * x + 7))) →
  (∃ h : ℝ, h = 6 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l1794_179434


namespace NUMINAMATH_GPT_evaluate_dollar_l1794_179428

variable {R : Type} [Field R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : dollar (2 * x + 3 * y) (3 * x - 4 * y) = x ^ 2 - 14 * x * y + 49 * y ^ 2 := by
  sorry

end NUMINAMATH_GPT_evaluate_dollar_l1794_179428


namespace NUMINAMATH_GPT_restaurant_meal_cost_l1794_179484

def cost_of_group_meal (total_people : Nat) (kids : Nat) (adult_meal_cost : Nat) : Nat :=
  let adults := total_people - kids
  adults * adult_meal_cost

theorem restaurant_meal_cost :
  cost_of_group_meal 9 2 2 = 14 := by
  sorry

end NUMINAMATH_GPT_restaurant_meal_cost_l1794_179484


namespace NUMINAMATH_GPT_eval_expression_l1794_179499

theorem eval_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1794_179499


namespace NUMINAMATH_GPT_no_tiling_with_seven_sided_convex_l1794_179412

noncomputable def Polygon := {n : ℕ // 3 ≤ n}

def convex (M : Polygon) : Prop := sorry

def tiles_plane (M : Polygon) : Prop := sorry

theorem no_tiling_with_seven_sided_convex (M : Polygon) (h_convex : convex M) (h_sides : 7 ≤ M.1) : ¬ tiles_plane M := sorry

end NUMINAMATH_GPT_no_tiling_with_seven_sided_convex_l1794_179412


namespace NUMINAMATH_GPT_range_of_m_l1794_179498

-- Definitions of propositions and their negations
def p (x : ℝ) : Prop := x + 2 ≥ 0 ∧ x - 10 ≤ 0
def q (x m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m ∧ m > 0
def not_p (x : ℝ) : Prop := x < -2 ∨ x > 10
def not_q (x m : ℝ) : Prop := x < (1 - m) ∨ x > (1 + m) ∧ m > 0

-- Statement that \neg p is a necessary but not sufficient condition for \neg q
def necessary_but_not_sufficient (x m : ℝ) : Prop := 
  (∀ x, not_q x m → not_p x) ∧ ¬(∀ x, not_p x → not_q x m)

-- The main theorem to prove
theorem range_of_m (m : ℝ) : (∀ x, necessary_but_not_sufficient x m) ↔ 9 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1794_179498


namespace NUMINAMATH_GPT_sum_last_two_digits_l1794_179430

theorem sum_last_two_digits (n : ℕ) (h1 : n = 20) : (9^n + 11^n) % 100 = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_l1794_179430


namespace NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1794_179452

variable (a b : ℝ)

theorem condition_neither_sufficient_nor_necessary 
    (h1 : ∃ a b : ℝ, a > b ∧ ¬(a^2 > b^2))
    (h2 : ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b)) :
  ¬((a > b) ↔ (a^2 > b^2)) :=
sorry

end NUMINAMATH_GPT_condition_neither_sufficient_nor_necessary_l1794_179452


namespace NUMINAMATH_GPT_number_of_ways_to_fold_cube_with_one_face_missing_l1794_179420

-- Definitions:
-- The polygon is initially in the shape of a cross with 5 congruent squares.
-- One additional square can be attached to any of the 12 possible edge positions around this polygon.
-- Define what it means for the resulting figure to fold into a cube with one face missing.

-- Statement:
theorem number_of_ways_to_fold_cube_with_one_face_missing 
  (initial_squares : ℕ)
  (additional_positions : ℕ)
  (valid_folding_positions : ℕ) : 
  initial_squares = 5 ∧ additional_positions = 12 → valid_folding_positions = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_fold_cube_with_one_face_missing_l1794_179420


namespace NUMINAMATH_GPT_even_function_derivative_at_zero_l1794_179431

-- Define an even function f and its differentiability at x = 0
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def differentiable_at_zero (f : ℝ → ℝ) : Prop := DifferentiableAt ℝ f 0

-- The theorem to prove that f'(0) = 0
theorem even_function_derivative_at_zero
  (f : ℝ → ℝ)
  (hf_even : is_even_function f)
  (hf_diff : differentiable_at_zero f) :
  deriv f 0 = 0 := 
sorry

end NUMINAMATH_GPT_even_function_derivative_at_zero_l1794_179431


namespace NUMINAMATH_GPT_solve_for_a_l1794_179482

theorem solve_for_a (x a : ℝ) (h : x = 3) (eq : 5 * x - a = 8) : a = 7 :=
by
  -- sorry to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_solve_for_a_l1794_179482


namespace NUMINAMATH_GPT_age_of_new_person_l1794_179457

theorem age_of_new_person (n : ℕ) (T A : ℕ) (h₁ : n = 10) (h₂ : T = 15 * n)
    (h₃ : (T + A) / (n + 1) = 17) : A = 37 := by
  sorry

end NUMINAMATH_GPT_age_of_new_person_l1794_179457


namespace NUMINAMATH_GPT_problem_statement_l1794_179489

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_statement : f (f (1/2)) = 1 :=
by
    sorry

end NUMINAMATH_GPT_problem_statement_l1794_179489


namespace NUMINAMATH_GPT_locus_eq_l1794_179493

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  5 * a^2 + 9 * b^2 + 80 * a - 400 = 0

theorem locus_eq (a b : ℝ) :
  (∃ r : ℝ, (a^2 + b^2 = (r + 2)^2) ∧ ((a - 1)^2 + b^2 = (5 - r)^2)) →
  locus_of_centers a b :=
by
  intro h
  sorry

end NUMINAMATH_GPT_locus_eq_l1794_179493


namespace NUMINAMATH_GPT_Angelina_speeds_l1794_179447

def distance_home_to_grocery := 960
def distance_grocery_to_gym := 480
def distance_gym_to_library := 720
def time_diff_grocery_to_gym := 40
def time_diff_gym_to_library := 20

noncomputable def initial_speed (v : ℝ) :=
  (distance_home_to_grocery : ℝ) = (v * (960 / v)) ∧
  (distance_grocery_to_gym : ℝ) = (2 * v * (240 / v)) ∧
  (distance_gym_to_library : ℝ) = (3 * v * (720 / v))

theorem Angelina_speeds (v : ℝ) :
  initial_speed v →
  v = 18 ∧ 2 * v = 36 ∧ 3 * v = 54 :=
by
  sorry

end NUMINAMATH_GPT_Angelina_speeds_l1794_179447


namespace NUMINAMATH_GPT_find_max_number_l1794_179494

noncomputable def increasing_sequence (a : ℕ → ℝ) := ∀ n m, n < m → a n < a m

noncomputable def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) (n : ℕ) := 
  (a n + d = a (n+1)) ∧ (a (n+1) + d = a (n+2)) ∧ (a (n+2) + d = a (n+3))

noncomputable def geometric_progression (a : ℕ → ℝ) (r : ℝ) (n : ℕ) := 
  (a (n+1) = a n * r) ∧ (a (n+2) = a (n+1) * r) ∧ (a (n+3) = a (n+2) * r)

theorem find_max_number (a : ℕ → ℝ):
  increasing_sequence a → 
  (∃ n, arithmetic_progression a 4 n) →
  (∃ n, arithmetic_progression a 36 n) →
  (∃ n, geometric_progression a (a (n+1) / a n) n) →
  a 7 = 126 := sorry

end NUMINAMATH_GPT_find_max_number_l1794_179494


namespace NUMINAMATH_GPT_inequality_proof_l1794_179424

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * (x - z) ^ 2 + y * (y - z) ^ 2 ≥ (x - z) * (y - z) * (x + y - z) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1794_179424


namespace NUMINAMATH_GPT_car_speed_l1794_179417

variable (v : ℝ)
variable (Distance : ℝ := 1)  -- distance in kilometers
variable (Speed_120 : ℝ := 120)  -- speed in kilometers per hour
variable (Time_120 : ℝ := Distance / Speed_120)  -- time in hours to travel 1 km at 120 km/h
variable (Time_120_sec : ℝ := Time_120 * 3600)  -- time in seconds to travel 1 km at 120 km/h
variable (Additional_time : ℝ := 2)  -- additional time in seconds
variable (Time_v_sec : ℝ := Time_120_sec + Additional_time)  -- time in seconds for unknown speed
variable (Time_v : ℝ := Time_v_sec / 3600)  -- time in hours for unknown speed

theorem car_speed (h : v = Distance / Time_v) : v = 112.5 :=
by
  -- The given proof steps will go here
  sorry

end NUMINAMATH_GPT_car_speed_l1794_179417


namespace NUMINAMATH_GPT_find_variable_value_l1794_179459

axiom variable_property (x : ℝ) (h : 4 + 1 / x ≠ 0) : 5 / (4 + 1 / x) = 1 → x = 1

-- Given condition: 5 / (4 + 1 / x) = 1
-- Prove: x = 1
theorem find_variable_value (x : ℝ) (h : 4 + 1 / x ≠ 0) (h1 : 5 / (4 + 1 / x) = 1) : x = 1 :=
variable_property x h h1

end NUMINAMATH_GPT_find_variable_value_l1794_179459


namespace NUMINAMATH_GPT_find_fourth_number_l1794_179488

variable (x : ℝ)

theorem find_fourth_number
  (h : 3 + 33 + 333 + x = 399.6) :
  x = 30.6 :=
sorry

end NUMINAMATH_GPT_find_fourth_number_l1794_179488


namespace NUMINAMATH_GPT_minimize_sum_l1794_179456

noncomputable def objective_function (x : ℝ) : ℝ := x + x^2

theorem minimize_sum : ∃ x : ℝ, (objective_function x = x + x^2) ∧ (∀ y : ℝ, objective_function y ≥ objective_function (-1/2)) :=
by
  sorry

end NUMINAMATH_GPT_minimize_sum_l1794_179456


namespace NUMINAMATH_GPT_dvd_cost_packs_l1794_179495

theorem dvd_cost_packs (cost_per_pack : ℕ) (number_of_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 12 → number_of_packs = 11 → total_money = (cost_per_pack * number_of_packs) → total_money = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_dvd_cost_packs_l1794_179495


namespace NUMINAMATH_GPT_partitions_equal_l1794_179409

namespace MathProof

-- Define the set of natural numbers
def nat := ℕ

-- Define the partition functions (placeholders)
def num_distinct_partitions (n : nat) : nat := sorry
def num_odd_partitions (n : nat) : nat := sorry

-- Statement of the theorem
theorem partitions_equal (n : nat) : 
  num_distinct_partitions n = num_odd_partitions n :=
sorry

end MathProof

end NUMINAMATH_GPT_partitions_equal_l1794_179409


namespace NUMINAMATH_GPT_deputy_more_enemies_than_friends_l1794_179404

theorem deputy_more_enemies_than_friends (deputies : Type) 
  (friendship hostility indifference : deputies → deputies → Prop)
  (h_symm_friend : ∀ (a b : deputies), friendship a b → friendship b a)
  (h_symm_hostile : ∀ (a b : deputies), hostility a b → hostility b a)
  (h_symm_indiff : ∀ (a b : deputies), indifference a b → indifference b a)
  (h_enemy_exists : ∀ (d : deputies), ∃ (e : deputies), hostility d e)
  (h_principle : ∀ (a b c : deputies), hostility a b → friendship b c → hostility a c) :
  ∃ (d : deputies), ∃ (f e : ℕ), f < e :=
sorry

end NUMINAMATH_GPT_deputy_more_enemies_than_friends_l1794_179404


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1794_179426

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | abs (x^2 - 1) ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = A :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1794_179426


namespace NUMINAMATH_GPT_garden_length_80_l1794_179449

-- Let the width of the garden be denoted by w and the length by l
-- Given conditions
def is_rectangular_garden (l w : ℝ) := l = 2 * w ∧ 2 * l + 2 * w = 240

-- We want to prove that the length of the garden is 80 yards
theorem garden_length_80 (w : ℝ) (h : is_rectangular_garden (2 * w) w) : 2 * w = 80 :=
by
  sorry

end NUMINAMATH_GPT_garden_length_80_l1794_179449


namespace NUMINAMATH_GPT_triangle_base_length_l1794_179438

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) 
  (h_height : height = 6) (h_area : area = 9) 
  (h_formula : area = (1/2) * base * height) : 
  base = 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l1794_179438


namespace NUMINAMATH_GPT_find_a_for_even_function_l1794_179401

theorem find_a_for_even_function (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x) 
  (h_value : f 3 = 3) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_for_even_function_l1794_179401


namespace NUMINAMATH_GPT_instantaneous_velocity_at_1_l1794_179497

noncomputable def particle_displacement (t : ℝ) : ℝ := t + Real.log t

theorem instantaneous_velocity_at_1 : 
  let v := fun t => deriv (particle_displacement) t
  v 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_1_l1794_179497


namespace NUMINAMATH_GPT_altitude_eq_4r_l1794_179463

variable (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

-- We define the geometrical relations and constraints
def AC_eq_BC (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (AC BC : ℝ) : Prop :=
AC = BC

def in_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (incircle_radius r : ℝ) : Prop :=
incircle_radius = r

def ex_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (excircle_radius r : ℝ) : Prop :=
excircle_radius = r

-- Main theorem to prove
theorem altitude_eq_4r 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (r : ℝ)
  (h : ℝ)
  (H1 : AC_eq_BC A B C D AC BC)
  (H2 : in_circle_radius_eq_r A B C D r r)
  (H3 : ex_circle_radius_eq_r A B C D r r) :
  h = 4 * r :=
  sorry

end NUMINAMATH_GPT_altitude_eq_4r_l1794_179463


namespace NUMINAMATH_GPT_living_room_curtain_length_l1794_179432

theorem living_room_curtain_length :
  let length_bolt := 16
  let width_bolt := 12
  let area_bolt := length_bolt * width_bolt
  let area_left := 160
  let area_cut := area_bolt - area_left
  let length_bedroom := 2
  let width_bedroom := 4
  let area_bedroom := length_bedroom * width_bedroom
  let area_living_room := area_cut - area_bedroom
  let width_living_room := 4
  area_living_room / width_living_room = 6 :=
by
  sorry

end NUMINAMATH_GPT_living_room_curtain_length_l1794_179432


namespace NUMINAMATH_GPT_slant_asymptote_sum_l1794_179407

theorem slant_asymptote_sum (x : ℝ) (hx : x ≠ 5) :
  (5 : ℝ) + (21 : ℝ) = 26 :=
by
  sorry

end NUMINAMATH_GPT_slant_asymptote_sum_l1794_179407


namespace NUMINAMATH_GPT_percentage_sum_l1794_179469

theorem percentage_sum (A B C : ℕ) (x y : ℕ)
  (hA : A = 120) (hB : B = 110) (hC : C = 100)
  (hAx : A = C * (1 + x / 100))
  (hBy : B = C * (1 + y / 100)) : x + y = 30 := 
by
  sorry

end NUMINAMATH_GPT_percentage_sum_l1794_179469


namespace NUMINAMATH_GPT_school_distance_is_seven_l1794_179436

-- Definitions based on conditions
def distance_to_school (x : ℝ) : Prop :=
  let monday_to_thursday_distance := 8 * x
  let friday_distance := 2 * x + 4
  let total_distance := monday_to_thursday_distance + friday_distance
  total_distance = 74

-- The problem statement to prove
theorem school_distance_is_seven : ∃ (x : ℝ), distance_to_school x ∧ x = 7 := 
by {
  sorry
}

end NUMINAMATH_GPT_school_distance_is_seven_l1794_179436


namespace NUMINAMATH_GPT_total_spending_correct_l1794_179403

-- Define the costs and number of children for each ride and snack
def cost_ferris_wheel := 5 * 5
def cost_roller_coaster := 7 * 3
def cost_merry_go_round := 3 * 8
def cost_bumper_cars := 4 * 6

def cost_ice_cream := 8 * 2 * 5
def cost_hot_dog := 6 * 4
def cost_pizza := 4 * 3

-- Calculate the total cost
def total_cost_rides := cost_ferris_wheel + cost_roller_coaster + cost_merry_go_round + cost_bumper_cars
def total_cost_snacks := cost_ice_cream + cost_hot_dog + cost_pizza
def total_spent := total_cost_rides + total_cost_snacks

-- The statement to prove
theorem total_spending_correct : total_spent = 170 := by
  sorry

end NUMINAMATH_GPT_total_spending_correct_l1794_179403


namespace NUMINAMATH_GPT_greatest_possible_sum_of_roots_l1794_179448

noncomputable def quadratic_roots (c b : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ α + β = c ∧ α * β = b ∧ |α - β| = 1

theorem greatest_possible_sum_of_roots :
  ∃ (c : ℝ), ( ∃ b : ℝ, quadratic_roots c b) ∧
             ( ∀ (d : ℝ), ( ∃ b : ℝ, quadratic_roots d b) → d ≤ 11 ) ∧ c = 11 :=
sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_roots_l1794_179448


namespace NUMINAMATH_GPT_price_of_other_frisbees_l1794_179474

theorem price_of_other_frisbees :
  ∃ F3 Fx Px : ℕ, F3 + Fx = 60 ∧ 3 * F3 + Px * Fx = 204 ∧ Fx ≥ 24 ∧ Px = 4 := 
by
  sorry

end NUMINAMATH_GPT_price_of_other_frisbees_l1794_179474


namespace NUMINAMATH_GPT_total_animals_seen_correct_l1794_179445

-- Define the number of beavers in the morning
def beavers_morning : ℕ := 35

-- Define the number of chipmunks in the morning
def chipmunks_morning : ℕ := 60

-- Define the number of beavers in the afternoon (tripled)
def beavers_afternoon : ℕ := 3 * beavers_morning

-- Define the number of chipmunks in the afternoon (decreased by 15)
def chipmunks_afternoon : ℕ := chipmunks_morning - 15

-- Calculate the total number of animals seen in the morning
def total_morning : ℕ := beavers_morning + chipmunks_morning

-- Calculate the total number of animals seen in the afternoon
def total_afternoon : ℕ := beavers_afternoon + chipmunks_afternoon

-- The total number of animals seen that day
def total_animals_seen : ℕ := total_morning + total_afternoon

theorem total_animals_seen_correct :
  total_animals_seen = 245 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_total_animals_seen_correct_l1794_179445


namespace NUMINAMATH_GPT_number_of_sequences_l1794_179423

-- Define the number of possible outcomes for a single coin flip
def coinFlipOutcomes : ℕ := 2

-- Define the number of flips
def numberOfFlips : ℕ := 8

-- Theorem statement: The number of distinct sequences when flipping a coin eight times is 256
theorem number_of_sequences (n : ℕ) (outcomes : ℕ) (h : outcomes = 2) (hn : n = 8) : outcomes ^ n = 256 := by
  sorry

end NUMINAMATH_GPT_number_of_sequences_l1794_179423


namespace NUMINAMATH_GPT_smallest_x_l1794_179477

theorem smallest_x (x : ℕ) (h₁ : x % 3 = 2) (h₂ : x % 4 = 3) (h₃ : x % 5 = 4) : x = 59 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l1794_179477


namespace NUMINAMATH_GPT_range_of_a_l1794_179467

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 + a * x + 1
noncomputable def quadratic_eq (x₀ a : ℝ) : Prop := x₀^2 - x₀ + a = 0

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, quadratic a x > 0) (q : ∃ x₀ : ℝ, quadratic_eq x₀ a) : 0 ≤ a ∧ a ≤ 1/4 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1794_179467


namespace NUMINAMATH_GPT_line_parallel_to_y_axis_l1794_179421

theorem line_parallel_to_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x + b * y + 1 = 0 → b = 0):
  a ≠ 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_line_parallel_to_y_axis_l1794_179421


namespace NUMINAMATH_GPT_gena_encoded_numbers_unique_l1794_179458

theorem gena_encoded_numbers_unique : 
  ∃ (B AN AX NO FF d : ℕ), (AN - B = d) ∧ (AX - AN = d) ∧ (NO - AX = d) ∧ (FF - NO = d) ∧ 
  [B, AN, AX, NO, FF] = [5, 12, 19, 26, 33] := sorry

end NUMINAMATH_GPT_gena_encoded_numbers_unique_l1794_179458


namespace NUMINAMATH_GPT_inequality_holds_for_all_x_l1794_179446

theorem inequality_holds_for_all_x (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_x_l1794_179446


namespace NUMINAMATH_GPT_probability_of_selecting_male_is_three_fifths_l1794_179476

-- Define the number of male and female students
def num_male_students : ℕ := 6
def num_female_students : ℕ := 4

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability of selecting a male student's ID
def probability_male_student : ℚ := num_male_students / total_students

-- Theorem: The probability of selecting a male student's ID is 3/5
theorem probability_of_selecting_male_is_three_fifths : probability_male_student = 3 / 5 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_probability_of_selecting_male_is_three_fifths_l1794_179476


namespace NUMINAMATH_GPT_sport_formulation_water_l1794_179419

theorem sport_formulation_water (corn_syrup_ounces : ℕ) (h_cs : corn_syrup_ounces = 3) : 
  ∃ water_ounces : ℕ, water_ounces = 45 :=
by
  -- The ratios for the "sport" formulation: Flavoring : Corn Syrup : Water = 1 : 4 : 60
  let flavoring_ratio := 1
  let corn_syrup_ratio := 4
  let water_ratio := 60
  -- The given corn syrup is 3 ounces which corresponds to corn_syrup_ratio parts
  have h_ratio : corn_syrup_ratio = 4 := rfl
  have h_flavoring_to_corn_syrup : flavoring_ratio / corn_syrup_ratio = 1 / 4 := by sorry
  have h_flavoring_to_water : flavoring_ratio / water_ratio = 1 / 60 := by sorry
  -- Set up the proportion
  have h_proportion : corn_syrup_ratio / corn_syrup_ounces = water_ratio / 45 := by sorry 
  -- Cross-multiply to solve for the water
  have h_cross_mul : 4 * 45 = 3 * 60 := by sorry
  exact ⟨45, rfl⟩

end NUMINAMATH_GPT_sport_formulation_water_l1794_179419


namespace NUMINAMATH_GPT_smallest_even_consecutive_sum_l1794_179451

theorem smallest_even_consecutive_sum (n : ℕ) (h_even : n % 2 = 0) (h_sum : n + (n + 2) + (n + 4) = 162) : n = 52 :=
sorry

end NUMINAMATH_GPT_smallest_even_consecutive_sum_l1794_179451


namespace NUMINAMATH_GPT_total_nephews_proof_l1794_179462

-- We declare the current number of nephews as unknown variables
variable (Alden_current Vihaan Shruti Nikhil : ℕ)

-- State the conditions as hypotheses
theorem total_nephews_proof
  (h1 : 70 = (1 / 3 : ℚ) * Alden_current)
  (h2 : Vihaan = Alden_current + 120)
  (h3 : Shruti = 2 * Vihaan)
  (h4 : Nikhil = Alden_current + Shruti - 40) :
  Alden_current + Vihaan + Shruti + Nikhil = 2030 := 
by
  sorry

end NUMINAMATH_GPT_total_nephews_proof_l1794_179462


namespace NUMINAMATH_GPT_expression_value_l1794_179480

theorem expression_value :
  3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := 
sorry

end NUMINAMATH_GPT_expression_value_l1794_179480


namespace NUMINAMATH_GPT_like_terms_monomials_l1794_179453

theorem like_terms_monomials (a b : ℕ) : (5 * (m^8) * (n^6) = -(3/4) * (m^(2*a)) * (n^(2*b))) → (a = 4 ∧ b = 3) := by
  sorry

end NUMINAMATH_GPT_like_terms_monomials_l1794_179453


namespace NUMINAMATH_GPT_total_space_compacted_l1794_179454

-- Definitions according to the conditions
def num_cans : ℕ := 60
def space_per_can_before : ℝ := 30
def compaction_rate : ℝ := 0.20

-- Theorem statement
theorem total_space_compacted : num_cans * (space_per_can_before * compaction_rate) = 360 := by
  sorry

end NUMINAMATH_GPT_total_space_compacted_l1794_179454


namespace NUMINAMATH_GPT_frank_remaining_money_l1794_179473

theorem frank_remaining_money
  (cheapest_lamp : ℕ)
  (most_expensive_factor : ℕ)
  (frank_money : ℕ)
  (cheapest_lamp_cost : cheapest_lamp = 20)
  (most_expensive_lamp_cost : most_expensive_factor = 3)
  (frank_current_money : frank_money = 90) :
  frank_money - (most_expensive_factor * cheapest_lamp) = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_frank_remaining_money_l1794_179473


namespace NUMINAMATH_GPT_dante_coconuts_l1794_179442

theorem dante_coconuts (P : ℕ) (D : ℕ) (S : ℕ) (hP : P = 14) (hD : D = 3 * P) (hS : S = 10) :
  (D - S) = 32 :=
by
  sorry

end NUMINAMATH_GPT_dante_coconuts_l1794_179442


namespace NUMINAMATH_GPT_operation_is_addition_l1794_179450

theorem operation_is_addition : (5 + (-5) = 0) :=
by
  sorry

end NUMINAMATH_GPT_operation_is_addition_l1794_179450


namespace NUMINAMATH_GPT_digit_6_count_1_to_700_l1794_179433

theorem digit_6_count_1_to_700 :
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  countNumbersWithDigit6 = 133 := 
by
  let countNumbersWithDigit6 := 700 - (7 * 9 * 9)
  show countNumbersWithDigit6 = 133
  sorry

end NUMINAMATH_GPT_digit_6_count_1_to_700_l1794_179433


namespace NUMINAMATH_GPT_learning_hours_difference_l1794_179402

/-- Define the hours Ryan spends on each language. -/
def hours_learned (lang : String) : ℝ :=
  if lang = "English" then 2 else
  if lang = "Chinese" then 5 else
  if lang = "Spanish" then 4 else
  if lang = "French" then 3 else
  if lang = "German" then 1.5 else 0

/-- Prove that Ryan spends 2.5 more hours learning Chinese and French combined
    than he does learning German and Spanish combined. -/
theorem learning_hours_difference :
  hours_learned "Chinese" + hours_learned "French" - (hours_learned "German" + hours_learned "Spanish") = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_learning_hours_difference_l1794_179402


namespace NUMINAMATH_GPT_horner_eval_v4_at_2_l1794_179471

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

theorem horner_eval_v4_at_2 : 
  let x := 2
  let v_0 := 1
  let v_1 := (v_0 * x) - 12 
  let v_2 := (v_1 * x) + 60 
  let v_3 := (v_2 * x) - 160 
  let v_4 := (v_3 * x) + 240 
  v_4 = 80 := 
by 
  sorry

end NUMINAMATH_GPT_horner_eval_v4_at_2_l1794_179471


namespace NUMINAMATH_GPT_max_length_of_each_piece_l1794_179415

theorem max_length_of_each_piece (a b c d : ℕ) (h1 : a = 48) (h2 : b = 72) (h3 : c = 108) (h4 : d = 120) : Nat.gcd (Nat.gcd a b) (Nat.gcd c d) = 12 := by
  sorry

end NUMINAMATH_GPT_max_length_of_each_piece_l1794_179415


namespace NUMINAMATH_GPT_width_of_plot_is_correct_l1794_179460

-- Definitions based on the given conditions
def cost_per_acre_per_month : ℝ := 60
def total_monthly_rent : ℝ := 600
def length_of_plot : ℝ := 360
def sq_feet_per_acre : ℝ := 43560

-- Theorems to be proved based on the conditions and the correct answer
theorem width_of_plot_is_correct :
  let number_of_acres := total_monthly_rent / cost_per_acre_per_month
  let total_sq_footage := number_of_acres * sq_feet_per_acre
  let width_of_plot := total_sq_footage / length_of_plot
  width_of_plot = 1210 :=
by 
  sorry

end NUMINAMATH_GPT_width_of_plot_is_correct_l1794_179460


namespace NUMINAMATH_GPT_fraction_interval_l1794_179465

theorem fraction_interval :
  (5 / 24 > 1 / 6) ∧ (5 / 24 < 1 / 4) ∧
  (¬ (5 / 12 > 1 / 6 ∧ 5 / 12 < 1 / 4)) ∧
  (¬ (5 / 36 > 1 / 6 ∧ 5 / 36 < 1 / 4)) ∧
  (¬ (5 / 60 > 1 / 6 ∧ 5 / 60 < 1 / 4)) ∧
  (¬ (5 / 48 > 1 / 6 ∧ 5 / 48 < 1 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_interval_l1794_179465


namespace NUMINAMATH_GPT_lunks_to_apples_l1794_179441

theorem lunks_to_apples :
  (∀ (a b c d e f : ℕ), (7 * b = 4 * a) → (3 * d = 5 * c) → c = 24 → f * e = d → e = 27) :=
by sorry

end NUMINAMATH_GPT_lunks_to_apples_l1794_179441


namespace NUMINAMATH_GPT_max_green_beads_l1794_179411

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end NUMINAMATH_GPT_max_green_beads_l1794_179411


namespace NUMINAMATH_GPT_room_width_l1794_179439

theorem room_width (W : ℝ) (L : ℝ := 17) (veranda_width : ℝ := 2) (veranda_area : ℝ := 132) :
  (21 * (W + veranda_width) - L * W = veranda_area) → W = 12 :=
by
  -- setup of the problem
  have total_length := L + 2 * veranda_width
  have total_width := W + 2 * veranda_width
  have area_room_incl_veranda := total_length * total_width - (L * W)
  -- the statement is already provided in the form of the theorem to be proven
  sorry

end NUMINAMATH_GPT_room_width_l1794_179439


namespace NUMINAMATH_GPT_kira_breakfast_time_l1794_179464

theorem kira_breakfast_time :
  let fry_time_per_sausage := 5 -- minutes per sausage
  let scramble_time_per_egg := 4 -- minutes per egg
  let sausages := 3
  let eggs := 6
  let time_to_fry := sausages * fry_time_per_sausage
  let time_to_scramble := eggs * scramble_time_per_egg
  (time_to_fry + time_to_scramble) = 39 := 
by
  sorry

end NUMINAMATH_GPT_kira_breakfast_time_l1794_179464


namespace NUMINAMATH_GPT_min_value_of_xy_ratio_l1794_179481

theorem min_value_of_xy_ratio :
  ∃ t : ℝ,
    (t = 2 ∨
    t = ((-1 + Real.sqrt 217) / 12) ∨
    t = ((-1 - Real.sqrt 217) / 12)) ∧
    min (min 2 ((-1 + Real.sqrt 217) / 12)) ((-1 - Real.sqrt 217) / 12) = -1.31 :=
sorry

end NUMINAMATH_GPT_min_value_of_xy_ratio_l1794_179481


namespace NUMINAMATH_GPT_find_ax5_by5_l1794_179435

variable (a b x y : ℝ)

theorem find_ax5_by5 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end NUMINAMATH_GPT_find_ax5_by5_l1794_179435


namespace NUMINAMATH_GPT_line_length_after_erasing_l1794_179444

-- Definition of the initial length in meters and the erased length in centimeters
def initial_length_meters : ℝ := 1.5
def erased_length_centimeters : ℝ := 15.25

-- Conversion factor from meters to centimeters
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- Definition of the initial length in centimeters
def initial_length_centimeters : ℝ := meters_to_centimeters initial_length_meters

-- Statement of the theorem
theorem line_length_after_erasing :
  initial_length_centimeters - erased_length_centimeters = 134.75 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_line_length_after_erasing_l1794_179444


namespace NUMINAMATH_GPT_values_of_a_and_b_l1794_179472

theorem values_of_a_and_b (a b : ℝ) :
  (∀ x : ℝ, x ≥ -1 → a * x^2 + b * x + a^2 - 1 ≤ 0) →
  a = 0 ∧ b = -1 :=
sorry

end NUMINAMATH_GPT_values_of_a_and_b_l1794_179472


namespace NUMINAMATH_GPT_find_original_integer_l1794_179425

theorem find_original_integer (a b c d : ℕ) 
    (h1 : (b + c + d) / 3 + 10 = 37) 
    (h2 : (a + c + d) / 3 + 10 = 31) 
    (h3 : (a + b + d) / 3 + 10 = 25) 
    (h4 : (a + b + c) / 3 + 10 = 19) : 
    d = 45 := 
    sorry

end NUMINAMATH_GPT_find_original_integer_l1794_179425
