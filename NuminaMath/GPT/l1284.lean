import Mathlib

namespace NUMINAMATH_GPT_complement_U_A_l1284_128401

def U := {x : ℝ | x < 2}
def A := {x : ℝ | x^2 < x}

theorem complement_U_A :
  (U \ A) = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
sorry

end NUMINAMATH_GPT_complement_U_A_l1284_128401


namespace NUMINAMATH_GPT_initial_oranges_l1284_128469

theorem initial_oranges (O : ℕ) (h1 : O + 6 - 3 = 6) : O = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_oranges_l1284_128469


namespace NUMINAMATH_GPT_lucas_change_l1284_128462

-- Define the given conditions as constants in Lean
def num_bananas : ℕ := 5
def cost_per_banana : ℝ := 0.70
def num_oranges : ℕ := 2
def cost_per_orange : ℝ := 0.80
def amount_paid : ℝ := 10.00

-- Define a noncomputable constant to represent the change received
noncomputable def change_received : ℝ := 
  amount_paid - (num_bananas * cost_per_banana + num_oranges * cost_per_orange)

-- State the theorem to be proved
theorem lucas_change : change_received = 4.90 := 
by 
  -- Dummy proof since the actual proof is not required
  sorry

end NUMINAMATH_GPT_lucas_change_l1284_128462


namespace NUMINAMATH_GPT_f_is_even_f_monotonic_increase_range_of_a_for_solutions_l1284_128402

-- Define the function f(x) = x^2 - 2a|x|
def f (a x : ℝ) : ℝ := x^2 - 2 * a * |x|

-- Given a > 0
variable (a : ℝ) (ha : a > 0)

-- 1. Prove that f(x) is an even function.
theorem f_is_even : ∀ x : ℝ, f a x = f a (-x) := sorry

-- 2. Prove the interval of monotonic increase for f(x) when x > 0 is [a, +∞).
theorem f_monotonic_increase (x : ℝ) (hx : x > 0) : a ≤ x → ∃ c : ℝ, x ≤ c := sorry

-- 3. Prove the range of values for a for which the equation f(x) = -1 has solutions is a ≥ 1.
theorem range_of_a_for_solutions : (∃ x : ℝ, f a x = -1) ↔ 1 ≤ a := sorry

end NUMINAMATH_GPT_f_is_even_f_monotonic_increase_range_of_a_for_solutions_l1284_128402


namespace NUMINAMATH_GPT_no_solution_when_k_equals_7_l1284_128460

noncomputable def no_solution_eq (k x : ℝ) : Prop :=
  (x - 3) / (x - 4) = (x - k) / (x - 8)
  
theorem no_solution_when_k_equals_7 :
  ∀ x : ℝ, x ≠ 4 → x ≠ 8 → ¬ no_solution_eq 7 x :=
by
  sorry

end NUMINAMATH_GPT_no_solution_when_k_equals_7_l1284_128460


namespace NUMINAMATH_GPT_sqrt_fraction_equiv_l1284_128440

-- Define the fractions
def frac1 : ℚ := 25 / 36
def frac2 : ℚ := 16 / 9

-- Define the expression under the square root
def sum_frac : ℚ := frac1 + (frac2 * 36 / 36)

-- State the problem
theorem sqrt_fraction_equiv : (Real.sqrt sum_frac) = Real.sqrt 89 / 6 :=
by
  -- Steps and proof are omitted; we use sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_sqrt_fraction_equiv_l1284_128440


namespace NUMINAMATH_GPT_expenditure_fraction_l1284_128476

variable (B : ℝ)
def cost_of_book (x y : ℝ) (B : ℝ) := x = 0.30 * (B - 2 * y)
def cost_of_coffee (x y : ℝ) (B : ℝ) := y = 0.10 * (B - x)

theorem expenditure_fraction (x y : ℝ) (B : ℝ) 
  (hx : cost_of_book x y B) 
  (hy : cost_of_coffee x y B) : 
  (x + y) / B = 31 / 94 :=
sorry

end NUMINAMATH_GPT_expenditure_fraction_l1284_128476


namespace NUMINAMATH_GPT_initial_alcohol_percentage_l1284_128408

theorem initial_alcohol_percentage (P : ℚ) (initial_volume : ℚ) (added_alcohol : ℚ) (added_water : ℚ)
  (final_percentage : ℚ) (final_volume : ℚ) (alcohol_volume_in_initial_solution : ℚ) :
  initial_volume = 40 ∧ 
  added_alcohol = 3.5 ∧ 
  added_water = 6.5 ∧ 
  final_percentage = 0.11 ∧ 
  final_volume = 50 ∧ 
  alcohol_volume_in_initial_solution = (P / 100) * initial_volume ∧ 
  alcohol_volume_in_initial_solution + added_alcohol = final_percentage * final_volume
  → P = 5 :=
by
  sorry

end NUMINAMATH_GPT_initial_alcohol_percentage_l1284_128408


namespace NUMINAMATH_GPT_polynomial_sum_l1284_128465

def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : p x + q x + r x = -4 * x^2 + 12 * x - 12 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l1284_128465


namespace NUMINAMATH_GPT_an_general_term_sum_bn_l1284_128432

open Nat

variable (a : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (T : ℕ → ℕ)

-- Conditions
axiom a3 : a 3 = 3
axiom S6 : S 6 = 21
axiom Sn : ∀ n, S n = n * (a 1 + a n) / 2

-- Define bn based on the given condition for bn = an + 2^n
def bn (n : ℕ) : ℕ := a n + 2^n

-- Define Tn based on the given condition for Tn.
def Tn (n : ℕ) : ℕ := (n * (n + 1)) / 2 + (2^(n + 1) - 2)

-- Prove the general term formula of the arithmetic sequence an
theorem an_general_term (n : ℕ) : a n = n :=
by
  sorry

-- Prove the sum of the first n terms of the sequence bn
theorem sum_bn (n : ℕ) : T n = Tn n :=
by
  sorry

end NUMINAMATH_GPT_an_general_term_sum_bn_l1284_128432


namespace NUMINAMATH_GPT_find_x_in_triangle_l1284_128404

theorem find_x_in_triangle 
  (P Q R S: Type) 
  (PQS_is_straight: PQS) 
  (angle_PQR: ℝ)
  (h1: angle_PQR = 110) 
  (angle_RQS : ℝ)
  (h2: angle_RQS = 70)
  (angle_QRS : ℝ)
  (h3: angle_QRS = 3 * angle_x)
  (angle_QSR : ℝ)
  (h4: angle_QSR = angle_x + 14) 
  (triangle_angles_sum : ∀ (a b c: ℝ), a + b + c = 180) : 
  angle_x = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_x_in_triangle_l1284_128404


namespace NUMINAMATH_GPT_percentage_both_correct_l1284_128425

variable (A B : Type) 

noncomputable def percentage_of_test_takers_correct_first : ℝ := 0.85
noncomputable def percentage_of_test_takers_correct_second : ℝ := 0.70
noncomputable def percentage_of_test_takers_neither_correct : ℝ := 0.05

theorem percentage_both_correct :
  percentage_of_test_takers_correct_first + 
  percentage_of_test_takers_correct_second - 
  (1 - percentage_of_test_takers_neither_correct) = 0.60 := by
  sorry

end NUMINAMATH_GPT_percentage_both_correct_l1284_128425


namespace NUMINAMATH_GPT_jessica_deposited_fraction_l1284_128443

-- Definitions based on conditions
def original_balance (B : ℝ) : Prop :=
  B * (3 / 5) = B - 200

def final_balance (B : ℝ) (F : ℝ) : Prop :=
  ((3 / 5) * B) + (F * ((3 / 5) * B)) = 360

-- Theorem statement proving that the fraction deposited is 1/5
theorem jessica_deposited_fraction (B : ℝ) (F : ℝ) (h1 : original_balance B) (h2 : final_balance B F) : F = 1 / 5 :=
  sorry

end NUMINAMATH_GPT_jessica_deposited_fraction_l1284_128443


namespace NUMINAMATH_GPT_eeshas_usual_time_l1284_128472

/-- Eesha's usual time to reach her office from home is 60 minutes,
given that she started 30 minutes late and reached her office
50 minutes late while driving 25% slower than her usual speed. -/
theorem eeshas_usual_time (T T' : ℝ) (h1 : T' = T / 0.75) (h2 : T' = T + 20) : T = 60 := by
  sorry

end NUMINAMATH_GPT_eeshas_usual_time_l1284_128472


namespace NUMINAMATH_GPT_percentage_of_girls_l1284_128468

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 400) (h2 : B = 80) :
  (G * 100) / (B + G) = 80 :=
by sorry

end NUMINAMATH_GPT_percentage_of_girls_l1284_128468


namespace NUMINAMATH_GPT_triangle_angle_bisector_sum_l1284_128428

theorem triangle_angle_bisector_sum (P Q R : ℝ × ℝ)
  (hP : P = (-8, 5)) (hQ : Q = (-15, -19)) (hR : R = (1, -7)) 
  (a b c : ℕ) (h : a + c = 89) 
  (gcd_abc : Int.gcd (Int.gcd a b) c = 1) :
  a + c = 89 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_bisector_sum_l1284_128428


namespace NUMINAMATH_GPT_proof_statements_BCD_l1284_128492

variable (a b : ℝ)

theorem proof_statements_BCD (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) :=
by
  sorry

end NUMINAMATH_GPT_proof_statements_BCD_l1284_128492


namespace NUMINAMATH_GPT_cookies_baked_on_monday_is_32_l1284_128442

-- Definitions for the problem.
variable (X : ℕ)

-- Conditions.
def cookies_baked_on_monday := X
def cookies_baked_on_tuesday := X / 2
def cookies_baked_on_wednesday := 3 * (X / 2) - 4

-- Total cookies at the end of three days.
def total_cookies := cookies_baked_on_monday X + cookies_baked_on_tuesday X + cookies_baked_on_wednesday X

-- Theorem statement to prove the number of cookies baked on Monday.
theorem cookies_baked_on_monday_is_32 : total_cookies X = 92 → cookies_baked_on_monday X = 32 :=
by
  -- We would add the proof steps here.
  sorry

end NUMINAMATH_GPT_cookies_baked_on_monday_is_32_l1284_128442


namespace NUMINAMATH_GPT_employees_cycle_l1284_128410

theorem employees_cycle (total_employees : ℕ) (drivers_percentage walkers_percentage cyclers_percentage: ℕ) (walk_cycle_ratio_walk walk_cycle_ratio_cycle: ℕ)
    (h_total : total_employees = 500)
    (h_drivers_perc : drivers_percentage = 35)
    (h_transit_perc : walkers_percentage = 25)
    (h_walkers_cyclers_ratio_walk : walk_cycle_ratio_walk = 3)
    (h_walkers_cyclers_ratio_cycle : walk_cycle_ratio_cycle = 7) :
    cyclers_percentage = 140 :=
by
  sorry

end NUMINAMATH_GPT_employees_cycle_l1284_128410


namespace NUMINAMATH_GPT_vector_scalar_operations_l1284_128418

-- Define the vectors
def v1 : ℤ × ℤ := (2, -9)
def v2 : ℤ × ℤ := (-1, -6)

-- Define the scalars
def c1 : ℤ := 4
def c2 : ℤ := 3

-- Define the scalar multiplication of vectors
def scale (c : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (c * v.1, c * v.2)

-- Define the vector subtraction
def sub (v w : ℤ × ℤ) : ℤ × ℤ := (v.1 - w.1, v.2 - w.2)

-- State the theorem
theorem vector_scalar_operations :
  sub (scale c1 v1) (scale c2 v2) = (11, -18) :=
by
  sorry

end NUMINAMATH_GPT_vector_scalar_operations_l1284_128418


namespace NUMINAMATH_GPT_cos_sq_sub_sin_sq_pi_div_12_l1284_128400

theorem cos_sq_sub_sin_sq_pi_div_12 : 
  (Real.cos (π / 12))^2 - (Real.sin (π / 12))^2 = Real.cos (π / 6) :=
by
  sorry

end NUMINAMATH_GPT_cos_sq_sub_sin_sq_pi_div_12_l1284_128400


namespace NUMINAMATH_GPT_rate_of_decrease_l1284_128424

theorem rate_of_decrease (x : ℝ) (h : 400 * (1 - x) ^ 2 = 361) : x = 0.05 :=
by {
  sorry -- The proof is omitted as requested.
}

end NUMINAMATH_GPT_rate_of_decrease_l1284_128424


namespace NUMINAMATH_GPT_inequality_holds_l1284_128406

theorem inequality_holds (x y : ℝ) : (y - x^2 < abs x) ↔ (y < x^2 + abs x) := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1284_128406


namespace NUMINAMATH_GPT_tangent_line_at_P_range_of_a_l1284_128450

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := a * (x - 1/x) - Real.log x

-- Problem (Ⅰ): Tangent line equation at P(1, f(1)) for a = 1
theorem tangent_line_at_P (x : ℝ) (h : x = 1) : (∃ y : ℝ, f x 1 = y ∧ x - y - 1 = 0) := sorry

-- Problem (Ⅱ): Range of a for f(x) ≥ 0 ∀ x ≥ 1
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x ≥ 1 → f x a ≥ 0) : a ≥ 1/2 := sorry

end NUMINAMATH_GPT_tangent_line_at_P_range_of_a_l1284_128450


namespace NUMINAMATH_GPT_accurate_bottle_weight_l1284_128483

-- Define the options as constants
def OptionA : ℕ := 500 -- milligrams
def OptionB : ℕ := 500 * 1000 -- grams
def OptionC : ℕ := 500 * 1000 * 1000 -- kilograms
def OptionD : ℕ := 500 * 1000 * 1000 * 1000 -- tons

-- Define a threshold range for the weight of a standard bottle of mineral water in grams
def typicalBottleWeightMin : ℕ := 400 -- for example
def typicalBottleWeightMax : ℕ := 600 -- for example

-- Translate the question and conditions into a proof statement
theorem accurate_bottle_weight : OptionB = 500 * 1000 :=
by
  -- Normally, we would add the necessary steps here to prove the statement
  sorry

end NUMINAMATH_GPT_accurate_bottle_weight_l1284_128483


namespace NUMINAMATH_GPT_total_apples_l1284_128427

-- Define the number of apples given to each person
def apples_per_person : ℝ := 15.0

-- Define the number of people
def number_of_people : ℝ := 3.0

-- Goal: Prove that the total number of apples is 45.0
theorem total_apples : apples_per_person * number_of_people = 45.0 := by
  sorry

end NUMINAMATH_GPT_total_apples_l1284_128427


namespace NUMINAMATH_GPT_sum_of_odd_powers_divisible_by_six_l1284_128445

theorem sum_of_odd_powers_divisible_by_six (a1 a2 a3 a4 : ℤ)
    (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) :
    ∀ k : ℕ, k % 2 = 1 → 6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
by
  intros k hk
  sorry

end NUMINAMATH_GPT_sum_of_odd_powers_divisible_by_six_l1284_128445


namespace NUMINAMATH_GPT_fractions_product_l1284_128495

theorem fractions_product :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end NUMINAMATH_GPT_fractions_product_l1284_128495


namespace NUMINAMATH_GPT_wendy_boxes_l1284_128466

theorem wendy_boxes (x : ℕ) (w_brother : ℕ) (total : ℕ) (candy_per_box : ℕ) 
    (h_w_brother : w_brother = 6) 
    (h_candy_per_box : candy_per_box = 3) 
    (h_total : total = 12) 
    (h_equation : 3 * x + w_brother = total) : 
    x = 2 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_wendy_boxes_l1284_128466


namespace NUMINAMATH_GPT_original_number_l1284_128409

theorem original_number (x : ℝ) (h : 1.50 * x = 165) : x = 110 :=
sorry

end NUMINAMATH_GPT_original_number_l1284_128409


namespace NUMINAMATH_GPT_chloe_total_score_l1284_128481

theorem chloe_total_score :
  let first_level_treasure_points := 9
  let first_level_bonus_points := 15
  let first_level_treasures := 6
  let second_level_treasure_points := 11
  let second_level_bonus_points := 20
  let second_level_treasures := 3

  let first_level_score := first_level_treasures * first_level_treasure_points + first_level_bonus_points
  let second_level_score := second_level_treasures * second_level_treasure_points + second_level_bonus_points

  first_level_score + second_level_score = 122 :=
by
  sorry

end NUMINAMATH_GPT_chloe_total_score_l1284_128481


namespace NUMINAMATH_GPT_reciprocal_of_abs_neg_two_l1284_128475

theorem reciprocal_of_abs_neg_two : 1 / |(-2: ℤ)| = (1 / 2: ℚ) := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_abs_neg_two_l1284_128475


namespace NUMINAMATH_GPT_find_integer_pairs_l1284_128447

theorem find_integer_pairs :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 2 = a^3 + 2 * b →
  (a = 1 ∧ b = 1) ∨ (a = 3 ∧ b = 25) ∨ (a = 4 ∧ b = 31) ∨ (a = 5 ∧ b = 41) ∨ (a = 8 ∧ b = 85) :=
by
  intros a b ha hb hab_eq
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1284_128447


namespace NUMINAMATH_GPT_tangent_line_with_smallest_slope_l1284_128474

-- Define the given curve
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 + 6 * x - 10

-- Define the derivative of the given curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 6

-- Define the equation of the tangent line with the smallest slope
def tangent_line (x y : ℝ) : Prop := 3 * x - y = 11

-- Prove that the equation of the tangent line with the smallest slope on the curve is 3x - y - 11 = 0
theorem tangent_line_with_smallest_slope :
  ∃ x y : ℝ, curve x = y ∧ curve_derivative x = 3 ∧ tangent_line x y :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_with_smallest_slope_l1284_128474


namespace NUMINAMATH_GPT_remainder_43_pow_43_plus_43_mod_44_l1284_128436

theorem remainder_43_pow_43_plus_43_mod_44 : (43^43 + 43) % 44 = 42 :=
by 
    sorry

end NUMINAMATH_GPT_remainder_43_pow_43_plus_43_mod_44_l1284_128436


namespace NUMINAMATH_GPT_arithmetic_seq_contains_geometric_seq_l1284_128487

theorem arithmetic_seq_contains_geometric_seq (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (∃ (ns : ℕ → ℕ) (k : ℝ), k ≠ 1 ∧ (∀ n, a + b * (ns (n + 1)) = k * (a + b * (ns n)))) ↔ (∃ (q : ℚ), a = q * b) :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_contains_geometric_seq_l1284_128487


namespace NUMINAMATH_GPT_original_ticket_price_l1284_128499

open Real

theorem original_ticket_price 
  (P : ℝ)
  (total_revenue : ℝ)
  (revenue_equation : total_revenue = 10 * 0.60 * P + 20 * 0.85 * P + 15 * P) 
  (total_revenue_val : total_revenue = 760) : 
  P = 20 := 
by
  sorry

end NUMINAMATH_GPT_original_ticket_price_l1284_128499


namespace NUMINAMATH_GPT_suraj_average_after_17th_innings_l1284_128407

theorem suraj_average_after_17th_innings (A : ℕ) :
  (16 * A + 92) / 17 = A + 4 -> A + 4 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_suraj_average_after_17th_innings_l1284_128407


namespace NUMINAMATH_GPT_Alice_wins_no_matter_what_Bob_does_l1284_128490

theorem Alice_wins_no_matter_what_Bob_does (a b c : ℝ) :
  (∀ d : ℝ, (b + d) ^ 2 - 4 * (a + d) * (c + d) ≤ 0) → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Alice_wins_no_matter_what_Bob_does_l1284_128490


namespace NUMINAMATH_GPT_max_cities_visited_l1284_128471

theorem max_cities_visited (n k : ℕ) : ∃ t, t = n - k :=
by
  sorry

end NUMINAMATH_GPT_max_cities_visited_l1284_128471


namespace NUMINAMATH_GPT_maximum_n_l1284_128431

noncomputable def a1 : ℝ := sorry -- define a1 solving a_5 equations
noncomputable def q : ℝ := sorry -- define q solving a_5 and a_6 + a_7 equations
noncomputable def sn (n : ℕ) : ℝ := (a1 * (1 - q^n)) / (1 - q)  -- S_n of geometric series with a1 and q
noncomputable def pin (n : ℕ) : ℝ := (a1 * (q^((1 + n) * n / 2 - (11 * n) / 2 + 19 / 2)))  -- Pi solely in terms of n, a1, and q

theorem maximum_n (n : ℕ) (h1 : (a1 : ℝ) > 0) (h2 : q > 0) (h3 : q ≠ 1)
(h4 : a1 * q^4 = 1 / 4) (h5 : a1 * q^5 + a1 * q^6 = 3 / 2) :
  ∃ n : ℕ, sn n > pin n ∧ ∀ m : ℕ, m > 13 → sn m ≤ pin m := sorry

end NUMINAMATH_GPT_maximum_n_l1284_128431


namespace NUMINAMATH_GPT_actual_length_of_road_l1284_128457

-- Define the conditions
def scale_factor : ℝ := 2500000
def length_on_map : ℝ := 6
def cm_to_km : ℝ := 100000

-- State the theorem
theorem actual_length_of_road : (length_on_map * scale_factor) / cm_to_km = 150 := by
  sorry

end NUMINAMATH_GPT_actual_length_of_road_l1284_128457


namespace NUMINAMATH_GPT_inequality_solution_m_range_l1284_128482

def f (x : ℝ) : ℝ := abs (x - 2)
def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem inequality_solution (a : ℝ) :
  (∀ x : ℝ, a = 1 → f x + a - 1 > 0 ↔ x ≠ 2) ∧
  (a > 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ True) ∧
  (a < 1 → ∀ x : ℝ, f x + a - 1 > 0 ↔ x < a + 1 ∨ x > 3 - a) :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m < 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_m_range_l1284_128482


namespace NUMINAMATH_GPT_triangle_sum_l1284_128497

-- Define the triangle operation
def triangle (a b c : ℕ) : ℕ := a + b + c

-- State the theorem
theorem triangle_sum :
  triangle 2 4 3 + triangle 1 6 5 = 21 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sum_l1284_128497


namespace NUMINAMATH_GPT_warehouse_can_release_100kg_l1284_128496

theorem warehouse_can_release_100kg (a b c d : ℕ) : 
  24 * a + 23 * b + 17 * c + 16 * d = 100 → True :=
by
  sorry

end NUMINAMATH_GPT_warehouse_can_release_100kg_l1284_128496


namespace NUMINAMATH_GPT_total_point_value_of_test_l1284_128464

theorem total_point_value_of_test (total_questions : ℕ) (five_point_questions : ℕ) 
  (ten_point_questions : ℕ) (points_5 : ℕ) (points_10 : ℕ) 
  (h1 : total_questions = 30) (h2 : five_point_questions = 20) 
  (h3 : ten_point_questions = total_questions - five_point_questions) 
  (h4 : points_5 = 5) (h5 : points_10 = 10) : 
  five_point_questions * points_5 + ten_point_questions * points_10 = 200 :=
by
  sorry

end NUMINAMATH_GPT_total_point_value_of_test_l1284_128464


namespace NUMINAMATH_GPT_highest_degree_divisibility_l1284_128411

-- Definition of the problem settings
def prime_number := 1991
def number_1 := 1990 ^ (1991 ^ 1002)
def number_2 := 1992 ^ (1501 ^ 1901)
def combined_number := number_1 + number_2

-- Statement of the proof to be formalized
theorem highest_degree_divisibility (k : ℕ) : k = 1001 ∧ prime_number ^ k ∣ combined_number := by
  sorry

end NUMINAMATH_GPT_highest_degree_divisibility_l1284_128411


namespace NUMINAMATH_GPT_ratio_ab_l1284_128458

theorem ratio_ab (a b : ℚ) (h : b / a = 5 / 13) : (a - b) / (a + b) = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_ab_l1284_128458


namespace NUMINAMATH_GPT_smart_charging_piles_growth_l1284_128453

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_smart_charging_piles_growth_l1284_128453


namespace NUMINAMATH_GPT_lowest_possible_number_of_students_l1284_128435

theorem lowest_possible_number_of_students : ∃ n : ℕ, (n > 0) ∧ (∃ k1 : ℕ, n = 10 * k1) ∧ (∃ k2 : ℕ, n = 24 * k2) ∧ n = 120 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_number_of_students_l1284_128435


namespace NUMINAMATH_GPT_sum_of_solutions_eqn_l1284_128486

theorem sum_of_solutions_eqn : 
  (∀ x : ℝ, -48 * x^2 + 100 * x + 200 = 0 → False) → 
  (-100 / -48) = (25 / 12) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eqn_l1284_128486


namespace NUMINAMATH_GPT_fifth_graders_more_than_seventh_l1284_128493

theorem fifth_graders_more_than_seventh (price_per_pencil : ℕ) (price_per_pencil_pos : price_per_pencil > 0)
    (total_cents_7th : ℕ) (total_cents_7th_val : total_cents_7th = 201)
    (total_cents_5th : ℕ) (total_cents_5th_val : total_cents_5th = 243)
    (pencil_price_div_7th : total_cents_7th % price_per_pencil = 0)
    (pencil_price_div_5th : total_cents_5th % price_per_pencil = 0) :
    (total_cents_5th / price_per_pencil - total_cents_7th / price_per_pencil = 14) := 
by
    sorry

end NUMINAMATH_GPT_fifth_graders_more_than_seventh_l1284_128493


namespace NUMINAMATH_GPT_exists_ij_aij_gt_ij_l1284_128461

theorem exists_ij_aij_gt_ij (a : ℕ → ℕ → ℕ) 
  (h_a_positive : ∀ i j, 0 < a i j)
  (h_a_distribution : ∀ k, (∃ S : Finset (ℕ × ℕ), S.card = 8 ∧ ∀ ij : ℕ × ℕ, ij ∈ S ↔ a ij.1 ij.2 = k)) :
  ∃ i j, a i j > i * j :=
by
  sorry

end NUMINAMATH_GPT_exists_ij_aij_gt_ij_l1284_128461


namespace NUMINAMATH_GPT_anne_ben_charlie_difference_l1284_128405

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def charlie_discount_rate : ℝ := 0.15

def anne_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def ben_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)
def charlie_total : ℝ := (original_price * (1 - charlie_discount_rate)) * (1 + sales_tax_rate)

def anne_minus_ben_minus_charlie : ℝ := anne_total - ben_total - charlie_total

theorem anne_ben_charlie_difference : anne_minus_ben_minus_charlie = -12.96 :=
by
  sorry

end NUMINAMATH_GPT_anne_ben_charlie_difference_l1284_128405


namespace NUMINAMATH_GPT_remainder_55_57_div_8_l1284_128463

def remainder (a b n : ℕ) := (a * b) % n

theorem remainder_55_57_div_8 : remainder 55 57 8 = 7 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_remainder_55_57_div_8_l1284_128463


namespace NUMINAMATH_GPT_solve_for_z_l1284_128449

theorem solve_for_z (z : ℂ) (i : ℂ) (h : i^2 = -1) : 3 + 2 * i * z = 5 - 3 * i * z → z = - (2 * i) / 5 :=
by
  intro h_equation
  -- Proof steps will be provided here.
  sorry

end NUMINAMATH_GPT_solve_for_z_l1284_128449


namespace NUMINAMATH_GPT_greatest_number_of_balloons_l1284_128413

-- Let p be the regular price of one balloon, and M be the total amount of money Orvin has
variable (p M : ℝ)

-- Initial condition: Orvin can buy 45 balloons at the regular price.
-- Thus, he has money M = 45 * p
def orvin_has_enough_money : Prop :=
  M = 45 * p

-- Special Sale condition: The first balloon costs p and the second balloon costs p/2,
-- so total cost for 2 balloons = 1.5 * p
def special_sale_condition : Prop :=
  ∀ pairs : ℝ, M / (1.5 * p) = pairs ∧ pairs * 2 = 60

-- Given the initial condition and the special sale condition, prove the greatest 
-- number of balloons Orvin could purchase is 60
theorem greatest_number_of_balloons (p : ℝ) (M : ℝ) (h1 : orvin_has_enough_money p M) (h2 : special_sale_condition p M) : 
∀ N : ℝ, N = 60 :=
sorry

end NUMINAMATH_GPT_greatest_number_of_balloons_l1284_128413


namespace NUMINAMATH_GPT_a_4_value_l1284_128417

def seq (n : ℕ) : ℚ :=
  if n = 0 then 0 -- To handle ℕ index starting from 0.
  else if n = 1 then 1
  else seq (n - 1) + 1 / ((n:ℚ) * (n-1))

noncomputable def a_4 : ℚ := seq 4

theorem a_4_value : a_4 = 7 / 4 := 
  by sorry

end NUMINAMATH_GPT_a_4_value_l1284_128417


namespace NUMINAMATH_GPT_profit_sharing_l1284_128430

-- Define constants and conditions
def Tom_investment : ℕ := 30000
def Tom_share : ℝ := 0.40

def Jose_investment : ℕ := 45000
def Jose_start_month : ℕ := 2
def Jose_share : ℝ := 0.30

def Sarah_investment : ℕ := 60000
def Sarah_start_month : ℕ := 5
def Sarah_share : ℝ := 0.20

def Ravi_investment : ℕ := 75000
def Ravi_start_month : ℕ := 8
def Ravi_share : ℝ := 0.10

def total_profit : ℕ := 120000

-- Define expected shares
def Tom_expected_share : ℕ := 48000
def Jose_expected_share : ℕ := 36000
def Sarah_expected_share : ℕ := 24000
def Ravi_expected_share : ℕ := 12000

-- Theorem statement
theorem profit_sharing :
  let Tom_contribution := Tom_investment * 12
  let Jose_contribution := Jose_investment * (12 - Jose_start_month)
  let Sarah_contribution := Sarah_investment * (12 - Sarah_start_month)
  let Ravi_contribution := Ravi_investment * (12 - Ravi_start_month)
  Tom_share * total_profit = Tom_expected_share ∧
  Jose_share * total_profit = Jose_expected_share ∧
  Sarah_share * total_profit = Sarah_expected_share ∧
  Ravi_share * total_profit = Ravi_expected_share := by {
    sorry
  }

end NUMINAMATH_GPT_profit_sharing_l1284_128430


namespace NUMINAMATH_GPT_percent_nurses_with_neither_l1284_128420

-- Define the number of nurses in each category
def total_nurses : ℕ := 150
def nurses_with_hbp : ℕ := 90
def nurses_with_ht : ℕ := 50
def nurses_with_both : ℕ := 30

-- Define a predicate that checks the conditions of the problem
theorem percent_nurses_with_neither :
  ((total_nurses - (nurses_with_hbp + nurses_with_ht - nurses_with_both)) * 100 : ℚ) / total_nurses = 2667 / 100 :=
by sorry

end NUMINAMATH_GPT_percent_nurses_with_neither_l1284_128420


namespace NUMINAMATH_GPT_sector_max_angle_l1284_128429

variables (r l : ℝ)

theorem sector_max_angle (h : 2 * r + l = 40) : (l / r) = 2 :=
sorry

end NUMINAMATH_GPT_sector_max_angle_l1284_128429


namespace NUMINAMATH_GPT_find_x_l1284_128426

theorem find_x (x : ℕ) (h : 5 * x + 4 * x + x + 2 * x = 360) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1284_128426


namespace NUMINAMATH_GPT_longest_diagonal_length_l1284_128470

-- Defining conditions
variable (d1 d2 : ℝ)
variable (x : ℝ)
variable (area : ℝ)
variable (h_area : area = 144)
variable (h_ratio : d1 = 4 * x)
variable (h_ratio' : d2 = 3 * x)
variable (h_area_eq : area = 1 / 2 * d1 * d2)

-- The Lean statement, asserting the length of the longest diagonal is 8 * sqrt(6)
theorem longest_diagonal_length (x : ℝ) (h_area : 1 / 2 * (4 * x) * (3 * x) = 144) :
  4 * x = 8 * Real.sqrt 6 := by
sorry

end NUMINAMATH_GPT_longest_diagonal_length_l1284_128470


namespace NUMINAMATH_GPT_rhombus_side_length_l1284_128473

-- Definitions
def is_rhombus_perimeter (P s : ℝ) : Prop := P = 4 * s

-- Theorem to prove
theorem rhombus_side_length (P : ℝ) (hP : P = 4) : ∃ s : ℝ, is_rhombus_perimeter P s ∧ s = 1 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l1284_128473


namespace NUMINAMATH_GPT_b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l1284_128484

variable (a b : ℕ)

-- Conditions
def is_multiple_of_5 (n : ℕ) : Prop := ∃ k : ℕ, n = 5 * k
def is_multiple_of_10 (n : ℕ) : Prop := ∃ k : ℕ, n = 10 * k

-- Given conditions in the problem
axiom h_a : is_multiple_of_5 a
axiom h_b : is_multiple_of_10 b

-- Statements to be proved
theorem b_is_multiple_of_5 : is_multiple_of_5 b :=
sorry

theorem a_plus_b_is_multiple_of_5 : is_multiple_of_5 (a + b) :=
sorry

end NUMINAMATH_GPT_b_is_multiple_of_5_a_plus_b_is_multiple_of_5_l1284_128484


namespace NUMINAMATH_GPT_value_of_expression_l1284_128478

theorem value_of_expression : (85 + 32 / 113) * 113 = 9635 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1284_128478


namespace NUMINAMATH_GPT_paws_on_ground_are_correct_l1284_128452

-- Problem statement
def num_paws_on_ground (total_dogs : ℕ) (half_on_all_fours : ℕ) (paws_on_all_fours : ℕ) (half_on_two_legs : ℕ) (paws_on_two_legs : ℕ) : ℕ :=
  half_on_all_fours * paws_on_all_fours + half_on_two_legs * paws_on_two_legs

theorem paws_on_ground_are_correct :
  let total_dogs := 12
  let half_on_all_fours := 6
  let half_on_two_legs := 6
  let paws_on_all_fours := 4
  let paws_on_two_legs := 2
  num_paws_on_ground total_dogs half_on_all_fours paws_on_all_fours half_on_two_legs paws_on_two_legs = 36 :=
by sorry

end NUMINAMATH_GPT_paws_on_ground_are_correct_l1284_128452


namespace NUMINAMATH_GPT_solution_z_sq_eq_neg_4_l1284_128438

theorem solution_z_sq_eq_neg_4 (x y : ℝ) (i : ℂ) (z : ℂ) (h : z = x + y * i) (hi : i^2 = -1) : 
  z^2 = -4 ↔ z = 2 * i ∨ z = -2 * i := 
by
  sorry

end NUMINAMATH_GPT_solution_z_sq_eq_neg_4_l1284_128438


namespace NUMINAMATH_GPT_max_area_quadrilateral_cdfg_l1284_128491

theorem max_area_quadrilateral_cdfg (s : ℝ) (x : ℝ)
  (h1 : s = 1) (h2 : x > 0) (h3 : x < s) (h4 : AE = x) (h5 : AF = x) : 
  ∃ x, x > 0 ∧ x < 1 ∧ (1 - x) * x ≤ 5 / 8 :=
sorry

end NUMINAMATH_GPT_max_area_quadrilateral_cdfg_l1284_128491


namespace NUMINAMATH_GPT_distance_between_stations_l1284_128423

theorem distance_between_stations (x y t : ℝ) 
(start_same_hour : t > 0)
(speed_slow_train : ∀ t, x = 16 * t)
(speed_fast_train : ∀ t, y = 21 * t)
(distance_difference : y = x + 60) : 
  x + y = 444 := 
sorry

end NUMINAMATH_GPT_distance_between_stations_l1284_128423


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_correct_l1284_128477

-- Definitions based on conditions
def equilateral_triangle_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 3

def isosceles_triangle_perimeter (side1 side2 base : ℕ) : ℕ :=
  side1 + side2 + base

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 45
def equilateral_triangle_side : ℕ := equilateral_triangle_side_length equilateral_triangle_perimeter

-- The side of the equilateral triangle is also a leg of the isosceles triangle
def isosceles_triangle_leg : ℕ := equilateral_triangle_side
def isosceles_triangle_base : ℕ := 10

-- The problem to prove
theorem isosceles_triangle_perimeter_correct : 
  isosceles_triangle_perimeter isosceles_triangle_leg isosceles_triangle_leg isosceles_triangle_base = 40 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_correct_l1284_128477


namespace NUMINAMATH_GPT_rikki_poetry_sales_l1284_128439

theorem rikki_poetry_sales :
  let words_per_5min := 25
  let total_minutes := 2 * 60
  let intervals := total_minutes / 5
  let total_words := words_per_5min * intervals
  let total_earnings := 6
  let price_per_word := total_earnings / total_words
  price_per_word = 0.01 :=
by
  sorry

end NUMINAMATH_GPT_rikki_poetry_sales_l1284_128439


namespace NUMINAMATH_GPT_inequality_x_y_l1284_128479

theorem inequality_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y + x * y = 3) : x + y ≥ 2 := 
  sorry

end NUMINAMATH_GPT_inequality_x_y_l1284_128479


namespace NUMINAMATH_GPT_jellybeans_initial_amount_l1284_128437

theorem jellybeans_initial_amount (x : ℝ) 
  (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end NUMINAMATH_GPT_jellybeans_initial_amount_l1284_128437


namespace NUMINAMATH_GPT_minimum_x_plus_y_l1284_128489

variable (x y : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1)

theorem minimum_x_plus_y (hx : 0 < x) (hy : 0 < y) (h : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1) : x + y ≥ 9 / 4 :=
sorry

end NUMINAMATH_GPT_minimum_x_plus_y_l1284_128489


namespace NUMINAMATH_GPT_range_of_F_l1284_128451

theorem range_of_F (A B C : ℝ) (h1 : 0 < A) (h2 : A ≤ B) (h3 : B ≤ C) (h4 : C < π / 2) :
  1 + (Real.sqrt 2) / 2 < (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) ∧
  (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) < 2 :=
  sorry

end NUMINAMATH_GPT_range_of_F_l1284_128451


namespace NUMINAMATH_GPT_range_of_a_l1284_128456

def decreasing_range (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ 4 → y ≤ 4 → x < y → (x^2 + 2 * (a - 1) * x + 2) ≥ (y^2 + 2 * (a - 1) * y + 2)

theorem range_of_a (a : ℝ) : decreasing_range a ↔ a ≤ -3 := 
  sorry

end NUMINAMATH_GPT_range_of_a_l1284_128456


namespace NUMINAMATH_GPT_maximum_n_l1284_128444

variable (x y z : ℝ)

theorem maximum_n (h1 : x + y + z = 12) (h2 : x * y + y * z + z * x = 30) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_maximum_n_l1284_128444


namespace NUMINAMATH_GPT_sequence_general_formula_l1284_128403

-- Definitions according to conditions in a)
def seq (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n + 1

def S (n : ℕ) (seq : ℕ → ℕ) : ℕ :=
  n * seq (n + 1) - 3 * n^2 - 4 * n

-- The proof goal
theorem sequence_general_formula (n : ℕ) (h : 0 < n) :
  seq n = 2 * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1284_128403


namespace NUMINAMATH_GPT_max_sides_of_polygon_in_1950_gon_l1284_128455

theorem max_sides_of_polygon_in_1950_gon (n : ℕ) (h : n = 1950) :
  ∃ (m : ℕ), (m ≤ 1949) ∧ (∀ k, k > m → k ≤ 1949) :=
sorry

end NUMINAMATH_GPT_max_sides_of_polygon_in_1950_gon_l1284_128455


namespace NUMINAMATH_GPT_general_term_an_l1284_128421

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 2
noncomputable def S_n (n : ℕ) : ℕ := n^2 + 3 * n

theorem general_term_an (n : ℕ) (h : 1 ≤ n) : a_n n = (S_n n) - (S_n (n-1)) :=
by sorry

end NUMINAMATH_GPT_general_term_an_l1284_128421


namespace NUMINAMATH_GPT_power_mean_inequality_l1284_128480

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end NUMINAMATH_GPT_power_mean_inequality_l1284_128480


namespace NUMINAMATH_GPT_product_of_decimal_numbers_l1284_128454

theorem product_of_decimal_numbers 
  (h : 213 * 16 = 3408) : 
  1.6 * 21.3 = 34.08 :=
by
  sorry

end NUMINAMATH_GPT_product_of_decimal_numbers_l1284_128454


namespace NUMINAMATH_GPT_mod_remainder_l1284_128498

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end NUMINAMATH_GPT_mod_remainder_l1284_128498


namespace NUMINAMATH_GPT_side_length_of_base_l1284_128422

variable (s : ℕ) -- side length of the square base
variable (A : ℕ) -- area of one lateral face
variable (h : ℕ) -- slant height

-- Given conditions
def area_of_lateral_face (s h : ℕ) : ℕ := 20 * s

axiom lateral_face_area_given : A = 120
axiom slant_height_given : h = 40

theorem side_length_of_base (A : ℕ) (h : ℕ) (s : ℕ) : 20 * s = A → s = 6 :=
by
  -- The proof part is omitted, only required the statement as per guidelines
  sorry

end NUMINAMATH_GPT_side_length_of_base_l1284_128422


namespace NUMINAMATH_GPT_perimeter_of_triangle_AF2B_l1284_128414

theorem perimeter_of_triangle_AF2B (a : ℝ) (m n : ℝ) (F1 F2 A B : ℝ × ℝ) 
  (h_hyperbola : ∀ x y : ℝ, (x^2 - 4*y^2 = 4) ↔ (x^2 / 4 - y^2 = 1)) 
  (h_mn : m + n = 3) 
  (h_AF1 : dist A F1 = m) 
  (h_BF1 : dist B F1 = n) 
  (h_AF2 : dist A F2 = 4 + m) 
  (h_BF2 : dist B F2 = 4 + n) 
  : dist A F1 + dist A F2 + dist B F2 + dist B F1 = 14 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_AF2B_l1284_128414


namespace NUMINAMATH_GPT_tangent_line_solution_l1284_128448

variables (x y : ℝ)

noncomputable def circle_equation (m : ℝ) : Prop :=
  x^2 + y^2 - 4 * x + m * y = 0

def point_on_circle (m : ℝ) : Prop :=
  circle_equation 1 1 m

noncomputable def tangent_line_equation (x y : ℝ) : Prop :=
  x - 2 * y + 1 = 0

theorem tangent_line_solution (m : ℝ) :
  point_on_circle m →
  m = 2 →
  tangent_line_equation 1 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_solution_l1284_128448


namespace NUMINAMATH_GPT_find_n_l1284_128416

   theorem find_n (n : ℕ) : 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 3 * y + z = n) → 
     (n = 34 ∨ n = 37) :=
   by
     intros
     sorry
   
end NUMINAMATH_GPT_find_n_l1284_128416


namespace NUMINAMATH_GPT_ages_correct_l1284_128415

variables (Son Daughter Wife Man Father : ℕ)

theorem ages_correct :
  (Man = Son + 20) ∧
  (Man = Daughter + 15) ∧
  (Man + 2 = 2 * (Son + 2)) ∧
  (Man + 2 = 3 * (Daughter + 2)) ∧
  (Wife = Man - 5) ∧
  (Wife + 6 = 2 * (Daughter + 6)) ∧
  (Father = Man + 32) →
  (Son = 7 ∧ Daughter = 12 ∧ Wife = 22 ∧ Man = 27 ∧ Father = 59) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ages_correct_l1284_128415


namespace NUMINAMATH_GPT_Jonathan_typing_time_l1284_128419

theorem Jonathan_typing_time
  (J : ℝ)
  (HJ : 0 < J)
  (rate_Jonathan : ℝ := 1 / J)
  (rate_Susan : ℝ := 1 / 30)
  (rate_Jack : ℝ := 1 / 24)
  (combined_rate : ℝ := 1 / 10)
  (combined_rate_eq : rate_Jonathan + rate_Susan + rate_Jack = combined_rate)
  : J = 40 :=
sorry

end NUMINAMATH_GPT_Jonathan_typing_time_l1284_128419


namespace NUMINAMATH_GPT_quadratic_function_symmetry_l1284_128494

theorem quadratic_function_symmetry
  (p : ℝ → ℝ)
  (h_sym : ∀ x, p (5.5 - x) = p (5.5 + x))
  (h_0 : p 0 = -4) :
  p 11 = -4 :=
by sorry

end NUMINAMATH_GPT_quadratic_function_symmetry_l1284_128494


namespace NUMINAMATH_GPT_length_width_difference_l1284_128434

theorem length_width_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 578) : L - W = 17 :=
sorry

end NUMINAMATH_GPT_length_width_difference_l1284_128434


namespace NUMINAMATH_GPT_sum_m_n_is_192_l1284_128485

def smallest_prime : ℕ := 2

def largest_four_divisors_under_200 : ℕ :=
  -- we assume this as 190 based on the provided problem's solution
  190

theorem sum_m_n_is_192 :
  smallest_prime = 2 →
  largest_four_divisors_under_200 = 190 →
  smallest_prime + largest_four_divisors_under_200 = 192 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sum_m_n_is_192_l1284_128485


namespace NUMINAMATH_GPT_parabola_equation_l1284_128412

theorem parabola_equation {p : ℝ} (hp : 0 < p)
  (h_cond : ∃ A B : ℝ × ℝ, (A.1^2 = 2 * A.2 * p) ∧ (B.1^2 = 2 * B.2 * p) ∧ (A.2 = A.1 - p / 2) ∧ (B.2 = B.1 - p / 2) ∧ (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4))
  : y^2 = 2 * x := sorry

end NUMINAMATH_GPT_parabola_equation_l1284_128412


namespace NUMINAMATH_GPT_bouquets_sold_on_Monday_l1284_128446

theorem bouquets_sold_on_Monday
  (tuesday_three_times_monday : ∀ (x : ℕ), bouquets_sold_Tuesday = 3 * x)
  (wednesday_third_of_tuesday : ∀ (bouquets_sold_Tuesday : ℕ), bouquets_sold_Wednesday = bouquets_sold_Tuesday / 3)
  (total_bouquets : bouquets_sold_Monday + bouquets_sold_Tuesday + bouquets_sold_Wednesday = 60)
  : bouquets_sold_Monday = 12 := 
sorry

end NUMINAMATH_GPT_bouquets_sold_on_Monday_l1284_128446


namespace NUMINAMATH_GPT_total_pencils_owned_l1284_128459

def SetA_pencils := 10
def SetB_pencils := 20
def SetC_pencils := 30

def friends_SetA_Buys := 3
def friends_SetB_Buys := 2
def friends_SetC_Buys := 2

def Chloe_SetA_Buys := 1
def Chloe_SetB_Buys := 1
def Chloe_SetC_Buys := 1

def total_friends_pencils := friends_SetA_Buys * SetA_pencils + friends_SetB_Buys * SetB_pencils + friends_SetC_Buys * SetC_pencils
def total_Chloe_pencils := Chloe_SetA_Buys * SetA_pencils + Chloe_SetB_Buys * SetB_pencils + Chloe_SetC_Buys * SetC_pencils
def total_pencils := total_friends_pencils + total_Chloe_pencils

theorem total_pencils_owned : total_pencils = 190 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_owned_l1284_128459


namespace NUMINAMATH_GPT_prove_a_eq_b_l1284_128441

theorem prove_a_eq_b (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h_eq : a^b = b^a) (h_a_lt_1 : a < 1) : a = b :=
by
  sorry

end NUMINAMATH_GPT_prove_a_eq_b_l1284_128441


namespace NUMINAMATH_GPT_most_stable_performance_l1284_128488

theorem most_stable_performance :
  ∀ (σ2_A σ2_B σ2_C σ2_D : ℝ), 
  σ2_A = 0.56 → 
  σ2_B = 0.78 → 
  σ2_C = 0.42 → 
  σ2_D = 0.63 → 
  σ2_C ≤ σ2_A ∧ σ2_C ≤ σ2_B ∧ σ2_C ≤ σ2_D :=
by
  intros σ2_A σ2_B σ2_C σ2_D hA hB hC hD
  sorry

end NUMINAMATH_GPT_most_stable_performance_l1284_128488


namespace NUMINAMATH_GPT_monica_total_savings_l1284_128433

noncomputable def weekly_savings (week: ℕ) : ℕ :=
  if week < 6 then 15 + 5 * week
  else if week < 11 then 40 - 5 * (week - 5)
  else weekly_savings (week % 10)

theorem monica_total_savings : 
  let cycle_savings := (15 + 20 + 25 + 30 + 35 + 40) + (40 + 35 + 30 + 25 + 20 + 15) - 40 
  let total_savings := 5 * cycle_savings
  total_savings = 1450 := by
  sorry

end NUMINAMATH_GPT_monica_total_savings_l1284_128433


namespace NUMINAMATH_GPT_number_of_family_members_l1284_128467

noncomputable def total_money : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 26 * 0.25

noncomputable def leftover_money : ℝ := 0.48

noncomputable def double_scoop_cost : ℝ := 3.0

noncomputable def amount_spent : ℝ := total_money - leftover_money

noncomputable def number_of_double_scoops : ℝ := amount_spent / double_scoop_cost

theorem number_of_family_members :
  number_of_double_scoops = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_family_members_l1284_128467
