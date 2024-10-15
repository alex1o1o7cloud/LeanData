import Mathlib

namespace NUMINAMATH_GPT_intersection_of_A_and_B_l170_17073

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l170_17073


namespace NUMINAMATH_GPT_maximize_profit_l170_17097

def cups_sold (p : ℝ) : ℝ :=
  150 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * cups_sold p

def cost : ℝ :=
  200

def profit (p : ℝ) : ℝ :=
  revenue p - cost

theorem maximize_profit (p : ℝ) (h : p ≤ 30) : p = 19 → profit p = 1206.25 :=
by
  sorry

end NUMINAMATH_GPT_maximize_profit_l170_17097


namespace NUMINAMATH_GPT_new_paint_intensity_l170_17007

theorem new_paint_intensity (V : ℝ) (h1 : V > 0) :
    let initial_intensity := 0.5
    let replaced_fraction := 0.4
    let replaced_intensity := 0.25
    let new_intensity := (0.3 + 0.1 * replaced_fraction)  -- derived from (0.6 * 0.5 + 0.4 * 0.25)
    new_intensity = 0.4 :=
by
    sorry

end NUMINAMATH_GPT_new_paint_intensity_l170_17007


namespace NUMINAMATH_GPT_p_twice_q_in_future_years_l170_17088

-- We define the ages of p and q
def p_current_age : ℕ := 33
def q_current_age : ℕ := 11

-- Third condition that is redundant given the values we already defined
def age_relation : Prop := (p_current_age = 3 * q_current_age)

-- Number of years in the future when p will be twice as old as q
def future_years_when_twice : ℕ := 11

-- Prove that in future_years_when_twice years, p will be twice as old as q
theorem p_twice_q_in_future_years :
  ∀ t : ℕ, t = future_years_when_twice → (p_current_age + t = 2 * (q_current_age + t)) := by
  sorry

end NUMINAMATH_GPT_p_twice_q_in_future_years_l170_17088


namespace NUMINAMATH_GPT_mikey_jelly_beans_l170_17054

theorem mikey_jelly_beans :
  let napoleon_jelly_beans := 17
  let sedrich_jelly_beans := napoleon_jelly_beans + 4
  let total_jelly_beans := napoleon_jelly_beans + sedrich_jelly_beans
  let twice_sum := 2 * total_jelly_beans
  ∃ mikey_jelly_beans, 4 * mikey_jelly_beans = twice_sum → mikey_jelly_beans = 19 :=
by
  intro napoleon_jelly_beans
  intro sedrich_jelly_beans
  intro total_jelly_beans
  intro twice_sum
  use 19
  sorry

end NUMINAMATH_GPT_mikey_jelly_beans_l170_17054


namespace NUMINAMATH_GPT_unique_pairs_pos_int_satisfy_eq_l170_17008

theorem unique_pairs_pos_int_satisfy_eq (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  a^(b^2) = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) := 
by
  sorry

end NUMINAMATH_GPT_unique_pairs_pos_int_satisfy_eq_l170_17008


namespace NUMINAMATH_GPT_xiaomings_possible_score_l170_17084

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end NUMINAMATH_GPT_xiaomings_possible_score_l170_17084


namespace NUMINAMATH_GPT_volume_ratio_l170_17095

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length * side_length * side_length

theorem volume_ratio 
  (hyungjin_side_length_cm : ℕ)
  (kyujun_side_length_m : ℕ)
  (h1 : hyungjin_side_length_cm = 100)
  (h2 : kyujun_side_length_m = 2) :
  volume_of_cube (kyujun_side_length_m * 100) = 8 * volume_of_cube hyungjin_side_length_cm :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_l170_17095


namespace NUMINAMATH_GPT_sqrt_rational_rational_l170_17064

theorem sqrt_rational_rational 
  (a b : ℚ) 
  (h : ∃ r : ℚ, r = (a : ℝ).sqrt + (b : ℝ).sqrt) : 
  (∃ p : ℚ, p = (a : ℝ).sqrt) ∧ (∃ q : ℚ, q = (b : ℝ).sqrt) := 
sorry

end NUMINAMATH_GPT_sqrt_rational_rational_l170_17064


namespace NUMINAMATH_GPT_ann_top_cost_l170_17009

noncomputable def cost_per_top (T : ℝ) := 75 = (5 * 7) + (2 * 10) + (4 * T)

theorem ann_top_cost : cost_per_top 5 :=
by {
  -- statement: prove cost per top given conditions
  sorry
}

end NUMINAMATH_GPT_ann_top_cost_l170_17009


namespace NUMINAMATH_GPT_find_flag_count_l170_17056

-- Definitions of conditions
inductive Color
| purple
| gold
| silver

-- Function to count valid flags
def countValidFlags : Nat :=
  let first_stripe_choices := 3
  let second_stripe_choices := 2
  let third_stripe_choices := 2
  first_stripe_choices * second_stripe_choices * third_stripe_choices

-- Statement to prove
theorem find_flag_count : countValidFlags = 12 := by
  sorry

end NUMINAMATH_GPT_find_flag_count_l170_17056


namespace NUMINAMATH_GPT_solve_for_t_l170_17038

theorem solve_for_t (s t : ℤ) (h1 : 11 * s + 7 * t = 160) (h2 : s = 2 * t + 4) : t = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_t_l170_17038


namespace NUMINAMATH_GPT_acute_triangle_sec_csc_inequality_l170_17091

theorem acute_triangle_sec_csc_inequality (A B C : ℝ) (h : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA90 : A < π / 2) (hB90 : B < π / 2) (hC90 : C < π / 2) :
  (1 / Real.cos A) + (1 / Real.cos B) + (1 / Real.cos C) ≥
  (1 / Real.sin (A / 2)) + (1 / Real.sin (B / 2)) + (1 / Real.sin (C / 2)) :=
by sorry

end NUMINAMATH_GPT_acute_triangle_sec_csc_inequality_l170_17091


namespace NUMINAMATH_GPT_effective_average_speed_l170_17094

def rowing_speed_with_stream := 16 -- km/h
def rowing_speed_against_stream := 6 -- km/h
def stream1_effect := 2 -- km/h
def stream2_effect := -1 -- km/h
def stream3_effect := 3 -- km/h
def opposing_wind := 1 -- km/h

theorem effective_average_speed :
  ((rowing_speed_with_stream + stream1_effect - opposing_wind) + 
   (rowing_speed_against_stream + stream2_effect - opposing_wind) + 
   (rowing_speed_with_stream + stream3_effect - opposing_wind)) / 3 = 13 := 
by
  sorry

end NUMINAMATH_GPT_effective_average_speed_l170_17094


namespace NUMINAMATH_GPT_find_a_in_triangle_l170_17099

theorem find_a_in_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) 
  : a = 4 :=
  sorry

end NUMINAMATH_GPT_find_a_in_triangle_l170_17099


namespace NUMINAMATH_GPT_sum_9_to_12_l170_17055

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variables {S : ℕ → ℝ} -- Define the sum function of the sequence

-- Define the conditions given in the problem
def S_4 : ℝ := 8
def S_8 : ℝ := 20

-- The goal is to show that the sum of the 9th to 12th terms is 16
theorem sum_9_to_12 : (a 9) + (a 10) + (a 11) + (a 12) = 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_9_to_12_l170_17055


namespace NUMINAMATH_GPT_find_k_l170_17025

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l170_17025


namespace NUMINAMATH_GPT_smallest_constant_for_triangle_sides_l170_17002

theorem smallest_constant_for_triangle_sides (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (triangle_condition : a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ N, (∀ a b c, (a + b > c ∧ b + c > a ∧ c + a > b) → (a^2 + b^2) / (a * b) < N) ∧ N = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_constant_for_triangle_sides_l170_17002


namespace NUMINAMATH_GPT_range_of_a_l170_17079

theorem range_of_a (a x : ℝ) (p : 0.5 ≤ x ∧ x ≤ 1) (q : (x - a) * (x - a - 1) > 0) :
  (0 ≤ a ∧ a ≤ 0.5) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l170_17079


namespace NUMINAMATH_GPT_rug_shorter_side_l170_17074

theorem rug_shorter_side (x : ℝ) :
  (64 - x * 7) / 64 = 0.78125 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_rug_shorter_side_l170_17074


namespace NUMINAMATH_GPT_two_digit_factors_count_l170_17068

-- Definition of the expression 10^8 - 1
def expr : ℕ := 10^8 - 1

-- Factorization of 10^8 - 1
def factored_expr : List ℕ := [73, 137, 101, 11, 3^2]

-- Define the condition for being a two-digit factor
def is_two_digit (n : ℕ) : Bool := n > 9 ∧ n < 100

-- Count the number of positive two-digit factors in the factorization of 10^8 - 1
def num_two_digit_factors : ℕ := List.length (factored_expr.filter is_two_digit)

-- The theorem stating our proof problem
theorem two_digit_factors_count : num_two_digit_factors = 2 := by
  sorry

end NUMINAMATH_GPT_two_digit_factors_count_l170_17068


namespace NUMINAMATH_GPT_intersection_A_B_l170_17016

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 3^x}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2^(-x)}

theorem intersection_A_B :
  A ∩ B = {p | p = (0, 1)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l170_17016


namespace NUMINAMATH_GPT_time_to_finish_by_p_l170_17000

theorem time_to_finish_by_p (P_rate Q_rate : ℝ) (worked_together_hours remaining_job_rate : ℝ) :
    P_rate = 1/3 ∧ Q_rate = 1/9 ∧ worked_together_hours = 2 ∧ remaining_job_rate = 1 - (worked_together_hours * (P_rate + Q_rate)) → 
    (remaining_job_rate / P_rate) * 60 = 20 := 
by
  sorry

end NUMINAMATH_GPT_time_to_finish_by_p_l170_17000


namespace NUMINAMATH_GPT_exam_fail_percentage_l170_17033

theorem exam_fail_percentage
  (total_candidates : ℕ := 2000)
  (girls : ℕ := 900)
  (pass_percent : ℝ := 0.32) :
  ((total_candidates - ((pass_percent * (total_candidates - girls)) + (pass_percent * girls))) / total_candidates) * 100 = 68 :=
by
  sorry

end NUMINAMATH_GPT_exam_fail_percentage_l170_17033


namespace NUMINAMATH_GPT_kyle_caught_fish_l170_17043

def total_fish := 36
def fish_carla := 8
def fish_total := total_fish - fish_carla

-- kelle and tasha same number of fish means they equally divide the total fish left after deducting carla's
def fish_each_kt := fish_total / 2

theorem kyle_caught_fish :
  fish_each_kt = 14 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_kyle_caught_fish_l170_17043


namespace NUMINAMATH_GPT_simplify_fraction_l170_17075

theorem simplify_fraction :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1)) + (3 / (Real.sqrt 5 + 2)))) =
  (1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5)) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l170_17075


namespace NUMINAMATH_GPT_inequality_proof_l170_17063

/-- Given a and b are positive and satisfy the inequality ab > 2007a + 2008b,
    prove that a + b > (sqrt 2007 + sqrt 2008)^2 -/
theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l170_17063


namespace NUMINAMATH_GPT_pebbles_ratio_l170_17077

variable (S : ℕ)

theorem pebbles_ratio :
  let initial_pebbles := 18
  let skipped_pebbles := 9
  let additional_pebbles := 30
  let final_pebbles := 39
  initial_pebbles - skipped_pebbles + additional_pebbles = final_pebbles →
  (skipped_pebbles : ℚ) / initial_pebbles = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pebbles_ratio_l170_17077


namespace NUMINAMATH_GPT_nicky_run_time_l170_17011

-- Define the constants according to the conditions in the problem
def head_start : ℕ := 100 -- Nicky's head start (meters)
def cr_speed : ℕ := 8 -- Cristina's speed (meters per second)
def ni_speed : ℕ := 4 -- Nicky's speed (meters per second)

-- Define the event of Cristina catching up to Nicky
def meets_at_time (t : ℕ) : Prop :=
  cr_speed * t = head_start + ni_speed * t

-- The proof statement
theorem nicky_run_time : ∃ t : ℕ, meets_at_time t ∧ t = 25 :=
by
  sorry

end NUMINAMATH_GPT_nicky_run_time_l170_17011


namespace NUMINAMATH_GPT_ordered_quadruple_solution_exists_l170_17053

theorem ordered_quadruple_solution_exists (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  a^2 * b = c ∧ b * c^2 = a ∧ c * a^2 = b ∧ a + b + c = d → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3) :=
by
  sorry

end NUMINAMATH_GPT_ordered_quadruple_solution_exists_l170_17053


namespace NUMINAMATH_GPT_find_sum_of_common_ratios_l170_17014

-- Definition of the problem conditions
def is_geometric_sequence (a b c : ℕ) (k : ℕ) (r : ℕ) : Prop :=
  b = k * r ∧ c = k * r * r

-- Main theorem statement
theorem find_sum_of_common_ratios (k p r a_2 a_3 b_2 b_3 : ℕ) 
  (hk : k ≠ 0)
  (hp_neq_r : p ≠ r)
  (hp_seq : is_geometric_sequence k a_2 a_3 k p)
  (hr_seq : is_geometric_sequence k b_2 b_3 k r)
  (h_eq : a_3 - b_3 = 3 * (a_2 - b_2)) :
  p + r = 3 :=
sorry

end NUMINAMATH_GPT_find_sum_of_common_ratios_l170_17014


namespace NUMINAMATH_GPT_find_a_plus_b_l170_17045

theorem find_a_plus_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_plus_b_l170_17045


namespace NUMINAMATH_GPT_complex_division_l170_17087

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 + i) = 1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l170_17087


namespace NUMINAMATH_GPT_nadine_total_cleaning_time_l170_17018

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end NUMINAMATH_GPT_nadine_total_cleaning_time_l170_17018


namespace NUMINAMATH_GPT_remaining_amount_to_pay_l170_17026

-- Define the constants and conditions
def total_cost : ℝ := 1300
def first_deposit : ℝ := 0.10 * total_cost
def second_deposit : ℝ := 2 * first_deposit
def promotional_discount : ℝ := 0.05 * total_cost
def interest_rate : ℝ := 0.02

-- Define the function to calculate the final payment
def final_payment (total_cost first_deposit second_deposit promotional_discount interest_rate : ℝ) : ℝ :=
  let total_paid := first_deposit + second_deposit
  let remaining_balance := total_cost - total_paid
  let remaining_after_discount := remaining_balance - promotional_discount
  remaining_after_discount * (1 + interest_rate)

-- Define the theorem to be proven
theorem remaining_amount_to_pay :
  final_payment total_cost first_deposit second_deposit promotional_discount interest_rate = 861.90 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_remaining_amount_to_pay_l170_17026


namespace NUMINAMATH_GPT_quadratic_eqns_mod_7_l170_17060

/-- Proving the solutions for quadratic equations in arithmetic modulo 7. -/
theorem quadratic_eqns_mod_7 :
  (¬ ∃ x : ℤ, (5 * x^2 + 3 * x + 1) % 7 = 0) ∧
  (∃! x : ℤ, (x^2 + 3 * x + 4) % 7 = 0 ∧ x % 7 = 2) ∧
  (∃ x1 x2 : ℤ, (x1 ^ 2 - 2 * x1 - 3) % 7 = 0 ∧ (x2 ^ 2 - 2 * x2 - 3) % 7 = 0 ∧ 
              x1 % 7 = 3 ∧ x2 % 7 = 6) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eqns_mod_7_l170_17060


namespace NUMINAMATH_GPT_rectangle_area_x_l170_17082

theorem rectangle_area_x (x : ℕ) (h1 : x > 0) (h2 : 5 * x = 45) : x = 9 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_rectangle_area_x_l170_17082


namespace NUMINAMATH_GPT_line_through_two_points_l170_17072

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ x y : ℝ, (x, y) = (-2, 4) ∨ (x, y) = (-1, 3) → y = m * x + b) ∧ b = 2 ∧ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_line_through_two_points_l170_17072


namespace NUMINAMATH_GPT_even_integers_in_form_3k_plus_4_l170_17031

theorem even_integers_in_form_3k_plus_4 (n : ℕ) :
  (20 ≤ n ∧ n ≤ 180 ∧ ∃ k : ℕ, n = 3 * k + 4) → 
  (∃ s : ℕ, s = 27) :=
by
  sorry

end NUMINAMATH_GPT_even_integers_in_form_3k_plus_4_l170_17031


namespace NUMINAMATH_GPT_sum_is_correct_l170_17048

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end NUMINAMATH_GPT_sum_is_correct_l170_17048


namespace NUMINAMATH_GPT_second_train_speed_l170_17093

theorem second_train_speed (v : ℝ) :
  (∃ t : ℝ, 20 * t = v * t + 75 ∧ 20 * t + v * t = 675) → v = 16 :=
by
  sorry

end NUMINAMATH_GPT_second_train_speed_l170_17093


namespace NUMINAMATH_GPT_parallel_vectors_cosine_identity_l170_17034

-- Defining the problem in Lean 4

theorem parallel_vectors_cosine_identity :
  ∀ α : ℝ, (∃ k : ℝ, (1 / 3, Real.tan α) = (k * Real.cos α, k)) →
  Real.cos (Real.pi / 2 + α) = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_cosine_identity_l170_17034


namespace NUMINAMATH_GPT_distance_between_centers_of_tangent_circles_l170_17057

theorem distance_between_centers_of_tangent_circles
  (R r d : ℝ) (h1 : R = 8) (h2 : r = 3) (h3 : d = R + r) : d = 11 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_distance_between_centers_of_tangent_circles_l170_17057


namespace NUMINAMATH_GPT_find_ab_l170_17062

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 2 →
  (3 * x - 2 < a + 1 ∧ 6 - 2 * x < b + 2)) →
  a = 3 ∧ b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l170_17062


namespace NUMINAMATH_GPT_b_value_rational_polynomial_l170_17098

theorem b_value_rational_polynomial (a b : ℚ) :
  (Polynomial.aeval (2 + Real.sqrt 3) (Polynomial.C (-15) + Polynomial.C b * X + Polynomial.C a * X^2 + X^3 : Polynomial ℚ) = 0) →
  b = -44 :=
by
  sorry

end NUMINAMATH_GPT_b_value_rational_polynomial_l170_17098


namespace NUMINAMATH_GPT_solution_correct_l170_17003

noncomputable def satisfies_conditions (f : ℤ → ℝ) : Prop :=
  (f 1 = 5 / 2) ∧ (f 0 ≠ 0) ∧ (∀ m n : ℤ, f m * f n = f (m + n) + f (m - n))

theorem solution_correct (f : ℤ → ℝ) :
  satisfies_conditions f → ∀ n : ℤ, f n = 2^n + (1/2)^n :=
by sorry

end NUMINAMATH_GPT_solution_correct_l170_17003


namespace NUMINAMATH_GPT_age_problem_l170_17058

variable (A B : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 5) : B = 35 := by
  sorry

end NUMINAMATH_GPT_age_problem_l170_17058


namespace NUMINAMATH_GPT_sum_of_first_nine_terms_l170_17029

theorem sum_of_first_nine_terms (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 = 3 * a 3 - 6) : 
  (9 * (a 0 + a 8)) / 2 = 27 := 
sorry

end NUMINAMATH_GPT_sum_of_first_nine_terms_l170_17029


namespace NUMINAMATH_GPT_bags_production_l170_17013

def machines_bags_per_minute (n : ℕ) : ℕ :=
  if n = 15 then 45 else 0 -- this definition is constrained by given condition

def bags_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  machines * (machines_bags_per_minute 15 / 15) * minutes

theorem bags_production (machines minutes : ℕ) (h : machines = 150 ∧ minutes = 8):
  bags_produced machines minutes = 3600 :=
by
  cases h with
  | intro h_machines h_minutes =>
    sorry

end NUMINAMATH_GPT_bags_production_l170_17013


namespace NUMINAMATH_GPT_women_exceed_men_l170_17006

variable (M W : ℕ)

theorem women_exceed_men (h1 : M + W = 24) (h2 : (M : ℚ) / (W : ℚ) = 0.6) : W - M = 6 :=
sorry

end NUMINAMATH_GPT_women_exceed_men_l170_17006


namespace NUMINAMATH_GPT_smallest_third_term_GP_l170_17046

theorem smallest_third_term_GP : 
  ∃ d : ℝ, 
    (11 + d) ^ 2 = 9 * (29 + 2 * d) ∧
    min (29 + 2 * 10) (29 + 2 * -14) = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_third_term_GP_l170_17046


namespace NUMINAMATH_GPT_sign_of_x_minus_y_l170_17004

theorem sign_of_x_minus_y (x y a : ℝ) (h1 : x + y > 0) (h2 : a < 0) (h3 : a * y > 0) : x - y > 0 := 
by 
  sorry

end NUMINAMATH_GPT_sign_of_x_minus_y_l170_17004


namespace NUMINAMATH_GPT_find_value_of_expression_l170_17070

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem find_value_of_expression (a b : ℝ) (h : quadratic_function a b (-1) = 0) :
  2 * a - 2 * b = -4 :=
sorry

end NUMINAMATH_GPT_find_value_of_expression_l170_17070


namespace NUMINAMATH_GPT_find_number_l170_17030

theorem find_number :
  ∃ x : Int, x - (28 - (37 - (15 - 20))) = 59 ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l170_17030


namespace NUMINAMATH_GPT_apples_number_l170_17001

def num_apples (A O B : ℕ) : Prop :=
  A = O + 27 ∧ O = B + 11 ∧ A + O + B = 301 → A = 122

theorem apples_number (A O B : ℕ) : num_apples A O B := by
  sorry

end NUMINAMATH_GPT_apples_number_l170_17001


namespace NUMINAMATH_GPT_minimum_value_of_f_l170_17020

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 - (Real.log x) / x

theorem minimum_value_of_f : (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l170_17020


namespace NUMINAMATH_GPT_parabola_equation_max_slope_OQ_l170_17089

-- Definition of the problem for part (1)
theorem parabola_equation (p : ℝ) (hp : p = 2) : (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) :=
by {
  sorry
}

-- Definition of the problem for part (2)
theorem max_slope_OQ (m n : ℝ) (hp : y^2 = 4 * x)
  (h_relate : ∀ P Q F : (ℝ × ℝ), P.1 * Q.1 + P.2 * Q.2 = 9 * (Q.1 - F.1) * (Q.2 - F.2))
  : (∀ Q : (ℝ × ℝ), max (Q.2 / Q.1) = 1/3) :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_equation_max_slope_OQ_l170_17089


namespace NUMINAMATH_GPT_calculate_square_add_subtract_l170_17028

theorem calculate_square_add_subtract (a b : ℤ) :
  (41 : ℤ)^2 = (40 : ℤ)^2 + 81 ∧ (39 : ℤ)^2 = (40 : ℤ)^2 - 79 :=
by
  sorry

end NUMINAMATH_GPT_calculate_square_add_subtract_l170_17028


namespace NUMINAMATH_GPT_ellipse_problem_l170_17051

theorem ellipse_problem :
  (∃ (k : ℝ) (a θ : ℝ), 
    (∀ x y : ℝ, y = k * (x + 3) → (x^2 / 25 + y^2 / 16 = 1)) ∧
    (a > -3) ∧
    (∃ x y : ℝ, (x = - (25 / 3) ∧ y = k * (x + 3)) ∧ 
                 (x = D_fst ∧ y = D_snd) ∧ -- Point D(a, θ)
                 (x = M_fst ∧ y = M_snd) ∧ -- Point M
                 (x = N_fst ∧ y = N_snd)) ∧ -- Point N
    (∃ x y : ℝ, (x = -3 ∧ y = 0))) → 
    a = 5 :=
sorry

end NUMINAMATH_GPT_ellipse_problem_l170_17051


namespace NUMINAMATH_GPT_find_q_l170_17036

noncomputable def common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 4 = 27 ∧ a 7 = -729 ∧ ∀ n m, a n = a m * q ^ (n - m)

theorem find_q {a : ℕ → ℝ} {q : ℝ} (h : common_ratio_of_geometric_sequence a q) :
  q = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_q_l170_17036


namespace NUMINAMATH_GPT_felicity_used_5_gallons_less_l170_17085

def adhesion_gas_problem : Prop :=
  ∃ A x : ℕ, (A + 23 = 30) ∧ (4 * A - x = 23) ∧ (x = 5)
  
theorem felicity_used_5_gallons_less :
  adhesion_gas_problem :=
by
  sorry

end NUMINAMATH_GPT_felicity_used_5_gallons_less_l170_17085


namespace NUMINAMATH_GPT_gcf_252_96_l170_17019

theorem gcf_252_96 : Int.gcd 252 96 = 12 := by
  sorry

end NUMINAMATH_GPT_gcf_252_96_l170_17019


namespace NUMINAMATH_GPT_quadratic_minimum_value_interval_l170_17040

theorem quadratic_minimum_value_interval (k : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x < 2 → (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (2*k^2 + 2*k - 1)) → (0 ≤ k ∧ k < 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_minimum_value_interval_l170_17040


namespace NUMINAMATH_GPT_quadratic_real_roots_range_l170_17078

theorem quadratic_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + 2 * x + 1 = 0 → 
    (∃ x1 x2 : ℝ, x = x1 ∧ x = x2 ∧ x1 = x2 → true)) → 
    m ≤ 2 ∧ m ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_range_l170_17078


namespace NUMINAMATH_GPT_mutually_exclusive_A_B_head_l170_17027

variables (A_head B_head B_end : Prop)

def mut_exclusive (P Q : Prop) : Prop := ¬(P ∧ Q)

theorem mutually_exclusive_A_B_head (A_head B_head : Prop) :
  mut_exclusive A_head B_head :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_A_B_head_l170_17027


namespace NUMINAMATH_GPT_find_width_fabric_width_is_3_l170_17092

variable (Area Length : ℝ)
variable (Width : ℝ)

theorem find_width (h1 : Area = 24) (h2 : Length = 8) :
  Width = Area / Length :=
sorry

theorem fabric_width_is_3 (h1 : Area = 24) (h2 : Length = 8) :
  (Area / Length) = 3 :=
by
  have h : Area / Length = 3 := by sorry
  exact h

end NUMINAMATH_GPT_find_width_fabric_width_is_3_l170_17092


namespace NUMINAMATH_GPT_flower_shop_february_roses_l170_17005

theorem flower_shop_february_roses (roses_oct : ℕ) (roses_nov : ℕ) (roses_dec : ℕ) (roses_jan : ℕ) (d : ℕ) :
  roses_oct = 108 →
  roses_nov = 120 →
  roses_dec = 132 →
  roses_jan = 144 →
  roses_nov - roses_oct = d →
  roses_dec - roses_nov = d →
  roses_jan - roses_dec = d →
  (roses_jan + d = 156) :=
by
  intros h_oct h_nov h_dec h_jan h_diff1 h_diff2 h_diff3
  rw [h_jan, h_diff1] at *
  sorry

end NUMINAMATH_GPT_flower_shop_february_roses_l170_17005


namespace NUMINAMATH_GPT_pure_alcohol_addition_l170_17052

theorem pure_alcohol_addition (x : ℝ) (h1 : 3 / 10 * 10 = 3)
    (h2 : 60 / 100 * (10 + x) = (3 + x) ) : x = 7.5 :=
sorry

end NUMINAMATH_GPT_pure_alcohol_addition_l170_17052


namespace NUMINAMATH_GPT_min_value_frac_sum_l170_17044

theorem min_value_frac_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 2) : 
  (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 3 * b = 2 ∧ (2 / a + 4 / b) = 14) :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_sum_l170_17044


namespace NUMINAMATH_GPT_average_age_when_youngest_born_l170_17086

theorem average_age_when_youngest_born (n : ℕ) (current_average_age youngest age_difference total_ages : ℝ)
  (hc1 : n = 7)
  (hc2 : current_average_age = 30)
  (hc3 : youngest = 6)
  (hc4 : age_difference = youngest * 6)
  (hc5 : total_ages = n * current_average_age - age_difference) :
  total_ages / n = 24.857
:= sorry

end NUMINAMATH_GPT_average_age_when_youngest_born_l170_17086


namespace NUMINAMATH_GPT_functional_equation_f2023_l170_17096

theorem functional_equation_f2023 (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_one : f 1 = 1) :
  f 2023 = 2023 := sorry

end NUMINAMATH_GPT_functional_equation_f2023_l170_17096


namespace NUMINAMATH_GPT_Johnson_farm_budget_l170_17083

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end NUMINAMATH_GPT_Johnson_farm_budget_l170_17083


namespace NUMINAMATH_GPT_pages_read_on_saturday_l170_17037

namespace BookReading

def total_pages : ℕ := 93
def pages_read_sunday : ℕ := 20
def pages_remaining : ℕ := 43

theorem pages_read_on_saturday :
  total_pages - (pages_read_sunday + pages_remaining) = 30 :=
by
  sorry

end BookReading

end NUMINAMATH_GPT_pages_read_on_saturday_l170_17037


namespace NUMINAMATH_GPT_find_a2018_l170_17076

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 0.5 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

-- The statement to prove
theorem find_a2018 (a : ℕ → ℝ) (h : seq a) : a 2018 = -1 := by
  sorry

end NUMINAMATH_GPT_find_a2018_l170_17076


namespace NUMINAMATH_GPT_complement_U_A_l170_17090

open Set

-- Definitions of the universal set U and the set A
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

-- Proof statement: the complement of A with respect to U is {3}
theorem complement_U_A : U \ A = {3} :=
by
  sorry

end NUMINAMATH_GPT_complement_U_A_l170_17090


namespace NUMINAMATH_GPT_chestnut_picking_l170_17015

theorem chestnut_picking 
  (P : ℕ)
  (h1 : 12 + P + (P + 2) = 26) :
  12 / P = 2 :=
sorry

end NUMINAMATH_GPT_chestnut_picking_l170_17015


namespace NUMINAMATH_GPT_total_money_given_by_father_is_100_l170_17050

-- Define the costs and quantities given in the problem statement.
def cost_per_sharpener := 5
def cost_per_notebook := 5
def cost_per_eraser := 4
def money_spent_on_highlighters := 30

def heaven_sharpeners := 2
def heaven_notebooks := 4
def brother_erasers := 10

-- Calculate the total amount of money given by their father.
def total_money_given : ℕ :=
  heaven_sharpeners * cost_per_sharpener +
  heaven_notebooks * cost_per_notebook +
  brother_erasers * cost_per_eraser +
  money_spent_on_highlighters

-- Lean statement to prove
theorem total_money_given_by_father_is_100 :
  total_money_given = 100 := by
  sorry

end NUMINAMATH_GPT_total_money_given_by_father_is_100_l170_17050


namespace NUMINAMATH_GPT_train_length_l170_17024

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (length_m : ℝ) 
  (h1 : speed_km_hr = 90) 
  (h2 : time_sec = 11) 
  (h3 : length_m = 275) :
  length_m = (speed_km_hr * 1000 / 3600) * time_sec :=
sorry

end NUMINAMATH_GPT_train_length_l170_17024


namespace NUMINAMATH_GPT_farmer_apples_count_l170_17041

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end NUMINAMATH_GPT_farmer_apples_count_l170_17041


namespace NUMINAMATH_GPT_divisibility_polynomial_l170_17042

variables {a m x n : ℕ}

theorem divisibility_polynomial (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) :=
by
  sorry

end NUMINAMATH_GPT_divisibility_polynomial_l170_17042


namespace NUMINAMATH_GPT_intersection_in_fourth_quadrant_l170_17049

theorem intersection_in_fourth_quadrant :
  (∃ x y : ℝ, y = -x ∧ y = 2 * x - 1 ∧ x = 1 ∧ y = -1) ∧ (1 > 0 ∧ -1 < 0) :=
by
  sorry

end NUMINAMATH_GPT_intersection_in_fourth_quadrant_l170_17049


namespace NUMINAMATH_GPT_largest_frog_weight_l170_17017

theorem largest_frog_weight (S L : ℕ) (h1 : L = 10 * S) (h2 : L = S + 108): L = 120 := by
  sorry

end NUMINAMATH_GPT_largest_frog_weight_l170_17017


namespace NUMINAMATH_GPT_least_possible_value_l170_17067

theorem least_possible_value (x y : ℝ) : (3 * x * y - 1)^2 + (x - y)^2 ≥ 1 := sorry

end NUMINAMATH_GPT_least_possible_value_l170_17067


namespace NUMINAMATH_GPT_total_gallons_in_tanks_l170_17023

def tank1_capacity : ℕ := 7000
def tank2_capacity : ℕ := 5000
def tank3_capacity : ℕ := 3000

def tank1_filled : ℚ := 3/4
def tank2_filled : ℚ := 4/5
def tank3_filled : ℚ := 1/2

theorem total_gallons_in_tanks :
  (tank1_capacity * tank1_filled + tank2_capacity * tank2_filled + tank3_capacity * tank3_filled : ℚ) = 10750 := 
by 
  sorry

end NUMINAMATH_GPT_total_gallons_in_tanks_l170_17023


namespace NUMINAMATH_GPT_students_in_front_of_Yuna_l170_17069

-- Defining the total number of students
def total_students : ℕ := 25

-- Defining the number of students behind Yuna
def students_behind_Yuna : ℕ := 9

-- Defining Yuna's position from the end of the line
def Yuna_position_from_end : ℕ := students_behind_Yuna + 1

-- Statement to prove the number of students in front of Yuna
theorem students_in_front_of_Yuna : (total_students - Yuna_position_from_end) = 15 := by
  sorry

end NUMINAMATH_GPT_students_in_front_of_Yuna_l170_17069


namespace NUMINAMATH_GPT_irreducible_fractions_properties_l170_17080

theorem irreducible_fractions_properties : 
  let f1 := 11 / 2
  let f2 := 11 / 6
  let f3 := 11 / 3
  let reciprocal_sum := (2 / 11) + (6 / 11) + (3 / 11)
  (f1 + f2 + f3 = 11) ∧ (reciprocal_sum = 1) :=
by
  sorry

end NUMINAMATH_GPT_irreducible_fractions_properties_l170_17080


namespace NUMINAMATH_GPT_parallel_vectors_l170_17010

variable (a b : ℝ × ℝ)
variable (m : ℝ)

theorem parallel_vectors (h₁ : a = (-6, 2)) (h₂ : b = (m, -3)) (h₃ : a.1 * b.2 = a.2 * b.1) : m = 9 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l170_17010


namespace NUMINAMATH_GPT_notebooks_bought_l170_17021

def dan_total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def pens_cost : ℕ := 1
def pencils_cost : ℕ := 1
def notebook_cost : ℕ := 3

theorem notebooks_bought :
  ∃ x : ℕ, dan_total_spent - (backpack_cost + pens_cost + pencils_cost) = x * notebook_cost ∧ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_notebooks_bought_l170_17021


namespace NUMINAMATH_GPT_find_angle_x_l170_17071

def angle_ABC := 124
def angle_BAD := 30
def angle_BDA := 28
def angle_ABD := 180 - angle_ABC
def angle_x := 180 - (angle_BAD + angle_ABD)

theorem find_angle_x : angle_x = 94 :=
by
  repeat { sorry }

end NUMINAMATH_GPT_find_angle_x_l170_17071


namespace NUMINAMATH_GPT_intersection_of_A_and_B_solve_inequality_l170_17047

-- Definitions based on conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | x^2 - 16 < 0}
def B : Set ℝ := {x : ℝ | x^2 - 4 * x + 3 ≥ 0}

-- Proof problem 1: Find A ∩ B
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4)} :=
sorry

-- Proof problem 2: Solve the inequality with respect to x
theorem solve_inequality (a : ℝ) :
  if a = 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = ∅
  else if a > 1 then
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | 1 < x ∧ x < a}
  else
    {x : ℝ | x^2 - (a+1) * x + a < 0} = {x : ℝ | a < x ∧ x < 1} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_solve_inequality_l170_17047


namespace NUMINAMATH_GPT_find_a_plus_d_l170_17059

theorem find_a_plus_d (a b c d : ℕ)
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3) : 
  a + d = 2 :=
by sorry

end NUMINAMATH_GPT_find_a_plus_d_l170_17059


namespace NUMINAMATH_GPT_children_tickets_l170_17022

theorem children_tickets (A C : ℝ) (h1 : A + C = 200) (h2 : 3 * A + 1.5 * C = 510) : C = 60 := by
  sorry

end NUMINAMATH_GPT_children_tickets_l170_17022


namespace NUMINAMATH_GPT_Alyssa_spending_correct_l170_17035

def cost_per_game : ℕ := 20

def last_year_in_person_games : ℕ := 13
def this_year_in_person_games : ℕ := 11
def this_year_streaming_subscription : ℕ := 120
def next_year_in_person_games : ℕ := 15
def next_year_streaming_subscription : ℕ := 150
def friends_count : ℕ := 2
def friends_join_games : ℕ := 5

def Alyssa_total_spending : ℕ :=
  (last_year_in_person_games * cost_per_game) +
  (this_year_in_person_games * cost_per_game) + this_year_streaming_subscription +
  (next_year_in_person_games * cost_per_game) + next_year_streaming_subscription -
  (friends_join_games * friends_count * cost_per_game)

theorem Alyssa_spending_correct : Alyssa_total_spending = 850 := by
  sorry

end NUMINAMATH_GPT_Alyssa_spending_correct_l170_17035


namespace NUMINAMATH_GPT_merck_hourly_rate_l170_17065

-- Define the relevant data from the problem
def hours_donaldsons : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def total_earnings : ℕ := 273

-- Define the total hours based on the conditions
def total_hours : ℕ := hours_donaldsons + hours_merck + hours_hille

-- Define what we want to prove:
def hourly_rate := total_earnings / total_hours

theorem merck_hourly_rate : hourly_rate = 273 / (7 + 6 + 3) := by
  sorry

end NUMINAMATH_GPT_merck_hourly_rate_l170_17065


namespace NUMINAMATH_GPT_regular_polygon_sides_l170_17039

theorem regular_polygon_sides (D : ℕ) (h : D = 30) :
  ∃ n : ℕ, D = n * (n - 3) / 2 ∧ n = 9 :=
by
  use 9
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l170_17039


namespace NUMINAMATH_GPT_pencils_across_diameter_l170_17032

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end NUMINAMATH_GPT_pencils_across_diameter_l170_17032


namespace NUMINAMATH_GPT_problem_inequality_l170_17061

theorem problem_inequality {n : ℕ} {a : ℕ → ℕ} (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j → (a j - a i) ∣ a i) 
  (h_sorted : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h_pos : ∀ i : ℕ, 1 ≤ i → i ≤ n → 0 < a i) 
  (i j : ℕ) (hi : 1 ≤ i) (hij : i < j) (hj : j ≤ n) : i * a j ≤ j * a i := 
sorry

end NUMINAMATH_GPT_problem_inequality_l170_17061


namespace NUMINAMATH_GPT_coats_count_l170_17081

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_coats_count_l170_17081


namespace NUMINAMATH_GPT_balloon_count_l170_17012

theorem balloon_count (gold_balloon silver_balloon black_balloon blue_balloon green_balloon total_balloon : ℕ) (h1 : gold_balloon = 141) 
                      (h2 : silver_balloon = (gold_balloon / 3) * 5) 
                      (h3 : black_balloon = silver_balloon / 2) 
                      (h4 : blue_balloon = black_balloon / 2) 
                      (h5 : green_balloon = (blue_balloon / 4) * 3) 
                      (h6 : total_balloon = gold_balloon + silver_balloon + black_balloon + blue_balloon + green_balloon): 
                      total_balloon = 593 :=
by 
  sorry

end NUMINAMATH_GPT_balloon_count_l170_17012


namespace NUMINAMATH_GPT_total_unique_items_l170_17066

-- Define the conditions
def shared_albums : ℕ := 12
def total_andrew_albums : ℕ := 23
def exclusive_andrew_memorabilia : ℕ := 5
def exclusive_john_albums : ℕ := 8

-- Define the number of unique items in Andrew's and John's collection 
def unique_andrew_albums : ℕ := total_andrew_albums - shared_albums
def unique_total_items : ℕ := unique_andrew_albums + exclusive_john_albums + exclusive_andrew_memorabilia

-- The proof goal
theorem total_unique_items : unique_total_items = 24 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_unique_items_l170_17066
