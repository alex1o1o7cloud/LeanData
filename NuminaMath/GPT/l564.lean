import Mathlib

namespace NUMINAMATH_GPT_theodoreEarningsCorrect_l564_56453

noncomputable def theodoreEarnings : ℝ := 
  let s := 10
  let ps := 20
  let w := 20
  let pw := 5
  let b := 15
  let pb := 15
  let m := 150
  let l := 200
  let t := 0.10
  let totalEarnings := (s * ps) + (w * pw) + (b * pb)
  let expenses := m + l
  let earningsBeforeTaxes := totalEarnings - expenses
  let taxes := t * earningsBeforeTaxes
  earningsBeforeTaxes - taxes

theorem theodoreEarningsCorrect :
  theodoreEarnings = 157.50 :=
by sorry

end NUMINAMATH_GPT_theodoreEarningsCorrect_l564_56453


namespace NUMINAMATH_GPT_contrapositive_roots_l564_56424

theorem contrapositive_roots {a b c : ℝ} (h : a ≠ 0) (hac : a * c ≤ 0) :
  ¬ (∀ x : ℝ, (a * x^2 - b * x + c = 0) → x > 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_roots_l564_56424


namespace NUMINAMATH_GPT_find_a_values_l564_56486

theorem find_a_values (a : ℝ) : 
  (∃ x : ℝ, (a * x^2 + (a - 3) * x + 1 = 0)) ∧ 
  (∀ x1 x2 : ℝ, (a * x1^2 + (a - 3) * x1 + 1 = 0 ∧ a * x2^2 + (a - 3) * x2 + 1 = 0 → x1 = x2)) 
  ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_values_l564_56486


namespace NUMINAMATH_GPT_fraction_numerator_l564_56438

theorem fraction_numerator (x : ℕ) (h1 : 4 * x - 4 > 0) (h2 : (x : ℚ) / (4 * x - 4) = 3 / 8) : x = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_numerator_l564_56438


namespace NUMINAMATH_GPT_horizontal_force_magnitude_l564_56445

-- We state our assumptions and goal
theorem horizontal_force_magnitude (W : ℝ) : 
  (∀ μ : ℝ, μ = (Real.sin (Real.pi / 6)) / (Real.cos (Real.pi / 6)) ∧ 
    (∀ P : ℝ, 
      (P * (Real.sin (Real.pi / 3))) = 
      ((μ * (W * (Real.cos (Real.pi / 6)) + P * (Real.cos (Real.pi / 3)))) + W * (Real.sin (Real.pi / 6))) →
      P = W * Real.sqrt 3)) :=
sorry

end NUMINAMATH_GPT_horizontal_force_magnitude_l564_56445


namespace NUMINAMATH_GPT_system_solve_l564_56420

theorem system_solve (x y : ℚ) (h1 : 2 * x + y = 3) (h2 : 3 * x - 2 * y = 12) : x + y = 3 / 7 :=
by
  -- The proof will go here, but we skip it for now.
  sorry

end NUMINAMATH_GPT_system_solve_l564_56420


namespace NUMINAMATH_GPT_g_g_g_g_of_2_eq_242_l564_56447

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 3 * x + 2

theorem g_g_g_g_of_2_eq_242 : g (g (g (g 2))) = 242 :=
by
  sorry

end NUMINAMATH_GPT_g_g_g_g_of_2_eq_242_l564_56447


namespace NUMINAMATH_GPT_P_iff_nonQ_l564_56478

-- Given conditions
def P (x y : ℝ) : Prop := x^2 + y^2 = 0
def Q (x y : ℝ) : Prop := x ≠ 0 ∨ y ≠ 0
def nonQ (x y : ℝ) : Prop := x = 0 ∧ y = 0

-- Main statement
theorem P_iff_nonQ (x y : ℝ) : P x y ↔ nonQ x y :=
sorry

end NUMINAMATH_GPT_P_iff_nonQ_l564_56478


namespace NUMINAMATH_GPT_inequality_part_1_inequality_part_2_l564_56498

noncomputable def f (x : ℝ) := |x - 2| + 2
noncomputable def g (x : ℝ) (m : ℝ) := m * |x|

theorem inequality_part_1 (x : ℝ) : f x > 5 ↔ x < -1 ∨ x > 5 := by
  sorry

theorem inequality_part_2 (m : ℝ) : (∀ x, f x ≥ g x m) ↔ m ≤ 1 := by
  sorry

end NUMINAMATH_GPT_inequality_part_1_inequality_part_2_l564_56498


namespace NUMINAMATH_GPT_coefficient_of_q_is_correct_l564_56458

theorem coefficient_of_q_is_correct (q' : ℕ → ℕ) : 
  (∀ q : ℕ, q' q = 3 * q - 3) ∧  q' (q' 7) = 306 → ∃ a : ℕ, (∀ q : ℕ, q' q = a * q - 3) ∧ a = 17 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_q_is_correct_l564_56458


namespace NUMINAMATH_GPT_smallest_k_for_abk_l564_56456

theorem smallest_k_for_abk : ∃ (k : ℝ), (∀ (a b : ℝ), a + b = k ∧ ab = k → k = 4) :=
sorry

end NUMINAMATH_GPT_smallest_k_for_abk_l564_56456


namespace NUMINAMATH_GPT_total_number_of_rats_l564_56415

theorem total_number_of_rats (Kenia Hunter Elodie Teagan : ℕ) 
  (h1 : Elodie = 30)
  (h2 : Elodie = Hunter + 10)
  (h3 : Kenia = 3 * (Hunter + Elodie))
  (h4 : Teagan = 2 * Elodie)
  (h5 : Teagan = Kenia - 5) : 
  Kenia + Hunter + Elodie + Teagan = 260 :=
by 
  sorry

end NUMINAMATH_GPT_total_number_of_rats_l564_56415


namespace NUMINAMATH_GPT_cost_of_items_l564_56471

variable (p q r : ℝ)

theorem cost_of_items :
  8 * p + 2 * q + r = 4.60 → 
  2 * p + 5 * q + r = 3.90 → 
  p + q + 3 * r = 2.75 → 
  4 * p + 3 * q + 2 * r = 7.4135 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_of_items_l564_56471


namespace NUMINAMATH_GPT_find_pq_l564_56422

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_pq (p q : ℕ) 
(hp : is_prime p) 
(hq : is_prime q) 
(h : is_prime (q^2 - p^2)) : 
  p * q = 6 :=
by sorry

end NUMINAMATH_GPT_find_pq_l564_56422


namespace NUMINAMATH_GPT_num_diamonds_in_G6_l564_56400

noncomputable def triangular_number (k : ℕ) : ℕ :=
  (k * (k + 1)) / 2

noncomputable def total_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ k => triangular_number (k + 1)))

theorem num_diamonds_in_G6 :
  total_diamonds 6 = 141 := by
  -- This will be proven
  sorry

end NUMINAMATH_GPT_num_diamonds_in_G6_l564_56400


namespace NUMINAMATH_GPT_estimate_pi_l564_56461

theorem estimate_pi (m : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (h1 : m = 56) (h2 : n = 200) (h3 : a = 1/2) (h4 : b = 1/4) :
  (m / n) = (π / 4 - 1 / 2) ↔ π = 78 / 25 :=
by
  sorry

end NUMINAMATH_GPT_estimate_pi_l564_56461


namespace NUMINAMATH_GPT_age_difference_l564_56407

variable (S M : ℕ)

theorem age_difference (hS : S = 28) (hM : M + 2 = 2 * (S + 2)) : M - S = 30 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l564_56407


namespace NUMINAMATH_GPT_find_a_l564_56455

theorem find_a (a x : ℝ) (h1 : 3 * a - x = x / 2 + 3) (h2 : x = 2) : a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l564_56455


namespace NUMINAMATH_GPT_eval_ceil_floor_sum_l564_56441

def ceil_floor_sum : ℤ :=
  ⌈(7:ℚ) / (3:ℚ)⌉ + ⌊-((7:ℚ) / (3:ℚ))⌋

theorem eval_ceil_floor_sum : ceil_floor_sum = 0 :=
sorry

end NUMINAMATH_GPT_eval_ceil_floor_sum_l564_56441


namespace NUMINAMATH_GPT_circle_equation_l564_56485

theorem circle_equation 
  (h k : ℝ) 
  (H_center : k = 2 * h)
  (H_tangent : ∃ (r : ℝ), (h - 1)^2 + (k - 0)^2 = r^2 ∧ r = k) :
  (x - 1)^2 + (y - 2)^2 = 4 := 
sorry

end NUMINAMATH_GPT_circle_equation_l564_56485


namespace NUMINAMATH_GPT_side_length_of_S2_l564_56411

theorem side_length_of_S2 :
  ∀ (r s : ℕ), 
    (2 * r + s = 2000) → 
    (2 * r + 5 * s = 3030) → 
    s = 258 :=
by
  intros r s h1 h2
  sorry

end NUMINAMATH_GPT_side_length_of_S2_l564_56411


namespace NUMINAMATH_GPT_inequality_solution_l564_56476

theorem inequality_solution (x : ℝ) : (1 - x > 0) ∧ ((x + 2) / 3 - 1 ≤ x) ↔ (-1/2 ≤ x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l564_56476


namespace NUMINAMATH_GPT_min_pieces_for_net_l564_56454

theorem min_pieces_for_net (n : ℕ) : ∃ (m : ℕ), m = n * (n + 1) := by
  sorry

end NUMINAMATH_GPT_min_pieces_for_net_l564_56454


namespace NUMINAMATH_GPT_initial_marbles_l564_56473

theorem initial_marbles (total_marbles now found: ℕ) (h_found: found = 7) (h_now: now = 28) : 
  total_marbles = now - found → total_marbles = 21 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_initial_marbles_l564_56473


namespace NUMINAMATH_GPT_total_apples_picked_l564_56401

def benny_apples : Nat := 2
def dan_apples : Nat := 9

theorem total_apples_picked : benny_apples + dan_apples = 11 := 
by
  sorry

end NUMINAMATH_GPT_total_apples_picked_l564_56401


namespace NUMINAMATH_GPT_martha_total_points_l564_56406

-- Define the costs and points
def cost_beef := 11 * 3
def cost_fruits_vegetables := 4 * 8
def cost_spices := 6 * 3
def cost_other := 37

def total_spending := cost_beef + cost_fruits_vegetables + cost_spices + cost_other

def points_per_dollar := 50 / 10
def base_points := total_spending * points_per_dollar
def bonus_points := if total_spending > 100 then 250 else 0

def total_points := base_points + bonus_points

-- The theorem to prove the question == answer given the conditions
theorem martha_total_points :
  total_points = 850 :=
by
  sorry

end NUMINAMATH_GPT_martha_total_points_l564_56406


namespace NUMINAMATH_GPT_trigonometric_inequality_l564_56482

theorem trigonometric_inequality (a b A B : ℝ) (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos 2 * x - B * Real.sin 2 * x ≥ 0) : 
  a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := 
sorry

end NUMINAMATH_GPT_trigonometric_inequality_l564_56482


namespace NUMINAMATH_GPT_max_plus_min_value_of_f_l564_56428

noncomputable def f (x : ℝ) : ℝ := (2 * (x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_plus_min_value_of_f :
  let M := ⨆ x, f x
  let m := ⨅ x, f x
  M + m = 4 :=
by 
  sorry

end NUMINAMATH_GPT_max_plus_min_value_of_f_l564_56428


namespace NUMINAMATH_GPT_circle_center_coordinates_l564_56483

theorem circle_center_coordinates :
  ∀ (x y : ℝ), x^2 + y^2 - 10 * x + 6 * y + 25 = 0 → (5, -3) = ((-(-10) / 2), (-6 / 2)) :=
by
  intros x y h
  have H : (5, -3) = ((-(-10) / 2), (-6 / 2)) := sorry
  exact H

end NUMINAMATH_GPT_circle_center_coordinates_l564_56483


namespace NUMINAMATH_GPT_a2009_equals_7_l564_56421

def sequence_element (n k : ℕ) : ℚ :=
  if k = 0 then 0 else (n - k + 1) / k

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem a2009_equals_7 : 
  let n := 63
  let m := 2009
  let subset_cumulative_count := cumulative_count n
  (2 * m = n * (n + 1) - 14 ∧
   m = subset_cumulative_count - 7 ∧ 
   sequence_element n 8 = 7) →
  sequence_element n (subset_cumulative_count - m + 1) = 7 :=
by
  -- proof steps to be filled here
  sorry

end NUMINAMATH_GPT_a2009_equals_7_l564_56421


namespace NUMINAMATH_GPT_ronald_profit_fraction_l564_56437

theorem ronald_profit_fraction:
  let initial_units : ℕ := 200
  let total_investment : ℕ := 3000
  let selling_price_per_unit : ℕ := 20
  let total_selling_price := initial_units * selling_price_per_unit
  let total_profit := total_selling_price - total_investment
  (total_profit : ℚ) / total_investment = (1 : ℚ) / 3 :=
by
  -- here we will put the steps needed to prove the theorem.
  sorry

end NUMINAMATH_GPT_ronald_profit_fraction_l564_56437


namespace NUMINAMATH_GPT_calculate_expression_l564_56488

theorem calculate_expression :
  (-0.25) ^ 2014 * (-4) ^ 2015 = -4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l564_56488


namespace NUMINAMATH_GPT_interest_rate_l564_56434

/-- 
Given a principal amount that doubles itself in 10 years at simple interest,
prove that the rate of interest per annum is 10%.
-/
theorem interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) (h1 : SI = P) (h2 : T = 10) (h3 : SI = P * R * T / 100) : 
  R = 10 := by
  sorry

end NUMINAMATH_GPT_interest_rate_l564_56434


namespace NUMINAMATH_GPT_walter_equal_share_l564_56405

-- Conditions
def jefferson_bananas : ℕ := 56
def walter_fewer_fraction : ℚ := 1 / 4
def walter_fewer_bananas := walter_fewer_fraction * jefferson_bananas
def walter_bananas := jefferson_bananas - walter_fewer_bananas
def total_bananas := walter_bananas + jefferson_bananas

-- Proof problem: Prove that Walter gets 49 bananas when they share the total number of bananas equally.
theorem walter_equal_share : total_bananas / 2 = 49 := 
by sorry

end NUMINAMATH_GPT_walter_equal_share_l564_56405


namespace NUMINAMATH_GPT_radius_decrease_l564_56439

theorem radius_decrease (r r' : ℝ) (A A' : ℝ) (h_original_area : A = π * r^2)
  (h_area_decrease : A' = 0.25 * A) (h_new_area : A' = π * r'^2) : r' = 0.5 * r :=
by
  sorry

end NUMINAMATH_GPT_radius_decrease_l564_56439


namespace NUMINAMATH_GPT_smallest_fraction_divides_exactly_l564_56418

theorem smallest_fraction_divides_exactly (a b c p q r m n : ℕ)
    (h1: a = 6) (h2: b = 5) (h3: c = 10) (h4: p = 7) (h5: q = 14) (h6: r = 21)
    (h1_frac: 6/7 = a/p) (h2_frac: 5/14 = b/q) (h3_frac: 10/21 = c/r)
    (h_lcm: m = Nat.lcm p (Nat.lcm q r)) (h_gcd: n = Nat.gcd a (Nat.gcd b c)) :
  (n/m) = 1/42 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_fraction_divides_exactly_l564_56418


namespace NUMINAMATH_GPT_conic_section_is_hyperbola_l564_56452

noncomputable def is_hyperbola (x y : ℝ) : Prop :=
  (x - 4) ^ 2 = 9 * (y + 3) ^ 2 + 27

theorem conic_section_is_hyperbola : ∀ x y : ℝ, is_hyperbola x y → "H" = "H" := sorry

end NUMINAMATH_GPT_conic_section_is_hyperbola_l564_56452


namespace NUMINAMATH_GPT_opposite_of_negative_six_is_six_l564_56419

-- Define what it means for one number to be the opposite of another.
def is_opposite (a b : Int) : Prop :=
  a = -b

-- The statement to be proved: the opposite number of -6 is 6.
theorem opposite_of_negative_six_is_six : is_opposite (-6) 6 :=
  by sorry

end NUMINAMATH_GPT_opposite_of_negative_six_is_six_l564_56419


namespace NUMINAMATH_GPT_bruno_initial_books_l564_56487

theorem bruno_initial_books (X : ℝ)
  (h1 : X - 4.5 + 10.25 = 39.75) :
  X = 34 := by
  sorry

end NUMINAMATH_GPT_bruno_initial_books_l564_56487


namespace NUMINAMATH_GPT_smallest_six_digit_number_exists_l564_56449

def three_digit_number (n : ℕ) := n % 4 = 2 ∧ n % 5 = 2 ∧ n % 6 = 2 ∧ 100 ≤ n ∧ n < 1000

def valid_six_digit_number (m n : ℕ) := 
  (m * 1000 + n) % 4 = 0 ∧ (m * 1000 + n) % 5 = 0 ∧ (m * 1000 + n) % 6 = 0 ∧ 
  three_digit_number n ∧ 0 ≤ m ∧ m < 1000

theorem smallest_six_digit_number_exists : 
  ∃ m n, valid_six_digit_number m n ∧ (∀ m' n', valid_six_digit_number m' n' → m * 1000 + n ≤ m' * 1000 + n') :=
sorry

end NUMINAMATH_GPT_smallest_six_digit_number_exists_l564_56449


namespace NUMINAMATH_GPT_investment_ratio_l564_56409

-- Definitions of all the conditions
variables (A B C profit b_share: ℝ)

-- Conditions based on the provided problem
def condition1 (n : ℝ) : Prop := A = n * B
def condition2 : Prop := B = (2 / 3) * C
def condition3 : Prop := profit = 4400
def condition4 : Prop := b_share = 800

-- The theorem we want to prove
theorem investment_ratio (n : ℝ) :
  (condition1 A B n) ∧ (condition2 B C) ∧ (condition3 profit) ∧ (condition4 b_share) → A / B = 3 :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l564_56409


namespace NUMINAMATH_GPT_values_of_k_for_exactly_one_real_solution_l564_56417

variable {k : ℝ}

def quadratic_eq (k : ℝ) : Prop := 3 * k^2 + 42 * k - 573 = 0

theorem values_of_k_for_exactly_one_real_solution :
  quadratic_eq k ↔ k = 8 ∨ k = -22 := by
  sorry

end NUMINAMATH_GPT_values_of_k_for_exactly_one_real_solution_l564_56417


namespace NUMINAMATH_GPT_matrix_power_100_l564_56495

def matrix_100_pow : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![200, 1]]

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![2, 1]]

theorem matrix_power_100 (A : Matrix (Fin 2) (Fin 2) ℤ) :
  A^100 = matrix_100_pow :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_100_l564_56495


namespace NUMINAMATH_GPT_mean_of_four_numbers_l564_56462

theorem mean_of_four_numbers (a b c d : ℚ) (h : a + b + c + d = 1/2) : (a + b + c + d) / 4 = 1 / 8 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_mean_of_four_numbers_l564_56462


namespace NUMINAMATH_GPT_arithmetic_sequence_a3_l564_56443

theorem arithmetic_sequence_a3 (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : a2 = a1 + (a1 + a5 - a1) / 4)
  (h2 : a3 = a1 + 2 * (a1 + a5 - a1) / 4) 
  (h3 : a4 = a1 + 3 * (a1 + a5 - a1) / 4) 
  (h4 : a5 = a1 + 4 * (a1 + a5 - a1) / 4)
  (h_sum : 5 * a3 = 15) : 
  a3 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a3_l564_56443


namespace NUMINAMATH_GPT_probability_of_popped_white_is_12_over_17_l564_56408

noncomputable def probability_white_given_popped (white_kernels yellow_kernels : ℚ) (pop_white pop_yellow : ℚ) : ℚ :=
  let p_white_popped := white_kernels * pop_white
  let p_yellow_popped := yellow_kernels * pop_yellow
  let p_popped := p_white_popped + p_yellow_popped
  p_white_popped / p_popped

theorem probability_of_popped_white_is_12_over_17 :
  probability_white_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_popped_white_is_12_over_17_l564_56408


namespace NUMINAMATH_GPT_application_methods_count_l564_56431

theorem application_methods_count (total_universities: ℕ) (universities_with_coinciding_exams: ℕ) (chosen_universities: ℕ) 
  (remaining_universities: ℕ) (remaining_combinations: ℕ) : 
  total_universities = 6 → universities_with_coinciding_exams = 2 → chosen_universities = 3 → 
  remaining_universities = 4 → remaining_combinations = 16 := 
by
  intros
  sorry

end NUMINAMATH_GPT_application_methods_count_l564_56431


namespace NUMINAMATH_GPT_salt_amount_evaporation_l564_56496

-- Define the conditions as constants
def total_volume : ℕ := 2 -- 2 liters
def salt_concentration : ℝ := 0.2 -- 20%

-- The volume conversion factor from liters to milliliters.
def liter_to_ml : ℕ := 1000

-- Define the statement to prove
theorem salt_amount_evaporation : total_volume * (salt_concentration * liter_to_ml) = 400 := 
by 
  -- We'll skip the proof steps here
  sorry

end NUMINAMATH_GPT_salt_amount_evaporation_l564_56496


namespace NUMINAMATH_GPT_problem1_l564_56491

theorem problem1 (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  |x - y| + |y - z| + |z - x| ≤ 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem1_l564_56491


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l564_56435

theorem solve_quadratic_1 (x : ℝ) : 3 * x^2 - 8 * x + 4 = 0 ↔ x = 2/3 ∨ x = 2 := by
  sorry

theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1)^2 = (x - 3)^2 ↔ x = 4/3 ∨ x = -2 := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l564_56435


namespace NUMINAMATH_GPT_oranges_needed_l564_56416

theorem oranges_needed 
  (total_fruit_needed : ℕ := 12) 
  (apples : ℕ := 3) 
  (bananas : ℕ := 4) : 
  total_fruit_needed - (apples + bananas) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_oranges_needed_l564_56416


namespace NUMINAMATH_GPT_johns_shell_arrangements_l564_56432

-- Define the total number of arrangements without considering symmetries
def totalArrangements := Nat.factorial 12

-- Define the number of equivalent arrangements due to symmetries
def symmetries := 6 * 2

-- Define the number of distinct arrangements
def distinctArrangements : Nat := totalArrangements / symmetries

-- State the theorem
theorem johns_shell_arrangements : distinctArrangements = 479001600 :=
by
  sorry

end NUMINAMATH_GPT_johns_shell_arrangements_l564_56432


namespace NUMINAMATH_GPT_calculate_allocations_l564_56450

variable (new_revenue : ℝ)
variable (ratio_employee_salaries ratio_stock_purchases ratio_rent ratio_marketing_costs : ℕ)

theorem calculate_allocations :
  let total_ratio := ratio_employee_salaries + ratio_stock_purchases + ratio_rent + ratio_marketing_costs
  let part_value := new_revenue / total_ratio
  let employee_salary_alloc := ratio_employee_salaries * part_value
  let rent_alloc := ratio_rent * part_value
  let marketing_costs_alloc := ratio_marketing_costs * part_value
  employee_salary_alloc + rent_alloc + marketing_costs_alloc = 7800 :=
by
  sorry

end NUMINAMATH_GPT_calculate_allocations_l564_56450


namespace NUMINAMATH_GPT_difference_white_black_l564_56494

def total_stones : ℕ := 928
def white_stones : ℕ := 713
def black_stones : ℕ := total_stones - white_stones

theorem difference_white_black :
  (white_stones - black_stones = 498) :=
by
  -- Leaving the proof for later
  sorry

end NUMINAMATH_GPT_difference_white_black_l564_56494


namespace NUMINAMATH_GPT_unique_integer_in_ranges_l564_56426

theorem unique_integer_in_ranges {x : ℤ} :
  1 < x ∧ x < 9 → 
  2 < x ∧ x < 15 → 
  -1 < x ∧ x < 7 → 
  0 < x ∧ x < 4 → 
  x + 1 < 5 → 
  x = 3 := by
  intros _ _ _ _ _
  sorry

end NUMINAMATH_GPT_unique_integer_in_ranges_l564_56426


namespace NUMINAMATH_GPT_tangent_parallel_line_coordinates_l564_56436

theorem tangent_parallel_line_coordinates :
  ∃ (m n : ℝ), 
    (∀ x : ℝ, (deriv (λ x => x^4 + x) x = 4 * x^3 + 1)) ∧ 
    (deriv (λ x => x^4 + x) m = -3) ∧ 
    (n = m^4 + m) ∧ 
    (m, n) = (-1, 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_line_coordinates_l564_56436


namespace NUMINAMATH_GPT_lemon_pie_degrees_l564_56425

def total_students : ℕ := 45
def chocolate_pie_students : ℕ := 15
def apple_pie_students : ℕ := 10
def blueberry_pie_students : ℕ := 7
def cherry_and_lemon_students := total_students - (chocolate_pie_students + apple_pie_students + blueberry_pie_students)
def lemon_pie_students := cherry_and_lemon_students / 2

theorem lemon_pie_degrees (students_nonnegative : lemon_pie_students ≥ 0) (students_rounding : lemon_pie_students = 7) :
  (lemon_pie_students * 360 / total_students) = 56 := 
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_lemon_pie_degrees_l564_56425


namespace NUMINAMATH_GPT_real_root_uncertainty_l564_56444

noncomputable def f (x m : ℝ) : ℝ := m * x^2 - 2 * (m + 2) * x + m + 5
noncomputable def g (x m : ℝ) : ℝ := (m - 5) * x^2 - 2 * (m + 2) * x + m

theorem real_root_uncertainty (m : ℝ) :
  (∀ x : ℝ, f x m ≠ 0) → 
  (m ≤ 5 → ∃ x : ℝ, g x m = 0 ∧ ∀ y : ℝ, y ≠ x → g y m = 0) ∧
  (m > 5 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g x₁ m = 0 ∧ g x₂ m = 0) :=
sorry

end NUMINAMATH_GPT_real_root_uncertainty_l564_56444


namespace NUMINAMATH_GPT_evaluate_F_2_f_3_l564_56497

def f (a : ℤ) : ℤ := a^2 - 1

def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 510 := by
  sorry

end NUMINAMATH_GPT_evaluate_F_2_f_3_l564_56497


namespace NUMINAMATH_GPT_usual_time_to_school_l564_56490

variables (R T : ℝ)

theorem usual_time_to_school :
  (3 / 2) * R * (T - 4) = R * T -> T = 12 :=
by sorry

end NUMINAMATH_GPT_usual_time_to_school_l564_56490


namespace NUMINAMATH_GPT_segment_combination_l564_56446

theorem segment_combination (x y : ℕ) :
  7 * x + 12 * y = 100 ↔ (x, y) = (4, 6) :=
by
  sorry

end NUMINAMATH_GPT_segment_combination_l564_56446


namespace NUMINAMATH_GPT_average_distance_scientific_notation_l564_56414

theorem average_distance_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ a * 10 ^ n = 384000000 ∧ a = 3.84 ∧ n = 8 :=
sorry

end NUMINAMATH_GPT_average_distance_scientific_notation_l564_56414


namespace NUMINAMATH_GPT_wire_length_l564_56466

variables (L M S W : ℕ)

def ratio_condition (L M S : ℕ) : Prop :=
  L * 2 = 7 * S ∧ M * 2 = 3 * S

def total_length (L M S : ℕ) : ℕ :=
  L + M + S

theorem wire_length (h : ratio_condition L M 16) : total_length L M 16 = 96 :=
by sorry

end NUMINAMATH_GPT_wire_length_l564_56466


namespace NUMINAMATH_GPT_cost_per_student_admission_l564_56499

-- Definitions based on the conditions.
def cost_to_rent_bus : ℕ := 100
def total_budget : ℕ := 350
def number_of_students : ℕ := 25

-- The theorem that we need to prove.
theorem cost_per_student_admission : (total_budget - cost_to_rent_bus) / number_of_students = 10 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_student_admission_l564_56499


namespace NUMINAMATH_GPT_inequality_solution_l564_56459

theorem inequality_solution (x : ℝ) :
  (abs ((x^2 - 5 * x + 4) / 3) < 1) ↔ 
  ((5 - Real.sqrt 21) / 2 < x) ∧ (x < (5 + Real.sqrt 21) / 2) := 
sorry

end NUMINAMATH_GPT_inequality_solution_l564_56459


namespace NUMINAMATH_GPT_calories_per_shake_l564_56493

theorem calories_per_shake (total_calories_per_day : ℕ) (breakfast_calories : ℕ)
  (lunch_percentage_increase : ℕ) (dinner_multiplier : ℕ) (number_of_shakes : ℕ)
  (daily_calories : ℕ) :
  total_calories_per_day = breakfast_calories +
                            (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100)) +
                            (2 * (breakfast_calories + (lunch_percentage_increase * breakfast_calories / 100))) →
  daily_calories = total_calories_per_day + number_of_shakes * (daily_calories - total_calories_per_day) / number_of_shakes →
  daily_calories = 3275 → breakfast_calories = 500 →
  lunch_percentage_increase = 25 →
  dinner_multiplier = 2 →
  number_of_shakes = 3 →
  (daily_calories - total_calories_per_day) / number_of_shakes = 300 := by 
  sorry

end NUMINAMATH_GPT_calories_per_shake_l564_56493


namespace NUMINAMATH_GPT_union_of_A_B_complement_intersection_l564_56465

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_B_complement_intersection_l564_56465


namespace NUMINAMATH_GPT_students_no_A_l564_56410

theorem students_no_A (T AH AM AHAM : ℕ) (h1 : T = 35) (h2 : AH = 10) (h3 : AM = 15) (h4 : AHAM = 5) :
  T - (AH + AM - AHAM) = 15 :=
by
  sorry

end NUMINAMATH_GPT_students_no_A_l564_56410


namespace NUMINAMATH_GPT_radius_of_circle_l564_56451

theorem radius_of_circle (r : ℝ) : 
  (∀ x : ℝ, (4 * x^2 + r = x) → (1 - 16 * r = 0)) → r = 1 / 16 :=
by
  intro H
  have h := H 0
  simp at h
  sorry

end NUMINAMATH_GPT_radius_of_circle_l564_56451


namespace NUMINAMATH_GPT_intersection_points_of_graphs_l564_56477

open Real

theorem intersection_points_of_graphs (f : ℝ → ℝ) (hf : Function.Injective f) :
  ∃! x : ℝ, (f (x^3) = f (x^6)) ∧ (x = -1 ∨ x = 0 ∨ x = 1) :=
by
  -- Provide the structure of the proof
  sorry

end NUMINAMATH_GPT_intersection_points_of_graphs_l564_56477


namespace NUMINAMATH_GPT_count_integers_with_sum_of_digits_18_l564_56404

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end NUMINAMATH_GPT_count_integers_with_sum_of_digits_18_l564_56404


namespace NUMINAMATH_GPT_parallel_vectors_l564_56469

open Real

theorem parallel_vectors (k : ℝ) 
  (a : ℝ × ℝ := (k-1, 1)) 
  (b : ℝ × ℝ := (k+3, k)) 
  (h : a.1 * b.2 = a.2 * b.1) : 
  k = 3 ∨ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l564_56469


namespace NUMINAMATH_GPT_heracles_age_l564_56412

theorem heracles_age
  (H : ℕ)
  (audrey_current_age : ℕ)
  (audrey_in_3_years : ℕ)
  (h1 : audrey_current_age = H + 7)
  (h2 : audrey_in_3_years = audrey_current_age + 3)
  (h3 : audrey_in_3_years = 2 * H)
  : H = 10 :=
by
  sorry

end NUMINAMATH_GPT_heracles_age_l564_56412


namespace NUMINAMATH_GPT_no_nat_n_exists_l564_56448

theorem no_nat_n_exists (n : ℕ) : ¬ ∃ n, ∃ k, n ^ 2012 - 1 = 2 ^ k := by
  sorry

end NUMINAMATH_GPT_no_nat_n_exists_l564_56448


namespace NUMINAMATH_GPT_roden_gold_fish_count_l564_56470

theorem roden_gold_fish_count
  (total_fish : ℕ)
  (blue_fish : ℕ)
  (gold_fish : ℕ)
  (h1 : total_fish = 22)
  (h2 : blue_fish = 7)
  (h3 : total_fish = blue_fish + gold_fish) : gold_fish = 15 :=
by
  sorry

end NUMINAMATH_GPT_roden_gold_fish_count_l564_56470


namespace NUMINAMATH_GPT_decimal_to_fraction_l564_56484

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end NUMINAMATH_GPT_decimal_to_fraction_l564_56484


namespace NUMINAMATH_GPT_problem1_l564_56489

theorem problem1 : 20 + (-14) - (-18) + 13 = 37 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l564_56489


namespace NUMINAMATH_GPT_sam_added_later_buckets_l564_56492

variable (initial_buckets : ℝ) (total_buckets : ℝ)

def buckets_added_later (initial_buckets total_buckets : ℝ) : ℝ :=
  total_buckets - initial_buckets

theorem sam_added_later_buckets :
  initial_buckets = 1 ∧ total_buckets = 9.8 → buckets_added_later initial_buckets total_buckets = 8.8 := by
  sorry

end NUMINAMATH_GPT_sam_added_later_buckets_l564_56492


namespace NUMINAMATH_GPT_unique_solution_l564_56474

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1)) ∧
  (2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1)) ∧
  (2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1))

theorem unique_solution : ∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) → system_of_equations x y z → (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  intro x y z hx hy hz h
  sorry

end NUMINAMATH_GPT_unique_solution_l564_56474


namespace NUMINAMATH_GPT_inequality_proof_l564_56413

theorem inequality_proof
  (p q a b c d e : Real)
  (hpq : 0 < p ∧ p ≤ a ∧ p ≤ b ∧ p ≤ c ∧ p ≤ d ∧ p ≤ e)
  (hq : a ≤ q ∧ b ≤ q ∧ c ≤ q ∧ d ≤ q ∧ e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e)
  ≤ 25 + 6 * (Real.sqrt (p / q) - Real.sqrt (q / p)) ^ 2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l564_56413


namespace NUMINAMATH_GPT_pentagon_area_l564_56429

-- Definitions of the side lengths of the pentagon
def side1 : ℕ := 12
def side2 : ℕ := 17
def side3 : ℕ := 25
def side4 : ℕ := 18
def side5 : ℕ := 17

-- Definitions for the rectangle and triangle dimensions
def rectangle_width : ℕ := side4
def rectangle_height : ℕ := side1
def triangle_base : ℕ := side4
def triangle_height : ℕ := side3 - side1

-- The area of the pentagon proof statement
theorem pentagon_area : rectangle_width * rectangle_height +
    (triangle_base * triangle_height) / 2 = 333 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_l564_56429


namespace NUMINAMATH_GPT_B_and_C_have_together_l564_56472

theorem B_and_C_have_together
  (A B C : ℕ)
  (h1 : A + B + C = 700)
  (h2 : A + C = 300)
  (h3 : C = 200) :
  B + C = 600 := by
  sorry

end NUMINAMATH_GPT_B_and_C_have_together_l564_56472


namespace NUMINAMATH_GPT_geometric_ratio_l564_56457

theorem geometric_ratio (a₁ q : ℝ) (h₀ : a₁ ≠ 0) (h₁ : a₁ + a₁ * q + a₁ * q^2 = 3 * a₁) : q = -2 ∨ q = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_ratio_l564_56457


namespace NUMINAMATH_GPT_park_bench_problem_l564_56481

/-- A single bench section at a park can hold either 8 adults or 12 children.
When N bench sections are connected end to end, an equal number of adults and 
children seated together will occupy all the bench space.
This theorem states that the smallest positive integer N such that this condition 
is satisfied is 3. -/
theorem park_bench_problem : ∃ N : ℕ, N > 0 ∧ (8 * N = 12 * N) ∧ N = 3 :=
by
  sorry

end NUMINAMATH_GPT_park_bench_problem_l564_56481


namespace NUMINAMATH_GPT_maximum_small_circles_l564_56427

-- Definitions for small circle radius, large circle radius, and the maximum number n.
def smallCircleRadius : ℝ := 1
def largeCircleRadius : ℝ := 11

-- Function to check if small circles can be placed without overlapping
def canPlaceCircles (n : ℕ) : Prop := n * 2 < 2 * Real.pi * (largeCircleRadius - smallCircleRadius)

theorem maximum_small_circles : ∀ n : ℕ, canPlaceCircles n → n ≤ 31 := by
  sorry

end NUMINAMATH_GPT_maximum_small_circles_l564_56427


namespace NUMINAMATH_GPT_min_students_orchestra_l564_56463

theorem min_students_orchestra (n : ℕ) 
  (h1 : n % 9 = 0)
  (h2 : n % 10 = 0)
  (h3 : n % 11 = 0) : 
  n ≥ 990 ∧ ∃ k, n = 990 * k :=
by
  sorry

end NUMINAMATH_GPT_min_students_orchestra_l564_56463


namespace NUMINAMATH_GPT_beavers_fraction_l564_56440

theorem beavers_fraction (total_beavers : ℕ) (swim_percentage : ℕ) (work_percentage : ℕ) (fraction_working : ℕ) : 
total_beavers = 4 → 
swim_percentage = 75 → 
work_percentage = 100 - swim_percentage → 
fraction_working = 1 →
(work_percentage * total_beavers) / 100 = fraction_working → 
fraction_working / total_beavers = 1 / 4 :=
by 
  intros h1 h2 h3 h4 h5 
  sorry

end NUMINAMATH_GPT_beavers_fraction_l564_56440


namespace NUMINAMATH_GPT_proof_problem_l564_56475

-- Variables representing the numbers a, b, and c
variables {a b c : ℝ}

-- Given condition
def given_condition (a b c : ℝ) : Prop :=
  (a^2 + b^2) / (b^2 + c^2) = a / c

-- Required to prove
def to_prove (a b c : ℝ) : Prop :=
  (a / b = b / c) → False

-- Theorem stating that the given condition does not imply the required assertion
theorem proof_problem (a b c : ℝ) (h : given_condition a b c) : to_prove a b c :=
sorry

end NUMINAMATH_GPT_proof_problem_l564_56475


namespace NUMINAMATH_GPT_pete_nickels_spent_l564_56479

-- Definitions based on conditions
def initial_amount_per_person : ℕ := 250 -- 250 cents for $2.50
def total_initial_amount : ℕ := 2 * initial_amount_per_person
def total_expense : ℕ := 200 -- they spent 200 cents in total
def raymond_dimes_left : ℕ := 7
def value_of_dime : ℕ := 10
def raymond_remaining_amount : ℕ := raymond_dimes_left * value_of_dime
def raymond_spent_amount : ℕ := total_expense - raymond_remaining_amount
def value_of_nickel : ℕ := 5

-- Theorem to prove Pete spent 14 nickels
theorem pete_nickels_spent : 
  (total_expense - raymond_spent_amount) / value_of_nickel = 14 :=
by
  sorry

end NUMINAMATH_GPT_pete_nickels_spent_l564_56479


namespace NUMINAMATH_GPT_neg_two_squared_result_l564_56433

theorem neg_two_squared_result : -2^2 = -4 :=
by
  sorry

end NUMINAMATH_GPT_neg_two_squared_result_l564_56433


namespace NUMINAMATH_GPT_slope_of_tangent_at_0_l564_56430

theorem slope_of_tangent_at_0 (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp (2 * x)) : 
  (deriv f 0) = 2 :=
sorry

end NUMINAMATH_GPT_slope_of_tangent_at_0_l564_56430


namespace NUMINAMATH_GPT_R_and_D_expense_corresponding_to_productivity_increase_l564_56467

/-- Given values for R&D expenses and increase in average labor productivity -/
def R_and_D_t : ℝ := 2640.92
def Delta_APL_t_plus_2 : ℝ := 0.81

/-- Statement to be proved: the R&D expense in million rubles corresponding 
    to an increase in average labor productivity by 1 million rubles per person -/
theorem R_and_D_expense_corresponding_to_productivity_increase : 
  R_and_D_t / Delta_APL_t_plus_2 = 3260 := 
by
  sorry

end NUMINAMATH_GPT_R_and_D_expense_corresponding_to_productivity_increase_l564_56467


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l564_56423

theorem right_triangle_hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c^2 = a^2 + b^2) : c = 26 :=
by
  -- sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l564_56423


namespace NUMINAMATH_GPT_games_did_not_work_l564_56460

theorem games_did_not_work 
  (games_from_friend : ℕ) 
  (games_from_garage_sale : ℕ) 
  (good_games : ℕ) 
  (total_games : ℕ := games_from_friend + games_from_garage_sale) 
  (did_not_work : ℕ := total_games - good_games) :
  games_from_friend = 41 ∧ 
  games_from_garage_sale = 14 ∧ 
  good_games = 24 → 
  did_not_work = 31 := 
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_games_did_not_work_l564_56460


namespace NUMINAMATH_GPT_total_flour_needed_l564_56403

noncomputable def katie_flour : ℝ := 3

noncomputable def sheila_flour : ℝ := katie_flour + 2

noncomputable def john_flour : ℝ := 1.5 * sheila_flour

theorem total_flour_needed :
  katie_flour + sheila_flour + john_flour = 15.5 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_needed_l564_56403


namespace NUMINAMATH_GPT_only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l564_56480

theorem only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c
  (n a b c : ℕ) (hn : n > 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hca : c > a) (hcb : c > b) (hab : a ≤ b) :
  n * a + n * b = n * c ↔ (n = 2 ∧ b = a ∧ c = a + 1) := by
  sorry

end NUMINAMATH_GPT_only_solution_to_n_mul_a_n_mul_b_eq_n_mul_c_l564_56480


namespace NUMINAMATH_GPT_triangle_side_c_l564_56464

theorem triangle_side_c
  (a b c : ℝ)
  (A B C : ℝ)
  (h_bc : b = 3)
  (h_sinC : Real.sin C = 56 / 65)
  (h_sinB : Real.sin B = 12 / 13)
  (h_Angles : A + B + C = π)
  (h_valid_triangle : ∀ {x y z : ℝ}, x + y > z ∧ x + z > y ∧ y + z > x):
  c = 14 / 5 :=
sorry

end NUMINAMATH_GPT_triangle_side_c_l564_56464


namespace NUMINAMATH_GPT_induction_base_case_l564_56402

theorem induction_base_case : (-1 : ℤ) + 3 - 5 + (-1)^2 * 1 = (-1 : ℤ) := sorry

end NUMINAMATH_GPT_induction_base_case_l564_56402


namespace NUMINAMATH_GPT_proof_expression_value_l564_56442

noncomputable def a : ℝ := 0.15
noncomputable def b : ℝ := 0.06
noncomputable def x : ℝ := a^3
noncomputable def y : ℝ := b^3
noncomputable def z : ℝ := a^2
noncomputable def w : ℝ := b^2

theorem proof_expression_value :
  ( (x - y) / (z + w) ) + 0.009 + w^4 = 0.1300341679616 := sorry

end NUMINAMATH_GPT_proof_expression_value_l564_56442


namespace NUMINAMATH_GPT_side_increase_percentage_l564_56468

theorem side_increase_percentage (s : ℝ) (p : ℝ) 
  (h : (s^2) * (1.5625) = (s * (1 + p / 100))^2) : p = 25 := 
sorry

end NUMINAMATH_GPT_side_increase_percentage_l564_56468
