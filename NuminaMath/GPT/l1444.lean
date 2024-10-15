import Mathlib

namespace NUMINAMATH_GPT_probability_of_three_tails_one_head_in_four_tosses_l1444_144400

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_probability_of_three_tails_one_head_in_four_tosses_l1444_144400


namespace NUMINAMATH_GPT_sara_total_spent_l1444_144488

def ticket_cost : ℝ := 10.62
def num_tickets : ℕ := 2
def rent_cost : ℝ := 1.59
def buy_cost : ℝ := 13.95
def total_spent : ℝ := 36.78

theorem sara_total_spent : (num_tickets * ticket_cost) + rent_cost + buy_cost = total_spent := by
  sorry

end NUMINAMATH_GPT_sara_total_spent_l1444_144488


namespace NUMINAMATH_GPT_solve_for_x_l1444_144454

theorem solve_for_x (x : ℝ) (h : 3 / 4 + 1 / x = 7 / 8) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1444_144454


namespace NUMINAMATH_GPT_final_limes_count_l1444_144480

def limes_initial : ℕ := 9
def limes_by_Sara : ℕ := 4
def limes_used_for_juice : ℕ := 5
def limes_given_to_neighbor : ℕ := 3

theorem final_limes_count :
  limes_initial + limes_by_Sara - limes_used_for_juice - limes_given_to_neighbor = 5 :=
by
  sorry

end NUMINAMATH_GPT_final_limes_count_l1444_144480


namespace NUMINAMATH_GPT_arithmetic_square_root_of_4_l1444_144444

theorem arithmetic_square_root_of_4 : ∃ x : ℕ, x * x = 4 ∧ x = 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_4_l1444_144444


namespace NUMINAMATH_GPT_contrapositive_of_proposition_is_false_l1444_144413

variables {a b : ℤ}

/-- Proposition: If a and b are both even, then a + b is even -/
def proposition (a b : ℤ) : Prop :=
  (∀ n m : ℤ, a = 2 * n ∧ b = 2 * m → ∃ k : ℤ, a + b = 2 * k)

/-- Contrapositive: If a and b are not both even, then a + b is not even -/
def contrapositive (a b : ℤ) : Prop :=
  ¬(∀ n m : ℤ, a = 2 * n ∧ b = 2 * m) → ¬(∃ k : ℤ, a + b = 2 * k)

/-- The contrapositive of the proposition "If a and b are both even, then a + b is even" -/
theorem contrapositive_of_proposition_is_false :
  (contrapositive a b) = false :=
sorry

end NUMINAMATH_GPT_contrapositive_of_proposition_is_false_l1444_144413


namespace NUMINAMATH_GPT_total_amount_of_check_l1444_144414

def numParts : Nat := 59
def price50DollarPart : Nat := 50
def price20DollarPart : Nat := 20
def num50DollarParts : Nat := 40

theorem total_amount_of_check : (num50DollarParts * price50DollarPart + (numParts - num50DollarParts) * price20DollarPart) = 2380 := by
  sorry

end NUMINAMATH_GPT_total_amount_of_check_l1444_144414


namespace NUMINAMATH_GPT_cos_A_minus_B_minus_3pi_div_2_l1444_144474

theorem cos_A_minus_B_minus_3pi_div_2 (A B : ℝ)
  (h1 : Real.tan B = 2 * Real.tan A)
  (h2 : Real.cos A * Real.sin B = 4 / 5) :
  Real.cos (A - B - 3 * Real.pi / 2) = 2 / 5 := 
sorry

end NUMINAMATH_GPT_cos_A_minus_B_minus_3pi_div_2_l1444_144474


namespace NUMINAMATH_GPT_cube_surface_area_l1444_144471

noncomputable def volume (x : ℝ) : ℝ := x ^ 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem cube_surface_area (x : ℝ) :
  surface_area x = 6 * x ^ 2 :=
by sorry

end NUMINAMATH_GPT_cube_surface_area_l1444_144471


namespace NUMINAMATH_GPT_simplify_expression_l1444_144420

theorem simplify_expression (m n : ℝ) (h : m ≠ 0) : 
  (m^(4/3) - 27 * m^(1/3) * n) / 
  (m^(2/3) + 3 * (m * n)^(1/3) + 9 * n^(2/3)) / 
  (1 - 3 * (n / m)^(1/3)) - 
  (m^2)^(1/3) = 0 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l1444_144420


namespace NUMINAMATH_GPT_negation_of_quadratic_prop_l1444_144459

theorem negation_of_quadratic_prop :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_quadratic_prop_l1444_144459


namespace NUMINAMATH_GPT_no_four_digit_number_ending_in_47_is_divisible_by_5_l1444_144440

theorem no_four_digit_number_ending_in_47_is_divisible_by_5 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n % 100 = 47 → n % 10 ≠ 0 ∧ n % 10 ≠ 5) := by
  intro n
  intro Hn
  intro H47
  sorry

end NUMINAMATH_GPT_no_four_digit_number_ending_in_47_is_divisible_by_5_l1444_144440


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_a5_sum_l1444_144431

theorem arithmetic_sequence_a4_a5_sum
  (a_n : ℕ → ℝ)
  (a1_a2_sum : a_n 1 + a_n 2 = -1)
  (a3_val : a_n 3 = 4)
  (h_arith : ∃ d : ℝ, ∀ (n : ℕ), a_n (n + 1) = a_n n + d) :
  a_n 4 + a_n 5 = 17 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_a5_sum_l1444_144431


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sums_l1444_144432

variable {a b : ℝ}

theorem min_value_of_reciprocal_sums (ha : a ≠ 0) (hb : b ≠ 0) (h : 4 * a ^ 2 + b ^ 2 = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) = 9 := by
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sums_l1444_144432


namespace NUMINAMATH_GPT_probability_red_ball_distribution_of_X_expected_value_of_X_l1444_144443

theorem probability_red_ball :
  let pB₁ := 2 / 3
  let pB₂ := 1 / 3
  let pA_B₁ := 1 / 2
  let pA_B₂ := 3 / 4
  (pB₁ * pA_B₁ + pB₂ * pA_B₂) = 7 / 12 := by
  sorry

theorem distribution_of_X :
  let p_minus2 := 1 / 12
  let p_0 := 1 / 12
  let p_1 := 11 / 24
  let p_3 := 7 / 48
  let p_4 := 5 / 24
  let p_6 := 1 / 48
  (p_minus2 = 1 / 12) ∧ (p_0 = 1 / 12) ∧ (p_1 = 11 / 24) ∧ (p_3 = 7 / 48) ∧ (p_4 = 5 / 24) ∧ (p_6 = 1 / 48) := by
  sorry

theorem expected_value_of_X :
  let E_X := (-2 * (1 / 12) + 0 * (1 / 12) + 1 * (11 / 24) + 3 * (7 / 48) + 4 * (5 / 24) + 6 * (1 / 48))
  E_X = 27 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_red_ball_distribution_of_X_expected_value_of_X_l1444_144443


namespace NUMINAMATH_GPT_simplify_sqrt_is_cos_20_l1444_144448

noncomputable def simplify_sqrt : ℝ :=
  let θ : ℝ := 160 * Real.pi / 180
  Real.sqrt (1 - Real.sin θ ^ 2)

theorem simplify_sqrt_is_cos_20 : simplify_sqrt = Real.cos (20 * Real.pi / 180) :=
  sorry

end NUMINAMATH_GPT_simplify_sqrt_is_cos_20_l1444_144448


namespace NUMINAMATH_GPT_abc_inequality_l1444_144475

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) ≥ (a * b + b * c + c * a)^2 :=
sorry

end NUMINAMATH_GPT_abc_inequality_l1444_144475


namespace NUMINAMATH_GPT_geometric_sequence_problem_l1444_144495

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * (Real.log x)
  else (Real.log x) / x

theorem geometric_sequence_problem
  (a : ℕ → ℝ) 
  (r : ℝ)
  (h1 : ∃ r > 0, ∀ n, a (n + 1) = r * a n)
  (h2 : a 3 * a 4 * a 5 = 1)
  (h3 : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1) :
  a 1 = Real.exp 2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l1444_144495


namespace NUMINAMATH_GPT_candies_share_equally_l1444_144451

theorem candies_share_equally (mark_candies : ℕ) (peter_candies : ℕ) (john_candies : ℕ)
  (h_mark : mark_candies = 30) (h_peter : peter_candies = 25) (h_john : john_candies = 35) :
  (mark_candies + peter_candies + john_candies) / 3 = 30 :=
by
  sorry

end NUMINAMATH_GPT_candies_share_equally_l1444_144451


namespace NUMINAMATH_GPT_a_completes_in_12_days_l1444_144482

def work_rate_a_b (r_A r_B : ℝ) := r_A + r_B = 1 / 3
def work_rate_b_c (r_B r_C : ℝ) := r_B + r_C = 1 / 2
def work_rate_a_c (r_A r_C : ℝ) := r_A + r_C = 1 / 3

theorem a_completes_in_12_days (r_A r_B r_C : ℝ) 
  (h1 : work_rate_a_b r_A r_B)
  (h2 : work_rate_b_c r_B r_C)
  (h3 : work_rate_a_c r_A r_C) : 
  1 / r_A = 12 :=
by
  sorry

end NUMINAMATH_GPT_a_completes_in_12_days_l1444_144482


namespace NUMINAMATH_GPT_determinant_eval_l1444_144464

open Matrix

noncomputable def matrix_example (α γ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, 3 * Real.sin γ],
    ![2 * Real.cos α, -Real.sin γ, 0]]

theorem determinant_eval (α γ : ℝ) :
  det (matrix_example α γ) = 10 * Real.sin α * Real.sin γ * Real.cos α :=
sorry

end NUMINAMATH_GPT_determinant_eval_l1444_144464


namespace NUMINAMATH_GPT_log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l1444_144458

-- Define irrational numbers in Lean
def irrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Prove that log base 2 of 3 is irrational
theorem log_two_three_irrational : irrational (Real.log 3 / Real.log 2) := 
sorry

-- Prove that log base sqrt(2) of 3 is irrational
theorem log_sqrt2_three_irrational : 
  irrational (Real.log 3 / (1/2 * Real.log 2)) := 
sorry

-- Prove that log base (5 + 3sqrt(2)) of (3 + 5sqrt(2)) is irrational
theorem log_five_plus_three_sqrt2_irrational :
  irrational (Real.log (3 + 5 * Real.sqrt 2) / Real.log (5 + 3 * Real.sqrt 2)) := 
sorry

end NUMINAMATH_GPT_log_two_three_irrational_log_sqrt2_three_irrational_log_five_plus_three_sqrt2_irrational_l1444_144458


namespace NUMINAMATH_GPT_find_abc_l1444_144449

theorem find_abc
  {a b c : ℤ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 30)
  (h2 : 1/a + 1/b + 1/c + 672/(a*b*c) = 1) :
  a * b * c = 2808 :=
sorry

end NUMINAMATH_GPT_find_abc_l1444_144449


namespace NUMINAMATH_GPT_sum_of_real_roots_l1444_144411

theorem sum_of_real_roots (P : Polynomial ℝ) (hP : P = Polynomial.C 1 * X^4 - Polynomial.C 8 * X - Polynomial.C 2) :
  P.roots.sum = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_real_roots_l1444_144411


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1444_144499

theorem solution_set_of_inequality (x : ℝ) (n : ℕ) (h1 : n ≤ x ∧ x < n + 1 ∧ 0 < n) :
  4 * (⌊x⌋ : ℝ)^2 - 36 * (⌊x⌋ : ℝ) + 45 < 0 ↔ ∃ k : ℕ, (2 ≤ k ∧ k < 8 ∧ ⌊x⌋ = k) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1444_144499


namespace NUMINAMATH_GPT_multiple_of_C_share_l1444_144423

theorem multiple_of_C_share (A B C k : ℝ) : 
  3 * A = k * C ∧ 4 * B = k * C ∧ C = 84 ∧ A + B + C = 427 → k = 7 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_C_share_l1444_144423


namespace NUMINAMATH_GPT_find_percentage_ryegrass_in_seed_mixture_X_l1444_144419

open Real

noncomputable def percentage_ryegrass_in_seed_mixture_X (R : ℝ) : Prop := 
  let proportion_X : ℝ := 2 / 3
  let percentage_Y_ryegrass : ℝ := 25 / 100
  let proportion_Y : ℝ := 1 / 3
  let final_percentage_ryegrass : ℝ := 35 / 100
  final_percentage_ryegrass = (R / 100 * proportion_X) + (percentage_Y_ryegrass * proportion_Y)

/-
  Given the conditions:
  - Seed mixture Y is 25 percent ryegrass.
  - A mixture of seed mixtures X (66.67% of the mixture) and Y (33.33% of the mixture) contains 35 percent ryegrass.

  Prove:
  The percentage of ryegrass in seed mixture X is 40%.
-/
theorem find_percentage_ryegrass_in_seed_mixture_X : 
  percentage_ryegrass_in_seed_mixture_X 40 := 
  sorry

end NUMINAMATH_GPT_find_percentage_ryegrass_in_seed_mixture_X_l1444_144419


namespace NUMINAMATH_GPT_percentage_decrease_to_gain_30_percent_profit_l1444_144496

theorem percentage_decrease_to_gain_30_percent_profit
  (C : ℝ) (P : ℝ) (S : ℝ) (S_new : ℝ) 
  (C_eq : C = 60)
  (S_eq : S = 1.25 * C)
  (S_new_eq1 : S_new = S - 12.60)
  (S_new_eq2 : S_new = 1.30 * (C - P * C)) : 
  P = 0.20 := by
  sorry

end NUMINAMATH_GPT_percentage_decrease_to_gain_30_percent_profit_l1444_144496


namespace NUMINAMATH_GPT_total_watermelon_weight_l1444_144447

theorem total_watermelon_weight :
  let w1 := 9.91
  let w2 := 4.112
  let w3 := 6.059
  w1 + w2 + w3 = 20.081 :=
by
  sorry

end NUMINAMATH_GPT_total_watermelon_weight_l1444_144447


namespace NUMINAMATH_GPT_find_g_8_l1444_144435

def g (x : ℝ) : ℝ := x^2 + x + 1

theorem find_g_8 : (∀ x : ℝ, g (2*x - 4) = x^2 + x + 1) → g 8 = 43 := 
by sorry

end NUMINAMATH_GPT_find_g_8_l1444_144435


namespace NUMINAMATH_GPT_total_volume_is_correct_l1444_144472

theorem total_volume_is_correct :
  let carl_side := 3
  let carl_count := 3
  let kate_side := 1.5
  let kate_count := 4
  let carl_volume := carl_count * carl_side ^ 3
  let kate_volume := kate_count * kate_side ^ 3
  carl_volume + kate_volume = 94.5 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_is_correct_l1444_144472


namespace NUMINAMATH_GPT_find_principal_amount_l1444_144425

-- Definitions based on conditions
def A : ℝ := 3969
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The statement to be proved
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + r/n)^(n * t) ∧ P = 3600 :=
by
  use 3600
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1444_144425


namespace NUMINAMATH_GPT_functions_satisfying_equation_l1444_144433

theorem functions_satisfying_equation 
  (f g h : ℝ → ℝ)
  (H : ∀ x y : ℝ, f x - g y = (x - y) * h (x + y)) :
  ∃ a b c : ℝ, 
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, g x = a * x^2 + b * x + c) ∧ 
    (∀ x : ℝ, h x = a * x + b) :=
sorry

end NUMINAMATH_GPT_functions_satisfying_equation_l1444_144433


namespace NUMINAMATH_GPT_quadratic_intersection_l1444_144412

theorem quadratic_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 7 * x - 7 = 0) ↔ k ≥ -7/4 ∧ k ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersection_l1444_144412


namespace NUMINAMATH_GPT_expression_values_l1444_144453

-- Define the conditions as a predicate
def conditions (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a^2 - b * c = b^2 - a * c ∧ b^2 - a * c = c^2 - a * b

-- The main theorem statement
theorem expression_values (a b c : ℝ) (h : conditions a b c) :
  (∃ x : ℝ, x = (a / (b + c) + 2 * b / (a + c) + 4 * c / (a + b)) ∧ (x = 7 / 2 ∨ x = -7)) :=
by
  sorry

end NUMINAMATH_GPT_expression_values_l1444_144453


namespace NUMINAMATH_GPT_charlotte_avg_speed_l1444_144491

def distance : ℕ := 60  -- distance in miles
def time : ℕ := 6       -- time in hours

theorem charlotte_avg_speed : (distance / time) = 10 := by
  sorry

end NUMINAMATH_GPT_charlotte_avg_speed_l1444_144491


namespace NUMINAMATH_GPT_entire_meal_cost_correct_l1444_144439

-- Define given conditions
def appetizer_cost : ℝ := 9.00
def entree_cost : ℝ := 20.00
def num_entrees : ℕ := 2
def dessert_cost : ℝ := 11.00
def tip_percentage : ℝ := 0.30

-- Calculate intermediate values
def total_cost_before_tip : ℝ := appetizer_cost + (entree_cost * num_entrees) + dessert_cost
def tip : ℝ := tip_percentage * total_cost_before_tip
def entire_meal_cost : ℝ := total_cost_before_tip + tip

-- Statement to be proved
theorem entire_meal_cost_correct : entire_meal_cost = 78.00 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_entire_meal_cost_correct_l1444_144439


namespace NUMINAMATH_GPT_evaluate_expression_l1444_144434

theorem evaluate_expression :
  let x := 1.93
  let y := 51.3
  let z := 0.47
  Float.round (x * (y + z)) = 100 := by
sorry

end NUMINAMATH_GPT_evaluate_expression_l1444_144434


namespace NUMINAMATH_GPT_find_vector_b_l1444_144438

def vector_collinear (a b : ℝ × ℝ) : Prop :=
    ∃ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2

theorem find_vector_b (a b : ℝ × ℝ) (h_collinear : vector_collinear a b) (h_dot : dot_product a b = -10) : b = (-4, 2) :=
    by
        sorry

end NUMINAMATH_GPT_find_vector_b_l1444_144438


namespace NUMINAMATH_GPT_maria_anna_ages_l1444_144486

theorem maria_anna_ages : 
  ∃ (x y : ℝ), x + y = 44 ∧ x = 2 * (y - (- (1/2) * x + (3/2) * ((2/3) * y))) ∧ x = 27.5 ∧ y = 16.5 := by 
  sorry

end NUMINAMATH_GPT_maria_anna_ages_l1444_144486


namespace NUMINAMATH_GPT_tan_theta_l1444_144427

theorem tan_theta (θ : ℝ) (h : Real.sin (θ / 2) - 2 * Real.cos (θ / 2) = 0) : Real.tan θ = -4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_theta_l1444_144427


namespace NUMINAMATH_GPT_cap_to_sunglasses_prob_l1444_144476

-- Define the conditions
def num_people_wearing_sunglasses : ℕ := 60
def num_people_wearing_caps : ℕ := 40
def prob_sunglasses_and_caps : ℚ := 1 / 3

-- Define the statement to prove
theorem cap_to_sunglasses_prob : 
  (num_people_wearing_sunglasses * prob_sunglasses_and_caps) / num_people_wearing_caps = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cap_to_sunglasses_prob_l1444_144476


namespace NUMINAMATH_GPT_sequence_sum_l1444_144408

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 0 < a n)
  → (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) 
  → (∀ n : ℕ, a (n+1)^2 = a n * a (n+2))
  → S 3 = 13
  → a 1 = 1
  → (a 3 + a 4) / (a 1 + a 2) = 9 :=
sorry

end NUMINAMATH_GPT_sequence_sum_l1444_144408


namespace NUMINAMATH_GPT_number_line_distance_l1444_144461

theorem number_line_distance (x : ℝ) : |x + 1| = 6 ↔ (x = 5 ∨ x = -7) :=
by
  sorry

end NUMINAMATH_GPT_number_line_distance_l1444_144461


namespace NUMINAMATH_GPT_triangle_first_side_length_l1444_144405

theorem triangle_first_side_length (x : ℕ) (h1 : x + 20 + 30 = 55) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_first_side_length_l1444_144405


namespace NUMINAMATH_GPT_range_of_m_l1444_144484

def f (x : ℝ) : ℝ := x^2 - 4 * x - 6

theorem range_of_m (m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ m → -10 ≤ f x ∧ f x ≤ -6) →
  2 ≤ m ∧ m ≤ 4 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1444_144484


namespace NUMINAMATH_GPT_eugene_pencils_left_l1444_144416

-- Define the total number of pencils Eugene initially has
def initial_pencils : ℝ := 234.0

-- Define the number of pencils Eugene gives away
def pencils_given_away : ℝ := 35.0

-- Define the expected number of pencils left
def expected_pencils_left : ℝ := 199.0

-- Prove the number of pencils left after giving away 35.0 equals 199.0
theorem eugene_pencils_left : initial_pencils - pencils_given_away = expected_pencils_left := by
  -- This is where the proof would go, if needed
  sorry

end NUMINAMATH_GPT_eugene_pencils_left_l1444_144416


namespace NUMINAMATH_GPT_circle_condition_l1444_144409

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 5 * m = 0) ↔ m < 1 := by
  sorry

end NUMINAMATH_GPT_circle_condition_l1444_144409


namespace NUMINAMATH_GPT_coprime_ab_and_a_plus_b_l1444_144430

theorem coprime_ab_and_a_plus_b (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a * b) (a + b) = 1 := by
  sorry

end NUMINAMATH_GPT_coprime_ab_and_a_plus_b_l1444_144430


namespace NUMINAMATH_GPT_proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l1444_144404

theorem proportion_false_if_x_is_0_75 (x : ℚ) (h1 : x = 0.75) : ¬ (x / 2 = 2 / 6) :=
by sorry

theorem correct_value_of_x_in_proportion (x : ℚ) (h1 : x / 2 = 2 / 6) : x = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_proportion_false_if_x_is_0_75_correct_value_of_x_in_proportion_l1444_144404


namespace NUMINAMATH_GPT_sequence_problem_l1444_144415

theorem sequence_problem (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, a (n + 2) = 1 / (a n + 1)) (h_eq : a 100 = a 96) :
  a 2018 + a 3 = (Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_problem_l1444_144415


namespace NUMINAMATH_GPT_triangle_area_l1444_144457

theorem triangle_area (d : ℝ) (h : d = 8 * Real.sqrt 10) (ang : ∀ {α β γ : ℝ}, α = 45 ∨ β = 45 ∨ γ = 45) :
  ∃ A : ℝ, A = 160 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1444_144457


namespace NUMINAMATH_GPT_original_profit_percentage_l1444_144469

theorem original_profit_percentage {P S : ℝ}
  (h1 : S = 1100)
  (h2 : P ≠ 0)
  (h3 : 1.17 * P = 1170) :
  (S - P) / P * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_profit_percentage_l1444_144469


namespace NUMINAMATH_GPT_find_x_l1444_144483

theorem find_x 
  (x y z : ℝ)
  (h1 : (20 + 40 + 60 + x) / 4 = (10 + 70 + y + z) / 4 + 9)
  (h2 : y + z = 110) 
  : x = 106 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l1444_144483


namespace NUMINAMATH_GPT_sqrt_expression_eq_l1444_144441

theorem sqrt_expression_eq :
  Real.sqrt (3 * (7 + 4 * Real.sqrt 3)) = 2 * Real.sqrt 3 + 3 := 
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_l1444_144441


namespace NUMINAMATH_GPT_find_y_l1444_144466

def custom_op (a b : ℤ) : ℤ := (a - 1) * (b - 1)

theorem find_y (y : ℤ) (h : custom_op y 10 = 90) : y = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1444_144466


namespace NUMINAMATH_GPT_find_total_amount_l1444_144421

variables (A B C : ℕ) (total_amount : ℕ) 

-- Conditions
def condition1 : Prop := B = 36
def condition2 : Prop := 100 * B / 45 = A
def condition3 : Prop := 100 * C / 30 = A

-- Proof statement
theorem find_total_amount (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 A C) :
  total_amount = 300 :=
sorry

end NUMINAMATH_GPT_find_total_amount_l1444_144421


namespace NUMINAMATH_GPT_price_reduction_l1444_144493

variable (T : ℝ) -- The original price of the television
variable (first_discount : ℝ) -- First discount in percentage
variable (second_discount : ℝ) -- Second discount in percentage

theorem price_reduction (h1 : first_discount = 0.4) (h2 : second_discount = 0.4) : 
  (1 - (1 - first_discount) * (1 - second_discount)) = 0.64 :=
by
  sorry

end NUMINAMATH_GPT_price_reduction_l1444_144493


namespace NUMINAMATH_GPT_glass_cannot_all_be_upright_l1444_144422

def glass_flip_problem :=
  ∀ (g : Fin 6 → ℤ),
    g 0 = 1 ∧ g 1 = 1 ∧ g 2 = 1 ∧ g 3 = 1 ∧ g 4 = 1 ∧ g 5 = -1 →
    (∀ (flip : Fin 4 → Fin 6 → ℤ),
      (∃ (i1 i2 i3 i4: Fin 6), 
        flip 0 = g i1 * -1 ∧ 
        flip 1 = g i2 * -1 ∧
        flip 2 = g i3 * -1 ∧
        flip 3 = g i4 * -1) →
      ∃ j, g j ≠ 1)

theorem glass_cannot_all_be_upright : glass_flip_problem :=
  sorry

end NUMINAMATH_GPT_glass_cannot_all_be_upright_l1444_144422


namespace NUMINAMATH_GPT_mass_15_implies_age_7_l1444_144479

-- Define the mass function m which depends on age a
variable (m : ℕ → ℕ)

-- Define the condition for the mass to be 15 kg
def is_age_when_mass_is_15 (a : ℕ) : Prop :=
  m a = 15

-- The problem statement to be proven
theorem mass_15_implies_age_7 : ∀ a, is_age_when_mass_is_15 m a → a = 7 :=
by
  -- Proof details would follow here
  sorry

end NUMINAMATH_GPT_mass_15_implies_age_7_l1444_144479


namespace NUMINAMATH_GPT_distance_between_lines_l1444_144446

noncomputable def distance_between_parallel_lines
  (a b m n : ℝ) : ℝ :=
  |m - n| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines
  (a b m n : ℝ) :
  distance_between_parallel_lines a b m n = 
  |m - n| / Real.sqrt (a^2 + b^2) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_lines_l1444_144446


namespace NUMINAMATH_GPT_find_some_number_l1444_144467

theorem find_some_number (some_number : ℝ) (h : (3.242 * some_number) / 100 = 0.045388) : some_number = 1.400 := 
sorry

end NUMINAMATH_GPT_find_some_number_l1444_144467


namespace NUMINAMATH_GPT_gcd_840_1764_gcd_98_63_l1444_144456

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by sorry

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by sorry

end NUMINAMATH_GPT_gcd_840_1764_gcd_98_63_l1444_144456


namespace NUMINAMATH_GPT_solve_fraction_equation_l1444_144489

def fraction_equation (x : ℝ) : Prop :=
  1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) + 2 / (x - 1) = 5

theorem solve_fraction_equation (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  fraction_equation x → 
  x = (-11 + Real.sqrt 257) / 4 ∨ x = (-11 - Real.sqrt 257) / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l1444_144489


namespace NUMINAMATH_GPT_translate_point_correct_l1444_144403

def P : ℝ × ℝ := (2, 3)

def translate_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 - d)

theorem translate_point_correct :
  translate_down (translate_left P 3) 4 = (-1, -1) :=
by
  sorry

end NUMINAMATH_GPT_translate_point_correct_l1444_144403


namespace NUMINAMATH_GPT_calculate_total_weight_l1444_144424

variable (a b c d : ℝ)

-- Conditions
def I_II_weight := a + b = 156
def III_IV_weight := c + d = 195
def I_III_weight := a + c = 174
def II_IV_weight := b + d = 186

theorem calculate_total_weight (I_II_weight : a + b = 156) (III_IV_weight : c + d = 195)
    (I_III_weight : a + c = 174) (II_IV_weight : b + d = 186) :
    a + b + c + d = 355.5 :=
by
    sorry

end NUMINAMATH_GPT_calculate_total_weight_l1444_144424


namespace NUMINAMATH_GPT_gumballs_problem_l1444_144487

theorem gumballs_problem 
  (L x : ℕ)
  (h1 : 19 ≤ (17 + L + x) / 3 ∧ (17 + L + x) / 3 ≤ 25)
  (h2 : ∃ x_min x_max, x_max - x_min = 18 ∧ x_min = 19 ∧ x = x_min ∨ x = x_max) : 
  L = 21 :=
sorry

end NUMINAMATH_GPT_gumballs_problem_l1444_144487


namespace NUMINAMATH_GPT_focal_length_ellipse_l1444_144402

theorem focal_length_ellipse :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 :=
by
  sorry

end NUMINAMATH_GPT_focal_length_ellipse_l1444_144402


namespace NUMINAMATH_GPT_dacid_physics_marks_l1444_144401

theorem dacid_physics_marks 
  (english : ℕ := 73)
  (math : ℕ := 69)
  (chem : ℕ := 64)
  (bio : ℕ := 82)
  (avg_marks : ℕ := 76)
  (num_subjects : ℕ := 5)
  : ∃ physics : ℕ, physics = 92 :=
by
  let total_marks := avg_marks * num_subjects
  let known_marks := english + math + chem + bio
  have physics := total_marks - known_marks
  use physics
  sorry

end NUMINAMATH_GPT_dacid_physics_marks_l1444_144401


namespace NUMINAMATH_GPT_min_socks_no_conditions_l1444_144490

theorem min_socks_no_conditions (m n : Nat) (h : (m * (m - 1) = 2 * (m + n) * (m + n - 1))) : 
  m + n ≥ 4 := sorry

end NUMINAMATH_GPT_min_socks_no_conditions_l1444_144490


namespace NUMINAMATH_GPT_find_area_of_triangle_l1444_144426

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (s * (s - a) * (s - b) * (s - c)).sqrt

theorem find_area_of_triangle :
  let a := 10
  let b := 10
  let c := 12
  triangle_area a b c = 48 := 
by 
  sorry

end NUMINAMATH_GPT_find_area_of_triangle_l1444_144426


namespace NUMINAMATH_GPT_sum_reciprocals_of_factors_12_l1444_144445

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocals_of_factors_12_l1444_144445


namespace NUMINAMATH_GPT_total_votes_l1444_144437

theorem total_votes (A B C V : ℝ)
  (h1 : A = B + 0.10 * V)
  (h2 : A = C + 0.15 * V)
  (h3 : A - 3000 = B + 3000)
  (h4 : B + 3000 = A - 0.10 * V)
  (h5 : B + 3000 = C + 0.05 * V)
  : V = 60000 := 
sorry

end NUMINAMATH_GPT_total_votes_l1444_144437


namespace NUMINAMATH_GPT_find_f_of_7_over_2_l1444_144485

variable (f : ℝ → ℝ)

axiom f_odd : ∀ x, f (-x) = -f x
axiom f_periodic : ∀ x, f (x + 2) = f (x - 2)
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = 3^x

theorem find_f_of_7_over_2 : f (7 / 2) = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_7_over_2_l1444_144485


namespace NUMINAMATH_GPT_kids_played_on_Wednesday_l1444_144463

def played_on_Monday : ℕ := 17
def played_on_Tuesday : ℕ := 15
def total_kids : ℕ := 34

theorem kids_played_on_Wednesday :
  total_kids - (played_on_Monday + played_on_Tuesday) = 2 :=
by sorry

end NUMINAMATH_GPT_kids_played_on_Wednesday_l1444_144463


namespace NUMINAMATH_GPT_arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l1444_144442

-- Statement for Question 1
theorem arrangement_ways_13_books : 
  (Nat.factorial 13) = 6227020800 := 
sorry

-- Statement for Question 2
theorem arrangement_ways_13_books_with_4_arithmetic_together :
  (Nat.factorial 10) * (Nat.factorial 4) = 87091200 := 
sorry

-- Statement for Question 3
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_together :
  (Nat.factorial 5) * (Nat.factorial 4) * (Nat.factorial 6) = 2073600 := 
sorry

-- Statement for Question 4
theorem arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together :
  (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 6) * (Nat.factorial 3) = 622080 := 
sorry

end NUMINAMATH_GPT_arrangement_ways_13_books_arrangement_ways_13_books_with_4_arithmetic_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_together_arrangement_ways_13_books_with_4_arithmetic_6_algebra_3_geometry_together_l1444_144442


namespace NUMINAMATH_GPT_veranda_area_correct_l1444_144429

-- Definitions of the room dimensions and veranda width
def room_length : ℝ := 18
def room_width : ℝ := 12
def veranda_width : ℝ := 2

-- Definition of the total length including veranda
def total_length : ℝ := room_length + 2 * veranda_width

-- Definition of the total width including veranda
def total_width : ℝ := room_width + 2 * veranda_width

-- Definition of the area of the entire space (room plus veranda)
def area_entire_space : ℝ := total_length * total_width

-- Definition of the area of the room
def area_room : ℝ := room_length * room_width

-- Definition of the area of the veranda
def area_veranda : ℝ := area_entire_space - area_room

-- Theorem statement to prove the area of the veranda
theorem veranda_area_correct : area_veranda = 136 := 
by
  sorry

end NUMINAMATH_GPT_veranda_area_correct_l1444_144429


namespace NUMINAMATH_GPT_range_alpha_minus_beta_over_2_l1444_144436

theorem range_alpha_minus_beta_over_2 (α β : ℝ) (h1 : -π / 2 ≤ α) (h2 : α < β) (h3 : β ≤ π / 2) :
  Set.Ico (-π / 2) 0 = {x : ℝ | ∃ α β : ℝ, -π / 2 ≤ α ∧ α < β ∧ β ≤ π / 2 ∧ x = (α - β) / 2} :=
by
  sorry

end NUMINAMATH_GPT_range_alpha_minus_beta_over_2_l1444_144436


namespace NUMINAMATH_GPT_quadratic_not_divisible_by_49_l1444_144418

theorem quadratic_not_divisible_by_49 (n : ℤ) : ¬ (n^2 + 3 * n + 4) % 49 = 0 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_not_divisible_by_49_l1444_144418


namespace NUMINAMATH_GPT_rectangle_perimeter_l1444_144497

-- Definitions based on the conditions
def length : ℕ := 15
def width : ℕ := 8

-- Definition of the perimeter function
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Statement of the theorem we need to prove
theorem rectangle_perimeter : perimeter length width = 46 := by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1444_144497


namespace NUMINAMATH_GPT_next_sales_amount_l1444_144492

theorem next_sales_amount
  (royalties1: ℝ)
  (sales1: ℝ)
  (royalties2: ℝ)
  (percentage_decrease: ℝ)
  (X: ℝ)
  (h1: royalties1 = 4)
  (h2: sales1 = 20)
  (h3: royalties2 = 9)
  (h4: percentage_decrease = 58.333333333333336 / 100)
  (h5: royalties2 / X = royalties1 / sales1 - ((royalties1 / sales1) * percentage_decrease)): 
  X = 108 := 
  by 
    -- Proof omitted
    sorry

end NUMINAMATH_GPT_next_sales_amount_l1444_144492


namespace NUMINAMATH_GPT_possible_values_a_l1444_144410

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 - a * x - 7 else a / x

theorem possible_values_a (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → -2 * x - a ≥ 0) ∧
  (∀ x : ℝ, x > 1 → -a / (x^2) ≥ 0) ∧
  (-8 - a ≤ a) →
  a = -2 ∨ a = -3 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_possible_values_a_l1444_144410


namespace NUMINAMATH_GPT_total_broken_marbles_l1444_144406

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end NUMINAMATH_GPT_total_broken_marbles_l1444_144406


namespace NUMINAMATH_GPT_exponent_subtraction_l1444_144477

variable {a : ℝ} {m n : ℕ}

theorem exponent_subtraction (hm : a ^ m = 12) (hn : a ^ n = 3) : a ^ (m - n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_exponent_subtraction_l1444_144477


namespace NUMINAMATH_GPT_divide_fractions_l1444_144407

theorem divide_fractions :
  (7 / 3) / (5 / 4) = (28 / 15) :=
by
  sorry

end NUMINAMATH_GPT_divide_fractions_l1444_144407


namespace NUMINAMATH_GPT_basketball_weight_l1444_144494

variable {b c : ℝ}

theorem basketball_weight (h1 : 8 * b = 4 * c) (h2 : 3 * c = 120) : b = 20 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_basketball_weight_l1444_144494


namespace NUMINAMATH_GPT_time_to_run_round_square_field_l1444_144465

theorem time_to_run_round_square_field
  (side : ℝ) (speed_km_hr : ℝ)
  (h_side : side = 45)
  (h_speed_km_hr : speed_km_hr = 9) : 
  (4 * side / (speed_km_hr * 1000 / 3600)) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_time_to_run_round_square_field_l1444_144465


namespace NUMINAMATH_GPT_third_student_number_l1444_144455

theorem third_student_number (A B C D : ℕ) 
  (h1 : A + B + C + D = 531) 
  (h2 : A + B = C + D + 31) 
  (h3 : C = D + 22) : 
  C = 136 := 
by
  sorry

end NUMINAMATH_GPT_third_student_number_l1444_144455


namespace NUMINAMATH_GPT_chocolate_bars_per_box_is_25_l1444_144462

-- Define the conditions
def total_chocolate_bars : Nat := 400
def total_small_boxes : Nat := 16

-- Define the statement to be proved
def chocolate_bars_per_small_box : Nat := total_chocolate_bars / total_small_boxes

theorem chocolate_bars_per_box_is_25
  (h1 : total_chocolate_bars = 400)
  (h2 : total_small_boxes = 16) :
  chocolate_bars_per_small_box = 25 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_chocolate_bars_per_box_is_25_l1444_144462


namespace NUMINAMATH_GPT_linear_function_decreasing_iff_l1444_144468

-- Define the conditions
def linear_function (m b x : ℝ) : ℝ := m * x + b

-- Define the condition for decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

-- The theorem to prove
theorem linear_function_decreasing_iff (m b : ℝ) :
  (is_decreasing (linear_function m b)) ↔ (m < 0) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_decreasing_iff_l1444_144468


namespace NUMINAMATH_GPT_find_k_min_value_quadratic_zero_l1444_144498

theorem find_k_min_value_quadratic_zero (x y k : ℝ) :
  (∃ (k : ℝ), ∀ (x y : ℝ), 5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 10 * x - 6 * y + 9 = 0) ↔ k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_min_value_quadratic_zero_l1444_144498


namespace NUMINAMATH_GPT_number_of_teams_l1444_144450

theorem number_of_teams (n : ℕ) (G : ℕ) (h1 : G = 28) (h2 : G = n * (n - 1) / 2) : n = 8 := 
  by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_number_of_teams_l1444_144450


namespace NUMINAMATH_GPT_area_of_parallelogram_l1444_144452

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 := by
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l1444_144452


namespace NUMINAMATH_GPT_linear_function_solution_l1444_144417

theorem linear_function_solution (k : ℝ) (h₁ : k ≠ 0) (h₂ : 0 = k * (-2) + 3) :
  ∃ x : ℝ, k * (x - 5) + 3 = 0 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_solution_l1444_144417


namespace NUMINAMATH_GPT_pond_87_5_percent_algae_free_on_day_17_l1444_144460

/-- The algae in a local pond doubles every day. -/
def algae_doubles_every_day (coverage : ℕ → ℝ) : Prop :=
  ∀ n, coverage (n + 1) = 2 * coverage n

/-- The pond is completely covered in algae on day 20. -/
def pond_completely_covered_on_day_20 (coverage : ℕ → ℝ) : Prop :=
  coverage 20 = 1

/-- Determine the day on which the pond was 87.5% algae-free. -/
theorem pond_87_5_percent_algae_free_on_day_17 (coverage : ℕ → ℝ)
  (h1 : algae_doubles_every_day coverage)
  (h2 : pond_completely_covered_on_day_20 coverage) :
  coverage 17 = 0.125 :=
sorry

end NUMINAMATH_GPT_pond_87_5_percent_algae_free_on_day_17_l1444_144460


namespace NUMINAMATH_GPT_trajectory_equation_l1444_144481

-- Define the condition that the distance to the coordinate axes is equal.
def equidistantToAxes (x y : ℝ) : Prop :=
  abs x = abs y

-- State the theorem that we need to prove.
theorem trajectory_equation (x y : ℝ) (h : equidistantToAxes x y) : y^2 = x^2 :=
by sorry

end NUMINAMATH_GPT_trajectory_equation_l1444_144481


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1444_144428

theorem ball_hits_ground_time :
  ∃ t : ℚ, -20 * t^2 + 30 * t + 50 = 0 ∧ t = 5 / 2 :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1444_144428


namespace NUMINAMATH_GPT_jacques_suitcase_weight_l1444_144473

noncomputable def suitcase_weight_on_return : ℝ := 
  let initial_weight := 12
  let perfume_weight := (5 * 1.2) / 16
  let chocolate_weight := 4 + 1.5 + 3.25
  let soap_weight := (2 * 5) / 16
  let jam_weight := (8 + 6 + 10 + 12) / 16
  let sculpture_weight := 3.5 * 2.20462
  let shirts_weight := (3 * 300 * 0.03527396) / 16
  let cookies_weight := (450 * 0.03527396) / 16
  let wine_weight := (190 * 0.03527396) / 16
  initial_weight + perfume_weight + chocolate_weight + soap_weight + jam_weight + sculpture_weight + shirts_weight + cookies_weight + wine_weight

theorem jacques_suitcase_weight : suitcase_weight_on_return = 35.111288 := 
by 
  -- Calculation to verify that the total is 35.111288
  sorry

end NUMINAMATH_GPT_jacques_suitcase_weight_l1444_144473


namespace NUMINAMATH_GPT_area_of_triangle_l1444_144478

theorem area_of_triangle (a b : ℝ) 
  (hypotenuse : ℝ) (median : ℝ)
  (h_side : hypotenuse = 2)
  (h_median : median = 1)
  (h_sum : a + b = 1 + Real.sqrt 3) 
  (h_pythagorean :(a^2 + b^2 = 4)): 
  (1/2 * a * b) = (Real.sqrt 3 / 2) := 
sorry

end NUMINAMATH_GPT_area_of_triangle_l1444_144478


namespace NUMINAMATH_GPT_average_minutes_correct_l1444_144470

noncomputable def average_minutes_run_per_day : ℚ :=
  let f (fifth_graders : ℕ) : ℚ := (48 * (4 * fifth_graders) + 30 * (2 * fifth_graders) + 10 * fifth_graders) / (4 * fifth_graders + 2 * fifth_graders + fifth_graders)
  f 1

theorem average_minutes_correct :
  average_minutes_run_per_day = 88 / 7 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_correct_l1444_144470
