import Mathlib

namespace NUMINAMATH_GPT_emily_has_7_times_more_oranges_than_sandra_l1755_175579

theorem emily_has_7_times_more_oranges_than_sandra
  (B S E : ℕ)
  (h1 : S = 3 * B)
  (h2 : B = 12)
  (h3 : E = 252) :
  ∃ k : ℕ, E = k * S ∧ k = 7 :=
by
  use 7
  sorry

end NUMINAMATH_GPT_emily_has_7_times_more_oranges_than_sandra_l1755_175579


namespace NUMINAMATH_GPT_solve_system_l1755_175512

theorem solve_system :
  ∃ (x y : ℤ), (x * (1/7 : ℚ)^2 = 7^3) ∧ (x + y = 7^2) ∧ (x = 16807) ∧ (y = -16758) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1755_175512


namespace NUMINAMATH_GPT_parallel_lines_slope_eq_l1755_175537

theorem parallel_lines_slope_eq (k : ℝ) : (∀ x : ℝ, 3 = 6 * k) → k = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_eq_l1755_175537


namespace NUMINAMATH_GPT_fractions_product_equals_54_l1755_175559

theorem fractions_product_equals_54 :
  (4 / 5) * (9 / 6) * (12 / 4) * (20 / 15) * (14 / 21) * (35 / 28) * (48 / 32) * (24 / 16) = 54 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_fractions_product_equals_54_l1755_175559


namespace NUMINAMATH_GPT_constant_term_in_binomial_expansion_max_coef_sixth_term_l1755_175586

theorem constant_term_in_binomial_expansion_max_coef_sixth_term 
  (n : ℕ) (h : n = 10) : 
  (∃ C : ℕ → ℕ → ℕ, C 10 2 * (Nat.sqrt 2) ^ 8 = 720) :=
sorry

end NUMINAMATH_GPT_constant_term_in_binomial_expansion_max_coef_sixth_term_l1755_175586


namespace NUMINAMATH_GPT_g_of_f_eq_l1755_175588

def f (A B x : ℝ) : ℝ := A * x^2 - B^2
def g (B x : ℝ) : ℝ := B * x + B^2

theorem g_of_f_eq (A B : ℝ) (hB : B ≠ 0) : 
  g B (f A B 1) = B * A - B^3 + B^2 := 
by
  sorry

end NUMINAMATH_GPT_g_of_f_eq_l1755_175588


namespace NUMINAMATH_GPT_system_solution_correct_l1755_175583

theorem system_solution_correct (b : ℝ) : (∃ x y : ℝ, (y = 3 * x - 5) ∧ (y = 2 * x + b) ∧ (x = 1) ∧ (y = -2)) ↔ b = -4 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_correct_l1755_175583


namespace NUMINAMATH_GPT_differentiate_and_evaluate_l1755_175521

theorem differentiate_and_evaluate (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) (x : ℝ) :
  (2*x - 1)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 = 12 :=
sorry

end NUMINAMATH_GPT_differentiate_and_evaluate_l1755_175521


namespace NUMINAMATH_GPT_arithmetic_seq_perfect_sixth_power_l1755_175530

theorem arithmetic_seq_perfect_sixth_power 
  (a h : ℤ)
  (seq : ∀ n : ℕ, ℤ)
  (h_seq : ∀ n, seq n = a + n * h)
  (h1 : ∃ s₁ x, seq s₁ = x^2)
  (h2 : ∃ s₂ y, seq s₂ = y^3) :
  ∃ k s, seq s = k^6 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_perfect_sixth_power_l1755_175530


namespace NUMINAMATH_GPT_negation_of_p_implies_a_gt_one_half_l1755_175594

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, a * x^2 + x + 1 / 2 ≤ 0

-- Define the statement that negation of p implies a > 1/2
theorem negation_of_p_implies_a_gt_one_half (a : ℝ) (h : ¬ p a) : a > 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_implies_a_gt_one_half_l1755_175594


namespace NUMINAMATH_GPT_benny_bought_books_l1755_175524

theorem benny_bought_books :
  ∀ (initial_books sold_books remaining_books bought_books : ℕ),
    initial_books = 22 →
    sold_books = initial_books / 2 →
    remaining_books = initial_books - sold_books →
    remaining_books + bought_books = 17 →
    bought_books = 6 :=
by
  intros initial_books sold_books remaining_books bought_books
  sorry

end NUMINAMATH_GPT_benny_bought_books_l1755_175524


namespace NUMINAMATH_GPT_smallest_n_l1755_175528

def n_expr (n : ℕ) : ℕ :=
  n * (2^7) * (3^2) * (7^3)

theorem smallest_n (n : ℕ) (h1: 25 ∣ n_expr n) (h2: 27 ∣ n_expr n) : n = 75 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1755_175528


namespace NUMINAMATH_GPT_range_of_a_l1755_175550

theorem range_of_a (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5 * x + 15 / 2 * a <= 0) -> a > 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1755_175550


namespace NUMINAMATH_GPT_question1_question2_l1755_175560

def energy_cost (units: ℕ) : ℝ :=
  if units <= 100 then
    units * 0.5
  else
    100 * 0.5 + (units - 100) * 0.8

theorem question1 :
  energy_cost 130 = 74 := by
  sorry

theorem question2 (units: ℕ) (H: energy_cost units = 90) :
  units = 150 := by
  sorry

end NUMINAMATH_GPT_question1_question2_l1755_175560


namespace NUMINAMATH_GPT_sale_saving_percentage_l1755_175532

theorem sale_saving_percentage (P : ℝ) : 
  let original_price := 8 * P
  let sale_price := 6 * P
  let amount_saved := original_price - sale_price
  let percentage_saved := (amount_saved / original_price) * 100
  percentage_saved = 25 :=
by
  sorry

end NUMINAMATH_GPT_sale_saving_percentage_l1755_175532


namespace NUMINAMATH_GPT_problem_BD_l1755_175518

variable (a b c : ℝ)

theorem problem_BD (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) :=
by
  sorry

end NUMINAMATH_GPT_problem_BD_l1755_175518


namespace NUMINAMATH_GPT_range_of_m_l1755_175534

variable (m : ℝ)

def hyperbola (m : ℝ) := (x y : ℝ) → (x^2 / (1 + m)) - (y^2 / (3 - m)) = 1

def eccentricity_condition (m : ℝ) := (2 / (Real.sqrt (1 + m)) > Real.sqrt 2)

theorem range_of_m (m : ℝ) (h1 : 1 + m > 0) (h2 : 3 - m > 0) (h3 : eccentricity_condition m) :
 -1 < m ∧ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1755_175534


namespace NUMINAMATH_GPT_Annette_Caitlin_total_weight_l1755_175587

variable (A C S : ℕ)

-- Conditions
axiom cond1 : C + S = 87
axiom cond2 : A = S + 8

-- Theorem
theorem Annette_Caitlin_total_weight : A + C = 95 := by
  sorry

end NUMINAMATH_GPT_Annette_Caitlin_total_weight_l1755_175587


namespace NUMINAMATH_GPT_compound_interest_calculation_l1755_175597

-- Define the variables used in the problem
def principal : ℝ := 8000
def annual_rate : ℝ := 0.05
def compound_frequency : ℕ := 1
def final_amount : ℝ := 9261
def years : ℝ := 3

-- Statement we need to prove
theorem compound_interest_calculation :
  final_amount = principal * (1 + annual_rate / compound_frequency) ^ (compound_frequency * years) :=
by 
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l1755_175597


namespace NUMINAMATH_GPT_sphere_radius_l1755_175527

theorem sphere_radius (r : ℝ) (π : ℝ)
    (h1 : Volume = (4 / 3) * π * r^3)
    (h2 : SurfaceArea = 4 * π * r^2)
    (h3 : Volume = SurfaceArea) :
    r = 3 :=
by
  -- Here starts the proof, but we use 'sorry' to skip it as per the instructions.
  sorry

end NUMINAMATH_GPT_sphere_radius_l1755_175527


namespace NUMINAMATH_GPT_find_possible_values_of_a_l1755_175540

noncomputable def find_a (x y a : ℝ) : Prop :=
  (x + y = a) ∧ (x^3 + y^3 = a) ∧ (x^5 + y^5 = a)

theorem find_possible_values_of_a (a : ℝ) :
  (∃ x y : ℝ, find_a x y a) ↔ (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_find_possible_values_of_a_l1755_175540


namespace NUMINAMATH_GPT_find_S12_l1755_175562

variable {a : Nat → Int} -- representing the arithmetic sequence {a_n}
variable {S : Nat → Int} -- representing the sums of the first n terms, S_n

-- Condition: a_1 = -9
axiom a1_def : a 1 = -9

-- Condition: (S_n / n) forms an arithmetic sequence
axiom arithmetic_s : ∃ d : Int, ∀ n : Nat, S n / n = -9 + (n - 1) * d

-- Condition: 2 = S9 / 9 - S7 / 7
axiom condition : S 9 / 9 - S 7 / 7 = 2

-- We want to prove: S_12 = 36
theorem find_S12 : S 12 = 36 := 
sorry

end NUMINAMATH_GPT_find_S12_l1755_175562


namespace NUMINAMATH_GPT_no_integer_roots_l1755_175555
open Polynomial

theorem no_integer_roots {p : ℤ[X]} (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_pa : p.eval a = 1) (h_pb : p.eval b = 1) (h_pc : p.eval c = 1) : 
  ∀ m : ℤ, p.eval m ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l1755_175555


namespace NUMINAMATH_GPT_problem_statement_l1755_175565

variable {P : ℕ → Prop}

theorem problem_statement
  (h1 : ∀ k, P k → P (k + 1))
  (h2 : ¬P 4)
  (n : ℕ) (hn : 1 ≤ n → n ≤ 4 → n ∈ Set.Icc 1 4) :
  ¬P n :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1755_175565


namespace NUMINAMATH_GPT_smaller_angle_at_7_15_l1755_175582

theorem smaller_angle_at_7_15 
  (hour_hand_rate : ℕ → ℝ)
  (minute_hand_rate : ℕ → ℝ)
  (hour_time : ℕ)
  (minute_time : ℕ)
  (top_pos : ℝ)
  (smaller_angle : ℝ) 
  (h1 : hour_hand_rate hour_time + (minute_time/60) * hour_hand_rate hour_time = 217.5)
  (h2 : minute_hand_rate minute_time = 90.0)
  (h3 : |217.5 - 90.0| = smaller_angle) :
  smaller_angle = 127.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_7_15_l1755_175582


namespace NUMINAMATH_GPT_number_of_integer_solutions_l1755_175519

theorem number_of_integer_solutions : 
  (∃ (sols : List (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ sols ↔ (1 : ℚ)/x + (1 : ℚ)/y = 1/7) ∧ sols.length = 5) := 
sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l1755_175519


namespace NUMINAMATH_GPT_angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l1755_175581

-- Definitions according to the given conditions
variables (A B C D E F M N P : Point)
  (hexagon_regular : is_regular_hexagon A B C D E F)
  (is_midpoint_M : is_midpoint M C D)
  (is_midpoint_N : is_midpoint N D E)
  (intersection_P : intersection_point P (line_through A M) (line_through B N))

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_60_degrees 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  angle (line_through A M) (line_through B N) = 60 := 
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_triangle_ABP_eq_area_quadrilateral_MDNP 
  (h1 : hexagon_regular)
  (h2 : is_midpoint_M)
  (h3 : is_midpoint_N)
  (h4 : intersection_P) :
  area (triangle A B P) = area (quadrilateral M D N P) := 
sorry

end NUMINAMATH_GPT_angle_AM_BN_60_degrees_area_triangle_ABP_eq_area_quadrilateral_MDNP_l1755_175581


namespace NUMINAMATH_GPT_solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l1755_175548

-- Case 1: a ≠ 0
theorem solve_eq_nonzero (a b : ℝ) (h : a ≠ 0) : ∃ x : ℝ, x = -b / a ∧ a * x + b = 0 :=
by
  sorry

-- Case 2: a = 0 and b = 0
theorem solve_eq_zero_zero (a b : ℝ) (h1 : a = 0) (h2 : b = 0) : ∀ x : ℝ, a * x + b = 0 :=
by
  sorry

-- Case 3: a = 0 and b ≠ 0
theorem solve_eq_zero_nonzero (a b : ℝ) (h1 : a = 0) (h2 : b ≠ 0) : ¬ ∃ x : ℝ, a * x + b = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_nonzero_solve_eq_zero_zero_solve_eq_zero_nonzero_l1755_175548


namespace NUMINAMATH_GPT_total_people_is_120_l1755_175511

def num_children : ℕ := 80

def num_adults (num_children : ℕ) : ℕ := num_children / 2

def total_people (num_children num_adults : ℕ) : ℕ := num_children + num_adults

theorem total_people_is_120 : total_people num_children (num_adults num_children) = 120 := by
  sorry

end NUMINAMATH_GPT_total_people_is_120_l1755_175511


namespace NUMINAMATH_GPT_inequality_solution_l1755_175507

noncomputable def solution_set : Set ℝ := {x : ℝ | x < 4 ∨ x > 5}

theorem inequality_solution (x : ℝ) :
  (x - 2) / (x - 4) ≤ 3 ↔ x ∈ solution_set :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1755_175507


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l1755_175574

noncomputable def cos_135_deg : Real := - (Real.sqrt 2) / 2

theorem cos_135_eq_neg_sqrt_2_div_2 : Real.cos (135 * Real.pi / 180) = cos_135_deg := sorry

noncomputable def point_Q : Real × Real :=
  (- (Real.sqrt 2) / 2, (Real.sqrt 2) / 2)

theorem point_Q_coordinates :
  ∃ (Q : Real × Real), Q = point_Q ∧ Q = (Real.cos (135 * Real.pi / 180), Real.sin (135 * Real.pi / 180)) := sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt_2_div_2_point_Q_coordinates_l1755_175574


namespace NUMINAMATH_GPT_find_number_of_children_l1755_175535

def admission_cost_adult : ℝ := 30
def admission_cost_child : ℝ := 15
def total_people : ℕ := 10
def soda_cost : ℝ := 5
def discount_rate : ℝ := 0.8
def total_paid : ℝ := 197

def total_cost_with_discount (adults children : ℕ) : ℝ :=
  discount_rate * (adults * admission_cost_adult + children * admission_cost_child)

theorem find_number_of_children (A C : ℕ) 
  (h1 : A + C = total_people)
  (h2 : total_cost_with_discount A C + soda_cost = total_paid) :
  C = 4 :=
sorry

end NUMINAMATH_GPT_find_number_of_children_l1755_175535


namespace NUMINAMATH_GPT_balance_of_three_squares_and_two_heartsuits_l1755_175516

-- Definitions
variable {x y z w : ℝ}

-- Given conditions
axiom h1 : 3 * x + 4 * y + z = 12 * w
axiom h2 : x = z + 2 * w

-- Problem to prove
theorem balance_of_three_squares_and_two_heartsuits :
  (3 * y + 2 * z) = (26 / 9) * w :=
sorry

end NUMINAMATH_GPT_balance_of_three_squares_and_two_heartsuits_l1755_175516


namespace NUMINAMATH_GPT_driers_drying_time_l1755_175510

noncomputable def drying_time (r1 r2 r3 : ℝ) : ℝ := 1 / (r1 + r2 + r3)

theorem driers_drying_time (Q : ℝ) (r1 r2 r3 : ℝ)
  (h1 : r1 = Q / 24) 
  (h2 : r2 = Q / 2) 
  (h3 : r3 = Q / 8) : 
  drying_time r1 r2 r3 = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_driers_drying_time_l1755_175510


namespace NUMINAMATH_GPT_class_funding_reached_l1755_175589

-- Definition of the conditions
def students : ℕ := 45
def goal : ℝ := 3000
def full_payment_students : ℕ := 25
def full_payment_amount : ℝ := 60
def merit_students : ℕ := 10
def merit_payment_per_student_euro : ℝ := 40
def euro_to_usd : ℝ := 1.20
def financial_needs_students : ℕ := 7
def financial_needs_payment_per_student_pound : ℝ := 30
def pound_to_usd : ℝ := 1.35
def discount_students : ℕ := 3
def discount_payment_per_student_cad : ℝ := 68
def cad_to_usd : ℝ := 0.80
def administrative_fee_yen : ℝ := 10000
def yen_to_usd : ℝ := 0.009

-- Definitions of amounts
def full_payment_amount_total : ℝ := full_payment_students * full_payment_amount
def merit_payment_amount_total : ℝ := merit_students * merit_payment_per_student_euro * euro_to_usd
def financial_needs_payment_amount_total : ℝ := financial_needs_students * financial_needs_payment_per_student_pound * pound_to_usd
def discount_payment_amount_total : ℝ := discount_students * discount_payment_per_student_cad * cad_to_usd
def administrative_fee_usd : ℝ := administrative_fee_yen * yen_to_usd

-- Definition of total collected
def total_collected : ℝ := 
  full_payment_amount_total + 
  merit_payment_amount_total + 
  financial_needs_payment_amount_total + 
  discount_payment_amount_total - 
  administrative_fee_usd

-- The final theorem statement
theorem class_funding_reached : total_collected = 2427.70 ∧ goal - total_collected = 572.30 := by
  sorry

end NUMINAMATH_GPT_class_funding_reached_l1755_175589


namespace NUMINAMATH_GPT_gcd_exponentiation_gcd_fermat_numbers_l1755_175536

-- Part (a)
theorem gcd_exponentiation (m n : ℕ) (a : ℕ) (h1 : m ≠ n) (h2 : a > 1) : 
  Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1 :=
by
sorry

-- Part (b)
def fermat_number (k : ℕ) : ℕ := 2^(2^k) + 1

theorem gcd_fermat_numbers (m n : ℕ) (h1 : m ≠ n) : 
  Nat.gcd (fermat_number m) (fermat_number n) = 1 :=
by
sorry

end NUMINAMATH_GPT_gcd_exponentiation_gcd_fermat_numbers_l1755_175536


namespace NUMINAMATH_GPT_volleyball_match_prob_A_win_l1755_175502

-- Definitions of given probabilities and conditions
def rally_scoring_system := true
def first_to_25_wins := true
def tie_at_24_24_continues_until_lead_by_2 := true
def prob_team_A_serves_win : ℚ := 2/3
def prob_team_B_serves_win : ℚ := 2/5
def outcomes_independent := true
def score_22_22_team_A_serves := true

-- The problem to prove
theorem volleyball_match_prob_A_win :
  rally_scoring_system ∧
  first_to_25_wins ∧
  tie_at_24_24_continues_until_lead_by_2 ∧
  prob_team_A_serves_win = 2/3 ∧
  prob_team_B_serves_win = 2/5 ∧
  outcomes_independent ∧
  score_22_22_team_A_serves →
  (prob_team_A_serves_win ^ 3 + (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win ^ 2 + prob_team_A_serves_win * (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win + prob_team_A_serves_win ^ 2 * (1 - prob_team_A_serves_win) * prob_team_B_serves_win) = 64/135 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_match_prob_A_win_l1755_175502


namespace NUMINAMATH_GPT_llesis_more_rice_l1755_175590

theorem llesis_more_rice :
  let total_rice := 50
  let llesis_fraction := 7 / 10
  let llesis_rice := total_rice * llesis_fraction
  let everest_rice := total_rice - llesis_rice
  llesis_rice - everest_rice = 20 := by
    sorry

end NUMINAMATH_GPT_llesis_more_rice_l1755_175590


namespace NUMINAMATH_GPT_ratio_of_capital_l1755_175573

variable (C A B : ℝ)
variable (h1 : B = 4 * C)
variable (h2 : B / (A + 5 * C) = 6000 / 16500)

theorem ratio_of_capital : A / B = 17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_capital_l1755_175573


namespace NUMINAMATH_GPT_lcm_16_24_l1755_175563

/-
  Prove that the least common multiple (LCM) of 16 and 24 is 48.
-/
theorem lcm_16_24 : Nat.lcm 16 24 = 48 :=
by
  sorry

end NUMINAMATH_GPT_lcm_16_24_l1755_175563


namespace NUMINAMATH_GPT_solve_inequality_l1755_175558

theorem solve_inequality :
  { x : ℝ | x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ x ≠ 7 ∧ 
    (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) } = 
  { x : ℝ | (x < -8) ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ (x > 8) } := sorry

end NUMINAMATH_GPT_solve_inequality_l1755_175558


namespace NUMINAMATH_GPT_find_x_axis_intercept_l1755_175585

theorem find_x_axis_intercept : ∃ x, 5 * 0 - 6 * x = 15 ∧ x = -2.5 := by
  -- The theorem states that there exists an x-intercept such that substituting y = 0 in the equation results in x = -2.5.
  sorry

end NUMINAMATH_GPT_find_x_axis_intercept_l1755_175585


namespace NUMINAMATH_GPT_derivative_value_at_pi_over_2_l1755_175508

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem derivative_value_at_pi_over_2 : deriv f (Real.pi / 2) = -1 :=
by
  sorry

end NUMINAMATH_GPT_derivative_value_at_pi_over_2_l1755_175508


namespace NUMINAMATH_GPT_david_course_hours_l1755_175531

def total_course_hours (weeks : ℕ) (class_hours_per_week : ℕ) (homework_hours_per_week : ℕ) : ℕ :=
  weeks * (class_hours_per_week + homework_hours_per_week)

theorem david_course_hours :
  total_course_hours 24 (3 + 3 + 4) 4 = 336 :=
by
  sorry

end NUMINAMATH_GPT_david_course_hours_l1755_175531


namespace NUMINAMATH_GPT_max_value_in_range_l1755_175554

noncomputable def x_range : Set ℝ := {x | -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 3}

noncomputable def expression (x : ℝ) : ℝ :=
  Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)

theorem max_value_in_range :
  ∀ x ∈ x_range, expression x ≤ (11 / 6) * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_value_in_range_l1755_175554


namespace NUMINAMATH_GPT_abs_neg_three_eq_three_l1755_175513

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_abs_neg_three_eq_three_l1755_175513


namespace NUMINAMATH_GPT_b_95_mod_49_l1755_175500

def b (n : ℕ) : ℕ := 5^n + 7^n + 3

theorem b_95_mod_49 : b 95 % 49 = 5 := 
by sorry

end NUMINAMATH_GPT_b_95_mod_49_l1755_175500


namespace NUMINAMATH_GPT_rabbits_total_distance_l1755_175553

theorem rabbits_total_distance :
  let white_speed := 15
  let brown_speed := 12
  let grey_speed := 18
  let black_speed := 10
  let time := 7
  let white_distance := white_speed * time
  let brown_distance := brown_speed * time
  let grey_distance := grey_speed * time
  let black_distance := black_speed * time
  let total_distance := white_distance + brown_distance + grey_distance + black_distance
  total_distance = 385 :=
by
  sorry

end NUMINAMATH_GPT_rabbits_total_distance_l1755_175553


namespace NUMINAMATH_GPT_prop1_prop2_prop3_l1755_175575

variables (a b c d : ℝ)

-- Proposition 1: ab > 0 ∧ bc - ad > 0 → (c/a - d/b > 0)
theorem prop1 (h1 : a * b > 0) (h2 : b * c - a * d > 0) : c / a - d / b > 0 :=
sorry

-- Proposition 2: ab > 0 ∧ (c/a - d/b > 0) → bc - ad > 0
theorem prop2 (h1 : a * b > 0) (h2 : c / a - d / b > 0) : b * c - a * d > 0 :=
sorry

-- Proposition 3: (bc - ad > 0) ∧ (c/a - d/b > 0) → ab > 0
theorem prop3 (h1 : b * c - a * d > 0) (h2 : c / a - d / b > 0) : a * b > 0 :=
sorry

end NUMINAMATH_GPT_prop1_prop2_prop3_l1755_175575


namespace NUMINAMATH_GPT_asymptote_equation_l1755_175514

theorem asymptote_equation {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  (a + Real.sqrt (a^2 + b^2) = 2 * b) →
  (4 * x = 3 * y) ∨ (4 * x = -3 * y) :=
by
  sorry

end NUMINAMATH_GPT_asymptote_equation_l1755_175514


namespace NUMINAMATH_GPT_average_speed_of_train_l1755_175506

theorem average_speed_of_train (x : ℝ) (h1 : 0 < x) : 
  let Time1 := x / 40
  let Time2 := x / 10
  let TotalDistance := 3 * x
  let TotalTime := x / 8
  (TotalDistance / TotalTime = 24) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l1755_175506


namespace NUMINAMATH_GPT_junk_items_count_l1755_175578

variable (total_items : ℕ)
variable (useful_percentage : ℚ := 0.20)
variable (heirloom_percentage : ℚ := 0.10)
variable (junk_percentage : ℚ := 0.70)
variable (useful_items : ℕ := 8)

theorem junk_items_count (huseful : useful_percentage * total_items = useful_items) : 
  junk_percentage * total_items = 28 :=
by
  sorry

end NUMINAMATH_GPT_junk_items_count_l1755_175578


namespace NUMINAMATH_GPT_problem_A_problem_C_problem_D_problem_E_l1755_175596

variable {a b c : ℝ}
variable (ha : a < 0) (hab : a < b) (hb : b < 0) (hc : 0 < c)

theorem problem_A (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * b > a * c :=
by sorry

theorem problem_C (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a * c < b * c :=
by sorry

theorem problem_D (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : a + c < b + c :=
by sorry

theorem problem_E (h : a < 0 ∧ a < b ∧ b < 0 ∧ 0 < c) : c / a > 1 :=
by sorry

end NUMINAMATH_GPT_problem_A_problem_C_problem_D_problem_E_l1755_175596


namespace NUMINAMATH_GPT_bottle_caps_weight_l1755_175501

theorem bottle_caps_weight :
  (∀ n : ℕ, n = 7 → 1 = 1) → -- 7 bottle caps weigh exactly 1 ounce
  (∀ m : ℕ, m = 2016 → 1 = 1) → -- Josh has 2016 bottle caps
  2016 / 7 = 288 := -- The weight of Josh's entire bottle cap collection is 288 ounces
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_bottle_caps_weight_l1755_175501


namespace NUMINAMATH_GPT_largest_integral_value_of_y_l1755_175538

theorem largest_integral_value_of_y : 
  (1 / 4 : ℝ) < (y / 7 : ℝ) ∧ (y / 7 : ℝ) < (3 / 5 : ℝ) → y ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integral_value_of_y_l1755_175538


namespace NUMINAMATH_GPT_area_before_halving_l1755_175533

theorem area_before_halving (A : ℝ) (h : A / 2 = 7) : A = 14 :=
sorry

end NUMINAMATH_GPT_area_before_halving_l1755_175533


namespace NUMINAMATH_GPT_smallest_of_x_y_z_l1755_175526

variables {a b c d : ℕ}

/-- Given that x, y, and z are in the ratio a, b, c respectively, 
    and their sum x + y + z equals d, and 0 < a < b < c,
    prove that the smallest of x, y, and z is da / (a + b + c). -/
theorem smallest_of_x_y_z (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : 0 < d)
    (h_sum : ∀ k : ℚ, x = k * a → y = k * b → z = k * c → x + y + z = d) : 
    (∃ k : ℚ, x = k * a ∧ y = k * b ∧ z = k * c ∧ k = d / (a + b + c) ∧ x = da / (a + b + c)) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_of_x_y_z_l1755_175526


namespace NUMINAMATH_GPT_intersection_A_B_l1755_175580

-- Conditions
def A : Set ℝ := {1, 2, 0.5}
def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = x^2}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {1} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1755_175580


namespace NUMINAMATH_GPT_amount_of_bill_l1755_175525

theorem amount_of_bill (TD R FV T : ℝ) (hTD : TD = 270) (hR : R = 16) (hT : T = 9/12) 
(h_formula : TD = (R * T * FV) / (100 + (R * T))) : FV = 2520 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_bill_l1755_175525


namespace NUMINAMATH_GPT_value_of_a_l1755_175569

theorem value_of_a (x y a : ℝ) (h1 : x - 2 * y = a - 6) (h2 : 2 * x + 5 * y = 2 * a) (h3 : x + y = 9) : a = 11 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1755_175569


namespace NUMINAMATH_GPT_number_of_liars_on_the_island_l1755_175592

-- Definitions for the conditions
def isKnight (person : ℕ) : Prop := sorry -- Placeholder, we know knights always tell the truth
def isLiar (person : ℕ) : Prop := sorry -- Placeholder, we know liars always lie
def population := 1000
def villages := 10
def minInhabitantsPerVillage := 2

-- Definitional property: each islander claims that all other villagers in their village are liars
def claimsAllOthersAreLiars (islander : ℕ) (village : ℕ) : Prop := 
  ∀ (other : ℕ), (other ≠ islander) → (isLiar other)

-- Main statement in Lean
theorem number_of_liars_on_the_island : ∃ liars, liars = 990 :=
by
  have total_population := population
  have number_of_villages := villages
  have min_people_per_village := minInhabitantsPerVillage
  have knight_prop := isKnight
  have liar_prop := isLiar
  have claim_prop := claimsAllOthersAreLiars
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_number_of_liars_on_the_island_l1755_175592


namespace NUMINAMATH_GPT_polygon_sides_diagonals_l1755_175576

theorem polygon_sides_diagonals (n : ℕ) 
  (h1 : 4 * (n * (n - 3)) = 14 * n)
  (h2 : (n + (n * (n - 3)) / 2) % 2 = 0)
  (h3 : n + n * (n - 3) / 2 > 50) : n = 12 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_diagonals_l1755_175576


namespace NUMINAMATH_GPT_ring_worth_l1755_175595

theorem ring_worth (R : ℝ) (h1 : (R + 2000 + 2 * R = 14000)) : R = 4000 :=
by 
  sorry

end NUMINAMATH_GPT_ring_worth_l1755_175595


namespace NUMINAMATH_GPT_average_large_basket_weight_l1755_175598

-- Definitions derived from the conditions
def small_basket_capacity := 25  -- Capacity of each small basket in kilograms
def num_small_baskets := 28      -- Number of small baskets used
def num_large_baskets := 10      -- Number of large baskets used
def leftover_weight := 50        -- Leftover weight in kilograms

-- Statement of the problem
theorem average_large_basket_weight :
  (small_basket_capacity * num_small_baskets - leftover_weight) / num_large_baskets = 65 :=
by
  sorry

end NUMINAMATH_GPT_average_large_basket_weight_l1755_175598


namespace NUMINAMATH_GPT_circle_through_A_B_C_l1755_175539

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (1, 12)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (-9, 2)

-- Definition of the expected standard equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 100

-- Theorem stating that the expected equation is the equation of the circle through points A, B, and C
theorem circle_through_A_B_C : 
  ∀ (x y : ℝ),
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → 
  circle_eq x y := sorry

end NUMINAMATH_GPT_circle_through_A_B_C_l1755_175539


namespace NUMINAMATH_GPT_hexagon_ratio_l1755_175520

theorem hexagon_ratio (A B : ℝ) (h₁ : A = 8) (h₂ : B = 2)
                      (A_above : ℝ) (h₃ : A_above = (3 + B))
                      (H : 3 + B = 1 / 2 * (A + B)) 
                      (XQ QY : ℝ) (h₄ : XQ + QY = 4)
                      (h₅ : 3 + B = 4 + B / 2) :
  XQ / QY = 2 := 
by
  sorry

end NUMINAMATH_GPT_hexagon_ratio_l1755_175520


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_33_l1755_175541

theorem smallest_four_digit_divisible_by_33 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 33 = 0 ∧ n = 1023 := by 
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_33_l1755_175541


namespace NUMINAMATH_GPT_graduate_degree_ratio_l1755_175599

theorem graduate_degree_ratio (G C N : ℕ) (h1 : C = (2 / 3 : ℚ) * N)
  (h2 : (G : ℚ) / (G + C) = 0.15789473684210525) :
  (G : ℚ) / N = 1 / 8 :=
  sorry

end NUMINAMATH_GPT_graduate_degree_ratio_l1755_175599


namespace NUMINAMATH_GPT_sequence_b_l1755_175572

theorem sequence_b (b : ℕ → ℕ) 
  (h1 : b 1 = 2) 
  (h2 : ∀ m n : ℕ, b (m + n) = b m + b n + 2 * m * n) : 
  b 10 = 110 :=
sorry

end NUMINAMATH_GPT_sequence_b_l1755_175572


namespace NUMINAMATH_GPT_son_working_alone_l1755_175568

theorem son_working_alone (M S : ℝ) (h1: M = 1 / 5) (h2: M + S = 1 / 3) : 1 / S = 7.5 :=
  by
  sorry

end NUMINAMATH_GPT_son_working_alone_l1755_175568


namespace NUMINAMATH_GPT_total_fishermen_count_l1755_175543

theorem total_fishermen_count (F T F1 F2 : ℕ) (hT : T = 10000) (hF1 : F1 = 19 * 400) (hF2 : F2 = 2400) (hTotal : F1 + F2 = T) : F = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_fishermen_count_l1755_175543


namespace NUMINAMATH_GPT_gcd_14m_21n_126_l1755_175547

theorem gcd_14m_21n_126 {m n : ℕ} (hm_pos : 0 < m) (hn_pos : 0 < n) (h_gcd : Nat.gcd m n = 18) : 
  Nat.gcd (14 * m) (21 * n) = 126 :=
by
  sorry

end NUMINAMATH_GPT_gcd_14m_21n_126_l1755_175547


namespace NUMINAMATH_GPT_coefficient_of_determination_indicates_better_fit_l1755_175566

theorem coefficient_of_determination_indicates_better_fit (R_squared : ℝ) (h1 : 0 ≤ R_squared) (h2 : R_squared ≤ 1) :
  R_squared = 1 → better_fitting_effect_of_regression_model :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_determination_indicates_better_fit_l1755_175566


namespace NUMINAMATH_GPT_problem_statement_l1755_175549

noncomputable def seq_sub_triples: ℚ :=
  let a := (5 / 6 : ℚ)
  let b := (1 / 6 : ℚ)
  let c := (1 / 4 : ℚ)
  a - b - c

theorem problem_statement : seq_sub_triples = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1755_175549


namespace NUMINAMATH_GPT_Kolya_walking_speed_l1755_175593

theorem Kolya_walking_speed
  (x : ℝ) 
  (h1 : x > 0) 
  (t_closing : ℝ := (3 * x) / 10) 
  (t_travel : ℝ := ((x / 10) + (x / 20))) 
  (remaining_time : ℝ := t_closing - t_travel)
  (walking_speed : ℝ := x / remaining_time)
  (correct_speed : ℝ := 20 / 3) :
  walking_speed = correct_speed := 
by 
  sorry

end NUMINAMATH_GPT_Kolya_walking_speed_l1755_175593


namespace NUMINAMATH_GPT_minimum_value_of_f_l1755_175557

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem minimum_value_of_f :
  f 2 = -3 ∧ (∀ x : ℝ, f x ≥ -3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1755_175557


namespace NUMINAMATH_GPT_average_age_combined_l1755_175522

-- Definitions of the given conditions
def avg_age_fifth_graders := 10
def number_fifth_graders := 40
def avg_age_parents := 40
def number_parents := 60

-- The theorem we need to prove
theorem average_age_combined : 
  (avg_age_fifth_graders * number_fifth_graders + avg_age_parents * number_parents) / (number_fifth_graders + number_parents) = 28 := 
by
  sorry

end NUMINAMATH_GPT_average_age_combined_l1755_175522


namespace NUMINAMATH_GPT_person_savings_l1755_175584

theorem person_savings (income expenditure savings : ℝ) 
  (h1 : income = 18000)
  (h2 : income / expenditure = 5 / 4)
  (h3 : savings = income - expenditure) : 
  savings = 3600 := 
sorry

end NUMINAMATH_GPT_person_savings_l1755_175584


namespace NUMINAMATH_GPT_no_positive_rational_solutions_l1755_175561

theorem no_positive_rational_solutions (n : ℕ) (h_pos_n : 0 < n) : 
  ¬ ∃ (x y : ℚ) (h_x_pos : 0 < x) (h_y_pos : 0 < y), x + y + (1/x) + (1/y) = 3 * n :=
by
  sorry

end NUMINAMATH_GPT_no_positive_rational_solutions_l1755_175561


namespace NUMINAMATH_GPT_min_value_of_f_min_value_at_x_1_l1755_175505

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + 1 / (2 - 3 * x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 35 :=
by
  sorry

-- As an additional statement, we can check the specific case at x = 1
theorem min_value_at_x_1 :
  f 1 = 35 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_min_value_at_x_1_l1755_175505


namespace NUMINAMATH_GPT_total_pages_to_read_l1755_175571

theorem total_pages_to_read 
  (total_books : ℕ)
  (pages_per_book : ℕ)
  (books_read_first_month : ℕ)
  (books_remaining_second_month : ℕ) :
  total_books = 14 →
  pages_per_book = 200 →
  books_read_first_month = 4 →
  books_remaining_second_month = (total_books - books_read_first_month) / 2 →
  ((total_books * pages_per_book) - ((books_read_first_month + books_remaining_second_month) * pages_per_book) = 1000) :=
by
  sorry

end NUMINAMATH_GPT_total_pages_to_read_l1755_175571


namespace NUMINAMATH_GPT_area_of_sector_l1755_175577

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_sector_l1755_175577


namespace NUMINAMATH_GPT_total_time_naomi_30webs_l1755_175591

-- Define the constants based on the given conditions
def time_katherine : ℕ := 20
def factor_naomi : ℚ := 5/4
def websites : ℕ := 30

-- Define the time taken by Naomi to build one website based on the conditions
def time_naomi (time_katherine : ℕ) (factor_naomi : ℚ) : ℚ :=
  factor_naomi * time_katherine

-- Define the total time Naomi took to build all websites
def total_time_naomi (time_naomi : ℚ) (websites : ℕ) : ℚ :=
  time_naomi * websites

-- Statement: Proving that the total number of hours Naomi took to create 30 websites is 750
theorem total_time_naomi_30webs : 
  total_time_naomi (time_naomi time_katherine factor_naomi) websites = 750 := 
sorry

end NUMINAMATH_GPT_total_time_naomi_30webs_l1755_175591


namespace NUMINAMATH_GPT_area_bounded_by_curve_and_line_l1755_175503

theorem area_bounded_by_curve_and_line :
  let curve_x (t : ℝ) := 10 * (t - Real.sin t)
  let curve_y (t : ℝ) := 10 * (1 - Real.cos t)
  let y_line := 15
  (∫ t in (2/3) * Real.pi..(4/3) * Real.pi, 100 * (1 - Real.cos t)^2) = 100 * Real.pi + 200 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_area_bounded_by_curve_and_line_l1755_175503


namespace NUMINAMATH_GPT_slopes_product_no_circle_MN_A_l1755_175546

-- Define the equation of the ellipse E and the specific points A and B
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the point P which lies on the ellipse
def P (x0 y0 : ℝ) : Prop := ellipse_eq x0 y0 ∧ x0 ≠ -2 ∧ x0 ≠ 2

-- Prove the product of the slopes of lines PA and PB
theorem slopes_product (x0 y0 : ℝ) (hP : P x0 y0) : 
  (y0 / (x0 + 2)) * (y0 / (x0 - 2)) = -1 / 4 := sorry

-- Define point Q
def Q : ℝ × ℝ := (-1, 0)

-- Define points M and N which are intersections of line and ellipse
def MN_line (t y : ℝ) : ℝ := t * y - 1

-- Prove there is no circle with diameter MN passing through A
theorem no_circle_MN_A (t : ℝ) : 
  ¬ ∃ M N : ℝ × ℝ, ellipse_eq M.1 M.2 ∧ ellipse_eq N.1 N.2 ∧
  (∃ x1 y1 x2 y2, (M = (x1, y1) ∧ N = (x2, y2)) ∧
  (MN_line t y1 = x1 ∧ MN_line t y2 = x2) ∧ 
  ((x1 + 2) * (x2 + 2) + y1 * y2 = 0)) := sorry

end NUMINAMATH_GPT_slopes_product_no_circle_MN_A_l1755_175546


namespace NUMINAMATH_GPT_roots_cubic_l1755_175545

theorem roots_cubic (a b c d r s t : ℂ) 
    (h1 : a ≠ 0)
    (h2 : r + s + t = -b / a)
    (h3 : r * s + r * t + s * t = c / a)
    (h4 : r * s * t = -d / a) :
    (1 / r^2) + (1 / s^2) + (1 / t^2) = (b^2 - 2 * a * c) / (d^2) :=
by
    sorry

end NUMINAMATH_GPT_roots_cubic_l1755_175545


namespace NUMINAMATH_GPT_part_a_part_b_l1755_175551

-- Part (a)
theorem part_a (a b : ℕ) (h : Nat.lcm a (a + 5) = Nat.lcm b (b + 5)) : a = b :=
sorry

-- Part (b)
theorem part_b (a b c : ℕ) (gcd_abc : Nat.gcd a (Nat.gcd b c) = 1) :
  Nat.lcm a b = Nat.lcm (a + c) (b + c) → False :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l1755_175551


namespace NUMINAMATH_GPT_smallest_even_natural_number_l1755_175567

theorem smallest_even_natural_number (a : ℕ) :
  ( ∃ a, a % 2 = 0 ∧
    (a + 1) % 3 = 0 ∧
    (a + 2) % 5 = 0 ∧
    (a + 3) % 7 = 0 ∧
    (a + 4) % 11 = 0 ∧
    (a + 5) % 13 = 0 ) → 
  a = 788 := by
  sorry

end NUMINAMATH_GPT_smallest_even_natural_number_l1755_175567


namespace NUMINAMATH_GPT_sara_gave_dan_limes_l1755_175529

theorem sara_gave_dan_limes (initial_limes : ℕ) (final_limes : ℕ) (d : ℕ) 
  (h1: initial_limes = 9) (h2: final_limes = 13) (h3: final_limes = initial_limes + d) : d = 4 := 
by sorry

end NUMINAMATH_GPT_sara_gave_dan_limes_l1755_175529


namespace NUMINAMATH_GPT_polynomial_factors_sum_l1755_175515

open Real

theorem polynomial_factors_sum
  (a b c : ℝ)
  (h1 : ∀ x, (x^2 + x + 2) * (a * x + b - a) + (c - a - b) * x + 5 + 2 * a - 2 * b = 0)
  (h2 : a * (1/2)^3 + b * (1/2)^2 + c * (1/2) - 25/16 = 0) :
  a + b + c = 45 / 11 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factors_sum_l1755_175515


namespace NUMINAMATH_GPT_flowers_per_bouquet_l1755_175523

theorem flowers_per_bouquet :
  let red_seeds := 125
  let yellow_seeds := 125
  let orange_seeds := 125
  let purple_seeds := 125
  let red_killed := 45
  let yellow_killed := 61
  let orange_killed := 30
  let purple_killed := 40
  let bouquets := 36
  let red_flowers := red_seeds - red_killed
  let yellow_flowers := yellow_seeds - yellow_killed
  let orange_flowers := orange_seeds - orange_killed
  let purple_flowers := purple_seeds - purple_killed
  let total_flowers := red_flowers + yellow_flowers + orange_flowers + purple_flowers
  let flowers_per_bouquet := total_flowers / bouquets
  flowers_per_bouquet = 9 :=
by
  sorry

end NUMINAMATH_GPT_flowers_per_bouquet_l1755_175523


namespace NUMINAMATH_GPT_total_number_of_apples_l1755_175564

namespace Apples

def red_apples : ℕ := 7
def green_apples : ℕ := 2
def total_apples : ℕ := red_apples + green_apples

theorem total_number_of_apples : total_apples = 9 := by
  -- Definition of total_apples is used directly from conditions.
  -- Conditions state there are 7 red apples and 2 green apples.
  -- Therefore, total_apples = 7 + 2 = 9.
  sorry

end Apples

end NUMINAMATH_GPT_total_number_of_apples_l1755_175564


namespace NUMINAMATH_GPT_price_arun_paid_l1755_175570

theorem price_arun_paid 
  (original_price : ℝ)
  (standard_concession_rate : ℝ) 
  (additional_concession_rate : ℝ)
  (reduced_price : ℝ)
  (final_price : ℝ) 
  (h1 : original_price = 2000)
  (h2 : standard_concession_rate = 0.30)
  (h3 : additional_concession_rate = 0.20)
  (h4 : reduced_price = original_price * (1 - standard_concession_rate))
  (h5 : final_price = reduced_price * (1 - additional_concession_rate)) :
  final_price = 1120 :=
by
  sorry

end NUMINAMATH_GPT_price_arun_paid_l1755_175570


namespace NUMINAMATH_GPT_neg_all_cups_full_l1755_175544

variable (x : Type) (cup : x → Prop) (full : x → Prop)

theorem neg_all_cups_full :
  ¬ (∀ x, cup x → full x) = ∃ x, cup x ∧ ¬ full x := by
sorry

end NUMINAMATH_GPT_neg_all_cups_full_l1755_175544


namespace NUMINAMATH_GPT_factors_180_count_l1755_175542

theorem factors_180_count : 
  ∃ (n : ℕ), 180 = 2^2 * 3^2 * 5^1 ∧ n = 18 ∧ 
  ∀ p a b c, 
  180 = p^a * p^b * p^c →
  (a+1) * (b+1) * (c+1) = 18 :=
by {
  sorry
}

end NUMINAMATH_GPT_factors_180_count_l1755_175542


namespace NUMINAMATH_GPT_smallest_integer_l1755_175504

theorem smallest_integer :
  ∃ (M : ℕ), M > 0 ∧
             M % 3 = 2 ∧
             M % 4 = 3 ∧
             M % 5 = 4 ∧
             M % 6 = 5 ∧
             M % 7 = 6 ∧
             M % 11 = 10 ∧
             M = 4619 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_l1755_175504


namespace NUMINAMATH_GPT_BANANA_arrangements_l1755_175509

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end NUMINAMATH_GPT_BANANA_arrangements_l1755_175509


namespace NUMINAMATH_GPT_external_tangent_twice_internal_tangent_l1755_175556

noncomputable def distance_between_centers (r R : ℝ) : ℝ :=
  Real.sqrt (R^2 + r^2 + (10/3) * R * r)

theorem external_tangent_twice_internal_tangent 
  (r R O₁O₂ AB CD : ℝ)
  (h₁ : AB = 2 * CD)
  (h₂ : AB^2 = O₁O₂^2 - (R - r)^2)
  (h₃ : CD^2 = O₁O₂^2 - (R + r)^2) :
  O₁O₂ = distance_between_centers r R :=
by
  sorry

end NUMINAMATH_GPT_external_tangent_twice_internal_tangent_l1755_175556


namespace NUMINAMATH_GPT_spherical_cap_surface_area_l1755_175517

theorem spherical_cap_surface_area (V : ℝ) (h : ℝ) (A : ℝ) (r : ℝ) 
  (volume_eq : V = (4 / 3) * π * r^3) 
  (cap_height : h = 2) 
  (sphere_volume : V = 288 * π) 
  (cap_surface_area : A = 2 * π * r * h) : 
  A = 24 * π := 
sorry

end NUMINAMATH_GPT_spherical_cap_surface_area_l1755_175517


namespace NUMINAMATH_GPT_boric_acid_solution_l1755_175552

theorem boric_acid_solution
  (amount_first_solution: ℝ) (percentage_first_solution: ℝ)
  (amount_second_solution: ℝ) (percentage_second_solution: ℝ)
  (final_amount: ℝ) (final_percentage: ℝ)
  (h1: amount_first_solution = 15)
  (h2: percentage_first_solution = 0.01)
  (h3: amount_second_solution = 15)
  (h4: final_amount = 30)
  (h5: final_percentage = 0.03)
  : percentage_second_solution = 0.05 := 
by
  sorry

end NUMINAMATH_GPT_boric_acid_solution_l1755_175552
