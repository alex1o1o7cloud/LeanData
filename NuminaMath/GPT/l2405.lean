import Mathlib

namespace NUMINAMATH_GPT_boys_skip_count_l2405_240528

theorem boys_skip_count 
  (x y : ℕ)
  (avg_jumps_boys : ℕ := 85)
  (avg_jumps_girls : ℕ := 92)
  (avg_jumps_all : ℕ := 88)
  (h1 : x = y + 10)
  (h2 : (85 * x + 92 * y) / (x + y) = 88) : x = 40 :=
  sorry

end NUMINAMATH_GPT_boys_skip_count_l2405_240528


namespace NUMINAMATH_GPT_no_square_number_divisible_by_six_between_50_and_120_l2405_240569

theorem no_square_number_divisible_by_six_between_50_and_120 :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n * n) ∧ (x % 6 = 0) ∧ (50 < x ∧ x < 120) := 
sorry

end NUMINAMATH_GPT_no_square_number_divisible_by_six_between_50_and_120_l2405_240569


namespace NUMINAMATH_GPT_min_value_of_x_plus_2y_l2405_240525

noncomputable def min_value_condition (x y : ℝ) : Prop :=
x > -1 ∧ y > 0 ∧ (1 / (x + 1) + 2 / y = 1)

theorem min_value_of_x_plus_2y (x y : ℝ) (h : min_value_condition x y) : x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_2y_l2405_240525


namespace NUMINAMATH_GPT_cos_sq_alpha_cos_sq_beta_range_l2405_240523

theorem cos_sq_alpha_cos_sq_beta_range
  (α β : ℝ)
  (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  (Real.cos α)^2 + (Real.cos β)^2 ∈ Set.Icc (14 / 9) 2 :=
sorry

end NUMINAMATH_GPT_cos_sq_alpha_cos_sq_beta_range_l2405_240523


namespace NUMINAMATH_GPT_total_crayons_l2405_240544

def box1_crayons := 3 * (8 + 4 + 5)
def box2_crayons := 4 * (7 + 6 + 3)
def box3_crayons := 2 * (11 + 5 + 2)
def unique_box_crayons := 9 + 2 + 7

theorem total_crayons : box1_crayons + box2_crayons + box3_crayons + unique_box_crayons = 169 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l2405_240544


namespace NUMINAMATH_GPT_general_term_formula_l2405_240506

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)
variable (a1 d : ℤ)

-- Given conditions
axiom a2_eq : a 2 = 8
axiom S10_eq : S 10 = 185
axiom S_def : ∀ n, S n = n * (a 1 + a n) / 2
axiom a_def : ∀ n, a (n + 1) = a 1 + n * d

-- Prove the general term formula
theorem general_term_formula : a n = 3 * n + 2 := sorry

end NUMINAMATH_GPT_general_term_formula_l2405_240506


namespace NUMINAMATH_GPT_uncommon_card_cost_l2405_240571

/--
Tom's deck contains 19 rare cards, 11 uncommon cards, and 30 common cards.
Each rare card costs $1.
Each common card costs $0.25.
The total cost of the deck is $32.
Prove that the cost of each uncommon card is $0.50.
-/
theorem uncommon_card_cost (x : ℝ): 
  let rare_count := 19
  let uncommon_count := 11
  let common_count := 30
  let rare_cost := 1
  let common_cost := 0.25
  let total_cost := 32
  (rare_count * rare_cost) + (common_count * common_cost) + (uncommon_count * x) = total_cost 
  → x = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_uncommon_card_cost_l2405_240571


namespace NUMINAMATH_GPT_prism_faces_even_or_odd_l2405_240522

theorem prism_faces_even_or_odd (n : ℕ) (hn : 3 ≤ n) : ¬ (2 + n) % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_prism_faces_even_or_odd_l2405_240522


namespace NUMINAMATH_GPT_minimum_beta_value_l2405_240593

variable (α β : Real)

-- Defining the conditions given in the problem
def sin_alpha_condition : Prop := Real.sin α = -Real.sqrt 2 / 2
def cos_alpha_minus_beta_condition : Prop := Real.cos (α - β) = 1 / 2
def beta_greater_than_zero : Prop := β > 0

-- The theorem to be proven
theorem minimum_beta_value (h1 : sin_alpha_condition α) (h2 : cos_alpha_minus_beta_condition α β) (h3 : beta_greater_than_zero β) : β = Real.pi / 12 := 
sorry

end NUMINAMATH_GPT_minimum_beta_value_l2405_240593


namespace NUMINAMATH_GPT_least_number_divisible_l2405_240586

theorem least_number_divisible (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 9 = 4) (h3 : n % 18 = 4) : n = 130 := sorry

end NUMINAMATH_GPT_least_number_divisible_l2405_240586


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_sum_l2405_240575

variables {a_n S_n b_n T_n : ℕ → ℚ} {a_3 S_3 a_5 b_3 T_3 : ℚ} {q : ℚ}

def is_arithmetic_sequence (a_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, a_n n = a_1 + (n - 1) * d

def sum_first_n_arithmetic (S_n : ℕ → ℚ) (a_1 d : ℚ) : Prop :=
∀ n, S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

def is_geometric_sequence (b_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, b_n n = b_1 * q^(n-1)

def sum_first_n_geometric (T_n : ℕ → ℚ) (b_1 q : ℚ) : Prop :=
∀ n, T_n n = if q = 1 then n * b_1 else b_1 * (1 - q^n) / (1 - q)

theorem arithmetic_sequence_formula {a_1 d : ℚ} (h_arith : is_arithmetic_sequence a_n a_1 d)
    (h_sum : sum_first_n_arithmetic S_n a_1 d) (h1 : a_n 3 = 5) (h2 : S_n 3 = 9) :
    ∀ n, a_n n = 2 * n - 1 := sorry

theorem geometric_sequence_sum {b_1 : ℚ} (h_geom : is_geometric_sequence b_n b_1 q)
    (h_sum : sum_first_n_geometric T_n b_1 q) (h3 : q > 0) (h4 : b_n 3 = a_n 5) (h5 : T_n 3 = 13) :
    ∀ n, T_n n = (3^n - 1) / 2 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_geometric_sequence_sum_l2405_240575


namespace NUMINAMATH_GPT_max_amount_xiao_li_spent_l2405_240548

theorem max_amount_xiao_li_spent (a m n : ℕ) :
  33 ≤ m ∧ m < n ∧ n ≤ 37 ∧
  ∃ (x y : ℕ), 
  (25 * (a - x) + m * (a - y) + n * (x + y + a) = 700) ∧ 
  (25 * x + m * y + n * (3*a - x - y) = 1200) ∧
  ( 675 <= 700 - 25) :=
sorry

end NUMINAMATH_GPT_max_amount_xiao_li_spent_l2405_240548


namespace NUMINAMATH_GPT_percentage_reduction_in_price_l2405_240573

-- Definitions for the conditions in the problem
def reduced_price_per_kg : ℕ := 30
def extra_oil_obtained_kg : ℕ := 10
def total_money_spent : ℕ := 1500

-- Definition of the original price per kg of oil
def original_price_per_kg : ℕ := 75

-- Statement to prove the percentage reduction
theorem percentage_reduction_in_price : 
  (original_price_per_kg - reduced_price_per_kg) * 100 / original_price_per_kg = 60 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_l2405_240573


namespace NUMINAMATH_GPT_proof_intersection_complement_l2405_240535

open Set

variable (U : Set ℝ) (A B : Set ℝ)

theorem proof_intersection_complement:
  U = univ ∧ A = {x | -1 < x ∧ x ≤ 5} ∧ B = {x | x < 2} →
  A ∩ (U \ B) = {x | 2 ≤ x ∧ x ≤ 5} :=
by
  intros h
  rcases h with ⟨hU, hA, hB⟩
  simp [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_proof_intersection_complement_l2405_240535


namespace NUMINAMATH_GPT_number_of_valid_sequences_l2405_240584

/--
The measures of the interior angles of a convex pentagon form an increasing arithmetic sequence.
Determine the number of such sequences possible if the pentagon is not equiangular, all of the angle
degree measures are positive integers less than 150 degrees, and the smallest angle is at least 60 degrees.
-/

theorem number_of_valid_sequences : ∃ n : ℕ, n = 5 ∧
  ∀ (x d : ℕ),
  x ≥ 60 ∧ x + 4 * d < 150 ∧ 5 * x + 10 * d = 540 ∧ (x + d ≠ x + 2 * d) := 
sorry

end NUMINAMATH_GPT_number_of_valid_sequences_l2405_240584


namespace NUMINAMATH_GPT_circle_units_diff_l2405_240518

-- Define the context where we verify the claim about the circle

noncomputable def radius : ℝ := 3
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Lean Theorem statement that needs to be proved
theorem circle_units_diff (r : ℝ) (h₀ : r = radius) :
  circumference r ≠ area r :=
by sorry

end NUMINAMATH_GPT_circle_units_diff_l2405_240518


namespace NUMINAMATH_GPT_daria_amount_owed_l2405_240566

variable (savings : ℝ)
variable (couch_price : ℝ)
variable (table_price : ℝ)
variable (lamp_price : ℝ)
variable (total_cost : ℝ)
variable (amount_owed : ℝ)

theorem daria_amount_owed (h_savings : savings = 500)
                          (h_couch : couch_price = 750)
                          (h_table : table_price = 100)
                          (h_lamp : lamp_price = 50)
                          (h_total_cost : total_cost = couch_price + table_price + lamp_price)
                          (h_amount_owed : amount_owed = total_cost - savings) :
                          amount_owed = 400 :=
by
  sorry

end NUMINAMATH_GPT_daria_amount_owed_l2405_240566


namespace NUMINAMATH_GPT_log_constant_expression_l2405_240590

theorem log_constant_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hcond : x^2 + y^2 = 18 * x * y) :
  ∃ k : ℝ, (Real.log (x - y) / Real.log (Real.sqrt 2) - (1 / 2) * (Real.log x / Real.log (Real.sqrt 2) + Real.log y / Real.log (Real.sqrt 2))) = k :=
sorry

end NUMINAMATH_GPT_log_constant_expression_l2405_240590


namespace NUMINAMATH_GPT_jogger_ahead_distance_l2405_240515

def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def train_length_m : ℝ := 120
def passing_time_s : ℝ := 31

theorem jogger_ahead_distance :
  let V_rel := (train_speed_kmh - jogger_speed_kmh) * (1000 / 3600)
  let Distance_train := V_rel * passing_time_s 
  Distance_train = 310 → 
  Distance_train = 190 + train_length_m :=
by
  intros
  sorry

end NUMINAMATH_GPT_jogger_ahead_distance_l2405_240515


namespace NUMINAMATH_GPT_basketball_team_count_l2405_240505

theorem basketball_team_count :
  (∃ n : ℕ, n = (Nat.choose 13 4) ∧ n = 715) :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_count_l2405_240505


namespace NUMINAMATH_GPT_find_sum_of_xy_l2405_240519

theorem find_sum_of_xy (x y : ℝ) (hx_ne_y : x ≠ y) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0)
  (h_equation : x^4 - 2018 * x^3 - 2018 * y^2 * x = y^4 - 2018 * y^3 - 2018 * y * x^2) :
  x + y = 2018 :=
sorry

end NUMINAMATH_GPT_find_sum_of_xy_l2405_240519


namespace NUMINAMATH_GPT_Irene_hours_worked_l2405_240539

open Nat

theorem Irene_hours_worked (x totalHours : ℕ) : 
  (500 + 20 * x = 700) → 
  (totalHours = 40 + x) → 
  totalHours = 50 :=
by
  sorry

end NUMINAMATH_GPT_Irene_hours_worked_l2405_240539


namespace NUMINAMATH_GPT_scientific_notation_chip_gate_width_l2405_240576

theorem scientific_notation_chip_gate_width :
  0.000000014 = 1.4 * 10^(-8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_chip_gate_width_l2405_240576


namespace NUMINAMATH_GPT_woody_writing_time_l2405_240591

open Real

theorem woody_writing_time (W : ℝ) 
  (h1 : ∃ n : ℝ, n * 12 = W * 12 + 3) 
  (h2 : 12 * W + (12 * W + 3) = 39) :
  W = 1.5 :=
by sorry

end NUMINAMATH_GPT_woody_writing_time_l2405_240591


namespace NUMINAMATH_GPT_range_of_a_l2405_240594

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end NUMINAMATH_GPT_range_of_a_l2405_240594


namespace NUMINAMATH_GPT_max_kopeyka_coins_l2405_240592

def coins (n : Nat) (k : Nat) : Prop :=
  k ≤ n / 4 + 1

theorem max_kopeyka_coins : coins 2001 501 :=
by
  sorry

end NUMINAMATH_GPT_max_kopeyka_coins_l2405_240592


namespace NUMINAMATH_GPT_total_miles_driven_l2405_240564

-- Define the required variables and their types
variables (avg1 avg2 : ℝ) (gallons1 gallons2 : ℝ) (miles1 miles2 : ℝ)

-- State the conditions
axiom sum_avg_mpg : avg1 + avg2 = 75
axiom first_car_gallons : gallons1 = 25
axiom second_car_gallons : gallons2 = 35
axiom first_car_avg_mpg : avg1 = 40

-- Declare the function to calculate miles driven
def miles_driven (avg_mpg gallons : ℝ) : ℝ := avg_mpg * gallons

-- Declare the theorem for proof
theorem total_miles_driven : miles_driven avg1 gallons1 + miles_driven avg2 gallons2 = 2225 := by
  sorry

end NUMINAMATH_GPT_total_miles_driven_l2405_240564


namespace NUMINAMATH_GPT_find_f2_l2405_240562

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 2) : f a b 2 = 0 := by
  sorry

end NUMINAMATH_GPT_find_f2_l2405_240562


namespace NUMINAMATH_GPT_function_satisfies_conditions_l2405_240540

def f (m n : ℕ) : ℕ := m * n

theorem function_satisfies_conditions :
  (∀ m n : ℕ, m ≥ 1 → n ≥ 1 → 2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  (∀ m : ℕ, f m 0 = 0) ∧
  (∀ n : ℕ, f 0 n = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_function_satisfies_conditions_l2405_240540


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2405_240534

noncomputable def my_expression (m : ℝ) : ℝ :=
  (m - (m + 9) / (m + 1)) / ((m ^ 2 + 3 * m) / (m + 1))

theorem simplify_and_evaluate : my_expression (Real.sqrt 3) = 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2405_240534


namespace NUMINAMATH_GPT_problem_proof_l2405_240580

noncomputable def problem (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y ≥ 0 ∧ x ^ 2019 + y = 1) → (x + y ^ 2019 > 1 - 1 / 300)

theorem problem_proof (x y : ℝ) : problem x y :=
by
  intros h
  sorry

end NUMINAMATH_GPT_problem_proof_l2405_240580


namespace NUMINAMATH_GPT_number_of_sides_l2405_240582

theorem number_of_sides (P l n : ℕ) (hP : P = 49) (hl : l = 7) (h : P = n * l) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sides_l2405_240582


namespace NUMINAMATH_GPT_number_of_20_paise_coins_l2405_240521

theorem number_of_20_paise_coins (x y : ℕ) (h1 : x + y = 324) (h2 : 20 * x + 25 * y = 7000) : x = 220 :=
  sorry

end NUMINAMATH_GPT_number_of_20_paise_coins_l2405_240521


namespace NUMINAMATH_GPT_length_of_plot_l2405_240578

-- Define the conditions
def width : ℝ := 60
def num_poles : ℕ := 60
def dist_between_poles : ℝ := 5
def num_intervals : ℕ := num_poles - 1
def perimeter : ℝ := num_intervals * dist_between_poles

-- Define the theorem and the correctness condition
theorem length_of_plot : 
  perimeter = 2 * (length + width) → 
  length = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_plot_l2405_240578


namespace NUMINAMATH_GPT_maximum_value_of_M_l2405_240533

noncomputable def M (x : ℝ) : ℝ :=
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x)

theorem maximum_value_of_M : 
  ∃ x : ℝ, M x = (Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_M_l2405_240533


namespace NUMINAMATH_GPT_min_value_of_f_inequality_a_b_l2405_240545

theorem min_value_of_f :
  ∃ m : ℝ, m = 4 ∧ (∀ x : ℝ, |x + 3| + |x - 1| ≥ m) :=
sorry

theorem inequality_a_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1 / a + 4 / b ≥ 9 / 4) :=
sorry

end NUMINAMATH_GPT_min_value_of_f_inequality_a_b_l2405_240545


namespace NUMINAMATH_GPT_ratio_x_w_l2405_240585

variable {x y z w : ℕ}

theorem ratio_x_w (h1 : x / y = 24) (h2 : z / y = 8) (h3 : z / w = 1 / 12) : x / w = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_ratio_x_w_l2405_240585


namespace NUMINAMATH_GPT_intersection_A_B_l2405_240568

def A : Set ℝ := {x | abs x < 2}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l2405_240568


namespace NUMINAMATH_GPT_find_m2n_plus_mn2_minus_mn_l2405_240527

def quadratic_roots (a b c : ℝ) (x y : ℝ) : Prop :=
  a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0

theorem find_m2n_plus_mn2_minus_mn :
  ∃ m n : ℝ, quadratic_roots 1 2015 (-1) m n ∧ m^2 * n + m * n^2 - m * n = 2016 :=
by
  sorry

end NUMINAMATH_GPT_find_m2n_plus_mn2_minus_mn_l2405_240527


namespace NUMINAMATH_GPT_problem_l2405_240583

theorem problem
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 320) :
  x * y = 64 ∧ x^3 + y^3 = 4160 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2405_240583


namespace NUMINAMATH_GPT_calculate_expression_l2405_240500

theorem calculate_expression (f : ℕ → ℝ) (h1 : ∀ a b, f (a + b) = f a * f b) (h2 : f 1 = 2) : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) = 6 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l2405_240500


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l2405_240552

theorem arithmetic_sequence_solution :
  ∃ (a1 d : ℤ), 
    (a1 + 3*d + (a1 + 4*d) + (a1 + 5*d) + (a1 + 6*d) = 56) ∧
    ((a1 + 3*d) * (a1 + 6*d) = 187) ∧
    (
      (a1 = 5 ∧ d = 2) ∨
      (a1 = 23 ∧ d = -2)
    ) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l2405_240552


namespace NUMINAMATH_GPT_max_value_of_gems_l2405_240507

/-- Conditions -/
structure Gem :=
  (weight : ℕ)
  (value : ℕ)

def Gem1 : Gem := ⟨3, 9⟩
def Gem2 : Gem := ⟨6, 20⟩
def Gem3 : Gem := ⟨2, 5⟩

-- Laura can carry maximum of 21 pounds.
def max_weight : ℕ := 21

-- She is able to carry at least 15 of each type
def min_count := 15

/-- Prove that the maximum value Laura can carry is $69 -/
theorem max_value_of_gems : ∃ (n1 n2 n3 : ℕ), (n1 >= min_count) ∧ (n2 >= min_count) ∧ (n3 >= min_count) ∧ 
  (Gem1.weight * n1 + Gem2.weight * n2 + Gem3.weight * n3 ≤ max_weight) ∧ 
  (Gem1.value * n1 + Gem2.value * n2 + Gem3.value * n3 = 69) :=
sorry

end NUMINAMATH_GPT_max_value_of_gems_l2405_240507


namespace NUMINAMATH_GPT_complement_A_union_B_range_of_m_l2405_240547

def setA : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.sqrt (x^2 - 5*x - 14) }
def setB : Set ℝ := { x : ℝ | ∃ y : ℝ, y = Real.log (-x^2 - 7*x - 12) }
def setC (m : ℝ) : Set ℝ := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1 }

theorem complement_A_union_B :
  (A ∪ B)ᶜ = Set.Ioo (-2 : ℝ) 7 :=
sorry

theorem range_of_m (m : ℝ) :
  (A ∪ setC m = A) → (m < 2 ∨ m ≥ 6) :=
sorry

end NUMINAMATH_GPT_complement_A_union_B_range_of_m_l2405_240547


namespace NUMINAMATH_GPT_initial_percentage_decrease_l2405_240595

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₁ : P > 0) (h₂ : 1.55 * (1 - x / 100) = 1.24) :
    x = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_percentage_decrease_l2405_240595


namespace NUMINAMATH_GPT_inequality_abc_l2405_240520

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1):
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b))) ≥ (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l2405_240520


namespace NUMINAMATH_GPT_min_additional_games_l2405_240574

def num_initial_games : ℕ := 4
def num_lions_won : ℕ := 3
def num_eagles_won : ℕ := 1
def win_threshold : ℝ := 0.90

theorem min_additional_games (M : ℕ) : (num_eagles_won + M) / (num_initial_games + M) ≥ win_threshold ↔ M ≥ 26 :=
by
  sorry

end NUMINAMATH_GPT_min_additional_games_l2405_240574


namespace NUMINAMATH_GPT_fixed_point_of_parabola_l2405_240532

theorem fixed_point_of_parabola (s : ℝ) : ∃ y : ℝ, y = 4 * 3^2 + s * 3 - 3 * s ∧ (3, y) = (3, 36) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_parabola_l2405_240532


namespace NUMINAMATH_GPT_count_multiples_5_or_7_but_not_35_l2405_240558

def count_multiples (n d : ℕ) : ℕ :=
  n / d

def inclusion_exclusion (a b c : ℕ) : ℕ :=
  a + b - c

theorem count_multiples_5_or_7_but_not_35 : 
  count_multiples 3000 5 + count_multiples 3000 7 - count_multiples 3000 35 = 943 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_5_or_7_but_not_35_l2405_240558


namespace NUMINAMATH_GPT_price_reduction_equation_l2405_240501

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_price_reduction_equation_l2405_240501


namespace NUMINAMATH_GPT_find_a2_l2405_240579

def arithmetic_sequence (a : ℕ → ℤ) (a1 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n + d 

def sum_arithmetic_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * (a 1 + a n) / 2

theorem find_a2 (a : ℕ → ℤ) (S : ℕ → ℤ) (a1 : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a a1 d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : a1 = -2010)
  (h4 : (S 2010) / 2010 - (S 2008) / 2008 = 2) :
  a 2 = -2008 :=
sorry

end NUMINAMATH_GPT_find_a2_l2405_240579


namespace NUMINAMATH_GPT_proof_problem_l2405_240513

theorem proof_problem (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 + 2 * a * b = 64 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2405_240513


namespace NUMINAMATH_GPT_inequality_l2405_240589

variable {a b c : ℝ}

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) : 
  a * (a - 1) + b * (b - 1) + c * (c - 1) ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_l2405_240589


namespace NUMINAMATH_GPT_B_joined_after_8_months_l2405_240511

-- Define the initial investments and time
def A_investment : ℕ := 36000
def B_investment : ℕ := 54000
def profit_ratio_A_B := 2 / 1

-- Define a proposition which states that B joined the business after x = 8 months
theorem B_joined_after_8_months (x : ℕ) (h : (A_investment * 12) / (B_investment * (12 - x)) = profit_ratio_A_B) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_B_joined_after_8_months_l2405_240511


namespace NUMINAMATH_GPT_train_length_l2405_240597

theorem train_length (s : ℝ) (t : ℝ) (h_s : s = 60) (h_t : t = 10) :
  ∃ L : ℝ, L = 166.7 := by
  sorry

end NUMINAMATH_GPT_train_length_l2405_240597


namespace NUMINAMATH_GPT_senior_discount_percentage_l2405_240553

theorem senior_discount_percentage 
    (cost_shorts : ℕ)
    (count_shorts : ℕ)
    (cost_shirts : ℕ)
    (count_shirts : ℕ)
    (amount_paid : ℕ)
    (total_cost : ℕ := (cost_shorts * count_shorts) + (cost_shirts * count_shirts))
    (discount_received : ℕ := total_cost - amount_paid)
    (discount_percentage : ℚ := (discount_received : ℚ) / total_cost * 100) :
    count_shorts = 3 ∧ cost_shorts = 15 ∧ count_shirts = 5 ∧ cost_shirts = 17 ∧ amount_paid = 117 →
    discount_percentage = 10 := 
by
    sorry

end NUMINAMATH_GPT_senior_discount_percentage_l2405_240553


namespace NUMINAMATH_GPT_find_initial_strawberries_l2405_240503

-- Define the number of strawberries after picking 35 more to be 63
def strawberries_after_picking := 63

-- Define the number of strawberries picked
def strawberries_picked := 35

-- Define the initial number of strawberries
def initial_strawberries := 28

-- State the theorem
theorem find_initial_strawberries (x : ℕ) (h : x + strawberries_picked = strawberries_after_picking) : x = initial_strawberries :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_initial_strawberries_l2405_240503


namespace NUMINAMATH_GPT_magnitude_quotient_l2405_240508

open Complex

theorem magnitude_quotient : 
  abs ((1 + 2 * I) / (2 - I)) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_magnitude_quotient_l2405_240508


namespace NUMINAMATH_GPT_sum_of_h_and_k_l2405_240537

theorem sum_of_h_and_k (foci1 foci2 : ℝ × ℝ) (pt : ℝ × ℝ) (a b h k : ℝ) 
  (h_positive : a > 0) (b_positive : b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x - h)^2 / a^2 + (y - k)^2 / b^2 = if (x, y) = pt then 1 else sorry)
  (foci_eq : foci1 = (1, 2) ∧ foci2 = (4, 2))
  (pt_eq : pt = (-1, 5)) :
  h + k = 4.5 :=
sorry

end NUMINAMATH_GPT_sum_of_h_and_k_l2405_240537


namespace NUMINAMATH_GPT_comparison_1_comparison_2_l2405_240502

noncomputable def expr1 := -(-((6: ℝ) / 7))
noncomputable def expr2 := -((abs (-((4: ℝ) / 5))))
noncomputable def expr3 := -((4: ℝ) / 5)
noncomputable def expr4 := -((2: ℝ) / 3)

theorem comparison_1 : expr1 > expr2 := sorry
theorem comparison_2 : expr3 < expr4 := sorry

end NUMINAMATH_GPT_comparison_1_comparison_2_l2405_240502


namespace NUMINAMATH_GPT_second_wrongly_copied_number_l2405_240510

theorem second_wrongly_copied_number 
  (avg_err : ℝ) 
  (total_nums : ℕ) 
  (sum_err : ℝ) 
  (first_err_corr : ℝ) 
  (correct_avg : ℝ) 
  (correct_num : ℝ) 
  (second_num_wrong : ℝ) :
  (avg_err = 40.2) → 
  (total_nums = 10) → 
  (sum_err = total_nums * avg_err) → 
  (first_err_corr = 16) → 
  (correct_avg = 40) → 
  (correct_num = 31) → 
  sum_err - first_err_corr + (correct_num - second_num_wrong) = total_nums * correct_avg → 
  second_num_wrong = 17 := 
by 
  intros h_avg h_total h_sum_err h_first_corr h_correct_avg h_correct_num h_corrected_sum 
  sorry

end NUMINAMATH_GPT_second_wrongly_copied_number_l2405_240510


namespace NUMINAMATH_GPT_band_member_share_l2405_240560

def num_people : ℕ := 500
def ticket_price : ℝ := 30
def band_share_percent : ℝ := 0.70
def num_band_members : ℕ := 4

theorem band_member_share : 
  (num_people * ticket_price * band_share_percent) / num_band_members = 2625 := by
  sorry

end NUMINAMATH_GPT_band_member_share_l2405_240560


namespace NUMINAMATH_GPT_minimum_value_of_f_at_zero_inequality_f_geq_term_l2405_240561

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

theorem minimum_value_of_f_at_zero (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → f a y ≥ f a x ∧ f a x = 0) → a = 2 :=
by
  sorry

theorem inequality_f_geq_term (x : ℝ) (hx : x > 1) : 
  f 2 x ≥ 1 / x - Real.exp (1 - x) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_at_zero_inequality_f_geq_term_l2405_240561


namespace NUMINAMATH_GPT_find_a5_div_a7_l2405_240517

variable {a : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {aₙ} is a positive geometric sequence.
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom pos_seq (n : ℕ) : 0 < a n

-- Given conditions
axiom a2a8_eq_6 : a 2 * a 8 = 6
axiom a4_plus_a6_eq_5 : a 4 + a 6 = 5
axiom decreasing_seq (n : ℕ) : a (n + 1) < a n

theorem find_a5_div_a7 : a 5 / a 7 = 3 / 2 := 
sorry

end NUMINAMATH_GPT_find_a5_div_a7_l2405_240517


namespace NUMINAMATH_GPT_complex_problem_l2405_240546

theorem complex_problem (z : ℂ) (h : (i * z + z) = 2) : z = 1 - i :=
sorry

end NUMINAMATH_GPT_complex_problem_l2405_240546


namespace NUMINAMATH_GPT_average_employees_per_week_l2405_240549

-- Define the number of employees hired each week
variables (x : ℕ)
noncomputable def employees_first_week := x + 200
noncomputable def employees_second_week := x
noncomputable def employees_third_week := x + 150
noncomputable def employees_fourth_week := 400

-- Given conditions as hypotheses
axiom h1 : employees_third_week / 2 = employees_fourth_week / 2
axiom h2 : employees_fourth_week = 400

-- Prove the average number of employees hired per week is 225
theorem average_employees_per_week :
  (employees_first_week + employees_second_week + employees_third_week + employees_fourth_week) / 4 = 225 :=
by
  sorry

end NUMINAMATH_GPT_average_employees_per_week_l2405_240549


namespace NUMINAMATH_GPT_trigonometric_expression_evaluation_l2405_240556

theorem trigonometric_expression_evaluation :
  (Real.cos (-585 * Real.pi / 180)) / 
  (Real.tan (495 * Real.pi / 180) + Real.sin (-690 * Real.pi / 180)) = Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_trigonometric_expression_evaluation_l2405_240556


namespace NUMINAMATH_GPT_min_value_of_expression_l2405_240565

open Real

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h_perp : (x - 1) * 1 + 3 * y = 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_ab_perp : (a - 1) * 1 + 3 * b = 0), (1 / a) + (1 / (3 * b)) ≥ m) :=
by
  use 4
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2405_240565


namespace NUMINAMATH_GPT_inequality_correct_l2405_240563

theorem inequality_correct (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c :=
sorry

end NUMINAMATH_GPT_inequality_correct_l2405_240563


namespace NUMINAMATH_GPT_sum_of_first_15_terms_l2405_240536

-- Given conditions: Sum of 4th and 12th term is 24
variable (a d : ℤ) (a_4 a_12 : ℤ)
variable (S : ℕ → ℤ)
variable (arithmetic_series_4_12_sum : 2 * a + 14 * d = 24)
variable (nth_term_def : ∀ n, a + (n - 1) * d = a_n)

-- Question: Sum of the first 15 terms of the progression
theorem sum_of_first_15_terms : S 15 = 180 := by
  sorry

end NUMINAMATH_GPT_sum_of_first_15_terms_l2405_240536


namespace NUMINAMATH_GPT_intersection_set_eq_l2405_240567

-- Define M
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 }

-- Define N
def N : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 / 4) + (p.2 / 3) = 1 }

-- Define the intersection of M and N
def M_intersection_N := { x : ℝ | -4 ≤ x ∧ x ≤ 4 }

-- The theorem to be proved
theorem intersection_set_eq : 
  { p : ℝ × ℝ | p ∈ M ∧ p ∈ N } = { p : ℝ × ℝ | p.1 ∈ M_intersection_N } :=
sorry

end NUMINAMATH_GPT_intersection_set_eq_l2405_240567


namespace NUMINAMATH_GPT_income_of_deceased_is_correct_l2405_240554

-- Definitions based on conditions
def family_income_before_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def family_income_after_death (avg_income: ℝ) (members: ℕ) : ℝ := avg_income * members
def income_of_deceased (total_before: ℝ) (total_after: ℝ) : ℝ := total_before - total_after

-- Given conditions
def avg_income_before : ℝ := 782
def avg_income_after : ℝ := 650
def num_members_before : ℕ := 4
def num_members_after : ℕ := 3

-- Mathematical statement
theorem income_of_deceased_is_correct : 
  income_of_deceased (family_income_before_death avg_income_before num_members_before) 
                     (family_income_after_death avg_income_after num_members_after) = 1178 :=
by
  sorry

end NUMINAMATH_GPT_income_of_deceased_is_correct_l2405_240554


namespace NUMINAMATH_GPT_tilly_counts_total_stars_l2405_240531

open Nat

def stars_to_east : ℕ := 120
def factor_west_stars : ℕ := 6
def stars_to_west : ℕ := factor_west_stars * stars_to_east
def total_stars : ℕ := stars_to_east + stars_to_west

theorem tilly_counts_total_stars :
  total_stars = 840 := by
  sorry

end NUMINAMATH_GPT_tilly_counts_total_stars_l2405_240531


namespace NUMINAMATH_GPT_hyperbola_solution_l2405_240599

noncomputable def hyperbola_focus_parabola_equiv_hyperbola : Prop :=
  ∀ (a b c : ℝ),
    -- Condition 1: One focus of the hyperbola coincides with the focus of the parabola y^2 = 4sqrt(7)x
    (c^2 = a^2 + b^2) ∧ (c^2 = 7) →

    -- Condition 2: The hyperbola intersects the line y = x - 1 at points M and N
    (∃ M N : ℝ × ℝ, (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1^2 / a^2) - (M.2^2 / b^2) = 1) ∧ ((N.1^2 / a^2) - (N.2^2 / b^2) = 1)) →

    -- Condition 3: The x-coordinate of the midpoint of MN is -2/3
    (∀ M N : ℝ × ℝ, 
    (M.2 = M.1 - 1) ∧ (N.2 = N.1 - 1) ∧ 
    ((M.1 + N.1) / 2 = -2/3)) →

    -- Conclusion: The standard equation of the hyperbola is x^2 / 2 - y^2 / 5 = 1
    a^2 = 2 ∧ b^2 = 5 ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 → (x^2 / 2) - (y^2 / 5) = 1)

-- Proof omitted
theorem hyperbola_solution : hyperbola_focus_parabola_equiv_hyperbola :=
by sorry

end NUMINAMATH_GPT_hyperbola_solution_l2405_240599


namespace NUMINAMATH_GPT_largest_possible_value_of_N_l2405_240512

theorem largest_possible_value_of_N :
  ∃ N : ℕ, (∀ d : ℕ, (d ∣ N) → (d = 1 ∨ d = N ∨ (∃ k : ℕ, d = 3 ∨ d=k ∨ d=441 / k))) ∧
            ((21 * 3) ∣ N) ∧
            (N = 441) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_value_of_N_l2405_240512


namespace NUMINAMATH_GPT_wedding_cost_correct_l2405_240550

def venue_cost : ℕ := 10000
def cost_per_guest : ℕ := 500
def john_guests : ℕ := 50
def wife_guest_increase : ℕ := john_guests * 60 / 100
def total_wedding_cost : ℕ := venue_cost + cost_per_guest * (john_guests + wife_guest_increase)

theorem wedding_cost_correct : total_wedding_cost = 50000 :=
by
  sorry

end NUMINAMATH_GPT_wedding_cost_correct_l2405_240550


namespace NUMINAMATH_GPT_greatest_possible_number_of_blue_chips_l2405_240516

-- Definitions based on conditions
def total_chips : Nat := 72

-- Definition of the relationship between red and blue chips where p is a prime number
def is_prime (n : Nat) : Prop := Nat.Prime n

def satisfies_conditions (r b p : Nat) : Prop :=
  r + b = total_chips ∧ r = b + p ∧ is_prime p

-- The statement to prove
theorem greatest_possible_number_of_blue_chips (r b p : Nat) 
  (h : satisfies_conditions r b p) : b = 35 := 
sorry

end NUMINAMATH_GPT_greatest_possible_number_of_blue_chips_l2405_240516


namespace NUMINAMATH_GPT_sqrt_36_eq_6_l2405_240577

theorem sqrt_36_eq_6 : Real.sqrt 36 = 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_36_eq_6_l2405_240577


namespace NUMINAMATH_GPT_ratio_of_divisors_l2405_240509

def M : Nat := 75 * 75 * 140 * 343

noncomputable def sumOfOddDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all odd divisors of n. (placeholder)
  sorry

noncomputable def sumOfEvenDivisors (n : Nat) : Nat := 
  -- Function that computes the sum of all even divisors of n. (placeholder)
  sorry

theorem ratio_of_divisors :
  let sumOdd := sumOfOddDivisors M
  let sumEven := sumOfEvenDivisors M
  sumOdd / sumEven = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_divisors_l2405_240509


namespace NUMINAMATH_GPT_find_triplet_solution_l2405_240542

theorem find_triplet_solution (m n x y : ℕ) (hm : 0 < m) (hcoprime : Nat.gcd m n = 1) 
 (heq : (x^2 + y^2)^m = (x * y)^n) : 
  ∃ a : ℕ, x = 2^a ∧ y = 2^a ∧ n = m + 1 :=
by sorry

end NUMINAMATH_GPT_find_triplet_solution_l2405_240542


namespace NUMINAMATH_GPT_volleyballs_count_l2405_240555

-- Definitions of sports item counts based on given conditions.
def soccer_balls := 20
def basketballs := soccer_balls + 5
def tennis_balls := 2 * soccer_balls
def baseballs := soccer_balls + 10
def hockey_pucks := tennis_balls / 2
def total_items := 180

-- Calculate the total number of known sports items.
def known_items_sum := soccer_balls + basketballs + tennis_balls + baseballs + hockey_pucks

-- Prove the number of volleyballs
theorem volleyballs_count : total_items - known_items_sum = 45 := by
  sorry

end NUMINAMATH_GPT_volleyballs_count_l2405_240555


namespace NUMINAMATH_GPT_equivalence_of_min_perimeter_and_cyclic_quadrilateral_l2405_240514

-- Definitions for points P, Q, R, S on sides of quadrilateral ABCD
-- Function definitions for conditions and equivalence of stated problems

variable {A B C D P Q R S : Type*} 

def is_on_side (P : Type*) (A B : Type*) : Prop := sorry
def is_interior_point (P : Type*) (A B : Type*) : Prop := sorry
def is_convex_quadrilateral (A B C D : Type*) : Prop := sorry
def is_cyclic_quadrilateral (A B C D : Type*) : Prop := sorry
def has_circumcenter_interior (A B C D : Type*) : Prop := sorry
def has_minimal_perimeter (P Q R S : Type*) : Prop := sorry

theorem equivalence_of_min_perimeter_and_cyclic_quadrilateral 
  (h1 : is_convex_quadrilateral A B C D) 
  (hP : is_on_side P A B ∧ is_interior_point P A B) 
  (hQ : is_on_side Q B C ∧ is_interior_point Q B C) 
  (hR : is_on_side R C D ∧ is_interior_point R C D) 
  (hS : is_on_side S D A ∧ is_interior_point S D A) :
  (∃ P' Q' R' S', has_minimal_perimeter P' Q' R' S') ↔ (is_cyclic_quadrilateral A B C D ∧ has_circumcenter_interior A B C D) :=
sorry

end NUMINAMATH_GPT_equivalence_of_min_perimeter_and_cyclic_quadrilateral_l2405_240514


namespace NUMINAMATH_GPT_height_after_five_years_l2405_240538

namespace PapayaTreeGrowth

def growth_first_year := true → ℝ
def growth_second_year (x : ℝ) := 1.5 * x
def growth_third_year (x : ℝ) := 1.5 * growth_second_year x
def growth_fourth_year (x : ℝ) := 2 * growth_third_year x
def growth_fifth_year (x : ℝ) := 0.5 * growth_fourth_year x

def total_growth (x : ℝ) := x + growth_second_year x + growth_third_year x +
                             growth_fourth_year x + growth_fifth_year x

theorem height_after_five_years (x : ℝ) (H : total_growth x = 23) : x = 2 :=
by
  sorry

end PapayaTreeGrowth

end NUMINAMATH_GPT_height_after_five_years_l2405_240538


namespace NUMINAMATH_GPT_find_a_for_min_l2405_240526

noncomputable def f (a x : ℝ) : ℝ := x^3 + 3 * a * x^2 - 6 * a * x + 2

theorem find_a_for_min {a x0 : ℝ} (hx0 : 1 < x0 ∧ x0 < 3) (h : ∀ x : ℝ, deriv (f a) x0 = 0) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_min_l2405_240526


namespace NUMINAMATH_GPT_take_home_pay_l2405_240598

def tax_rate : ℝ := 0.10
def total_pay : ℝ := 650

theorem take_home_pay : total_pay - (total_pay * tax_rate) = 585 := by
  sorry

end NUMINAMATH_GPT_take_home_pay_l2405_240598


namespace NUMINAMATH_GPT_parking_savings_l2405_240529

theorem parking_savings
  (weekly_rent : ℕ := 10)
  (monthly_rent : ℕ := 40)
  (weeks_in_year : ℕ := 52)
  (months_in_year : ℕ := 12)
  : weekly_rent * weeks_in_year - monthly_rent * months_in_year = 40 := 
by
  sorry

end NUMINAMATH_GPT_parking_savings_l2405_240529


namespace NUMINAMATH_GPT_matrix_sum_correct_l2405_240551

def mat1 : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -1], ![3, 7]]
def mat2 : Matrix (Fin 2) (Fin 2) ℤ := ![![ -6, 8], ![5, -2]]
def mat_sum : Matrix (Fin 2) (Fin 2) ℤ := ![![-2, 7], ![8, 5]]

theorem matrix_sum_correct : mat1 + mat2 = mat_sum :=
by
  rw [mat1, mat2]
  sorry

end NUMINAMATH_GPT_matrix_sum_correct_l2405_240551


namespace NUMINAMATH_GPT_num_people_comparison_l2405_240587

def num_people_1st_session (a : ℝ) : Prop := a > 0 -- Define the number for first session
def num_people_2nd_session (a : ℝ) : ℝ := 1.1 * a -- Define the number for second session
def num_people_3rd_session (a : ℝ) : ℝ := 0.99 * a -- Define the number for third session

theorem num_people_comparison (a b : ℝ) 
    (h1 : b = 0.99 * a): 
    a > b := 
by 
  -- insert the proof here
  sorry 

end NUMINAMATH_GPT_num_people_comparison_l2405_240587


namespace NUMINAMATH_GPT_students_in_lower_grades_l2405_240570

noncomputable def seniors : ℕ := 300
noncomputable def percentage_cars_seniors : ℝ := 0.40
noncomputable def percentage_cars_remaining : ℝ := 0.10
noncomputable def total_percentage_cars : ℝ := 0.15

theorem students_in_lower_grades (X : ℝ) :
  (0.15 * (300 + X) = 120 + 0.10 * X) → X = 1500 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_students_in_lower_grades_l2405_240570


namespace NUMINAMATH_GPT_find_a_l2405_240543

variable (a : ℝ)

def A := ({1, 2, a} : Set ℝ)
def B := ({1, a^2 - a} : Set ℝ)

theorem find_a (h : B a ⊆ A a) : a = -1 ∨ a = 0 :=
  sorry

end NUMINAMATH_GPT_find_a_l2405_240543


namespace NUMINAMATH_GPT_complex_triple_sum_eq_sqrt3_l2405_240596

noncomputable section

open Complex

theorem complex_triple_sum_eq_sqrt3 {a b c : ℂ} (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : a + b + c ≠ 0) (h5 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3) : abs (a + b + c) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_complex_triple_sum_eq_sqrt3_l2405_240596


namespace NUMINAMATH_GPT_problem_solution_l2405_240530

theorem problem_solution (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3) →
  (a₁ + a₂ + a₃ = 19) :=
by
  -- Given condition: for any real number x, x^3 = a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3
  -- We need to prove: a₁ + a₂ + a₃ = 19
  sorry

end NUMINAMATH_GPT_problem_solution_l2405_240530


namespace NUMINAMATH_GPT_nico_reads_wednesday_l2405_240559

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end NUMINAMATH_GPT_nico_reads_wednesday_l2405_240559


namespace NUMINAMATH_GPT_union_M_N_eq_N_l2405_240557

def M := {x : ℝ | x^2 - 2 * x ≤ 0}
def N := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

theorem union_M_N_eq_N : M ∪ N = N := 
sorry

end NUMINAMATH_GPT_union_M_N_eq_N_l2405_240557


namespace NUMINAMATH_GPT_number_with_all_8s_is_divisible_by_13_l2405_240541

theorem number_with_all_8s_is_divisible_by_13 :
  ∀ (N : ℕ), (N = 8 * (10^1974 - 1) / 9) → 13 ∣ N :=
by
  sorry

end NUMINAMATH_GPT_number_with_all_8s_is_divisible_by_13_l2405_240541


namespace NUMINAMATH_GPT_value_of_a_l2405_240504

theorem value_of_a (a : ℝ) (x : ℝ) (h : 2 * x + 3 * a = -1) (hx : x = 1) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2405_240504


namespace NUMINAMATH_GPT_math_problem_l2405_240588

variable (a a' b b' c c' : ℝ)

theorem math_problem 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b * b) 
  (h3 : a' * c' ≥ b' * b') : 
  (a + a') * (c + c') ≥ (b + b') * (b + b') := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l2405_240588


namespace NUMINAMATH_GPT_cameras_not_in_both_l2405_240581

-- Definitions for the given conditions
def shared_cameras : ℕ := 12
def sarah_cameras : ℕ := 24
def mike_unique_cameras : ℕ := 9

-- The proof statement
theorem cameras_not_in_both : (sarah_cameras - shared_cameras) + mike_unique_cameras = 21 := by
  sorry

end NUMINAMATH_GPT_cameras_not_in_both_l2405_240581


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l2405_240572

theorem polynomial_coeff_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) (h : (2 * x - 3) ^ 5 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4 + a₅ * x ^ 5) :
  a₁ + 2 * a₂ + 3 * a₃ + 4 * a₄ + 5 * a₅ = 160 :=
sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l2405_240572


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2405_240524

theorem isosceles_triangle_largest_angle (α : ℝ) (β : ℝ)
  (h1 : 0 < α) (h2 : α = 30) (h3 : β = 30):
  ∃ γ : ℝ, γ = 180 - 2 * α ∧ γ = 120 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2405_240524
