import Mathlib

namespace NUMINAMATH_GPT_yadav_spends_50_percent_on_clothes_and_transport_l304_30413

variable (S : ℝ)
variable (monthly_savings : ℝ := 46800 / 12)
variable (clothes_transport_expense : ℝ := 3900)
variable (remaining_salary : ℝ := 0.40 * S)

theorem yadav_spends_50_percent_on_clothes_and_transport (h1 : remaining_salary = 2 * 3900) :
  (clothes_transport_expense / remaining_salary) * 100 = 50 :=
by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_yadav_spends_50_percent_on_clothes_and_transport_l304_30413


namespace NUMINAMATH_GPT_average_increase_fraction_l304_30457

-- First, we define the given conditions:
def incorrect_mark : ℕ := 82
def correct_mark : ℕ := 62
def number_of_students : ℕ := 80

-- We state the theorem to prove that the fraction by which the average marks increased is 1/4. 
theorem average_increase_fraction (incorrect_mark correct_mark : ℕ) (number_of_students : ℕ) :
  (incorrect_mark - correct_mark) / number_of_students = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_average_increase_fraction_l304_30457


namespace NUMINAMATH_GPT_determine_c_l304_30458

theorem determine_c {f : ℝ → ℝ} (c : ℝ) (h : ∀ x, f x = 2 / (3 * x + c))
  (hf_inv : ∀ x, (f⁻¹ x) = (3 - 6 * x) / x) : c = 18 :=
by sorry

end NUMINAMATH_GPT_determine_c_l304_30458


namespace NUMINAMATH_GPT_layoffs_payment_l304_30436

theorem layoffs_payment :
  let total_employees := 450
  let salary_2000_employees := 150
  let salary_2500_employees := 200
  let salary_3000_employees := 100
  let first_round_2000_layoffs := 0.20 * salary_2000_employees
  let first_round_2500_layoffs := 0.25 * salary_2500_employees
  let first_round_3000_layoffs := 0.15 * salary_3000_employees
  let remaining_2000_after_first_round := salary_2000_employees - first_round_2000_layoffs
  let remaining_2500_after_first_round := salary_2500_employees - first_round_2500_layoffs
  let remaining_3000_after_first_round := salary_3000_employees - first_round_3000_layoffs
  let second_round_2000_layoffs := 0.10 * remaining_2000_after_first_round
  let second_round_2500_layoffs := 0.15 * remaining_2500_after_first_round
  let second_round_3000_layoffs := 0.05 * remaining_3000_after_first_round
  let remaining_2000_after_second_round := remaining_2000_after_first_round - second_round_2000_layoffs
  let remaining_2500_after_second_round := remaining_2500_after_first_round - second_round_2500_layoffs
  let remaining_3000_after_second_round := remaining_3000_after_first_round - second_round_3000_layoffs
  let total_payment := remaining_2000_after_second_round * 2000 + remaining_2500_after_second_round * 2500 + remaining_3000_after_second_round * 3000
  total_payment = 776500 := sorry

end NUMINAMATH_GPT_layoffs_payment_l304_30436


namespace NUMINAMATH_GPT_average_monthly_balance_correct_l304_30463

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 250
def april_balance : ℕ := 250
def may_balance : ℕ := 150
def june_balance : ℕ := 100

def total_balance : ℕ :=
  january_balance + february_balance + march_balance + april_balance + may_balance + june_balance

def number_of_months : ℕ := 6

def average_monthly_balance : ℕ :=
  total_balance / number_of_months

theorem average_monthly_balance_correct :
  average_monthly_balance = 175 := by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_correct_l304_30463


namespace NUMINAMATH_GPT_price_of_tea_mixture_l304_30482

theorem price_of_tea_mixture 
  (p1 p2 p3 : ℝ) 
  (q1 q2 q3 : ℝ) 
  (h_p1 : p1 = 126) 
  (h_p2 : p2 = 135) 
  (h_p3 : p3 = 173.5) 
  (h_q1 : q1 = 1) 
  (h_q2 : q2 = 1) 
  (h_q3 : q3 = 2) : 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3) = 152 := 
by 
  sorry

end NUMINAMATH_GPT_price_of_tea_mixture_l304_30482


namespace NUMINAMATH_GPT_average_speed_l304_30471

theorem average_speed (d1 d2 t1 t2 : ℝ) 
  (h1 : d1 = 100) 
  (h2 : d2 = 80) 
  (h3 : t1 = 1) 
  (h4 : t2 = 1) : 
  (d1 + d2) / (t1 + t2) = 90 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_l304_30471


namespace NUMINAMATH_GPT_values_of_a_for_equation_l304_30456

theorem values_of_a_for_equation :
  ∃ S : Finset ℤ, (∀ a ∈ S, |3 * a + 7| + |3 * a - 5| = 12) ∧ S.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_values_of_a_for_equation_l304_30456


namespace NUMINAMATH_GPT_prosecutor_cases_knight_or_liar_l304_30491

-- Define the conditions as premises
variable (X : Prop)
variable (Y : Prop)
variable (prosecutor : Prop) -- Truthfulness of the prosecutor (true for knight, false for liar)

-- Define the statements made by the prosecutor
axiom statement1 : X  -- "X is guilty."
axiom statement2 : ¬ (X ∧ Y)  -- "Both X and Y cannot both be guilty."

-- Lean 4 statement for the proof problem
theorem prosecutor_cases_knight_or_liar (h1 : prosecutor) (h2 : ¬prosecutor) : 
  (prosecutor ∧ X ∧ ¬Y) :=
by sorry

end NUMINAMATH_GPT_prosecutor_cases_knight_or_liar_l304_30491


namespace NUMINAMATH_GPT_cylinder_height_l304_30400

theorem cylinder_height (r₁ r₂ : ℝ) (S : ℝ) (hR : r₁ = 3) (hL : r₂ = 4) (hS : S = 100 * Real.pi) : 
  (∃ h : ℝ, h = 7 ∨ h = 1) :=
by 
  sorry

end NUMINAMATH_GPT_cylinder_height_l304_30400


namespace NUMINAMATH_GPT_problem_statement_l304_30496
noncomputable def not_divisible (n : ℕ) : Prop := ∃ k : ℕ, (5^n - 3^n) = (2^n + 65) * k
theorem problem_statement (n : ℕ) (h : 0 < n) : ¬ not_divisible n := sorry

end NUMINAMATH_GPT_problem_statement_l304_30496


namespace NUMINAMATH_GPT_increase_in_area_l304_30431

-- Define the initial side length and the increment.
def initial_side_length : ℕ := 6
def increment : ℕ := 1

-- Define the original area of the land.
def original_area : ℕ := initial_side_length * initial_side_length

-- Define the new side length after the increase.
def new_side_length : ℕ := initial_side_length + increment

-- Define the new area of the land.
def new_area : ℕ := new_side_length * new_side_length

-- Define the theorem that states the increase in area.
theorem increase_in_area : new_area - original_area = 13 := by
  sorry

end NUMINAMATH_GPT_increase_in_area_l304_30431


namespace NUMINAMATH_GPT_steve_assignments_fraction_l304_30464

theorem steve_assignments_fraction (h_sleep: ℝ) (h_school: ℝ) (h_family: ℝ) (total_hours: ℝ) : 
  h_sleep = 1/3 ∧ 
  h_school = 1/6 ∧ 
  h_family = 10 ∧ 
  total_hours = 24 → 
  (2 / total_hours = 1 / 12) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_steve_assignments_fraction_l304_30464


namespace NUMINAMATH_GPT_sequence_transformation_possible_l304_30465

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end NUMINAMATH_GPT_sequence_transformation_possible_l304_30465


namespace NUMINAMATH_GPT_largest_b_for_denom_has_nonreal_roots_l304_30406

theorem largest_b_for_denom_has_nonreal_roots :
  ∃ b : ℤ, 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) 
  ∧ (∀ b' : ℤ, (∀ x : ℝ, x^2 + (b' : ℝ) * x + 12 ≠ 0) → b' ≤ b)
  ∧ b = 6 :=
sorry

end NUMINAMATH_GPT_largest_b_for_denom_has_nonreal_roots_l304_30406


namespace NUMINAMATH_GPT_probability_computation_l304_30429

-- Definitions of individual success probabilities
def probability_Xavier_solving_problem : ℚ := 1 / 4
def probability_Yvonne_solving_problem : ℚ := 2 / 3
def probability_William_solving_problem : ℚ := 7 / 10
def probability_Zelda_solving_problem : ℚ := 5 / 8
def probability_Zelda_notsolving_problem : ℚ := 1 - probability_Zelda_solving_problem

-- The target probability that only Xavier, Yvonne, and William, but not Zelda, will solve the problem
def target_probability : ℚ := (1 / 4) * (2 / 3) * (7 / 10) * (3 / 8)

-- The simplified form of the computed probability
def simplified_target_probability : ℚ := 7 / 160

-- Lean 4 statement to prove the equality of the computed and the target probabilities
theorem probability_computation :
  target_probability = simplified_target_probability := by
  sorry

end NUMINAMATH_GPT_probability_computation_l304_30429


namespace NUMINAMATH_GPT_amy_initial_money_l304_30467

-- Define the conditions
variable (left_fair : ℕ) (spent : ℕ)

-- Define the proof problem statement
theorem amy_initial_money (h1 : left_fair = 11) (h2 : spent = 4) : left_fair + spent = 15 := 
by sorry

end NUMINAMATH_GPT_amy_initial_money_l304_30467


namespace NUMINAMATH_GPT_find_a_l304_30433

-- Defining the problem conditions
def rational_eq (x a : ℝ) :=
  x / (x - 3) - 2 * a / (x - 3) = 2

def extraneous_root (x : ℝ) : Prop :=
  x = 3

-- Theorem: Given the conditions, prove that a = 3 / 2
theorem find_a (a : ℝ) : (∃ x, extraneous_root x ∧ rational_eq x a) → a = 3 / 2 :=
  by
    sorry

end NUMINAMATH_GPT_find_a_l304_30433


namespace NUMINAMATH_GPT_retirement_fund_increment_l304_30460

theorem retirement_fund_increment (k y : ℝ) (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27) : k * Real.sqrt y = 810 := by
  sorry

end NUMINAMATH_GPT_retirement_fund_increment_l304_30460


namespace NUMINAMATH_GPT_min_distance_of_complex_numbers_l304_30462

open Complex

theorem min_distance_of_complex_numbers
  (z w : ℂ)
  (h₁ : abs (z + 1 + 3 * Complex.I) = 1)
  (h₂ : abs (w - 7 - 8 * Complex.I) = 3) :
  ∃ d, d = Real.sqrt 185 - 4 ∧ ∀ Z W : ℂ, abs (Z + 1 + 3 * Complex.I) = 1 → abs (W - 7 - 8 * Complex.I) = 3 → abs (Z - W) ≥ d :=
sorry

end NUMINAMATH_GPT_min_distance_of_complex_numbers_l304_30462


namespace NUMINAMATH_GPT_value_of_otimes_l304_30414

variable (a b : ℚ)

/-- Define the operation ⊗ -/
def otimes (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Given conditions -/
axiom condition1 : otimes a b 1 (-3) = 2 

/-- Target proof -/
theorem value_of_otimes : otimes a b 2 (-6) = 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_otimes_l304_30414


namespace NUMINAMATH_GPT_sum_gcd_lcm_63_2898_l304_30444

theorem sum_gcd_lcm_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 :=
by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_63_2898_l304_30444


namespace NUMINAMATH_GPT_profit_percentage_l304_30495

-- Define the selling price
def selling_price : ℝ := 900

-- Define the profit
def profit : ℝ := 100

-- Define the cost price as selling price minus profit
def cost_price : ℝ := selling_price - profit

-- Statement of the profit percentage calculation
theorem profit_percentage : (profit / cost_price) * 100 = 12.5 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_l304_30495


namespace NUMINAMATH_GPT_mia_spent_per_parent_l304_30494

theorem mia_spent_per_parent (amount_sibling : ℕ) (num_siblings : ℕ) (total_spent : ℕ) 
  (num_parents : ℕ) : 
  amount_sibling = 30 → num_siblings = 3 → total_spent = 150 → num_parents = 2 → 
  (total_spent - num_siblings * amount_sibling) / num_parents = 30 :=
by
  sorry

end NUMINAMATH_GPT_mia_spent_per_parent_l304_30494


namespace NUMINAMATH_GPT_find_a_if_x_is_1_root_l304_30455

theorem find_a_if_x_is_1_root {a : ℝ} (h : (1 : ℝ)^2 + a * 1 - 2 = 0) : a = 1 :=
by sorry

end NUMINAMATH_GPT_find_a_if_x_is_1_root_l304_30455


namespace NUMINAMATH_GPT_triangle_side_length_l304_30424

noncomputable def sine (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180) -- Define sine function explicitly (degrees to radians)

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (hA : A = 30) (hC : C = 45) (ha : a = 4) :
  c = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l304_30424


namespace NUMINAMATH_GPT_determine_y_l304_30440

variable (x y : ℝ)

theorem determine_y (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := 
by 
  sorry

end NUMINAMATH_GPT_determine_y_l304_30440


namespace NUMINAMATH_GPT_train_length_250_meters_l304_30498

open Real

noncomputable def speed_in_ms (speed_km_hr: ℝ): ℝ :=
  speed_km_hr * (1000 / 3600)

noncomputable def length_of_train (speed: ℝ) (time: ℝ): ℝ :=
  speed * time

theorem train_length_250_meters (speed_km_hr: ℝ) (time_seconds: ℝ) :
  speed_km_hr = 40 → time_seconds = 22.5 → length_of_train (speed_in_ms speed_km_hr) time_seconds = 250 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_length_250_meters_l304_30498


namespace NUMINAMATH_GPT_triangle_integral_y_difference_l304_30453

theorem triangle_integral_y_difference :
  ∀ (y : ℕ), (3 ≤ y ∧ y ≤ 15) → (∃ y_min y_max : ℕ, y_min = 3 ∧ y_max = 15 ∧ (y_max - y_min = 12)) :=
by
  intro y
  intro h
  -- skipped proof
  sorry

end NUMINAMATH_GPT_triangle_integral_y_difference_l304_30453


namespace NUMINAMATH_GPT_fraction_of_work_left_l304_30483

theorem fraction_of_work_left 
  (A_days : ℝ) (B_days : ℝ) (work_days : ℝ) 
  (A_work_rate : A_days = 15) 
  (B_work_rate : B_days = 30) 
  (work_duration : work_days = 4)
  : (1 - (work_days * ((1 / A_days) + (1 / B_days)))) = 3 / 5 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_work_left_l304_30483


namespace NUMINAMATH_GPT_system_of_equations_value_l304_30484

theorem system_of_equations_value (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 4 * y - 10 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 96 / 13 := 
sorry

end NUMINAMATH_GPT_system_of_equations_value_l304_30484


namespace NUMINAMATH_GPT_total_capacity_of_bowl_l304_30423

theorem total_capacity_of_bowl (L C : ℕ) (h1 : L / C = 3 / 5) (h2 : C = L + 18) : L + C = 72 := 
by
  sorry

end NUMINAMATH_GPT_total_capacity_of_bowl_l304_30423


namespace NUMINAMATH_GPT_sqrt3_pow_log_sqrt3_8_eq_8_l304_30418

theorem sqrt3_pow_log_sqrt3_8_eq_8 : (Real.sqrt 3) ^ (Real.log 8 / Real.log (Real.sqrt 3)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_sqrt3_pow_log_sqrt3_8_eq_8_l304_30418


namespace NUMINAMATH_GPT_sum_binomial_2k_eq_2_2n_l304_30410

open scoped BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binomial_2k_eq_2_2n (n : ℕ) :
  ∑ k in Finset.range (n + 1), 2^k * binomial_coeff (2*n - k) n = 2^(2*n) := 
by
  sorry

end NUMINAMATH_GPT_sum_binomial_2k_eq_2_2n_l304_30410


namespace NUMINAMATH_GPT_weights_problem_l304_30437

theorem weights_problem
  (weights : Fin 10 → ℝ)
  (h1 : ∀ (i j k l a b c : Fin 10), i ≠ j → i ≠ k → i ≠ l → i ≠ a → i ≠ b → i ≠ c →
    j ≠ k → j ≠ l → j ≠ a → j ≠ b → j ≠ c →
    k ≠ l → k ≠ a → k ≠ b → k ≠ c → 
    l ≠ a → l ≠ b → l ≠ c →
    a ≠ b → a ≠ c →
    b ≠ c →
    weights i + weights j + weights k + weights l > weights a + weights b + weights c)
  (h2 : ∀ (i j : Fin 9), weights i ≤ weights (i + 1)) :
  ∀ (i j k a b : Fin 10), i ≠ j → i ≠ k → i ≠ a → i ≠ b → j ≠ k → j ≠ a → j ≠ b → k ≠ a → k ≠ b → a ≠ b → 
    weights i + weights j + weights k > weights a + weights b := 
sorry

end NUMINAMATH_GPT_weights_problem_l304_30437


namespace NUMINAMATH_GPT_positive_integers_between_300_and_1000_squared_l304_30404

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end NUMINAMATH_GPT_positive_integers_between_300_and_1000_squared_l304_30404


namespace NUMINAMATH_GPT_problem_inequality_l304_30474

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem problem_inequality (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 ≠ x2) :
  (f x2 - f x1) / (x2 - x1) < (1 + Real.log ((x1 + x2) / 2)) :=
sorry

end NUMINAMATH_GPT_problem_inequality_l304_30474


namespace NUMINAMATH_GPT_justify_misha_decision_l304_30450

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end NUMINAMATH_GPT_justify_misha_decision_l304_30450


namespace NUMINAMATH_GPT_min_points_on_dodecahedron_min_points_on_icosahedron_l304_30401

-- Definitions for the dodecahedron problem
def dodecahedron_has_12_faces : Prop := true
def each_vertex_in_dodecahedron_belongs_to_3_faces : Prop := true

-- Proof statement for dodecahedron
theorem min_points_on_dodecahedron : dodecahedron_has_12_faces ∧ each_vertex_in_dodecahedron_belongs_to_3_faces → ∃ n, n = 4 :=
by
  sorry

-- Definitions for the icosahedron problem
def icosahedron_has_20_faces : Prop := true
def icosahedron_has_12_vertices : Prop := true
def each_vertex_in_icosahedron_belongs_to_5_faces : Prop := true
def vertices_of_icosahedron_grouped_into_6_pairs : Prop := true

-- Proof statement for icosahedron
theorem min_points_on_icosahedron : 
  icosahedron_has_20_faces ∧ icosahedron_has_12_vertices ∧ each_vertex_in_icosahedron_belongs_to_5_faces ∧ vertices_of_icosahedron_grouped_into_6_pairs → ∃ n, n = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_points_on_dodecahedron_min_points_on_icosahedron_l304_30401


namespace NUMINAMATH_GPT_general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l304_30441

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

def c_sequence (a b : ℕ → ℤ) (n : ℕ) : ℤ := a n - b n

def sum_c_sequence (c : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum c

theorem general_term_formula_for_b_n (a b : ℕ → ℤ) (n : ℕ) 
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14) :
  b n = 2 * n - 1 :=
sorry

theorem sum_of_first_n_terms_of_c_n (a b : ℕ → ℤ) (n : ℕ)
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14)
  (h7 : ∀ n : ℕ, c_sequence a b n = a n - b n) :
  sum_c_sequence (c_sequence a b) n = (3 ^ n) / 2 - n ^ 2 - 1 / 2 :=
sorry

end NUMINAMATH_GPT_general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l304_30441


namespace NUMINAMATH_GPT_problem_result_l304_30421

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end NUMINAMATH_GPT_problem_result_l304_30421


namespace NUMINAMATH_GPT_base7_to_base10_conversion_l304_30468

theorem base7_to_base10_conversion (n: ℕ) (H: n = 3652) : 
  (3 * 7^3 + 6 * 7^2 + 5 * 7^1 + 2 * 7^0 = 1360) := by
  sorry

end NUMINAMATH_GPT_base7_to_base10_conversion_l304_30468


namespace NUMINAMATH_GPT_true_proposition_l304_30477

theorem true_proposition : 
  (∃ x0 : ℝ, x0 > 0 ∧ 3^x0 + x0 = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, abs x - a * x = abs (-x) - a * (-x)) := by
  sorry

end NUMINAMATH_GPT_true_proposition_l304_30477


namespace NUMINAMATH_GPT_christine_final_throw_difference_l304_30439

def christine_first_throw : ℕ := 20
def janice_first_throw : ℕ := christine_first_throw - 4
def christine_second_throw : ℕ := christine_first_throw + 10
def janice_second_throw : ℕ := janice_first_throw * 2
def janice_final_throw : ℕ := christine_first_throw + 17
def highest_throw : ℕ := 37

theorem christine_final_throw_difference :
  ∃ x : ℕ, christine_second_throw + x = highest_throw ∧ x = 7 := by 
sorry

end NUMINAMATH_GPT_christine_final_throw_difference_l304_30439


namespace NUMINAMATH_GPT_determine_k_l304_30448

theorem determine_k (k : ℝ) (h : 2 - 2^2 = k * (2)^2 + 1) : k = -3/4 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l304_30448


namespace NUMINAMATH_GPT_greatest_divisor_less_than_30_l304_30403

theorem greatest_divisor_less_than_30 :
  (∃ d, d ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} ∧ ∀ m, m ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} → m ≤ d) → 
  18 ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} :=
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_less_than_30_l304_30403


namespace NUMINAMATH_GPT_ineq_five_times_x_minus_six_gt_one_l304_30422

variable {x : ℝ}

theorem ineq_five_times_x_minus_six_gt_one (x : ℝ) : 5 * x - 6 > 1 :=
sorry

end NUMINAMATH_GPT_ineq_five_times_x_minus_six_gt_one_l304_30422


namespace NUMINAMATH_GPT_distance_from_house_to_work_l304_30466

-- Definitions for the conditions
variables (D : ℝ) (speed_to_work speed_back_work : ℝ) (time_to_work time_back_work total_time : ℝ)

-- Specific conditions in the problem
noncomputable def conditions : Prop :=
  (speed_back_work = 20) ∧
  (speed_to_work = speed_back_work / 2) ∧
  (time_to_work = D / speed_to_work) ∧
  (time_back_work = D / speed_back_work) ∧
  (total_time = 6) ∧
  (time_to_work + time_back_work = total_time)

-- The statement to prove the distance D is 40 km given the conditions
theorem distance_from_house_to_work (h : conditions D speed_to_work speed_back_work time_to_work time_back_work total_time) : D = 40 :=
sorry

end NUMINAMATH_GPT_distance_from_house_to_work_l304_30466


namespace NUMINAMATH_GPT_probability_of_less_than_20_l304_30445

variable (total_people : ℕ) (people_over_30 : ℕ)
variable (people_under_20 : ℕ) (probability_under_20 : ℝ)

noncomputable def group_size := total_people = 150
noncomputable def over_30 := people_over_30 = 90
noncomputable def under_20 := people_under_20 = total_people - people_over_30

theorem probability_of_less_than_20
  (total_people_eq : total_people = 150)
  (people_over_30_eq : people_over_30 = 90)
  (people_under_20_eq : people_under_20 = 60)
  (under_20_eq : 60 = total_people - people_over_30) :
  probability_under_20 = people_under_20 / total_people := by
  sorry

end NUMINAMATH_GPT_probability_of_less_than_20_l304_30445


namespace NUMINAMATH_GPT_annual_interest_rate_l304_30499

theorem annual_interest_rate (r : ℝ): 
  (1000 * r * 4.861111111111111 + 1400 * r * 4.861111111111111 = 350) → 
  r = 0.03 :=
sorry

end NUMINAMATH_GPT_annual_interest_rate_l304_30499


namespace NUMINAMATH_GPT_inequality_solution_l304_30443

theorem inequality_solution (x : ℝ) : (5 * x + 3 > 9 - 3 * x ∧ x ≠ 3) ↔ (x > 3 / 4 ∧ x ≠ 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_l304_30443


namespace NUMINAMATH_GPT_first_shipment_weight_l304_30485

variable (first_shipment : ℕ)
variable (total_dishes_made : ℕ := 13)
variable (couscous_per_dish : ℕ := 5)
variable (second_shipment : ℕ := 45)
variable (same_day_shipment : ℕ := 13)

theorem first_shipment_weight :
  13 * 5 = 65 → second_shipment ≠ first_shipment → 
  first_shipment + same_day_shipment = 65 →
  first_shipment = 65 :=
by
  sorry

end NUMINAMATH_GPT_first_shipment_weight_l304_30485


namespace NUMINAMATH_GPT_sam_morning_run_distance_l304_30430

variable (n : ℕ) (x : ℝ)

theorem sam_morning_run_distance (h : x + 2 * n * x + 12 = 18) : x = 6 / (1 + 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_sam_morning_run_distance_l304_30430


namespace NUMINAMATH_GPT_peter_situps_eq_24_l304_30435

noncomputable def situps_peter_did : ℕ :=
  let ratio_peter_greg := 3 / 4
  let situps_greg := 32
  let situps_peter := (3 * situps_greg) / 4
  situps_peter

theorem peter_situps_eq_24 : situps_peter_did = 24 := 
by 
  let h := situps_peter_did
  show h = 24
  sorry

end NUMINAMATH_GPT_peter_situps_eq_24_l304_30435


namespace NUMINAMATH_GPT_num_triangles_with_area_2_l304_30438

-- Define the grid and points
def is_grid_point (x y : ℕ) : Prop := x ≤ 3 ∧ y ≤ 3

-- Function to calculate the area of a triangle using vertices (x1, y1), (x2, y2), and (x3, y3)
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℕ) : ℤ := 
  (x1 * y2 + x2 * y3 + x3 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x1)

-- Check if the area is 2 (since we are dealing with a lattice grid, 
-- we can consider non-fractional form by multiplying by 2 to avoid half-area)
def has_area_2 (x1 y1 x2 y2 x3 y3 : ℕ) : Prop :=
  abs (area_of_triangle x1 y1 x2 y2 x3 y3) = 4

-- Define the main theorem that needs to be proved
theorem num_triangles_with_area_2 : 
  ∃ (n : ℕ), n = 64 ∧
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ), 
  is_grid_point x1 y1 ∧ is_grid_point x2 y2 ∧ is_grid_point x3 y3 ∧ 
  has_area_2 x1 y1 x2 y2 x3 y3 → n = 64 :=
sorry

end NUMINAMATH_GPT_num_triangles_with_area_2_l304_30438


namespace NUMINAMATH_GPT_head_start_proofs_l304_30411

def HeadStartAtoB : ℕ := 150
def HeadStartAtoC : ℕ := 310
def HeadStartAtoD : ℕ := 400

def HeadStartBtoC : ℕ := HeadStartAtoC - HeadStartAtoB
def HeadStartCtoD : ℕ := HeadStartAtoD - HeadStartAtoC
def HeadStartBtoD : ℕ := HeadStartAtoD - HeadStartAtoB

theorem head_start_proofs :
  (HeadStartBtoC = 160) ∧
  (HeadStartCtoD = 90) ∧
  (HeadStartBtoD = 250) :=
by
  sorry

end NUMINAMATH_GPT_head_start_proofs_l304_30411


namespace NUMINAMATH_GPT_find_quotient_l304_30461

-- Variables for larger number L and smaller number S
variables (L S: ℕ)

-- Conditions as definitions
def condition1 := L - S = 1325
def condition2 (quotient: ℕ) := L = S * quotient + 5
def condition3 := L = 1650

-- Statement to prove the quotient is 5
theorem find_quotient : ∃ (quotient: ℕ), condition1 L S ∧ condition2 L S quotient ∧ condition3 L → quotient = 5 := by
  sorry

end NUMINAMATH_GPT_find_quotient_l304_30461


namespace NUMINAMATH_GPT_simplify_fraction_l304_30459

theorem simplify_fraction (a b c d : ℕ) (h₁ : a = 2) (h₂ : b = 462) (h₃ : c = 29) (h₄ : d = 42) :
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) = 107 / 154 :=
by {
  sorry
}

end NUMINAMATH_GPT_simplify_fraction_l304_30459


namespace NUMINAMATH_GPT_problem1_problem2_l304_30473

-- Definitions of sets A and B
def setA : Set ℝ := { x | x^2 - 8 * x + 15 = 0 }
def setB (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }

-- Problem 1: If a = 1/5, B is a subset of A.
theorem problem1 : setB (1 / 5) ⊆ setA := sorry

-- Problem 2: If A ∩ B = B, then C = {0, 1/3, 1/5}.
def setC : Set ℝ := { a | a = 0 ∨ a = 1 / 3 ∨ a = 1 / 5 }

theorem problem2 (a : ℝ) : (setA ∩ setB a = setB a) ↔ (a ∈ setC) := sorry

end NUMINAMATH_GPT_problem1_problem2_l304_30473


namespace NUMINAMATH_GPT_birth_date_of_id_number_l304_30409

def extract_birth_date (id_number : String) := 
  let birth_str := id_number.drop 6 |>.take 8
  let year := birth_str.take 4
  let month := birth_str.drop 4 |>.take 2
  let day := birth_str.drop 6
  (year, month, day)

theorem birth_date_of_id_number :
  extract_birth_date "320106194607299871" = ("1946", "07", "29") := by
  sorry

end NUMINAMATH_GPT_birth_date_of_id_number_l304_30409


namespace NUMINAMATH_GPT_percent_time_in_meetings_l304_30489

-- Define the conditions
def work_day_minutes : ℕ := 10 * 60  -- Total minutes in a 10-hour work day is 600 minutes
def first_meeting_minutes : ℕ := 60  -- The first meeting took 60 minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes  -- The second meeting took three times as long as the first meeting

-- Total time spent in meetings
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes  -- 60 + 180 = 240 minutes

-- The task is to prove that Makarla spent 40% of her work day in meetings.
theorem percent_time_in_meetings : (total_meeting_minutes / work_day_minutes : ℚ) * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_percent_time_in_meetings_l304_30489


namespace NUMINAMATH_GPT_factor_expression_l304_30493

theorem factor_expression (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l304_30493


namespace NUMINAMATH_GPT_fifth_selected_ID_is_01_l304_30454

noncomputable def populationIDs : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

noncomputable def randomNumberTable : List (List ℕ) :=
  [[78, 16, 65, 72,  8, 2, 63, 14,  7, 2, 43, 69, 97, 28,  1, 98],
   [32,  4, 92, 34, 49, 35, 82,  0, 36, 23, 48, 69, 69, 38, 74, 81]]

noncomputable def selectedIDs (table : List (List ℕ)) : List ℕ :=
  [8, 2, 14, 7, 1]  -- Derived from the selection method

theorem fifth_selected_ID_is_01 : (selectedIDs randomNumberTable).get! 4 = 1 := by
  sorry

end NUMINAMATH_GPT_fifth_selected_ID_is_01_l304_30454


namespace NUMINAMATH_GPT_more_whistles_sean_than_charles_l304_30408

def whistles_sean : ℕ := 223
def whistles_charles : ℕ := 128

theorem more_whistles_sean_than_charles : (whistles_sean - whistles_charles) = 95 :=
by
  sorry

end NUMINAMATH_GPT_more_whistles_sean_than_charles_l304_30408


namespace NUMINAMATH_GPT_triangle_reflection_not_necessarily_perpendicular_l304_30446

theorem triangle_reflection_not_necessarily_perpendicular
  (P Q R : ℝ × ℝ)
  (hP : 0 ≤ P.1 ∧ 0 ≤ P.2)
  (hQ : 0 ≤ Q.1 ∧ 0 ≤ Q.2)
  (hR : 0 ≤ R.1 ∧ 0 ≤ R.2)
  (not_on_y_eq_x_P : P.1 ≠ P.2)
  (not_on_y_eq_x_Q : Q.1 ≠ Q.2)
  (not_on_y_eq_x_R : R.1 ≠ R.2) :
  ¬ (∃ (mPQ mPQ' : ℝ), 
      mPQ = (Q.2 - P.2) / (Q.1 - P.1) ∧ 
      mPQ' = (Q.1 - P.1) / (Q.2 - P.2) ∧ 
      mPQ * mPQ' = -1) :=
sorry

end NUMINAMATH_GPT_triangle_reflection_not_necessarily_perpendicular_l304_30446


namespace NUMINAMATH_GPT_minimize_reciprocals_l304_30427

theorem minimize_reciprocals (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 30) :
  (a = 10 ∧ b = 5) → ∀ x y : ℕ, (x > 0) → (y > 0) → (x + 4 * y = 30) → (1 / (x : ℝ) + 1 / (y : ℝ) ≥ 1 / 10 + 1 / 5) := 
by {
  sorry
}

end NUMINAMATH_GPT_minimize_reciprocals_l304_30427


namespace NUMINAMATH_GPT_solution_set_of_inequality_l304_30469

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x - 3) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x < 3} := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l304_30469


namespace NUMINAMATH_GPT_dividend_percentage_l304_30442

theorem dividend_percentage (face_value : ℝ) (investment : ℝ) (roi : ℝ) (dividend_percentage : ℝ) 
    (h1 : face_value = 40) 
    (h2 : investment = 20) 
    (h3 : roi = 0.25) : dividend_percentage = 12.5 := 
  sorry

end NUMINAMATH_GPT_dividend_percentage_l304_30442


namespace NUMINAMATH_GPT_count_three_digit_numbers_with_digit_sum_24_l304_30497

-- Define the conditions:
def isThreeDigitNumber (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  (100 * a + 10 * b + c ≥ 100)

def digitSumEquals24 (a b c : ℕ) : Prop :=
  a + b + c = 24

-- State the theorem:
theorem count_three_digit_numbers_with_digit_sum_24 :
  (∃ (count : ℕ), count = 10 ∧ 
   ∀ (a b c : ℕ), isThreeDigitNumber a b c ∧ digitSumEquals24 a b c → (count = 10)) :=
sorry

end NUMINAMATH_GPT_count_three_digit_numbers_with_digit_sum_24_l304_30497


namespace NUMINAMATH_GPT_chess_player_max_consecutive_win_prob_l304_30426

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end NUMINAMATH_GPT_chess_player_max_consecutive_win_prob_l304_30426


namespace NUMINAMATH_GPT_max_value_fraction_l304_30487

theorem max_value_fraction (x y : ℝ) : 
  (2 * x + 3 * y + 2) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 17 :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l304_30487


namespace NUMINAMATH_GPT_polynomial_divisibility_l304_30425

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (h_pos : 0 < n) :
  ∃ Q : Polynomial ℝ, (P * P + Q * Q) % (X * X + 1)^n = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l304_30425


namespace NUMINAMATH_GPT_range_of_a_l304_30472

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 1 then x else a * x^2 + 2 * x

theorem range_of_a (R : Set ℝ) :
  (∀ x : ℝ, f x a ∈ R) → (a ∈ Set.Icc (-1 : ℝ) 0) :=
sorry

end NUMINAMATH_GPT_range_of_a_l304_30472


namespace NUMINAMATH_GPT_bicycle_spokes_count_l304_30451

theorem bicycle_spokes_count (bicycles wheels spokes : ℕ) 
       (h1 : bicycles = 4) 
       (h2 : wheels = 2) 
       (h3 : spokes = 10) : 
       bicycles * (wheels * spokes) = 80 :=
by
  sorry

end NUMINAMATH_GPT_bicycle_spokes_count_l304_30451


namespace NUMINAMATH_GPT_inequality_for_M_cap_N_l304_30480

def f (x : ℝ) := 2 * |x - 1| + x - 1
def g (x : ℝ) := 16 * x^2 - 8 * x + 1

def M := {x : ℝ | 0 ≤ x ∧ x ≤ 4 / 3}
def N := {x : ℝ | -1 / 4 ≤ x ∧ x ≤ 3 / 4}
def M_cap_N := {x : ℝ | 0 ≤ x ∧ x ≤ 3 / 4}

theorem inequality_for_M_cap_N (x : ℝ) (hx : x ∈ M_cap_N) : x^2 * f x + x * (f x)^2 ≤ 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_for_M_cap_N_l304_30480


namespace NUMINAMATH_GPT_division_of_neg_six_by_three_l304_30490

theorem division_of_neg_six_by_three : (-6) / 3 = -2 := by
  sorry

end NUMINAMATH_GPT_division_of_neg_six_by_three_l304_30490


namespace NUMINAMATH_GPT_proof_inequality_l304_30405

noncomputable def inequality_proof (α : ℝ) (a b : ℝ) (m : ℕ) : Prop :=
  (0 < α) → (α < Real.pi / 2) →
  (m ≥ 1) →
  (0 < a) → (0 < b) →
  (a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2))

-- Statement of the proof problem
theorem proof_inequality (α : ℝ) (a b : ℝ) (m : ℕ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : 1 ≤ m) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ 
    (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2) :=
by
  sorry

end NUMINAMATH_GPT_proof_inequality_l304_30405


namespace NUMINAMATH_GPT_first_nonzero_digit_one_div_139_l304_30481

theorem first_nonzero_digit_one_div_139 :
  ∀ n : ℕ, (n > 0 → (∀ m : ℕ, (m > 0 → (m * 10^n) ∣ (10^n * 1 - 1) ∧ n ∣ (139 * 10 ^ (n + 1)) ∧ 10^(n+1 - 1) * 1 - 1 < 10^n))) :=
sorry

end NUMINAMATH_GPT_first_nonzero_digit_one_div_139_l304_30481


namespace NUMINAMATH_GPT_valid_integer_values_n_l304_30428

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem valid_integer_values_n : ∃ (n_values : ℕ), n_values = 3 ∧
  ∀ n : ℤ, is_integer (3200 * (2 / 5) ^ (2 * n)) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_valid_integer_values_n_l304_30428


namespace NUMINAMATH_GPT_RS_segment_length_l304_30486

theorem RS_segment_length (P Q R S : ℝ) (r1 r2 : ℝ) (hP : P = 0) (hQ : Q = 10) (rP : r1 = 6) (rQ : r2 = 4) :
    (∃ PR QR SR : ℝ, PR = 6 ∧ QR = 4 ∧ SR = 6) → (R - S = 12) :=
by
  sorry

end NUMINAMATH_GPT_RS_segment_length_l304_30486


namespace NUMINAMATH_GPT_maxim_is_correct_l304_30416

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end NUMINAMATH_GPT_maxim_is_correct_l304_30416


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l304_30449

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 1 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {2} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l304_30449


namespace NUMINAMATH_GPT_model_lighthouse_height_l304_30407

theorem model_lighthouse_height (h_actual : ℝ) (V_actual : ℝ) (V_model : ℝ) (h_actual_val : h_actual = 60) (V_actual_val : V_actual = 150000) (V_model_val : V_model = 0.15) :
  (h_actual * (V_model / V_actual)^(1/3)) = 0.6 :=
by
  rw [h_actual_val, V_actual_val, V_model_val]
  sorry

end NUMINAMATH_GPT_model_lighthouse_height_l304_30407


namespace NUMINAMATH_GPT_ratio_pentagon_area_l304_30492

noncomputable def square_side_length := 1
noncomputable def square_area := (square_side_length : ℝ)^2
noncomputable def total_area := 3 * square_area
noncomputable def area_triangle (base height : ℝ) := 0.5 * base * height
noncomputable def GC := 2 / 3 * square_side_length
noncomputable def HD := 2 / 3 * square_side_length
noncomputable def area_GJC := area_triangle GC square_side_length
noncomputable def area_HDJ := area_triangle HD square_side_length
noncomputable def area_AJKCB := square_area - (area_GJC + area_HDJ)

theorem ratio_pentagon_area :
  (area_AJKCB / total_area) = 1 / 9 := 
sorry

end NUMINAMATH_GPT_ratio_pentagon_area_l304_30492


namespace NUMINAMATH_GPT_product_form_l304_30488

theorem product_form (b a : ℤ) :
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := 
sorry

end NUMINAMATH_GPT_product_form_l304_30488


namespace NUMINAMATH_GPT_toy_poodle_height_l304_30419

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end NUMINAMATH_GPT_toy_poodle_height_l304_30419


namespace NUMINAMATH_GPT_Morio_age_when_Michiko_was_born_l304_30478

theorem Morio_age_when_Michiko_was_born (Teresa_age_now : ℕ) (Teresa_age_when_Michiko_born : ℕ) (Morio_age_now : ℕ)
  (hTeresa : Teresa_age_now = 59) (hTeresa_born : Teresa_age_when_Michiko_born = 26) (hMorio : Morio_age_now = 71) :
  Morio_age_now - (Teresa_age_now - Teresa_age_when_Michiko_born) = 38 :=
by
  sorry

end NUMINAMATH_GPT_Morio_age_when_Michiko_was_born_l304_30478


namespace NUMINAMATH_GPT_external_angle_bisector_lengths_l304_30415

noncomputable def f_a (a b c : ℝ) : ℝ := 4 * Real.sqrt 3
noncomputable def f_b (b : ℝ) : ℝ := 6 / Real.sqrt 7
noncomputable def f_c (a b c : ℝ) : ℝ := 4 * Real.sqrt 3

theorem external_angle_bisector_lengths (a b c : ℝ) 
  (ha : a = 5 - Real.sqrt 7)
  (hb : b = 6)
  (hc : c = 5 + Real.sqrt 7) :
  f_a a b c = 4 * Real.sqrt 3 ∧
  f_b b = 6 / Real.sqrt 7 ∧
  f_c a b c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_external_angle_bisector_lengths_l304_30415


namespace NUMINAMATH_GPT_range_of_a_l304_30475

variable (x a : ℝ)

def p : Prop := x^2 - 2 * x - 3 ≥ 0

def q : Prop := x^2 - (2 * a - 1) * x + a * (a - 1) ≥ 0

def sufficient_but_not_necessary (p q : Prop) : Prop := 
  (p → q) ∧ ¬(q → p)

theorem range_of_a (a : ℝ) : (∃ x, sufficient_but_not_necessary (p x) (q a x)) → (0 ≤ a ∧ a ≤ 3) := 
sorry

end NUMINAMATH_GPT_range_of_a_l304_30475


namespace NUMINAMATH_GPT_part1_part2_l304_30470

variable {a b c m t y1 y2 : ℝ}

-- Condition: point (2, m) lies on the parabola y = ax^2 + bx + c where axis of symmetry is x = t
def point_lies_on_parabola (a b c m : ℝ) := m = a * 2^2 + b * 2 + c

-- Condition: axis of symmetry x = t
def axis_of_symmetry (a b t : ℝ) := t = -b / (2 * a)

-- Condition: m = c
theorem part1 (a c : ℝ) (h : m = c) (h₀ : point_lies_on_parabola a (-2 * a) c m) :
  axis_of_symmetry a (-2 * a) 1 :=
by sorry

-- Additional Condition: c < m
def c_lt_m (c m : ℝ) := c < m

-- Points (-1, y1) and (3, y2) lie on the parabola y = ax^2 + bx + c
def points_on_parabola (a b c y1 y2 : ℝ) :=
  y1 = a * (-1)^2 + b * (-1) + c ∧ y2 = a * 3^2 + b * 3 + c

-- Comparison result
theorem part2 (a : ℝ) (h₁ : c_lt_m c m) (h₂ : 2 * a + (-2 * a) > 0) (h₂' : points_on_parabola a (-2 * a) c y1 y2) :
  y2 > y1 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l304_30470


namespace NUMINAMATH_GPT_total_weight_of_watermelons_l304_30476

theorem total_weight_of_watermelons (w1 w2 : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) :
  w1 + w2 = 14.02 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_watermelons_l304_30476


namespace NUMINAMATH_GPT_black_balls_number_l304_30420

-- Define the given conditions and the problem statement as Lean statements
theorem black_balls_number (n : ℕ) (h : (2 : ℝ) / (n + 2 : ℝ) = 0.4) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_black_balls_number_l304_30420


namespace NUMINAMATH_GPT_find_value_correct_l304_30452

-- Definitions for the given conditions
def equation1 (a b : ℚ) : Prop := 3 * a - b = 8
def equation2 (a b : ℚ) : Prop := 4 * b + 7 * a = 13

-- Definition for the question
def find_value (a b : ℚ) : ℚ := 2 * a + b

-- Statement of the proof
theorem find_value_correct (a b : ℚ) (h1 : equation1 a b) (h2 : equation2 a b) : find_value a b = 73 / 19 := 
by 
  sorry

end NUMINAMATH_GPT_find_value_correct_l304_30452


namespace NUMINAMATH_GPT_largest_shaded_area_figure_C_l304_30447

noncomputable def area_of_square (s : ℝ) : ℝ := s^2
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def shaded_area_of_figure_A : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_B : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_C : ℝ := Real.pi - 2

theorem largest_shaded_area_figure_C : shaded_area_of_figure_C > shaded_area_of_figure_A ∧ shaded_area_of_figure_C > shaded_area_of_figure_B := by
  sorry

end NUMINAMATH_GPT_largest_shaded_area_figure_C_l304_30447


namespace NUMINAMATH_GPT_system_solutions_range_b_l304_30417

theorem system_solutions_range_b (b : ℝ) :
  (∀ x y : ℝ, x^2 - y^2 = 0 → x^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0 ∨ y = b) →
  b ≥ 2 ∨ b ≤ -2 :=
sorry

end NUMINAMATH_GPT_system_solutions_range_b_l304_30417


namespace NUMINAMATH_GPT_octagon_area_in_square_l304_30412

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem octagon_area_in_square :
  ∀ (s : ℝ), ∀ (area_square : ℝ), ∀ (area_octagon : ℝ),
  (s * 4 = 160) →
  (s = 40) →
  (area_square = s * s) →
  (area_square = 1600) →
  (∃ (area_triangle : ℝ), area_triangle = 50 ∧ 8 * area_triangle = 400) →
  (area_octagon = area_square - 400) →
  (area_octagon = 1200) :=
by
  intros s area_square area_octagon h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_octagon_area_in_square_l304_30412


namespace NUMINAMATH_GPT_max_m_value_l304_30434

theorem max_m_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → ((2 / a) + (1 / b) ≥ (m / (2 * a + b)))) → m ≤ 9 :=
sorry

end NUMINAMATH_GPT_max_m_value_l304_30434


namespace NUMINAMATH_GPT_subway_distance_per_minute_l304_30479

theorem subway_distance_per_minute :
  let total_distance := 120 -- kilometers
  let total_time := 110 -- minutes (1 hour and 50 minutes)
  let bus_time := 70 -- minutes (1 hour and 10 minutes)
  let bus_distance := (14 * 40.8) / 6 -- kilometers
  let subway_distance := total_distance - bus_distance -- kilometers
  let subway_time := total_time - bus_time -- minutes
  let distance_per_minute := subway_distance / subway_time
  distance_per_minute = 0.62 := 
by
  sorry

end NUMINAMATH_GPT_subway_distance_per_minute_l304_30479


namespace NUMINAMATH_GPT_perfect_match_of_products_l304_30432

theorem perfect_match_of_products
  (x : ℕ)  -- number of workers assigned to produce nuts
  (h1 : 22 - x ≥ 0)  -- ensuring non-negative number of workers for screws
  (h2 : 1200 * (22 - x) = 2 * 2000 * x) :  -- the condition for perfect matching
  (2 * 1200 * (22 - x) = 2000 * x) :=  -- the correct equation
by sorry

end NUMINAMATH_GPT_perfect_match_of_products_l304_30432


namespace NUMINAMATH_GPT_TamekaBoxesRelation_l304_30402

theorem TamekaBoxesRelation 
  (S : ℤ)
  (h1 : 40 + S + S / 2 = 145) :
  S - 40 = 30 :=
by
  sorry

end NUMINAMATH_GPT_TamekaBoxesRelation_l304_30402
