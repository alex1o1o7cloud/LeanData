import Mathlib

namespace NUMINAMATH_GPT_average_pages_per_day_is_correct_l2181_218177

-- Definitions based on the given conditions
def first_book_pages := 249
def first_book_days := 3

def second_book_pages := 379
def second_book_days := 5

def third_book_pages := 480
def third_book_days := 6

-- Definition of total pages read
def total_pages := first_book_pages + second_book_pages + third_book_pages

-- Definition of total days spent reading
def total_days := first_book_days + second_book_days + third_book_days

-- Definition of expected average pages per day
def expected_average_pages_per_day := 79.14

-- The theorem to prove
theorem average_pages_per_day_is_correct : (total_pages.toFloat / total_days.toFloat) = expected_average_pages_per_day :=
by
  sorry

end NUMINAMATH_GPT_average_pages_per_day_is_correct_l2181_218177


namespace NUMINAMATH_GPT_total_pages_read_l2181_218179

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end NUMINAMATH_GPT_total_pages_read_l2181_218179


namespace NUMINAMATH_GPT_inequality_example_l2181_218144

theorem inequality_example (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (habc_sum : a + b + c = 3) :
  18 * ((1 / ((3 - a) * (4 - a))) + (1 / ((3 - b) * (4 - b))) + (1 / ((3 - c) * (4 - c)))) + 2 * (a * b + b * c + c * a) ≥ 15 :=
by
  sorry

end NUMINAMATH_GPT_inequality_example_l2181_218144


namespace NUMINAMATH_GPT_solve_for_x_l2181_218103

-- Define the equation
def equation (x : ℝ) : Prop := (x^2 + 3 * x + 4) / (x + 5) = x + 6

-- Prove that x = -13 / 4 satisfies the equation
theorem solve_for_x : ∃ x : ℝ, equation x ∧ x = -13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2181_218103


namespace NUMINAMATH_GPT_opposite_of_negative_2023_l2181_218181

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end NUMINAMATH_GPT_opposite_of_negative_2023_l2181_218181


namespace NUMINAMATH_GPT_calc_expr_eq_l2181_218192

theorem calc_expr_eq : 2 + 3 / (4 + 5 / 6) = 76 / 29 := 
by 
  sorry

end NUMINAMATH_GPT_calc_expr_eq_l2181_218192


namespace NUMINAMATH_GPT_trees_left_after_typhoon_l2181_218199

theorem trees_left_after_typhoon (trees_grown : ℕ) (trees_died : ℕ) (h1 : trees_grown = 17) (h2 : trees_died = 5) : (trees_grown - trees_died = 12) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_trees_left_after_typhoon_l2181_218199


namespace NUMINAMATH_GPT_inequality_l2181_218168

theorem inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) : 
  (a / b) + (b / c) + (c / a) + (b / a) + (a / c) + (c / b) + 6 ≥ 
  2 * Real.sqrt 2 * (Real.sqrt ((1 - a) / a) + Real.sqrt ((1 - b) / b) + Real.sqrt ((1 - c) / c)) :=
sorry

end NUMINAMATH_GPT_inequality_l2181_218168


namespace NUMINAMATH_GPT_matrix_product_l2181_218149

-- Define matrix A
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 3, 2], ![1, -3, 4]]

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, 0], ![2, 0, 4], ![3, 0, 1]]

-- Define the expected result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![9, 6, -1], ![12, 0, 14], ![7, 3, -8]]

-- The statement to prove
theorem matrix_product : A * B = C :=
by
  sorry

end NUMINAMATH_GPT_matrix_product_l2181_218149


namespace NUMINAMATH_GPT_B_representation_l2181_218185

def A : Set ℤ := {-1, 2, 3, 4}

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

def B : Set ℤ := {y | ∃ x ∈ A, y = f x}

theorem B_representation : B = {2, 5, 10} :=
by {
  -- Proof to be provided
  sorry
}

end NUMINAMATH_GPT_B_representation_l2181_218185


namespace NUMINAMATH_GPT_elvis_writing_time_per_song_l2181_218100

-- Define the conditions based on the problem statement
def total_studio_time_minutes := 300   -- 5 hours converted to minutes
def songs := 10
def recording_time_per_song := 12
def total_editing_time := 30

-- Define the total recording time
def total_recording_time := songs * recording_time_per_song

-- Define the total time available for writing songs
def total_writing_time := total_studio_time_minutes - total_recording_time - total_editing_time

-- Define the time to write each song
def time_per_song_writing := total_writing_time / songs

-- State the proof goal
theorem elvis_writing_time_per_song : time_per_song_writing = 15 := by
  sorry

end NUMINAMATH_GPT_elvis_writing_time_per_song_l2181_218100


namespace NUMINAMATH_GPT_largest_angle_is_176_l2181_218135

-- Define the angles of the pentagon
def angle1 (y : ℚ) : ℚ := y
def angle2 (y : ℚ) : ℚ := 2 * y + 2
def angle3 (y : ℚ) : ℚ := 3 * y - 3
def angle4 (y : ℚ) : ℚ := 4 * y + 4
def angle5 (y : ℚ) : ℚ := 5 * y - 5

-- Define the function to calculate the largest angle
def largest_angle (y : ℚ) : ℚ := 5 * y - 5

-- Problem statement: Prove that the largest angle in the pentagon is 176 degrees
theorem largest_angle_is_176 (y : ℚ) (h : angle1 y + angle2 y + angle3 y + angle4 y + angle5 y = 540) :
  largest_angle y = 176 :=
by sorry

end NUMINAMATH_GPT_largest_angle_is_176_l2181_218135


namespace NUMINAMATH_GPT_correct_product_exists_l2181_218124

variable (a b : ℕ)

theorem correct_product_exists
  (h1 : a < 100)
  (h2 : 10 * (a % 10) + a / 10 = 14)
  (h3 : 14 * b = 182) : a * b = 533 := sorry

end NUMINAMATH_GPT_correct_product_exists_l2181_218124


namespace NUMINAMATH_GPT_average_one_half_one_fourth_one_eighth_l2181_218146

theorem average_one_half_one_fourth_one_eighth : 
  ((1 / 2.0 + 1 / 4.0 + 1 / 8.0) / 3.0) = 7 / 24 := 
by sorry

end NUMINAMATH_GPT_average_one_half_one_fourth_one_eighth_l2181_218146


namespace NUMINAMATH_GPT_greatest_integer_less_than_neg_eight_over_three_l2181_218122

theorem greatest_integer_less_than_neg_eight_over_three :
  ∃ (z : ℤ), (z < -8 / 3) ∧ ∀ w : ℤ, (w < -8 / 3) → w ≤ z := by
  sorry

end NUMINAMATH_GPT_greatest_integer_less_than_neg_eight_over_three_l2181_218122


namespace NUMINAMATH_GPT_Robie_l2181_218114

def initial_bags (X : ℕ) := (X - 2) + 3 = 4

theorem Robie's_initial_bags (X : ℕ) (h : initial_bags X) : X = 3 :=
by
  unfold initial_bags at h
  sorry

end NUMINAMATH_GPT_Robie_l2181_218114


namespace NUMINAMATH_GPT_find_value_of_expression_l2181_218184

theorem find_value_of_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : x^2 + 2*x + 3 = 4 := by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l2181_218184


namespace NUMINAMATH_GPT_num_of_possible_outcomes_l2181_218123

def participants : Fin 6 := sorry  -- Define the participants as elements of Fin 6

theorem num_of_possible_outcomes : (6 * 5 * 4 = 120) :=
by {
  -- Prove this mathematical statement
  rfl
}

end NUMINAMATH_GPT_num_of_possible_outcomes_l2181_218123


namespace NUMINAMATH_GPT_tank_empty_time_l2181_218183

noncomputable def capacity : ℝ := 5760
noncomputable def leak_rate_time : ℝ := 6
noncomputable def inlet_rate_per_minute : ℝ := 4

-- leak rate calculation
noncomputable def leak_rate : ℝ := capacity / leak_rate_time

-- inlet rate calculation in litres per hour
noncomputable def inlet_rate : ℝ := inlet_rate_per_minute * 60

-- net emptying rate calculation
noncomputable def net_empty_rate : ℝ := leak_rate - inlet_rate

-- time to empty the tank calculation
noncomputable def time_to_empty : ℝ := capacity / net_empty_rate

-- The statement to prove
theorem tank_empty_time : time_to_empty = 8 :=
by
  -- Definition step
  have h1 : leak_rate = capacity / leak_rate_time := rfl
  have h2 : inlet_rate = inlet_rate_per_minute * 60 := rfl
  have h3 : net_empty_rate = leak_rate - inlet_rate := rfl
  have h4 : time_to_empty = capacity / net_empty_rate := rfl

  -- Final proof (skipped with sorry)
  sorry

end NUMINAMATH_GPT_tank_empty_time_l2181_218183


namespace NUMINAMATH_GPT_Margo_total_distance_walked_l2181_218106

theorem Margo_total_distance_walked :
  ∀ (d : ℝ),
  (5 * (d / 5) + 3 * (d / 3) = 1) →
  (2 * d = 3.75) :=
by
  sorry

end NUMINAMATH_GPT_Margo_total_distance_walked_l2181_218106


namespace NUMINAMATH_GPT_M_greater_than_N_l2181_218132

variable (a : ℝ)

def M := 2 * a^2 - 4 * a
def N := a^2 - 2 * a - 3

theorem M_greater_than_N : M a > N a := by
  sorry

end NUMINAMATH_GPT_M_greater_than_N_l2181_218132


namespace NUMINAMATH_GPT_f_monotonicity_l2181_218150

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem f_monotonicity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x : ℝ, x > 0 → deriv (f a) x > 0) ∧ (∀ x : ℝ, x < 0 → deriv (f a) x < 0) :=
by
  sorry

end NUMINAMATH_GPT_f_monotonicity_l2181_218150


namespace NUMINAMATH_GPT_perimeter_of_triangle_l2181_218182

-- Define the side lengths of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 7

-- Define the third side of the triangle, which is an even number and satisfies the triangle inequality conditions
def side3 : ℕ := 6

-- Define the theorem to prove the perimeter of the triangle
theorem perimeter_of_triangle : side1 + side2 + side3 = 15 := by
  -- The proof is omitted for brevity
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l2181_218182


namespace NUMINAMATH_GPT_barrels_oil_difference_l2181_218139

/--
There are two barrels of oil, A and B.
1. $\frac{1}{3}$ of the oil is poured from barrel A into barrel B.
2. $\frac{1}{5}$ of the oil is poured from barrel B back into barrel A.
3. Each barrel contains 24kg of oil after the transfers.

Prove that originally, barrel A had 6 kg more oil than barrel B.
-/
theorem barrels_oil_difference :
  ∃ (x y : ℝ), (y = 48 - x) ∧
  (24 = (2 / 3) * x + (1 / 5) * (48 - x + (1 / 3) * x)) ∧
  (24 = (48 - x + (1 / 3) * x) * (4 / 5)) ∧
  (x - y = 6) :=
by
  sorry

end NUMINAMATH_GPT_barrels_oil_difference_l2181_218139


namespace NUMINAMATH_GPT_marsh_ducks_l2181_218117

theorem marsh_ducks (D : ℕ) (h1 : 58 = D + 21) : D = 37 := 
by {
  sorry
}

end NUMINAMATH_GPT_marsh_ducks_l2181_218117


namespace NUMINAMATH_GPT_solve_abs_eq_2005_l2181_218195

theorem solve_abs_eq_2005 (x : ℝ) : |2005 * x - 2005| = 2005 ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_abs_eq_2005_l2181_218195


namespace NUMINAMATH_GPT_find_x_solution_l2181_218148

theorem find_x_solution (x : ℝ) 
  (h : ∑' n:ℕ, ((-1)^(n+1)) * (2 * n + 1) * x^n = 16) : 
  x = -15/16 :=
sorry

end NUMINAMATH_GPT_find_x_solution_l2181_218148


namespace NUMINAMATH_GPT_no_leopards_in_circus_l2181_218153

theorem no_leopards_in_circus (L T : ℕ) (N : ℕ) (h₁ : L = N / 5) (h₂ : T = 5 * (N - T)) : 
  ∀ A, A = L + N → A = T + (N - T) → ¬ ∃ x, x ≠ L ∧ x ≠ T ∧ x ≠ (N - L - T) :=
by
  sorry

end NUMINAMATH_GPT_no_leopards_in_circus_l2181_218153


namespace NUMINAMATH_GPT_imaginary_part_of_fraction_l2181_218101

theorem imaginary_part_of_fraction (i : ℂ) (h : i^2 = -1) : ( (i^2) / (2 * i - 1) ).im = (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_fraction_l2181_218101


namespace NUMINAMATH_GPT_correct_operation_l2181_218133

variable (a b : ℝ)

theorem correct_operation : 3 * a^2 * b - b * a^2 = 2 * a^2 * b := 
sorry

end NUMINAMATH_GPT_correct_operation_l2181_218133


namespace NUMINAMATH_GPT_peach_count_l2181_218127

theorem peach_count (n : ℕ) : n % 4 = 2 ∧ n % 6 = 4 ∧ n % 8 = 6 ∧ 120 ≤ n ∧ n ≤ 150 → n = 142 :=
sorry

end NUMINAMATH_GPT_peach_count_l2181_218127


namespace NUMINAMATH_GPT_round_table_chairs_l2181_218138

theorem round_table_chairs :
  ∃ x : ℕ, (2 * x + 2 * 7 = 26) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_round_table_chairs_l2181_218138


namespace NUMINAMATH_GPT_smallest_pos_mult_of_31_mod_97_l2181_218140

theorem smallest_pos_mult_of_31_mod_97 {k : ℕ} (h : 31 * k % 97 = 6) : 31 * k = 2015 :=
sorry

end NUMINAMATH_GPT_smallest_pos_mult_of_31_mod_97_l2181_218140


namespace NUMINAMATH_GPT_find_varphi_l2181_218145

theorem find_varphi
  (ϕ : ℝ)
  (h : ∃ k : ℤ, ϕ = (π / 8) + (k * π / 2)) :
  ϕ = π / 8 :=
by
  sorry

end NUMINAMATH_GPT_find_varphi_l2181_218145


namespace NUMINAMATH_GPT_age_ratio_in_six_years_l2181_218163

-- Definitions for Claire's and Pete's current ages
variables (c p : ℕ)

-- Conditions given in the problem
def condition1 : Prop := c - 3 = 2 * (p - 3)
def condition2 : Prop := p - 7 = (1 / 4) * (c - 7)

-- The proof problem statement
theorem age_ratio_in_six_years (c p : ℕ) (h1 : condition1 c p) (h2 : condition2 c p) : 
  (c + 6) = 3 * (p + 6) :=
sorry

end NUMINAMATH_GPT_age_ratio_in_six_years_l2181_218163


namespace NUMINAMATH_GPT_fibonacci_recurrence_l2181_218125

theorem fibonacci_recurrence (f : ℕ → ℝ) (a b : ℝ) 
  (h₀ : f 0 = 1) 
  (h₁ : f 1 = 1) 
  (h₂ : ∀ n, f (n + 2) = f (n + 1) + f n)
  (h₃ : a + b = 1) 
  (h₄ : a * b = -1) 
  (h₅ : a > b) 
  : ∀ n, f n = (a ^ (n + 1) - b ^ (n + 1)) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_fibonacci_recurrence_l2181_218125


namespace NUMINAMATH_GPT_reciprocal_of_neg3_l2181_218156

theorem reciprocal_of_neg3 : (1 / (-3) = -1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg3_l2181_218156


namespace NUMINAMATH_GPT_power_of_four_l2181_218119

theorem power_of_four (x : ℕ) (h : 5^29 * 4^x = 2 * 10^29) : x = 15 := by
  sorry

end NUMINAMATH_GPT_power_of_four_l2181_218119


namespace NUMINAMATH_GPT_dot_product_solution_1_l2181_218187

variable (a b : ℝ × ℝ)
variable (k : ℝ)

def two_a_add_b (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 + b.1, 2 * a.2 + b.2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_solution_1 :
  let a := (1, -1)
  let b := (-1, 2)
  dot_product (two_a_add_b a b) a = 1 := by
sorry

end NUMINAMATH_GPT_dot_product_solution_1_l2181_218187


namespace NUMINAMATH_GPT_three_configuration_m_separable_l2181_218171

theorem three_configuration_m_separable
  {n m : ℕ} (A : Finset (Fin n)) (h : m ≥ n / 2) :
  ∀ (C : Finset (Fin n)), C.card = 3 → ∃ B : Finset (Fin n), B.card = m ∧ (∀ c ∈ C, ∃ b ∈ B, c ≠ b) :=
by
  sorry

end NUMINAMATH_GPT_three_configuration_m_separable_l2181_218171


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2181_218121

-- Defining the variables with given values
def a : ℚ := 1 / 2
def b : ℚ := -2

-- Expression to be simplified and evaluated
def expression : ℚ := (2 * a + b) ^ 2 - (2 * a - b) * (a + b) - 2 * (a - 2 * b) * (a + 2 * b)

-- The main theorem
theorem simplify_and_evaluate : expression = 37 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2181_218121


namespace NUMINAMATH_GPT_smallest_n_7n_eq_n7_mod_3_l2181_218186

theorem smallest_n_7n_eq_n7_mod_3 : ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 3]) ∧ ∀ m : ℕ, m > 0 → (7^m ≡ m^7 [MOD 3] → m ≥ n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_7n_eq_n7_mod_3_l2181_218186


namespace NUMINAMATH_GPT_animath_extortion_l2181_218110

noncomputable def max_extortion (n : ℕ) : ℕ :=
2^n - n - 1 

theorem animath_extortion (n : ℕ) :
  ∃ steps : ℕ, steps < (2^n - n - 1) :=
sorry

end NUMINAMATH_GPT_animath_extortion_l2181_218110


namespace NUMINAMATH_GPT_largest_real_number_l2181_218180

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_real_number_l2181_218180


namespace NUMINAMATH_GPT_find_relationship_l2181_218155

noncomputable def log_equation (c d : ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 1 → 6 * (Real.log (x) / Real.log (c))^2 + 5 * (Real.log (x) / Real.log (d))^2 = 12 * (Real.log (x))^2 / (Real.log (c) * Real.log (d))

theorem find_relationship (c d : ℝ) :
  log_equation c d → 
    (d = c ^ (5 / (6 + Real.sqrt 6)) ∨ d = c ^ (5 / (6 - Real.sqrt 6))) :=
by
  sorry

end NUMINAMATH_GPT_find_relationship_l2181_218155


namespace NUMINAMATH_GPT_johns_overall_profit_l2181_218194

def cost_price_grinder : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_grinder : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10

noncomputable def loss_amount_grinder : ℝ := loss_percent_grinder * cost_price_grinder
noncomputable def selling_price_grinder : ℝ := cost_price_grinder - loss_amount_grinder

noncomputable def profit_amount_mobile : ℝ := profit_percent_mobile * cost_price_mobile
noncomputable def selling_price_mobile : ℝ := cost_price_mobile + profit_amount_mobile

noncomputable def total_cost_price : ℝ := cost_price_grinder + cost_price_mobile
noncomputable def total_selling_price : ℝ := selling_price_grinder + selling_price_mobile
noncomputable def overall_profit : ℝ := total_selling_price - total_cost_price

theorem johns_overall_profit :
  overall_profit = 50 := 
by
  sorry

end NUMINAMATH_GPT_johns_overall_profit_l2181_218194


namespace NUMINAMATH_GPT_heximal_to_binary_k_value_l2181_218128

theorem heximal_to_binary_k_value (k : ℕ) (h : 10 * (6^3) + k * 6 + 5 = 239) : 
  k = 3 :=
by
  sorry

end NUMINAMATH_GPT_heximal_to_binary_k_value_l2181_218128


namespace NUMINAMATH_GPT_determine_OP_l2181_218164

theorem determine_OP
  (a b c d e : ℝ)
  (h_dist_OA : a > 0)
  (h_dist_OB : b > 0)
  (h_dist_OC : c > 0)
  (h_dist_OD : d > 0)
  (h_dist_OE : e > 0)
  (h_c_le_d : c ≤ d)
  (P : ℝ)
  (hP : c ≤ P ∧ P ≤ d)
  (h_ratio : ∀ (P : ℝ) (hP : c ≤ P ∧ P ≤ d), (a - P) / (P - e) = (c - P) / (P - d)) :
  P = (ce - ad) / (a - c + e - d) :=
sorry

end NUMINAMATH_GPT_determine_OP_l2181_218164


namespace NUMINAMATH_GPT_unique_solution_c_exceeds_s_l2181_218118

-- Problem Conditions
def steers_cost : ℕ := 35
def cows_cost : ℕ := 40
def total_budget : ℕ := 1200

-- Definition of the solution conditions
def valid_purchase (s c : ℕ) : Prop := 
  steers_cost * s + cows_cost * c = total_budget ∧ s > 0 ∧ c > 0

-- Statement to prove
theorem unique_solution_c_exceeds_s :
  ∃ s c : ℕ, valid_purchase s c ∧ c > s ∧ ∀ (s' c' : ℕ), valid_purchase s' c' → s' = 8 ∧ c' = 17 :=
sorry

end NUMINAMATH_GPT_unique_solution_c_exceeds_s_l2181_218118


namespace NUMINAMATH_GPT_ellipse_equation_with_foci_l2181_218107

theorem ellipse_equation_with_foci (M N P : ℝ × ℝ)
  (area_triangle : Real) (tan_M tan_N : ℝ)
  (h₁ : area_triangle = 1)
  (h₂ : tan_M = 1 / 2)
  (h₃ : tan_N = -2) :
  ∃ (a b : ℝ), (4 * x^2) / (15 : ℝ) + y^2 / (3 : ℝ) = 1 :=
by
  -- Definitions to meet given conditions would be here
  sorry

end NUMINAMATH_GPT_ellipse_equation_with_foci_l2181_218107


namespace NUMINAMATH_GPT_area_ratio_BDF_FDCE_l2181_218162

-- Define the vertices of the triangle
variables {A B C : Point}
-- Define the points on the sides and midpoints
variables {E D F : Point}
-- Define angles and relevant properties
variables (angle_CBA : Angle B C A = 72)
variables (midpoint_E : Midpoint E A C)
variables (ratio_D : RatioSegment B D D C = 2)
-- Define intersection point F
variables (intersect_F : IntersectLineSegments (LineSegment A D) (LineSegment B E) = F)

theorem area_ratio_BDF_FDCE (h_angle : angle_CBA = 72) 
  (h_midpoint_E : midpoint_E) (h_ratio_D : ratio_D) (h_intersect_F : intersect_F)
  : area_ratio (Triangle.area B D F) (Quadrilateral.area F D C E) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_area_ratio_BDF_FDCE_l2181_218162


namespace NUMINAMATH_GPT_largest_trailing_zeros_l2181_218109

def count_trailing_zeros (n : Nat) : Nat :=
  if n = 0 then 0
  else Nat.min (Nat.factorial (n / 10)) (Nat.factorial (n / 5))

theorem largest_trailing_zeros :
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^5 * 3^4 * 5^6)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (2^4 * 3^4 * 5^5)) ∧
  (count_trailing_zeros (4^3 * 5^6 * 6^5) > count_trailing_zeros (4^2 * 5^4 * 6^3)) :=
  sorry

end NUMINAMATH_GPT_largest_trailing_zeros_l2181_218109


namespace NUMINAMATH_GPT_set_complement_l2181_218160

variable {U : Set ℝ} (A : Set ℝ)

theorem set_complement :
  (U = {x : ℝ | x > 1}) →
  (A ⊆ U) →
  (U \ A = {x : ℝ | x > 9}) →
  (A = {x : ℝ | 1 < x ∧ x ≤ 9}) :=
by
  intros hU hA hC
  sorry

end NUMINAMATH_GPT_set_complement_l2181_218160


namespace NUMINAMATH_GPT_compute_expression_l2181_218191

theorem compute_expression : 7^2 - 2 * 6 + (3^2 - 1) = 45 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2181_218191


namespace NUMINAMATH_GPT_distance_from_A_to_directrix_l2181_218105

open Real

noncomputable def distance_from_point_to_directrix (p : ℝ) : ℝ :=
  1 + p / 2

theorem distance_from_A_to_directrix : 
  ∃ (p : ℝ), (sqrt 5)^2 = 2 * p ∧ distance_from_point_to_directrix p = 9 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_distance_from_A_to_directrix_l2181_218105


namespace NUMINAMATH_GPT_count_valid_ys_l2181_218116

theorem count_valid_ys : 
  ∃ ys : Finset ℤ, ys.card = 4 ∧ ∀ y ∈ ys, (y - 3 > 0) ∧ ((y + 3) * (y - 3) * (y^2 + 9) < 2000) :=
by
  sorry

end NUMINAMATH_GPT_count_valid_ys_l2181_218116


namespace NUMINAMATH_GPT_least_number_of_equal_cubes_l2181_218197

def cuboid_dimensions := (18, 27, 36)
def ratio := (1, 2, 3)

theorem least_number_of_equal_cubes :
  ∃ n, n = 648 ∧
  ∃ a b c : ℕ,
    (a, b, c) = (3, 6, 9) ∧
    (18 % a = 0 ∧ 27 % b = 0 ∧ 36 % c = 0) ∧
    18 * 27 * 36 = n * (a * b * c) :=
sorry

end NUMINAMATH_GPT_least_number_of_equal_cubes_l2181_218197


namespace NUMINAMATH_GPT_average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l2181_218152

-- Conditions from the problem statement
def initial_daily_sales : ℕ := 20
def profit_per_box : ℕ := 40
def additional_sales_per_yuan_reduction : ℕ := 2

-- Part 1: New average daily sales after a 10 yuan reduction
theorem average_daily_sales_after_10_yuan_reduction :
  (initial_daily_sales + 10 * additional_sales_per_yuan_reduction) = 40 :=
  sorry

-- Part 2: Price reduction needed to achieve a daily sales profit of 1200 yuan
theorem price_reduction_for_1200_yuan_profit :
  ∃ (x : ℕ), 
  (profit_per_box - x) * (initial_daily_sales + x * additional_sales_per_yuan_reduction) = 1200 ∧ x = 20 :=
  sorry

end NUMINAMATH_GPT_average_daily_sales_after_10_yuan_reduction_price_reduction_for_1200_yuan_profit_l2181_218152


namespace NUMINAMATH_GPT_distinct_shading_patterns_l2181_218120

/-- How many distinct patterns can be made by shading exactly three of the sixteen squares 
    in a 4x4 grid, considering that patterns which can be matched by flips and/or turns are 
    not considered different? The answer is 8. -/
theorem distinct_shading_patterns : 
  (number_of_distinct_patterns : ℕ) = 8 :=
by
  /- Define the 4x4 Grid and the condition of shading exactly three squares, considering 
     flips and turns -/
  sorry

end NUMINAMATH_GPT_distinct_shading_patterns_l2181_218120


namespace NUMINAMATH_GPT_base_conversion_min_sum_l2181_218130

theorem base_conversion_min_sum (a b : ℕ) (h : 3 * a + 5 = 5 * b + 3)
    (h_mod: 3 * a - 2 ≡ 0 [MOD 5])
    (valid_base_a : a >= 2)
    (valid_base_b : b >= 2):
  a + b = 14 := sorry

end NUMINAMATH_GPT_base_conversion_min_sum_l2181_218130


namespace NUMINAMATH_GPT_correct_inequality_l2181_218166

theorem correct_inequality :
  1.6 ^ 0.3 > 0.9 ^ 3.1 :=
sorry

end NUMINAMATH_GPT_correct_inequality_l2181_218166


namespace NUMINAMATH_GPT_jerry_task_duration_l2181_218141

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end NUMINAMATH_GPT_jerry_task_duration_l2181_218141


namespace NUMINAMATH_GPT_solve_for_y_l2181_218157

theorem solve_for_y (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2181_218157


namespace NUMINAMATH_GPT_Janet_pages_per_day_l2181_218112

variable (J : ℕ)

-- Conditions
def belinda_pages_per_day : ℕ := 30
def janet_extra_pages_per_6_weeks : ℕ := 2100
def days_in_6_weeks : ℕ := 42

-- Prove that Janet reads 80 pages a day
theorem Janet_pages_per_day (h : J * days_in_6_weeks = (belinda_pages_per_day * days_in_6_weeks) + janet_extra_pages_per_6_weeks) : J = 80 := 
by sorry

end NUMINAMATH_GPT_Janet_pages_per_day_l2181_218112


namespace NUMINAMATH_GPT_distance_eq_3_implies_points_l2181_218189

-- Definition of the distance of point A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement translating the problem
theorem distance_eq_3_implies_points (x : ℝ) (h : distance_to_origin x = 3) :
  x = 3 ∨ x = -3 :=
sorry

end NUMINAMATH_GPT_distance_eq_3_implies_points_l2181_218189


namespace NUMINAMATH_GPT_unique_rs_exists_l2181_218175

theorem unique_rs_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (gcd_ab : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), (0 < r ∧ r < b) ∧ (0 < s ∧ s < a) ∧ (a * r - b * s = 1) :=
  sorry

end NUMINAMATH_GPT_unique_rs_exists_l2181_218175


namespace NUMINAMATH_GPT_joan_seashells_l2181_218108

/-- Prove that Joan has 36 seashells given the initial conditions. -/
theorem joan_seashells :
  let initial_seashells := 79
  let given_mike := 63
  let found_more := 45
  let traded_seashells := 20
  let lost_seashells := 5
  (initial_seashells - given_mike + found_more - traded_seashells - lost_seashells) = 36 :=
by
  sorry

end NUMINAMATH_GPT_joan_seashells_l2181_218108


namespace NUMINAMATH_GPT_kylie_daisies_l2181_218136

theorem kylie_daisies :
  let initial_daisies := 5
  let additional_daisies := 9
  let total_daisies := initial_daisies + additional_daisies
  let daisies_left := total_daisies / 2
  daisies_left = 7 :=
by
  sorry

end NUMINAMATH_GPT_kylie_daisies_l2181_218136


namespace NUMINAMATH_GPT_susan_age_l2181_218178

theorem susan_age (S J B : ℝ) 
  (h1 : S = 2 * J)
  (h2 : S + J + B = 60) 
  (h3 : B = J + 10) : 
  S = 25 := sorry

end NUMINAMATH_GPT_susan_age_l2181_218178


namespace NUMINAMATH_GPT_possible_value_m_l2181_218126

theorem possible_value_m (x m : ℝ) (h : ∃ x : ℝ, 2 * x^2 + 5 * x - m = 0) : m ≥ -25 / 8 := sorry

end NUMINAMATH_GPT_possible_value_m_l2181_218126


namespace NUMINAMATH_GPT_domain_of_function_l2181_218151

noncomputable def is_domain_of_function (x : ℝ) : Prop :=
  (4 - x^2 ≥ 0) ∧ (x ≠ 1)

theorem domain_of_function :
  {x : ℝ | is_domain_of_function x} = {x : ℝ | -2 ≤ x ∧ x < 1} ∪ {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2181_218151


namespace NUMINAMATH_GPT_caps_production_l2181_218147

def caps1 : Int := 320
def caps2 : Int := 400
def caps3 : Int := 300

def avg_caps (caps1 caps2 caps3 : Int) : Int := (caps1 + caps2 + caps3) / 3

noncomputable def total_caps_after_four_weeks : Int :=
  caps1 + caps2 + caps3 + avg_caps caps1 caps2 caps3

theorem caps_production : total_caps_after_four_weeks = 1360 :=
by
  sorry

end NUMINAMATH_GPT_caps_production_l2181_218147


namespace NUMINAMATH_GPT_last_digit_fifth_power_l2181_218169

theorem last_digit_fifth_power (R : ℤ) : (R^5 - R) % 10 = 0 := 
sorry

end NUMINAMATH_GPT_last_digit_fifth_power_l2181_218169


namespace NUMINAMATH_GPT_binomial_expansion_problem_l2181_218193

theorem binomial_expansion_problem :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ),
    (1 + 2 * x) ^ 11 =
      a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
      a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 +
      a_9 * x^9 + a_10 * x^10 + a_11 * x^11 →
    a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 + 5 * a_5 - 6 * a_6 +
    7 * a_7 - 8 * a_8 + 9 * a_9 - 10 * a_10 + 11 * a_11 = 22 :=
by
  -- The proof is omitted for this exercise
  sorry

end NUMINAMATH_GPT_binomial_expansion_problem_l2181_218193


namespace NUMINAMATH_GPT_invitational_tournament_l2181_218167

theorem invitational_tournament (x : ℕ) (h : 2 * (x * (x - 1) / 2) = 56) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_invitational_tournament_l2181_218167


namespace NUMINAMATH_GPT_remainder_of_sum_l2181_218102

open Nat

theorem remainder_of_sum :
  (12345 + 12347 + 12349 + 12351 + 12353 + 12355 + 12357) % 16 = 9 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l2181_218102


namespace NUMINAMATH_GPT_num_ways_to_choose_starting_lineup_l2181_218158

-- Define conditions as Lean definitions
def team_size : ℕ := 12
def outfield_players : ℕ := 4

-- Define the function to compute the number of ways to choose the starting lineup
def choose_starting_lineup (team_size : ℕ) (outfield_players : ℕ) : ℕ :=
  team_size * Nat.choose (team_size - 1) outfield_players

-- The theorem to prove that the number of ways to choose the lineup is 3960
theorem num_ways_to_choose_starting_lineup : choose_starting_lineup team_size outfield_players = 3960 :=
  sorry

end NUMINAMATH_GPT_num_ways_to_choose_starting_lineup_l2181_218158


namespace NUMINAMATH_GPT_sum_of_coordinates_x_l2181_218196

-- Given points Y and Z
def Y : ℝ × ℝ := (2, 8)
def Z : ℝ × ℝ := (0, -4)

-- Given ratio conditions
def ratio_condition (X Y Z : ℝ × ℝ) : Prop :=
  dist X Z / dist X Y = 1/3 ∧ dist Z Y / dist X Y = 1/3

-- Define X, ensuring Z is the midpoint of XY
def X : ℝ × ℝ := (4, 20)

-- Prove that sum of coordinates of X is 10
theorem sum_of_coordinates_x (h : ratio_condition X Y Z) : (X.1 + X.2) = 10 := 
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_x_l2181_218196


namespace NUMINAMATH_GPT_A_days_to_complete_work_l2181_218104

noncomputable def work (W : ℝ) (A_work_per_day B_work_per_day : ℝ) (days_A days_B days_B_alone : ℝ) : ℝ :=
  A_work_per_day * days_A + B_work_per_day * days_B

theorem A_days_to_complete_work 
  (W : ℝ)
  (A_work_per_day B_work_per_day : ℝ)
  (days_A days_B days_B_alone : ℝ)
  (h1 : days_A = 5)
  (h2 : days_B = 12)
  (h3 : days_B_alone = 18)
  (h4 : B_work_per_day = W / days_B_alone)
  (h5 : work W A_work_per_day B_work_per_day days_A days_B days_B_alone = W) :
  W / A_work_per_day = 15 := 
sorry

end NUMINAMATH_GPT_A_days_to_complete_work_l2181_218104


namespace NUMINAMATH_GPT_hypotenuse_length_l2181_218198

theorem hypotenuse_length
    (a b c : ℝ)
    (h1: a^2 + b^2 + c^2 = 2450)
    (h2: b = a + 7)
    (h3: c^2 = a^2 + b^2) :
    c = 35 := sorry

end NUMINAMATH_GPT_hypotenuse_length_l2181_218198


namespace NUMINAMATH_GPT_final_solution_sugar_percentage_l2181_218176

-- Define the conditions of the problem
def initial_solution_sugar_percentage : ℝ := 0.10
def replacement_fraction : ℝ := 0.25
def second_solution_sugar_percentage : ℝ := 0.26

-- Define the Lean statement that proves the final sugar percentage
theorem final_solution_sugar_percentage:
  (0.10 * (1 - 0.25) + 0.26 * 0.25) * 100 = 14 :=
by
  sorry

end NUMINAMATH_GPT_final_solution_sugar_percentage_l2181_218176


namespace NUMINAMATH_GPT_find_positive_number_l2181_218174

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_find_positive_number_l2181_218174


namespace NUMINAMATH_GPT_negation_of_P_l2181_218165

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x > Real.sin x

-- Formulate the negation of P
def neg_P : Prop := ∃ x : ℝ, x ≤ Real.sin x

-- State the theorem to be proved
theorem negation_of_P (hP : P) : neg_P :=
sorry

end NUMINAMATH_GPT_negation_of_P_l2181_218165


namespace NUMINAMATH_GPT_nonneg_int_repr_l2181_218131

theorem nonneg_int_repr (n : ℕ) : ∃ (a b c : ℕ), (0 < a ∧ a < b ∧ b < c) ∧ n = a^2 + b^2 - c^2 :=
sorry

end NUMINAMATH_GPT_nonneg_int_repr_l2181_218131


namespace NUMINAMATH_GPT_sum_of_digits_smallest_N_l2181_218172

/-- Define the probability Q(N) -/
def Q (N : ℕ) : ℚ :=
  ((2 * N) / 3 + 1) / (N + 1)

/-- Main mathematical statement to be proven in Lean 4 -/

theorem sum_of_digits_smallest_N (N : ℕ) (h1 : N > 9) (h2 : N % 6 = 0) (h3 : Q N < 7 / 10) : 
  (N.digits 10).sum = 3 :=
  sorry

end NUMINAMATH_GPT_sum_of_digits_smallest_N_l2181_218172


namespace NUMINAMATH_GPT_orchard_harvest_l2181_218143

theorem orchard_harvest (sacks_per_section : ℕ) (sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 45 → sections = 8 → total_sacks = sacks_per_section * sections → total_sacks = 360 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end NUMINAMATH_GPT_orchard_harvest_l2181_218143


namespace NUMINAMATH_GPT_inequalities_proof_l2181_218115

variables (x y z : ℝ)

def p := x + y + z
def q := x * y + y * z + z * x
def r := x * y * z

theorem inequalities_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (p x y z) ^ 2 ≥ 3 * (q x y z) ∧
  (p x y z) ^ 3 ≥ 27 * (r x y z) ∧
  (p x y z) * (q x y z) ≥ 9 * (r x y z) ∧
  (q x y z) ^ 2 ≥ 3 * (p x y z) * (r x y z) ∧
  (p x y z) ^ 2 * (q x y z) + 3 * (p x y z) * (r x y z) ≥ 4 * (q x y z) ^ 2 ∧
  (p x y z) ^ 3 + 9 * (r x y z) ≥ 4 * (p x y z) * (q x y z) ∧
  (p x y z) * (q x y z) ^ 2 ≥ 2 * (p x y z) ^ 2 * (r x y z) + 3 * (q x y z) * (r x y z) ∧
  (p x y z) * (q x y z) ^ 2 + 3 * (q x y z) * (r x y z) ≥ 4 * (p x y z) ^ 2 * (r x y z) ∧
  2 * (q x y z) ^ 3 + 9 * (r x y z) ^ 2 ≥ 7 * (p x y z) * (q x y z) * (r x y z) ∧
  (p x y z) ^ 4 + 4 * (q x y z) ^ 2 + 6 * (p x y z) * (r x y z) ≥ 5 * (p x y z) ^ 2 * (q x y z) :=
by sorry

end NUMINAMATH_GPT_inequalities_proof_l2181_218115


namespace NUMINAMATH_GPT_exists_positive_b_l2181_218142

theorem exists_positive_b (m p : ℕ) (hm : 0 < m) (hp : Prime p)
  (h1 : m^2 ≡ 2 [MOD p])
  (ha : ∃ a : ℕ, 0 < a ∧ a^2 ≡ 2 - m [MOD p]) :
  ∃ b : ℕ, 0 < b ∧ b^2 ≡ m + 2 [MOD p] := 
  sorry

end NUMINAMATH_GPT_exists_positive_b_l2181_218142


namespace NUMINAMATH_GPT_license_plates_count_l2181_218161

/-
Problem:
I want to choose a license plate that is 4 characters long,
where the first character is a letter,
the last two characters are either a letter or a digit,
and the second character can be a letter or a digit 
but must be the same as either the first or the third character.
Additionally, the fourth character must be different from the first three characters.
-/

def is_letter (c : Char) : Prop := c.isAlpha
def is_digit_or_letter (c : Char) : Prop := c.isAlpha || c.isDigit
noncomputable def count_license_plates : ℕ :=
  let first_char_options := 26
  let third_char_options := 36
  let second_char_options := 2
  let fourth_char_options := 34
  first_char_options * third_char_options * second_char_options * fourth_char_options

theorem license_plates_count : count_license_plates = 59904 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l2181_218161


namespace NUMINAMATH_GPT_valid_pin_count_l2181_218159

def total_pins : ℕ := 10^5

def restricted_pins (seq : List ℕ) : ℕ :=
  if seq = [3, 1, 4, 1] then 10 else 0

def valid_pins (seq : List ℕ) : ℕ :=
  total_pins - restricted_pins seq

theorem valid_pin_count :
  valid_pins [3, 1, 4, 1] = 99990 :=
by
  sorry

end NUMINAMATH_GPT_valid_pin_count_l2181_218159


namespace NUMINAMATH_GPT_well_depth_l2181_218111

variable (d : ℝ)

-- Conditions
def total_time (t₁ t₂ : ℝ) : Prop := t₁ + t₂ = 8.5
def stone_fall (t₁ : ℝ) : Prop := d = 16 * t₁^2 
def sound_travel (t₂ : ℝ) : Prop := t₂ = d / 1100

theorem well_depth : 
  ∃ t₁ t₂ : ℝ, total_time t₁ t₂ ∧ stone_fall d t₁ ∧ sound_travel d t₂ → d = 918.09 := 
by
  sorry

end NUMINAMATH_GPT_well_depth_l2181_218111


namespace NUMINAMATH_GPT_problem_solution_l2181_218129

variable (a : ℝ)
def ellipse_p (a : ℝ) : Prop := (0 < a) ∧ (a < 5)
def quadratic_q (a : ℝ) : Prop := (-3 ≤ a) ∧ (a ≤ 3)
def p_or_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∨ ((-3 ≤ a) ∧ (a ≤ 3)))
def p_and_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∧ ((-3 ≤ a) ∧ (a ≤ 3)))

theorem problem_solution (a : ℝ) :
  (ellipse_p a → 0 < a ∧ a < 5) ∧ 
  (¬(ellipse_p a) ∧ quadratic_q a → -3 ≤ a ∧ a ≤ 0) ∧
  (p_or_q a ∧ ¬(p_and_q a) → 3 < a ∧ a < 5 ∨ (-3 ≤ a ∧ a ≤ 0)) :=
  by
  sorry

end NUMINAMATH_GPT_problem_solution_l2181_218129


namespace NUMINAMATH_GPT_george_run_speed_l2181_218170

theorem george_run_speed (usual_distance : ℝ) (usual_speed : ℝ) (today_first_distance : ℝ) (today_first_speed : ℝ)
  (remaining_distance : ℝ) (expected_time : ℝ) :
  usual_distance = 1.5 →
  usual_speed = 3 →
  today_first_distance = 1 →
  today_first_speed = 2.5 →
  remaining_distance = 0.5 →
  expected_time = usual_distance / usual_speed →
  today_first_distance / today_first_speed + remaining_distance / (remaining_distance / (expected_time - today_first_distance / today_first_speed)) = expected_time →
  remaining_distance / (expected_time - today_first_distance / today_first_speed) = 5 :=
by sorry

end NUMINAMATH_GPT_george_run_speed_l2181_218170


namespace NUMINAMATH_GPT_average_minutes_per_player_is_2_l2181_218190

def total_player_footage := 130 + 145 + 85 + 60 + 180
def total_additional_content := 120 + 90 + 30
def pause_transition_time := 15 * (5 + 3) -- 5 players + game footage + interviews + opening/closing scenes - 1
def total_film_time := total_player_footage + total_additional_content + pause_transition_time
def number_of_players := 5
def average_seconds_per_player := total_player_footage / number_of_players
def average_minutes_per_player := average_seconds_per_player / 60

theorem average_minutes_per_player_is_2 :
  average_minutes_per_player = 2 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_average_minutes_per_player_is_2_l2181_218190


namespace NUMINAMATH_GPT_cube_plane_intersection_distance_l2181_218113

theorem cube_plane_intersection_distance :
  let vertices := [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]
  let P := (0, 3, 0)
  let Q := (2, 0, 0)
  let R := (2, 6, 6)
  let plane_equation := 3 * x - 2 * y - 2 * z + 6 = 0
  let S := (2, 0, 6)
  let T := (0, 6, 3)
  dist S T = 7 := sorry

end NUMINAMATH_GPT_cube_plane_intersection_distance_l2181_218113


namespace NUMINAMATH_GPT_positive_number_eq_576_l2181_218134

theorem positive_number_eq_576 (x : ℝ) (h : 0 < x) (h_eq : (2 / 3) * x = (25 / 216) * (1 / x)) : x = 5.76 := 
by 
  sorry

end NUMINAMATH_GPT_positive_number_eq_576_l2181_218134


namespace NUMINAMATH_GPT_maximum_n_l2181_218173

def arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d

def is_positive_first_term (a : ℕ → ℤ) : Prop :=
  a 0 > 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n-1))) / 2

def roots_of_equation (a1006 a1007 : ℤ) : Prop :=
  a1006 * a1007 = -2011 ∧ a1006 + a1007 = 2012

theorem maximum_n (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence_max_n a S 1007)
  (h2 : is_positive_first_term a)
  (h3 : sum_of_first_n_terms a S)
  (h4 : ∃ a1006 a1007, roots_of_equation a1006 a1007 ∧ a 1006 = a1006 ∧ a 1007 = a1007) :
  ∃ n, S n > 0 → n ≤ 1007 := 
sorry

end NUMINAMATH_GPT_maximum_n_l2181_218173


namespace NUMINAMATH_GPT_not_all_zero_iff_at_least_one_nonzero_l2181_218188

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by 
  sorry

end NUMINAMATH_GPT_not_all_zero_iff_at_least_one_nonzero_l2181_218188


namespace NUMINAMATH_GPT_contrapositive_even_statement_l2181_218154

-- Translate the conditions to Lean 4 definitions
def is_even (n : Int) : Prop := ∃ k : Int, n = 2 * k

theorem contrapositive_even_statement (a b : Int) :
  (¬ is_even (a + b) → ¬ (is_even a ∧ is_even b)) ↔ 
  (is_even a ∧ is_even b → is_even (a + b)) :=
by sorry

end NUMINAMATH_GPT_contrapositive_even_statement_l2181_218154


namespace NUMINAMATH_GPT_arith_seq_formula_l2181_218137

noncomputable def arith_seq (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n + a (n + 2) = 4 * n + 6

theorem arith_seq_formula (a : ℕ → ℤ) (h : arith_seq a) : ∀ n : ℕ, a n = 2 * n + 1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_arith_seq_formula_l2181_218137
