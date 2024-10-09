import Mathlib

namespace largest_possible_value_n_l461_46150

theorem largest_possible_value_n (n : ℕ) (h : ∀ m : ℕ, m ≠ n → n % m = 0 → m ≤ 35) : n = 35 :=
sorry

end largest_possible_value_n_l461_46150


namespace evaluate_expression_l461_46185

theorem evaluate_expression (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x - 4 = 26 :=
by {
    sorry
}

end evaluate_expression_l461_46185


namespace larger_number_is_1641_l461_46131

theorem larger_number_is_1641 (L S : ℕ) (h1 : L - S = 1370) (h2 : L = 6 * S + 15) : L = 1641 :=
by
  sorry

end larger_number_is_1641_l461_46131


namespace negation_of_proposition_l461_46143

theorem negation_of_proposition (a b c : ℝ) :
  ¬ (a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c ≠ 3 → a^2 + b^2 + c^2 < 3) :=
sorry

end negation_of_proposition_l461_46143


namespace no_positive_solution_for_special_k_l461_46129
open Nat

theorem no_positive_solution_for_special_k (p : ℕ) (hp : p.Prime) (hmod : p % 4 = 3) :
    ¬ ∃ n m k : ℕ, (n > 0) ∧ (m > 0) ∧ (k = p^2) ∧ (n^2 + m^2 = k * (m^4 + n)) :=
sorry

end no_positive_solution_for_special_k_l461_46129


namespace compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l461_46176

theorem compare_neg5_neg2 : -5 < -2 :=
by sorry

theorem compare_neg_third_neg_half : -(1/3) > -(1/2) :=
by sorry

theorem compare_absneg5_0 : abs (-5) > 0 :=
by sorry

end compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l461_46176


namespace find_balls_l461_46155

theorem find_balls (x y : ℕ) (h1 : (x + y : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) - 1 / 15)
                   (h2 : (y + 18 : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) * 11 / 10) :
  x = 12 ∧ y = 15 :=
sorry

end find_balls_l461_46155


namespace polygon_interior_angle_sum_l461_46182

theorem polygon_interior_angle_sum (n : ℕ) (hn : 3 ≤ n) :
  (n - 2) * 180 + 180 = 2007 → n = 13 := by
  sorry

end polygon_interior_angle_sum_l461_46182


namespace ice_cream_vendor_l461_46124

theorem ice_cream_vendor (M : ℕ) (h3 : 50 - (3 / 5) * 50 = 20) (h4 : (2 / 3) * M = 2 * M / 3) 
  (h5 : (50 - 30) + M - (2 * M / 3) = 38) :
  M = 12 :=
by
  sorry

end ice_cream_vendor_l461_46124


namespace least_number_to_add_divisible_l461_46160

theorem least_number_to_add_divisible (n d : ℕ) (h1 : n = 929) (h2 : d = 30) : 
  ∃ x, (n + x) % d = 0 ∧ x = 1 := 
by 
  sorry

end least_number_to_add_divisible_l461_46160


namespace codecracker_total_combinations_l461_46188

theorem codecracker_total_combinations (colors slots : ℕ) (h_colors : colors = 6) (h_slots : slots = 5) :
  colors ^ slots = 7776 :=
by
  rw [h_colors, h_slots]
  norm_num

end codecracker_total_combinations_l461_46188


namespace reading_time_difference_in_minutes_l461_46177

noncomputable def xanthia_reading_speed : ℝ := 120 -- pages per hour
noncomputable def molly_reading_speed : ℝ := 60 -- pages per hour
noncomputable def book_length : ℝ := 360 -- pages

theorem reading_time_difference_in_minutes :
  let time_for_xanthia := book_length / xanthia_reading_speed
  let time_for_molly := book_length / molly_reading_speed
  let difference_in_hours := time_for_molly - time_for_xanthia
  difference_in_hours * 60 = 180 :=
by
  sorry

end reading_time_difference_in_minutes_l461_46177


namespace integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l461_46103

noncomputable def count_integer_solutions (n : ℕ) : ℕ :=
  if n = 1992 then 90
  else if n = 1993 then 6
  else if n = 1994 then 6
  else 0

theorem integer_solutions_count_correct_1992 :
  count_integer_solutions 1992 = 90 :=
by
  sorry

theorem integer_solutions_count_correct_1993 :
  count_integer_solutions 1993 = 6 :=
by
  sorry

theorem integer_solutions_count_correct_1994 :
  count_integer_solutions 1994 = 6 :=
by
  sorry

example :
  count_integer_solutions 1992 = 90 ∧
  count_integer_solutions 1993 = 6 ∧
  count_integer_solutions 1994 = 6 :=
by
  exact ⟨integer_solutions_count_correct_1992, integer_solutions_count_correct_1993, integer_solutions_count_correct_1994⟩

end integer_solutions_count_correct_1992_integer_solutions_count_correct_1993_integer_solutions_count_correct_1994_l461_46103


namespace exponential_inequality_l461_46168

theorem exponential_inequality (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m < a^n) : ¬ (m < n) := 
sorry

end exponential_inequality_l461_46168


namespace multiple_of_three_l461_46195

theorem multiple_of_three (a b : ℤ) : ∃ k : ℤ, (a + b = 3 * k) ∨ (ab = 3 * k) ∨ (a - b = 3 * k) :=
sorry

end multiple_of_three_l461_46195


namespace product_of_undefined_x_l461_46120

-- Define the quadratic equation condition
def quad_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- The main theorem to prove the product of all x such that the expression is undefined
theorem product_of_undefined_x :
  (∃ x₁ x₂ : ℝ, quad_eq 1 4 3 x₁ ∧ quad_eq 1 4 3 x₂ ∧ x₁ * x₂ = 3) :=
by
  sorry

end product_of_undefined_x_l461_46120


namespace least_positive_integer_x_20y_l461_46146

theorem least_positive_integer_x_20y (x y : ℤ) (h : Int.gcd x (20 * y) = 4) : 
  ∃ k : ℕ, k > 0 ∧ k * (x + 20 * y) = 4 := 
sorry

end least_positive_integer_x_20y_l461_46146


namespace solve_expression_l461_46174

theorem solve_expression :
  (27 ^ (2 / 3) - 2 ^ (Real.log 3 / Real.log 2) * (Real.logb 2 (1 / 8)) +
    Real.logb 10 4 + Real.logb 10 25 = 20) :=
by
  sorry

end solve_expression_l461_46174


namespace g_solution_l461_46152

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2
axiom g_functional : ∀ x y : ℝ, g (x * y) = g ((x^2 + y^2) / 2) + (x - y)^2 + x^2

theorem g_solution :
  ∀ x : ℝ, g x = 2 - 2 * x := sorry

end g_solution_l461_46152


namespace simplify_expression_l461_46132

theorem simplify_expression: 3 * Real.sqrt 48 - 6 * Real.sqrt (1 / 3) + (Real.sqrt 3 - 1) ^ 2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end simplify_expression_l461_46132


namespace sum_of_edges_l461_46147

theorem sum_of_edges (a r : ℝ) 
  (h_volume : (a^3 = 512))
  (h_surface_area : (2 * (a^2 / r + a^2 + a^2 * r) = 384))
  (h_geometric_progression : true) :
  (4 * ((a / r) + a + (a * r)) = 96) :=
by
  -- It is only necessary to provide the theorem statement
  sorry

end sum_of_edges_l461_46147


namespace no_function_f_satisfies_condition_l461_46112

theorem no_function_f_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + y^2 :=
by
  sorry

end no_function_f_satisfies_condition_l461_46112


namespace sqrt_3_between_inequalities_l461_46100

theorem sqrt_3_between_inequalities (n : ℕ) (h1 : 1 + (3 : ℝ) / (n + 1) < Real.sqrt 3) (h2 : Real.sqrt 3 < 1 + (3 : ℝ) / n) : n = 4 := 
sorry

end sqrt_3_between_inequalities_l461_46100


namespace algebraic_expression_value_l461_46123

theorem algebraic_expression_value (x : ℝ) (h : x^2 - 2*x - 2 = 0) : 3*x^2 - 6*x + 9 = 15 := by
  sorry

end algebraic_expression_value_l461_46123


namespace range_of_m_l461_46104

-- Definition of the propositions and conditions
def p (m : ℝ) : Prop := m > 2
def q (m : ℝ) : Prop := 1 ≤ m ∧ m ≤ 3
def prop (m : ℝ) : Prop := (¬(p m ∧ q m) ∧ (p m ∨ q m))

-- The proof statement showing the range of m
theorem range_of_m (m : ℝ) : prop m ↔ (1 ≤ m ∧ m ≤ 2) ∨ (m > 3) :=
by
  sorry

end range_of_m_l461_46104


namespace negation_of_proposition_l461_46117

theorem negation_of_proposition :
  (¬ (∃ x_0 : ℝ, x_0 ≤ 0 ∧ x_0^2 ≥ 0)) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
sorry

end negation_of_proposition_l461_46117


namespace right_triangle_legs_l461_46191

theorem right_triangle_legs (c a b : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : ab = c^2 / 4) :
  a = c * (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ b = c * (Real.sqrt 6 - Real.sqrt 2) / 4 := 
sorry

end right_triangle_legs_l461_46191


namespace log_inequality_sqrt_inequality_l461_46135

-- Proof problem for part (1)
theorem log_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.log ((a + b) / 2) ≥ (Real.log a + Real.log b) / 2 :=
sorry

-- Proof problem for part (2)
theorem sqrt_inequality :
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end log_inequality_sqrt_inequality_l461_46135


namespace clea_escalator_time_standing_l461_46130

noncomputable def escalator_time (c : ℕ) : ℝ :=
  let s := (7 * c) / 5
  let d := 72 * c
  let t := d / s
  t

theorem clea_escalator_time_standing (c : ℕ) (h1 : 72 * c = 72 * c) (h2 : 30 * (c + (7 * c) / 5) = 72 * c): escalator_time c = 51 :=
by
  sorry

end clea_escalator_time_standing_l461_46130


namespace gum_lcm_l461_46197

theorem gum_lcm (strawberry blueberry cherry : ℕ) (h₁ : strawberry = 6) (h₂ : blueberry = 5) (h₃ : cherry = 8) :
  Nat.lcm (Nat.lcm strawberry blueberry) cherry = 120 :=
by
  rw [h₁, h₂, h₃]
  -- LCM(6, 5, 8) = LCM(LCM(6, 5), 8)
  sorry

end gum_lcm_l461_46197


namespace ratio_initial_to_doubled_l461_46161

theorem ratio_initial_to_doubled (x : ℕ) (h : 3 * (2 * x + 9) = 63) : x / (2 * x) = 1 / 2 := 
by
  sorry

end ratio_initial_to_doubled_l461_46161


namespace log_identity_l461_46184

noncomputable def my_log (base x : ℝ) := Real.log x / Real.log base

theorem log_identity (x : ℝ) (h : x > 0) (h1 : x ≠ 1) : 
  (my_log 4 x) * (my_log x 5) = my_log 4 5 :=
by
  sorry

end log_identity_l461_46184


namespace min_value_l461_46106

variable (d : ℕ) (a_n S_n : ℕ → ℕ)
variable (a1 : ℕ) (H1 : d ≠ 0)
variable (H2 : a1 = 1)
variable (H3 : (a_n 3)^2 = a1 * (a_n 13))
variable (H4 : a_n n = a1 + (n - 1) * d)
variable (H5 : S_n n = (n * (a1 + a_n n)) / 2)

theorem min_value (n : ℕ) (Hn : 1 ≤ n) : 
  ∃ n, ∀ m, 1 ≤ m → (2 * S_n n + 16) / (a_n n + 3) ≥ (2 * S_n m + 16) / (a_n m + 3) ∧ (2 * S_n n + 16) / (a_n n + 3) = 4 :=
sorry

end min_value_l461_46106


namespace problem_Ashwin_Sah_l461_46186

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah_l461_46186


namespace problem1_solution_problem2_solution_l461_46170

noncomputable def problem1 (α : ℝ) (h : Real.tan α = -2) : Real :=
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)

theorem problem1_solution (α : ℝ) (h : Real.tan α = -2) : problem1 α h = 5 := by
  sorry

noncomputable def problem2 (α : ℝ) (h : Real.tan α = -2) : Real :=
  1 / (Real.sin α * Real.cos α)

theorem problem2_solution (α : ℝ) (h : Real.tan α = -2) : problem2 α h = -5 / 2 := by
  sorry

end problem1_solution_problem2_solution_l461_46170


namespace motorist_travel_distance_l461_46126

def total_distance_traveled (time_first_half time_second_half speed_first_half speed_second_half : ℕ) : ℕ :=
  (speed_first_half * time_first_half) + (speed_second_half * time_second_half)

theorem motorist_travel_distance :
  total_distance_traveled 3 3 60 48 = 324 :=
by sorry

end motorist_travel_distance_l461_46126


namespace smallest_base10_integer_exists_l461_46101

theorem smallest_base10_integer_exists : ∃ (n a b : ℕ), a > 2 ∧ b > 2 ∧ n = 1 * a + 3 ∧ n = 3 * b + 1 ∧ n = 10 := by
  sorry

end smallest_base10_integer_exists_l461_46101


namespace polynomial_value_at_2_l461_46154

def f (x : ℕ) : ℕ := 8 * x^7 + 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem polynomial_value_at_2 : f 2 = 1397 := by
  sorry

end polynomial_value_at_2_l461_46154


namespace jessica_age_proof_l461_46137

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l461_46137


namespace largest_n_digit_number_divisible_by_61_correct_l461_46183

def largest_n_digit_number (n : ℕ) : ℕ :=
10^n - 1

def largest_n_digit_number_divisible_by_61 (n : ℕ) : ℕ :=
largest_n_digit_number n - (largest_n_digit_number n % 61)

theorem largest_n_digit_number_divisible_by_61_correct (n : ℕ) :
  ∃ k : ℕ, largest_n_digit_number_divisible_by_61 n = 61 * k :=
by
  sorry

end largest_n_digit_number_divisible_by_61_correct_l461_46183


namespace find_triple_l461_46162
-- Import necessary libraries

-- Define the required predicates and conditions
def satisfies_conditions (x y z : ℕ) : Prop :=
  x ≤ y ∧ y ≤ z ∧ x^3 * (y^3 + z^3) = 2012 * (x * y * z + 2)

-- The main theorem statement
theorem find_triple : 
  ∀ (x y z : ℕ), satisfies_conditions x y z → (x, y, z) = (2, 251, 252) :=
by
  sorry

end find_triple_l461_46162


namespace train_pass_jogger_time_l461_46122

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 60
noncomputable def initial_distance_m : ℝ := 350
noncomputable def train_length_m : ℝ := 250

noncomputable def relative_speed_m_per_s : ℝ := 
  ((train_speed_km_per_hr - jogger_speed_km_per_hr) * 1000) / 3600

noncomputable def total_distance_m : ℝ := initial_distance_m + train_length_m

noncomputable def time_to_pass_s : ℝ := total_distance_m / relative_speed_m_per_s

theorem train_pass_jogger_time :
  abs (time_to_pass_s - 42.35) < 0.01 :=
by 
  sorry

end train_pass_jogger_time_l461_46122


namespace ellipse_problem_l461_46140

noncomputable def point_coordinates (x y b : ℝ) : Prop :=
  x = 1 ∧ y = 1 ∧ (4 * x^2 = 4) ∧ (4 * b^2 / (4 + b^2) = 1)

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 - b^2)) / a

theorem ellipse_problem (b : ℝ) (h₁ : 4 * b^2 / (4 + b^2) = 1) :
  ∃ x y, point_coordinates x y b 
  ∧ eccentricity 2 b = Real.sqrt 6 / 3 := 
by 
  sorry

end ellipse_problem_l461_46140


namespace chess_pieces_present_l461_46158

theorem chess_pieces_present (total_pieces : ℕ) (missing_pieces : ℕ) (h1 : total_pieces = 32) (h2 : missing_pieces = 4) : (total_pieces - missing_pieces) = 28 := 
by sorry

end chess_pieces_present_l461_46158


namespace proper_subset_A_B_l461_46138

theorem proper_subset_A_B (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 2 → x < a) ∧ (∃ b, b < a ∧ ¬(1 < b ∧ b < 2)) ↔ 2 ≤ a :=
by
  sorry

end proper_subset_A_B_l461_46138


namespace problems_per_page_l461_46105

theorem problems_per_page (pages_math pages_reading total_problems x : ℕ) (h1 : pages_math = 2) (h2 : pages_reading = 4) (h3 : total_problems = 30) : 
  (pages_math + pages_reading) * x = total_problems → x = 5 := by
  sorry

end problems_per_page_l461_46105


namespace value_of_y_l461_46145

theorem value_of_y (x y z : ℕ) (h1 : 3 * x = 3 / 4 * y) (h2 : x + z = 24) (h3 : z = 8) : y = 64 :=
by
  -- Proof omitted
  sorry

end value_of_y_l461_46145


namespace find_female_employees_l461_46199

-- Definitions from conditions
def total_employees (E : ℕ) := True
def female_employees (F : ℕ) := True
def male_employees (M : ℕ) := True
def female_managers (F_mgrs : ℕ) := F_mgrs = 280
def fraction_of_managers : ℚ := 2 / 5
def fraction_of_male_managers : ℚ := 2 / 5

-- Statements as conditions in Lean
def managers_total (E M : ℕ) := (fraction_of_managers * E : ℚ) = (fraction_of_male_managers * M : ℚ) + 280
def employees_total (E F M : ℕ) := E = F + M

-- The proof target
theorem find_female_employees (E F M : ℕ) (F_mgrs : ℕ)
    (h1 : female_managers F_mgrs)
    (h2 : managers_total E M)
    (h3 : employees_total E F M) : F = 700 := by
  sorry

end find_female_employees_l461_46199


namespace smallest_integer_represented_as_AA6_and_BB8_l461_46141

def valid_digit_in_base (d : ℕ) (b : ℕ) : Prop := d < b

theorem smallest_integer_represented_as_AA6_and_BB8 :
  ∃ (n : ℕ) (A B : ℕ),
  valid_digit_in_base A 6 ∧ valid_digit_in_base B 8 ∧ 
  n = 7 * A ∧ n = 9 * B ∧ n = 63 :=
by
  sorry

end smallest_integer_represented_as_AA6_and_BB8_l461_46141


namespace not_function_of_x_l461_46144

theorem not_function_of_x : 
  ∃ x : ℝ, ∃ y1 y2 : ℝ, (|y1| = 2 * x ∧ |y2| = 2 * x ∧ y1 ≠ y2) := sorry

end not_function_of_x_l461_46144


namespace sequence_bound_100_l461_46165

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ≥ 2, a n = a (n - 1) + 1 / a (n - 1)

theorem sequence_bound_100 (a : ℕ → ℝ) (h : seq a) : 
  14 < a 100 ∧ a 100 < 18 := 
sorry

end sequence_bound_100_l461_46165


namespace find_c_if_lines_parallel_l461_46178

theorem find_c_if_lines_parallel (c : ℝ) : 
  (∀ x : ℝ, 5 * x - 3 = (3 * c) * x + 1) → 
  c = 5 / 3 :=
by
  intro h
  sorry

end find_c_if_lines_parallel_l461_46178


namespace polynomial_solution_l461_46110

noncomputable def roots (a b c : ℤ) : Set ℝ :=
  { x : ℝ | a * x ^ 2 + b * x + c = 0 }

theorem polynomial_solution :
  let x1 := (1 + Real.sqrt 13) / 2
  let x2 := (1 - Real.sqrt 13) / 2
  x1 ∈ roots 1 (-1) (-3) → x2 ∈ roots 1 (-1) (-3) →
  ((x1^5 - 20) * (3*x2^4 - 2*x2 - 35) = -1063) :=
by
  sorry

end polynomial_solution_l461_46110


namespace ratio_of_cows_to_bulls_l461_46156

-- Define the total number of cattle
def total_cattle := 555

-- Define the number of bulls
def number_of_bulls := 405

-- Compute the number of cows
def number_of_cows := total_cattle - number_of_bulls

-- Define the expected ratio of cows to bulls
def expected_ratio_cows_to_bulls := (10, 27)

-- Prove that the ratio of cows to bulls is equal to the expected ratio
theorem ratio_of_cows_to_bulls : 
  (number_of_cows / (gcd number_of_cows number_of_bulls), number_of_bulls / (gcd number_of_cows number_of_bulls)) = expected_ratio_cows_to_bulls :=
sorry

end ratio_of_cows_to_bulls_l461_46156


namespace relatively_prime_powers_of_two_l461_46125

theorem relatively_prime_powers_of_two (a : ℤ) (h₁ : a % 2 = 1) (n m : ℕ) (h₂ : n ≠ m) :
  Int.gcd (a^(2^n) + 2^(2^n)) (a^(2^m) + 2^(2^m)) = 1 :=
by
  sorry

end relatively_prime_powers_of_two_l461_46125


namespace midpoint_ellipse_trajectory_l461_46134

theorem midpoint_ellipse_trajectory (x y x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 / 12) + (y0 / 8) = 1 →
  (x1^2 / 24) + (y1^2 / 16) = 1 →
  (x2^2 / 24) + (y2^2 / 16) = 1 →
  x = (x1 + x2) / 2 →
  y = (y1 + y2) / 2 →
  ∃ x y, ((x - 1)^2 / (5 / 2)) + ((y - 1)^2 / (5 / 3)) = 1 :=
by
  sorry

end midpoint_ellipse_trajectory_l461_46134


namespace natural_numbers_satisfy_equation_l461_46111

theorem natural_numbers_satisfy_equation:
  ∀ (n k : ℕ), (k^5 + 5 * n^4 = 81 * k) ↔ (n = 2 ∧ k = 1) :=
by
  sorry

end natural_numbers_satisfy_equation_l461_46111


namespace platform_length_correct_l461_46167

noncomputable def platform_length : ℝ :=
  let T := 180
  let v_kmph := 72
  let t := 20
  let v_ms := v_kmph * 1000 / 3600
  let total_distance := v_ms * t
  total_distance - T

theorem platform_length_correct : platform_length = 220 := by
  sorry

end platform_length_correct_l461_46167


namespace parker_net_income_after_taxes_l461_46139

noncomputable def parker_income : Real := sorry

theorem parker_net_income_after_taxes :
  let daily_pay := 63
  let hours_per_day := 8
  let hourly_rate := daily_pay / hours_per_day
  let overtime_rate := 1.5 * hourly_rate
  let overtime_hours_per_weekend_day := 3
  let weekends_in_6_weeks := 6
  let days_per_week := 7
  let total_days_in_6_weeks := days_per_week * weekends_in_6_weeks
  let regular_earnings := daily_pay * total_days_in_6_weeks
  let total_overtime_earnings := overtime_rate * overtime_hours_per_weekend_day * 2 * weekends_in_6_weeks
  let gross_income := regular_earnings + total_overtime_earnings
  let tax_rate := 0.1
  let net_income_after_taxes := gross_income * (1 - tax_rate)
  net_income_after_taxes = 2764.125 := by sorry

end parker_net_income_after_taxes_l461_46139


namespace min_points_to_guarantee_victory_l461_46107

noncomputable def points_distribution (pos : ℕ) : ℕ :=
  match pos with
  | 1 => 7
  | 2 => 4
  | 3 => 2
  | _ => 0

def max_points_per_race : ℕ := 7
def num_races : ℕ := 3

theorem min_points_to_guarantee_victory : ∃ min_points, min_points = 18 ∧ 
  (∀ other_points, other_points < 18) := 
by {
  sorry
}

end min_points_to_guarantee_victory_l461_46107


namespace hyperbola_eccentricity_is_sqrt2_l461_46148

noncomputable def hyperbola_eccentricity (a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) 
(hyp3 : b = a) : ℝ :=
    let c := Real.sqrt (2) * a
    c / a

theorem hyperbola_eccentricity_is_sqrt2 
(a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) (hyp3 : b = a) :
hyperbola_eccentricity a b hyp1 hyp2 hyp3 = Real.sqrt 2 := sorry

end hyperbola_eccentricity_is_sqrt2_l461_46148


namespace root_in_interval_l461_46194

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval : 
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intros h1 h2
  sorry

end root_in_interval_l461_46194


namespace find_principal_amount_l461_46181

theorem find_principal_amount
  (P r : ℝ) -- P for Principal amount, r for interest rate
  (simple_interest : 800 = P * r / 100 * 2) -- Condition 1: Simple Interest Formula
  (compound_interest : 820 = P * ((1 + r / 100) ^ 2 - 1)) -- Condition 2: Compound Interest Formula
  : P = 8000 := 
sorry

end find_principal_amount_l461_46181


namespace escalator_length_l461_46179

theorem escalator_length
  (escalator_speed : ℝ)
  (person_speed : ℝ)
  (time_taken : ℝ)
  (combined_speed := escalator_speed + person_speed)
  (distance := combined_speed * time_taken) :
  escalator_speed = 10 → person_speed = 4 → time_taken = 8 → distance = 112 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end escalator_length_l461_46179


namespace binomial_coeff_sum_l461_46109

theorem binomial_coeff_sum {a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ}
  (h : (1 - x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 128 :=
by
  sorry

end binomial_coeff_sum_l461_46109


namespace reduced_price_per_kg_l461_46196

variables (P P' : ℝ)

-- Given conditions
def condition1 := P' = P / 2
def condition2 := 800 / P' = 800 / P + 5

-- Proof problem statement
theorem reduced_price_per_kg (P P' : ℝ) (h1 : condition1 P P') (h2 : condition2 P P') :
  P' = 80 :=
by
  sorry

end reduced_price_per_kg_l461_46196


namespace find_side_length_S2_l461_46151

-- Define the variables and conditions
variables (r s : ℕ)
def is_solution (r s : ℕ) : Prop :=
  2 * r + s = 2160 ∧ 2 * r + 3 * s = 3450

-- Define the problem statement
theorem find_side_length_S2 (r s : ℕ) (h : is_solution r s) : s = 645 :=
sorry

end find_side_length_S2_l461_46151


namespace ratio_docking_to_license_l461_46119

noncomputable def Mitch_savings : ℕ := 20000
noncomputable def boat_cost_per_foot : ℕ := 1500
noncomputable def license_and_registration_fees : ℕ := 500
noncomputable def max_boat_length : ℕ := 12

theorem ratio_docking_to_license :
  let remaining_amount := Mitch_savings - license_and_registration_fees
  let cost_of_longest_boat := boat_cost_per_foot * max_boat_length
  let docking_fees := remaining_amount - cost_of_longest_boat
  docking_fees / license_and_registration_fees = 3 :=
by
  sorry

end ratio_docking_to_license_l461_46119


namespace total_annual_gain_l461_46166

-- Definitions based on given conditions
variable (A B C : Type) [Field ℝ]

-- Assume initial investments and time factors
variable (x : ℝ) (A_share : ℝ := 5000) -- A's share is Rs. 5000

-- Total annual gain to be proven
theorem total_annual_gain (x : ℝ) (A_share B_share C_share Total_Profit : ℝ) :
  A_share = 5000 → 
  B_share = (2 * x) * (6 / 12) → 
  C_share = (3 * x) * (4 / 12) → 
  (A_share / (x * 12)) * Total_Profit = 5000 → -- A's determined share from profit
  Total_Profit = 15000 := 
by 
  sorry

end total_annual_gain_l461_46166


namespace range_of_x_l461_46102

theorem range_of_x (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
by {
  sorry
}

end range_of_x_l461_46102


namespace repeating_block_digits_l461_46172

theorem repeating_block_digits (n d : ℕ) (h1 : n = 3) (h2 : d = 11) : 
  (∃ repeating_block_length, repeating_block_length = 2) := by
  sorry

end repeating_block_digits_l461_46172


namespace calculate_expression_is_correct_l461_46193

noncomputable def calculate_expression : ℝ :=
  -(-2) + 2 * Real.cos (Real.pi / 3) + (-1 / 8)⁻¹ + (Real.pi - 3.14) ^ 0

theorem calculate_expression_is_correct :
  calculate_expression = -4 :=
by
  -- the conditions as definitions
  have h1 : Real.cos (Real.pi / 3) = 1 / 2 := by sorry
  have h2 : (Real.pi - 3.14) ^ 0 = 1 := by sorry
  -- use these conditions to prove the main statement
  sorry

end calculate_expression_is_correct_l461_46193


namespace determine_b_div_a_l461_46189

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem determine_b_div_a
  (a b : ℝ)
  (hf_deriv : ∀ x : ℝ, (deriv (f a b)) x = 3 * x^2 + 2 * a * x + b)
  (hf_max : f a b 1 = 10)
  (hf_deriv_at_1 : (deriv (f a b)) 1 = 0) :
  b / a = -3 / 2 :=
sorry

end determine_b_div_a_l461_46189


namespace grading_options_count_l461_46187

theorem grading_options_count :
  (4 ^ 15) = 1073741824 :=
by
  sorry

end grading_options_count_l461_46187


namespace green_disks_count_l461_46136

-- Definitions of the conditions given in the problem
def total_disks : ℕ := 14
def red_disks (g : ℕ) : ℕ := 2 * g
def blue_disks (g : ℕ) : ℕ := g / 2

-- The theorem statement to prove
theorem green_disks_count (g : ℕ) (h : 2 * g + g + g / 2 = total_disks) : g = 4 :=
sorry

end green_disks_count_l461_46136


namespace find_m_values_l461_46159

theorem find_m_values :
  ∃ m : ℝ, (∀ (α β : ℝ), (3 * α^2 + m * α - 4 = 0 ∧ 3 * β^2 + m * β - 4 = 0) ∧ (α * β = -4 / 3) ∧ (α + β = -m / 3) ∧ (α * β = 2 * (α^3 + β^3))) ↔
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
sorry

end find_m_values_l461_46159


namespace range_of_a_l461_46175

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * ((a + 2) * x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ deriv (f a) x = 0 ∧ deriv (f a) y = 0) ↔ a < -1 ∨ a > 2 :=
by
  sorry

end range_of_a_l461_46175


namespace spheres_in_base_l461_46128

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end spheres_in_base_l461_46128


namespace base3_to_base5_conversion_l461_46108

-- Define the conversion from base 3 to decimal
def base3_to_decimal (n : ℕ) : ℕ :=
  n % 10 * 1 + (n / 10 % 10) * 3 + (n / 100 % 10) * 9 + (n / 1000 % 10) * 27 + (n / 10000 % 10) * 81

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : ℕ :=
  n % 5 + (n / 5 % 5) * 10 + (n / 25 % 5) * 100

-- The initial number in base 3
def initial_number_base3 : ℕ := 10121

-- The final number in base 5
def final_number_base5 : ℕ := 342

-- The theorem that states the conversion result
theorem base3_to_base5_conversion :
  decimal_to_base5 (base3_to_decimal initial_number_base3) = final_number_base5 :=
by
  sorry

end base3_to_base5_conversion_l461_46108


namespace rd_expense_necessary_for_increase_l461_46153

theorem rd_expense_necessary_for_increase :
  ∀ (R_and_D_t : ℝ) (delta_APL_t1 : ℝ),
  R_and_D_t = 3289.31 → delta_APL_t1 = 1.55 →
  R_and_D_t / delta_APL_t1 = 2122 := 
by
  intros R_and_D_t delta_APL_t1 hR hD
  rw [hR, hD]
  norm_num
  sorry

end rd_expense_necessary_for_increase_l461_46153


namespace classroom_not_1_hectare_l461_46164

def hectare_in_sq_meters : ℕ := 10000
def classroom_area_approx : ℕ := 60

theorem classroom_not_1_hectare : ¬ (classroom_area_approx = hectare_in_sq_meters) :=
by 
  sorry

end classroom_not_1_hectare_l461_46164


namespace expression_eq_49_l461_46163

theorem expression_eq_49 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 := 
by 
  sorry

end expression_eq_49_l461_46163


namespace speed_of_man_in_still_water_l461_46173

theorem speed_of_man_in_still_water (v_m v_s : ℝ) (h1 : v_m + v_s = 18) (h2 : v_m - v_s = 13) : v_m = 15.5 :=
by {
  -- Proof is not required as per the instructions
  sorry
}

end speed_of_man_in_still_water_l461_46173


namespace percentage_failed_both_l461_46169

theorem percentage_failed_both (p_hindi p_english p_pass_both x : ℝ)
  (h₁ : p_hindi = 0.25)
  (h₂ : p_english = 0.5)
  (h₃ : p_pass_both = 0.5)
  (h₄ : (p_hindi + p_english - x) = 0.5) : 
  x = 0.25 := 
sorry

end percentage_failed_both_l461_46169


namespace log_expression_evaluation_l461_46127

theorem log_expression_evaluation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (Real.log (x^2) / Real.log (y^8)) * (Real.log (y^3) / Real.log (x^7)) * (Real.log (x^4) / Real.log (y^5)) * (Real.log (y^5) / Real.log (x^4)) * (Real.log (x^7) / Real.log (y^3)) =
  (1 / 4) * (Real.log x / Real.log y) := 
by
  sorry

end log_expression_evaluation_l461_46127


namespace order_of_abc_l461_46171

theorem order_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a^2 + b^2 < a^2 + c^2) (h2 : a^2 + c^2 < b^2 + c^2) : a < b ∧ b < c := 
by
  sorry

end order_of_abc_l461_46171


namespace number_of_movies_in_series_l461_46180

variables (watched_movies remaining_movies total_movies : ℕ)

theorem number_of_movies_in_series 
  (h_watched : watched_movies = 4) 
  (h_remaining : remaining_movies = 4) :
  total_movies = watched_movies + remaining_movies :=
by
  sorry

end number_of_movies_in_series_l461_46180


namespace bob_total_distance_l461_46113

theorem bob_total_distance:
  let time1 := 1.5
  let speed1 := 60
  let time2 := 2
  let speed2 := 45
  (time1 * speed1) + (time2 * speed2) = 180 := 
  by
  sorry

end bob_total_distance_l461_46113


namespace candy_distribution_l461_46133

theorem candy_distribution (candy_total friends : ℕ) (candies : List ℕ) :
  candy_total = 47 ∧ friends = 5 ∧ List.length candies = friends ∧
  (∀ n ∈ candies, n = 9) → (47 % 5 = 2) :=
by
  sorry

end candy_distribution_l461_46133


namespace symmetric_points_sum_l461_46190

theorem symmetric_points_sum (a b : ℝ) (P Q : ℝ × ℝ) 
    (hP : P = (3, a)) (hQ : Q = (b, 2))
    (symm : P = (-Q.1, Q.2)) : a + b = -1 := by
  sorry

end symmetric_points_sum_l461_46190


namespace temperature_at_4km_l461_46198

theorem temperature_at_4km (ground_temp : ℤ) (drop_rate : ℤ) (altitude : ℕ) (ΔT : ℤ) : 
  ground_temp = 15 ∧ drop_rate = -5 ∧ ΔT = altitude * drop_rate ∧ altitude = 4 → 
  ground_temp + ΔT = -5 :=
by
  sorry

end temperature_at_4km_l461_46198


namespace total_servings_of_vegetables_l461_46157

def carrot_plant_serving : ℕ := 4
def num_green_bean_plants : ℕ := 10
def num_carrot_plants : ℕ := 8
def num_corn_plants : ℕ := 12
def num_tomato_plants : ℕ := 15
def corn_plant_serving : ℕ := 5 * carrot_plant_serving
def green_bean_plant_serving : ℕ := corn_plant_serving / 2
def tomato_plant_serving : ℕ := carrot_plant_serving + 3

theorem total_servings_of_vegetables :
  (num_carrot_plants * carrot_plant_serving) +
  (num_corn_plants * corn_plant_serving) +
  (num_green_bean_plants * green_bean_plant_serving) +
  (num_tomato_plants * tomato_plant_serving) = 477 := by
  sorry

end total_servings_of_vegetables_l461_46157


namespace find_m_l461_46116

def f (x m : ℝ) : ℝ := x ^ 2 - 3 * x + m
def g (x m : ℝ) : ℝ := 2 * x ^ 2 - 6 * x + 5 * m

theorem find_m (m : ℝ) (h : 3 * f 3 m = 2 * g 3 m) : m = 0 :=
by sorry

end find_m_l461_46116


namespace candidates_appeared_equal_l461_46118

theorem candidates_appeared_equal 
  (A_candidates B_candidates : ℕ)
  (A_selected B_selected : ℕ)
  (h1 : 6 * A_candidates = A_selected * 100)
  (h2 : 7 * B_candidates = B_selected * 100)
  (h3 : B_selected = A_selected + 83)
  (h4 : A_candidates = B_candidates):
  A_candidates = 8300 :=
by
  sorry

end candidates_appeared_equal_l461_46118


namespace function_passes_through_fixed_point_l461_46115

noncomputable def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) + 7

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  given_function a 1 = 8 :=
by
  sorry

end function_passes_through_fixed_point_l461_46115


namespace pi_is_irrational_l461_46114

theorem pi_is_irrational (π : ℝ) (h : π = Real.pi) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ π = a / b :=
by
  sorry

end pi_is_irrational_l461_46114


namespace vectors_parallel_l461_46149

theorem vectors_parallel (m : ℝ) : 
    (∃ k : ℝ, (m, 4) = (k * 5, k * -2)) → m = -10 := 
by
  sorry

end vectors_parallel_l461_46149


namespace prime_pairs_solution_l461_46121

theorem prime_pairs_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  7 * p * q^2 + p = q^3 + 43 * p^3 + 1 ↔ (p = 2 ∧ q = 7) :=
by
  sorry

end prime_pairs_solution_l461_46121


namespace train_length_l461_46192

theorem train_length (speed_kmph : ℝ) (time_sec : ℝ) (h_speed : speed_kmph = 36) (h_time : time_sec = 6.5) : 
  (speed_kmph * 1000 / 3600) * time_sec = 65 := 
by {
  -- Placeholder for proof
  sorry
}

end train_length_l461_46192


namespace manuscript_typing_cost_l461_46142

theorem manuscript_typing_cost :
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 :=
by
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  have : cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 := sorry
  exact this

end manuscript_typing_cost_l461_46142
