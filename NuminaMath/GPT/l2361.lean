import Mathlib

namespace shaded_area_correct_l2361_236138

noncomputable def side_length : ℝ := 24
noncomputable def radius : ℝ := side_length / 4
noncomputable def area_of_square : ℝ := side_length ^ 2
noncomputable def area_of_one_circle : ℝ := Real.pi * radius ^ 2
noncomputable def total_area_of_circles : ℝ := 5 * area_of_one_circle
noncomputable def shaded_area : ℝ := area_of_square - total_area_of_circles

theorem shaded_area_correct :
  shaded_area = 576 - 180 * Real.pi := by
  sorry

end shaded_area_correct_l2361_236138


namespace problem_solution_l2361_236164

theorem problem_solution (x y z : ℝ)
  (h1 : 1/x + 1/y + 1/z = 2)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) :
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 :=
sorry

end problem_solution_l2361_236164


namespace value_is_6_l2361_236148

-- We know the conditions that the least number which needs an increment is 858
def least_number : ℕ := 858

-- Define the numbers 24, 32, 36, and 54
def num1 : ℕ := 24
def num2 : ℕ := 32
def num3 : ℕ := 36
def num4 : ℕ := 54

-- Define the LCM function to compute the least common multiple
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Define the LCM of the four numbers
def lcm_all : ℕ := lcm (lcm num1 num2) (lcm num3 num4)

-- Compute the value that needs to be added
def value_to_be_added : ℕ := lcm_all - least_number

-- Prove that this value equals to 6
theorem value_is_6 : value_to_be_added = 6 := by
  -- Proof would go here
  sorry

end value_is_6_l2361_236148


namespace annual_interest_rate_l2361_236109

theorem annual_interest_rate (principal total_paid: ℝ) (h_principal : principal = 150) (h_total_paid : total_paid = 162) : 
  ((total_paid - principal) / principal) * 100 = 8 :=
by
  sorry

end annual_interest_rate_l2361_236109


namespace polygon_sides_l2361_236186

theorem polygon_sides (n : ℕ) : 
  (∃ D, D = 104) ∧ (D = (n - 1) * (n - 4) / 2)  → n = 17 :=
by
  sorry

end polygon_sides_l2361_236186


namespace f_2007_l2361_236149

noncomputable def f : ℕ → ℝ :=
  sorry

axiom functional_eq (x y : ℕ) : f (x + y) = f x * f y

axiom f_one : f 1 = 2

theorem f_2007 : f 2007 = 2 ^ 2007 :=
by
  sorry

end f_2007_l2361_236149


namespace arithmetic_geometric_sequence_l2361_236116

theorem arithmetic_geometric_sequence (a b : ℝ)
  (h1 : 2 * a = 1 + b)
  (h2 : b^2 = a)
  (h3 : a ≠ b) : a = 1 / 4 :=
by
  sorry

end arithmetic_geometric_sequence_l2361_236116


namespace smallest_integer_n_l2361_236169

theorem smallest_integer_n (n : ℕ) : (1 / 2 : ℝ) < n / 9 ↔ n ≥ 5 := 
sorry

end smallest_integer_n_l2361_236169


namespace new_angle_after_rotation_l2361_236129

def initial_angle : ℝ := 25
def rotation_clockwise : ℝ := 350
def equivalent_rotation := rotation_clockwise - 360  -- equivalent to -10 degrees

theorem new_angle_after_rotation :
  initial_angle + equivalent_rotation = 15 := by
  sorry

end new_angle_after_rotation_l2361_236129


namespace coincide_foci_of_parabola_and_hyperbola_l2361_236175

theorem coincide_foci_of_parabola_and_hyperbola (p : ℝ) (hpos : p > 0) :
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ y^2 = 2 * p * x) →
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ (x^2 / 12) - (y^2 / 4) = 1) →
  p = 8 := 
sorry

end coincide_foci_of_parabola_and_hyperbola_l2361_236175


namespace smallest_positive_multiple_of_3_4_5_is_60_l2361_236173

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end smallest_positive_multiple_of_3_4_5_is_60_l2361_236173


namespace min_arg_z_l2361_236168

noncomputable def z (x y : ℝ) := x + y * Complex.I

def satisfies_condition (x y : ℝ) : Prop :=
  Complex.abs (z x y + 3 - Real.sqrt 3 * Complex.I) = Real.sqrt 3

theorem min_arg_z (x y : ℝ) (h : satisfies_condition x y) :
  Complex.arg (z x y) = 5 * Real.pi / 6 := 
sorry

end min_arg_z_l2361_236168


namespace sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l2361_236101

theorem sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30 :
  (7^30 + 13^30) % 100 = 0 := 
sorry

end sum_of_last_two_digits_of_7_pow_30_plus_13_pow_30_l2361_236101


namespace eval_sqrt_4_8_pow_12_l2361_236163

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end eval_sqrt_4_8_pow_12_l2361_236163


namespace find_total_students_l2361_236189

theorem find_total_students (n : ℕ) : n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 → n = 509 :=
by 
  sorry

end find_total_students_l2361_236189


namespace polynomial_proof_l2361_236134

noncomputable def f (x a b c : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - a*b*c

theorem polynomial_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) (h3 : f a a b c = 0) (h4 : f b a b c = 0) (h5 : f c a b c = 0) : 
  f 0 a b c * f 1 a b c < 0 ∧ f 0 a b c * f 3 a b c > 0 :=
by 
  sorry

end polynomial_proof_l2361_236134


namespace number_of_sections_l2361_236146

-- Definitions based on the conditions in a)
def num_reels : Nat := 3
def length_per_reel : Nat := 100
def section_length : Nat := 10

-- The math proof problem statement
theorem number_of_sections :
  (num_reels * length_per_reel) / section_length = 30 := by
  sorry

end number_of_sections_l2361_236146


namespace cyclic_inequality_l2361_236136

theorem cyclic_inequality
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) :=
by
  sorry

end cyclic_inequality_l2361_236136


namespace range_of_a_for_two_zeros_l2361_236103

theorem range_of_a_for_two_zeros (a : ℝ) :
  (∀ x : ℝ, (x + 1) * Real.exp x - a = 0 → -- There's no need to delete this part, see below note 
                                              -- The question of "exactly" is virtually ensured by other parts of the Lean theories
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
                (x₁ + 1) * Real.exp x₁ - a = 0 ∧
                (x₂ + 1) * Real.exp x₂ - a = 0) → 
  (-1 / Real.exp 2 < a ∧ a < 0) :=
sorry

end range_of_a_for_two_zeros_l2361_236103


namespace sum_of_numbers_in_50th_row_l2361_236111

-- Defining the array and the row sum
def row_sum (n : ℕ) : ℕ :=
  2^n

-- Proposition stating that the 50th row sum is equal to 2^50
theorem sum_of_numbers_in_50th_row : row_sum 50 = 2^50 :=
by sorry

end sum_of_numbers_in_50th_row_l2361_236111


namespace avg_marks_l2361_236141

theorem avg_marks (P C M : ℕ) (h : P + C + M = P + 150) : (C + M) / 2 = 75 :=
by
  -- Proof goes here
  sorry

end avg_marks_l2361_236141


namespace absolute_value_c_l2361_236131

noncomputable def condition_polynomial (a b c : ℤ) : Prop :=
  a * (↑(Complex.ofReal 3) + Complex.I)^4 +
  b * (↑(Complex.ofReal 3) + Complex.I)^3 +
  c * (↑(Complex.ofReal 3) + Complex.I)^2 +
  b * (↑(Complex.ofReal 3) + Complex.I) +
  a = 0

noncomputable def coprime_integers (a b c : ℤ) : Prop :=
  Int.gcd (Int.gcd a b) c = 1

theorem absolute_value_c (a b c : ℤ) (h1 : condition_polynomial a b c) (h2 : coprime_integers a b c) :
  |c| = 97 :=
sorry

end absolute_value_c_l2361_236131


namespace selling_price_for_target_profit_l2361_236197

-- Defining the conditions
def purchase_price : ℝ := 200
def annual_cost : ℝ := 40000
def annual_sales_volume (x : ℝ) := 800 - x
def annual_profit (x : ℝ) : ℝ := (x - purchase_price) * annual_sales_volume x - annual_cost

-- The theorem to prove
theorem selling_price_for_target_profit : ∃ x : ℝ, annual_profit x = 40000 ∧ x = 400 :=
by
  sorry

end selling_price_for_target_profit_l2361_236197


namespace chris_mixed_raisins_l2361_236165

-- Conditions
variables (R C : ℝ)

-- 1. Chris mixed some pounds of raisins with 3 pounds of nuts.
-- 2. A pound of nuts costs 3 times as much as a pound of raisins.
-- 3. The total cost of the raisins was 0.25 of the total cost of the mixture.

-- Problem statement: Prove that R = 3 given the conditions
theorem chris_mixed_raisins :
  R * C = 0.25 * (R * C + 3 * 3 * C) → R = 3 :=
by
  sorry

end chris_mixed_raisins_l2361_236165


namespace inequality_solution_set_l2361_236170

theorem inequality_solution_set :
  { x : ℝ | -x^2 + 2*x > 0 } = { x : ℝ | 0 < x ∧ x < 2 } :=
sorry

end inequality_solution_set_l2361_236170


namespace number_of_distinct_products_l2361_236120

   -- We define the set S
   def S : Finset ℕ := {2, 3, 5, 11, 13}

   -- We define what it means to have a distinct product of two or more elements
   def distinctProducts (s : Finset ℕ) : Finset ℕ :=
     (s.powerset.filter (λ t => 2 ≤ t.card)).image (λ t => t.prod id)

   -- We state the theorem that there are exactly 26 distinct products
   theorem number_of_distinct_products : (distinctProducts S).card = 26 :=
   sorry
   
end number_of_distinct_products_l2361_236120


namespace number_of_sequences_l2361_236142

theorem number_of_sequences : 
  let n : ℕ := 7
  let ones : ℕ := 5
  let twos : ℕ := 2
  let comb := Nat.choose
  (ones + twos = n) ∧  
  comb (ones + 1) twos + comb (ones + 1) (twos - 1) = 21 := 
  by sorry

end number_of_sequences_l2361_236142


namespace frosting_cupcakes_l2361_236180

noncomputable def rate_cagney := 1 / 25  -- Cagney's rate in cupcakes per second
noncomputable def rate_lacey := 1 / 20  -- Lacey's rate in cupcakes per second

noncomputable def break_time := 30      -- Break time in seconds
noncomputable def work_period := 180    -- Work period in seconds before a break
noncomputable def total_time := 600     -- Total time in seconds (10 minutes)

noncomputable def combined_rate := rate_cagney + rate_lacey -- Combined rate in cupcakes per second

-- Effective work time after considering breaks
noncomputable def effective_work_time :=
  total_time - (total_time / work_period) * break_time

-- Total number of cupcakes frosted in the effective work time
noncomputable def total_cupcakes := combined_rate * effective_work_time

theorem frosting_cupcakes : total_cupcakes = 48 :=
by
  sorry

end frosting_cupcakes_l2361_236180


namespace simplify_expression_l2361_236172

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x :=
by
  sorry

end simplify_expression_l2361_236172


namespace max_value_2ab_sqrt2_plus_2ac_l2361_236121

theorem max_value_2ab_sqrt2_plus_2ac (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 :=
sorry

end max_value_2ab_sqrt2_plus_2ac_l2361_236121


namespace two_digit_number_solution_l2361_236190

theorem two_digit_number_solution : ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ 10 * x + y = 10 * 5 + 3 ∧ 10 * y + x = 10 * 3 + 5 ∧ 3 * z = 3 * 15 ∧ 2 * z = 2 * 15 := by
  sorry

end two_digit_number_solution_l2361_236190


namespace parabola_min_perimeter_l2361_236181

noncomputable def focus_of_parabola (p : ℝ) (hp : p > 0) : ℝ × ℝ :=
(1, 0)

noncomputable def A : ℝ × ℝ := (3, 2)

noncomputable def is_on_parabola (P : ℝ × ℝ) (p : ℝ) : Prop :=
P.2 ^ 2 = 2 * p * P.1

noncomputable def area_of_triangle (A P F : ℝ × ℝ) : ℝ :=
0.5 * abs (A.1 * (P.2 - F.2) + P.1 * (F.2 - A.2) + F.1 * (A.2 - P.2))

noncomputable def perimeter (A P F : ℝ × ℝ) : ℝ := 
abs (A.1 - P.1) + abs (A.1 - F.1) + abs (P.1 - F.1)

theorem parabola_min_perimeter 
  {p : ℝ} (hp : p > 0)
  (A : ℝ × ℝ) (ha : A = (3,2))
  (P : ℝ × ℝ) (hP : is_on_parabola P p)
  {F : ℝ × ℝ} (hF : F = focus_of_parabola p hp)
  (harea : area_of_triangle A P F = 1)
  (hmin : ∀ P', is_on_parabola P' p → 
    perimeter A P' F ≥ perimeter A P F) :
  abs (P.1 - F.1) = 5/2 :=
sorry

end parabola_min_perimeter_l2361_236181


namespace largest_x_floor_condition_l2361_236143

theorem largest_x_floor_condition :
  ∃ x : ℝ, (⌊x⌋ : ℝ) / x = 8 / 9 ∧
      (∀ y : ℝ, (⌊y⌋ : ℝ) / y = 8 / 9 → y ≤ x) →
  x = 63 / 8 :=
by
  sorry

end largest_x_floor_condition_l2361_236143


namespace A_share_in_profit_l2361_236110

/-
Given:
1. a_contribution (A's amount contributed in Rs. 5000) and duration (in months 8)
2. b_contribution (B's amount contributed in Rs. 6000) and duration (in months 5)
3. total_profit (Total profit in Rs. 8400)

Prove that A's share in the total profit is Rs. 4800.
-/

theorem A_share_in_profit 
  (a_contribution : ℝ) (a_months : ℝ) 
  (b_contribution : ℝ) (b_months : ℝ) 
  (total_profit : ℝ) :
  a_contribution = 5000 → 
  a_months = 8 → 
  b_contribution = 6000 → 
  b_months = 5 → 
  total_profit = 8400 → 
  (a_contribution * a_months / (a_contribution * a_months + b_contribution * b_months) * total_profit) = 4800 := 
by {
  sorry
}

end A_share_in_profit_l2361_236110


namespace C_pow_eq_target_l2361_236176

open Matrix

-- Define the specific matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

-- Define the target matrix for the formula we need to prove
def C_power_50 : Matrix (Fin 2) (Fin 2) ℤ := !![101, 50; -200, -99]

-- Prove that C^50 equals to the target matrix
theorem C_pow_eq_target (n : ℕ) (h : n = 50) : C ^ n = C_power_50 := by
  rw [h]
  sorry

end C_pow_eq_target_l2361_236176


namespace value_of_1_plus_i_cubed_l2361_236171

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- Condition: i^2 = -1
lemma i_squared : i ^ 2 = -1 := by
  unfold i
  exact Complex.I_sq

-- The proof statement
theorem value_of_1_plus_i_cubed : 1 + i ^ 3 = 1 - i := by
  sorry

end value_of_1_plus_i_cubed_l2361_236171


namespace sin_square_alpha_minus_pi_div_4_l2361_236135

theorem sin_square_alpha_minus_pi_div_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α - Real.pi / 4) ^ 2 = 1 / 6 := 
sorry

end sin_square_alpha_minus_pi_div_4_l2361_236135


namespace dad_eyes_l2361_236187

def mom_eyes : ℕ := 1
def kids_eyes : ℕ := 3 * 4
def total_eyes : ℕ := 16

theorem dad_eyes :
  mom_eyes + kids_eyes + (total_eyes - (mom_eyes + kids_eyes)) = total_eyes :=
by 
  -- The proof part is omitted as per instructions
  sorry

example : (total_eyes - (mom_eyes + kids_eyes)) = 3 :=
by 
  -- The proof part is omitted as per instructions
  sorry

end dad_eyes_l2361_236187


namespace plane_through_points_eq_l2361_236133

-- Define the points M, N, P
def M := (1, 2, 0)
def N := (1, -1, 2)
def P := (0, 1, -1)

-- Define the target plane equation
def target_plane_eq (x y z : ℝ) := 5 * x - 2 * y + 3 * z - 1 = 0

-- Main theorem statement
theorem plane_through_points_eq :
  ∀ (x y z : ℝ),
    (∃ A B C : ℝ,
      A * (x - 1) + B * (y - 2) + C * z = 0 ∧
      A * (1 - 1) + B * (-1 - 2) + C * (2 - 0) = 0 ∧
      A * (0 - 1) + B * (1 - 2) + C * (-1 - 0) = 0) →
    target_plane_eq x y z :=
by
  sorry

end plane_through_points_eq_l2361_236133


namespace solution_set_of_inequality_l2361_236153

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) * (x - 2) ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end solution_set_of_inequality_l2361_236153


namespace quadratic_equation_terms_l2361_236179

theorem quadratic_equation_terms (x : ℝ) :
  (∃ a b c : ℝ, a = 3 ∧ b = -6 ∧ c = -7 ∧ a * x^2 + b * x + c = 0) →
  (∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = a * x^2 - 6 * x - 7) ∧
  (∃ (c : ℝ), c = -7 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = 3 * x^2 - 6 * x + c) :=
by
  sorry

end quadratic_equation_terms_l2361_236179


namespace log_product_identity_l2361_236196

theorem log_product_identity :
    (Real.log 9 / Real.log 8) * (Real.log 32 / Real.log 9) = 5 / 3 := 
by 
  sorry

end log_product_identity_l2361_236196


namespace observe_three_cell_types_l2361_236108

def biology_experiment
  (material : Type) (dissociation_fixative : material) (acetic_orcein_stain : material) (press_slide : Prop) : Prop :=
  ∃ (testes : material) (steps : material → Prop),
    steps testes ∧ press_slide ∧ (steps dissociation_fixative) ∧ (steps acetic_orcein_stain)

theorem observe_three_cell_types (material : Type)
  (dissociation_fixative acetic_orcein_stain : material)
  (press_slide : Prop)
  (steps : material → Prop) :
  biology_experiment material dissociation_fixative acetic_orcein_stain press_slide →
  ∃ (metaphase_of_mitosis metaphase_of_first_meiosis metaphase_of_second_meiosis : material), 
    steps metaphase_of_mitosis ∧ steps metaphase_of_first_meiosis ∧ steps metaphase_of_second_meiosis :=
sorry

end observe_three_cell_types_l2361_236108


namespace identify_false_statement_l2361_236127

-- Definitions for the conditions
def isMultipleOf (n k : Nat) : Prop := ∃ m, n = k * m

def conditions : Prop :=
  isMultipleOf 12 2 ∧
  isMultipleOf 123 3 ∧
  isMultipleOf 1234 4 ∧
  isMultipleOf 12345 5 ∧
  isMultipleOf 123456 6

-- The statement which proves which condition is false
theorem identify_false_statement : conditions → ¬ (isMultipleOf 1234 4) :=
by
  intros h
  sorry

end identify_false_statement_l2361_236127


namespace solve_problem_l2361_236114

variable (f : ℝ → ℝ)

axiom f_property : ∀ x : ℝ, f (x + 1) = x^2 - 2 * x

theorem solve_problem : f 2 = -1 :=
by
  sorry

end solve_problem_l2361_236114


namespace hyperbola_asymptotes_eq_l2361_236162

theorem hyperbola_asymptotes_eq (M : ℝ) :
  (4 / 3 = 5 / Real.sqrt M) → M = 225 / 16 :=
by
  intro h
  sorry

end hyperbola_asymptotes_eq_l2361_236162


namespace product_of_roots_eq_20_l2361_236158

open Real

theorem product_of_roots_eq_20 :
  (∀ x : ℝ, (x^2 + 18 * x + 30 = 2 * sqrt (x^2 + 18 * x + 45)) → 
  (x^2 + 18 * x + 20 = 0)) → 
  ∀ α β : ℝ, (α ≠ β ∧ α * β = 20) :=
by
  intros h x hx
  sorry

end product_of_roots_eq_20_l2361_236158


namespace arrangement_count_equivalent_problem_l2361_236157

noncomputable def number_of_unique_arrangements : Nat :=
  let n : Nat := 6 -- Number of balls and boxes
  let match_3_boxes_ways := Nat.choose n 3 -- Choosing 3 boxes out of 6
  let permute_remaining_boxes := 2 -- Permutations of the remaining 3 boxes such that no numbers match
  match_3_boxes_ways * permute_remaining_boxes

theorem arrangement_count_equivalent_problem :
  number_of_unique_arrangements = 40 := by
  sorry

end arrangement_count_equivalent_problem_l2361_236157


namespace whole_numbers_count_between_cubic_roots_l2361_236117

theorem whole_numbers_count_between_cubic_roots : 
  ∃ (n : ℕ) (h₁ : 3^3 < 50 ∧ 50 < 4^3) (h₂ : 7^3 < 500 ∧ 500 < 8^3), 
  n = 4 :=
by
  sorry

end whole_numbers_count_between_cubic_roots_l2361_236117


namespace square_inscription_l2361_236151

theorem square_inscription (a b : ℝ) (s1 s2 : ℝ)
  (h_eq_side_smaller : s1 = 4)
  (h_eq_side_larger : s2 = 3 * Real.sqrt 2)
  (h_sum_segments : a + b = s2)
  (h_eq_sum_squares : a^2 + b^2 = (4 * Real.sqrt 2)^2) :
  a * b = -7 := 
by sorry

end square_inscription_l2361_236151


namespace black_more_than_blue_l2361_236166

noncomputable def number_of_pencils := 8
noncomputable def number_of_blue_pens := 2 * number_of_pencils
noncomputable def number_of_red_pens := number_of_pencils - 2
noncomputable def total_pens := 48

-- Given the conditions
def satisfies_conditions (K B P : ℕ) : Prop :=
  P = number_of_pencils ∧
  B = number_of_blue_pens ∧
  K + B + number_of_red_pens = total_pens

-- Prove the number of more black pens than blue pens
theorem black_more_than_blue (K B P : ℕ) : satisfies_conditions K B P → (K - B) = 10 := by
  sorry

end black_more_than_blue_l2361_236166


namespace problem1_problem2_l2361_236174

-- Using the conditions from a) and the correct answers from b):
-- 1. Given an angle α with a point P(-4,3) on its terminal side

theorem problem1 (α : ℝ) (x y r : ℝ) (h₁ : x = -4) (h₂ : y = 3) (h₃ : r = 5) 
  (hx : r = Real.sqrt (x^2 + y^2)) 
  (hsin : Real.sin α = y / r) 
  (hcos : Real.cos α = x / r) 
  : (Real.cos (π / 2 + α) * Real.sin (-π - α)) / (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3 / 4 :=
by sorry

-- 2. Let k be an integer
theorem problem2 (α : ℝ) (k : ℤ)
  : (Real.sin (k * π - α) * Real.cos ((k + 1) * π - α)) / (Real.sin ((k - 1) * π + α) * Real.cos (k * π + α)) = -1 :=
by sorry

end problem1_problem2_l2361_236174


namespace truth_values_of_p_and_q_l2361_236184

variable {p q : Prop}

theorem truth_values_of_p_and_q (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end truth_values_of_p_and_q_l2361_236184


namespace swim_team_more_people_l2361_236145

theorem swim_team_more_people :
  let car1_people := 5
  let car2_people := 4
  let van1_people := 3
  let van2_people := 3
  let van3_people := 5
  let minibus_people := 10

  let car_max_capacity := 6
  let van_max_capacity := 8
  let minibus_max_capacity := 15

  let actual_people := car1_people + car2_people + van1_people + van2_people + van3_people + minibus_people
  let max_capacity := 2 * car_max_capacity + 3 * van_max_capacity + minibus_max_capacity
  (max_capacity - actual_people : ℕ) = 21 := 
  by
    sorry

end swim_team_more_people_l2361_236145


namespace fraction_simplification_l2361_236192

theorem fraction_simplification : 
  ((2 * 7) * (6 * 14)) / ((14 * 6) * (2 * 7)) = 1 :=
by
  sorry

end fraction_simplification_l2361_236192


namespace union_of_A_and_B_l2361_236188

/-- Given sets A and B defined as follows: A = {x | -1 <= x <= 3} and B = {x | 0 < x < 4}.
Prove that their union A ∪ B is the interval [-1, 4). -/
theorem union_of_A_and_B :
  let A := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let B := {x : ℝ | 0 < x ∧ x < 4}
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} :=
by
  sorry

end union_of_A_and_B_l2361_236188


namespace hcf_of_48_and_64_is_16_l2361_236159

theorem hcf_of_48_and_64_is_16
  (lcm_value : Nat)
  (hcf_value : Nat)
  (a : Nat)
  (b : Nat)
  (h_lcm : lcm_value = Nat.lcm a b)
  (hcf_def : hcf_value = Nat.gcd a b)
  (h_lcm_value : lcm_value = 192)
  (h_a : a = 48)
  (h_b : b = 64)
  : hcf_value = 16 := by
  sorry

end hcf_of_48_and_64_is_16_l2361_236159


namespace octal_sum_l2361_236147

open Nat

def octal_to_decimal (oct : ℕ) : ℕ :=
  match oct with
  | 0 => 0
  | n => let d3 := (n / 100) % 10
         let d2 := (n / 10) % 10
         let d1 := n % 10
         d3 * 8^2 + d2 * 8^1 + d1 * 8^0

def decimal_to_octal (dec : ℕ) : ℕ :=
  let rec aux (n : ℕ) (mul : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else aux (n / 8) (mul * 10) (acc + (n % 8) * mul)
  aux dec 1 0

theorem octal_sum :
  let a := 451
  let b := 167
  octal_to_decimal 451 + octal_to_decimal 167 = octal_to_decimal 640 := sorry

end octal_sum_l2361_236147


namespace fraction_inspected_by_Jane_l2361_236152

theorem fraction_inspected_by_Jane (P : ℝ) (x y : ℝ) 
    (h1: 0.007 * x * P + 0.008 * y * P = 0.0075 * P) 
    (h2: x + y = 1) : y = 0.5 :=
by sorry

end fraction_inspected_by_Jane_l2361_236152


namespace coefficient_of_monomial_l2361_236177

theorem coefficient_of_monomial : 
  ∀ (m n : ℝ), -((2 * Real.pi) / 3) * m * (n ^ 5) = -((2 * Real.pi) / 3) * m * (n ^ 5) :=
by
  sorry

end coefficient_of_monomial_l2361_236177


namespace monotonicity_of_g_l2361_236195

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) / (x ^ 2)

theorem monotonicity_of_g (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x : ℝ, x > 0 → (g a x) < (g a (x + 1))) ∧ (∀ x : ℝ, x < 0 → (g a x) > (g a (x - 1))) :=
  sorry

end monotonicity_of_g_l2361_236195


namespace comprehensive_score_correct_l2361_236115

def comprehensive_score
  (study_score hygiene_score discipline_score participation_score : ℕ)
  (study_weight hygiene_weight discipline_weight participation_weight : ℚ) : ℚ :=
  study_score * study_weight +
  hygiene_score * hygiene_weight +
  discipline_score * discipline_weight +
  participation_score * participation_weight

theorem comprehensive_score_correct :
  let study_score := 80
  let hygiene_score := 90
  let discipline_score := 84
  let participation_score := 70
  let study_weight := 0.4
  let hygiene_weight := 0.25
  let discipline_weight := 0.25
  let participation_weight := 0.1
  comprehensive_score study_score hygiene_score discipline_score participation_score
                      study_weight hygiene_weight discipline_weight participation_weight
  = 82.5 :=
by 
  sorry

#eval comprehensive_score 80 90 84 70 0.4 0.25 0.25 0.1  -- output should be 82.5

end comprehensive_score_correct_l2361_236115


namespace rectangle_area_l2361_236104

theorem rectangle_area (w l : ℕ) (h1 : l = w + 8) (h2 : 2 * l + 2 * w = 176) :
  l * w = 1920 :=
by
  sorry

end rectangle_area_l2361_236104


namespace find_e_l2361_236198

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) (h1 : 3 + d + e + f = -6)
  (h2 : - f / 3 = -6)
  (h3 : 9 = f)
  (h4 : - d / 3 = -18) : e = -72 :=
by
  sorry

end find_e_l2361_236198


namespace geometric_sequence_seventh_term_l2361_236113

theorem geometric_sequence_seventh_term (a r : ℕ) (h₁ : a = 6) (h₂ : a * r^4 = 486) : a * r^6 = 4374 :=
by
  -- The proof is not required, hence we use sorry.
  sorry

end geometric_sequence_seventh_term_l2361_236113


namespace solve_fraction_eq_zero_l2361_236150

theorem solve_fraction_eq_zero (x : ℝ) (h : x - 2 ≠ 0) : (x + 1) / (x - 2) = 0 ↔ x = -1 :=
by
  sorry

end solve_fraction_eq_zero_l2361_236150


namespace perimeter_inequality_l2361_236105

-- Define the problem parameters
variables {R S : ℝ}  -- radius and area of the inscribed polygon
variables (P : ℝ)    -- perimeter of the convex polygon formed by chosen points

-- Define the various conditions
def circle_with_polygon (r : ℝ) := r > 0 -- Circle with positive radius
def polygon_with_area (s : ℝ) := s > 0 -- Polygon with positive area

-- Main theorem to be proven
theorem perimeter_inequality (hR : circle_with_polygon R) (hS : polygon_with_area S) :
  P ≥ (2 * S / R) :=
sorry

end perimeter_inequality_l2361_236105


namespace calculate_fraction_l2361_236107

theorem calculate_fraction :
  ( (12^4 + 484) * (24^4 + 484) * (36^4 + 484) * (48^4 + 484) * (60^4 + 484) )
  /
  ( (6^4 + 484) * (18^4 + 484) * (30^4 + 484) * (42^4 + 484) * (54^4 + 484) )
  = 181 := by
  sorry

end calculate_fraction_l2361_236107


namespace area_of_shaded_region_l2361_236155

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l2361_236155


namespace polynomial_coefficient_l2361_236144

theorem polynomial_coefficient :
  ∀ d : ℝ, (2 * (2 : ℝ)^4 + 3 * (2 : ℝ)^3 + d * (2 : ℝ)^2 - 4 * (2 : ℝ) + 15 = 0) ↔ (d = -15.75) :=
by
  sorry

end polynomial_coefficient_l2361_236144


namespace max_d_n_l2361_236106

def sequence_a (n : ℕ) : ℤ := 100 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (sequence_a n) (sequence_a (n + 1))

theorem max_d_n : ∃ n, d_n n = 401 :=
by
  -- Placeholder for the actual proof
  sorry

end max_d_n_l2361_236106


namespace robert_salary_loss_l2361_236194

variable (S : ℝ)

theorem robert_salary_loss : 
  let decreased_salary := 0.80 * S
  let increased_salary := decreased_salary * 1.20
  let percentage_loss := 100 - (increased_salary / S) * 100
  percentage_loss = 4 :=
by
  sorry

end robert_salary_loss_l2361_236194


namespace sum_of_roots_l2361_236167

theorem sum_of_roots (a b c : ℝ) (h : 3 * x^2 - 7 * x + 2 = 0) : -b / a = 7 / 3 :=
by sorry

end sum_of_roots_l2361_236167


namespace train_length_correct_l2361_236199

noncomputable def length_of_train (speed_train_kmph : ℕ) (time_to_cross_bridge_sec : ℝ) (length_of_bridge_m : ℝ) : ℝ :=
let speed_train_mps := (speed_train_kmph : ℝ) * (1000 / 3600)
let total_distance := speed_train_mps * time_to_cross_bridge_sec
total_distance - length_of_bridge_m

theorem train_length_correct :
  length_of_train 90 32.99736021118311 660 = 164.9340052795778 :=
by
  have speed_train_mps : ℝ := 90 * (1000 / 3600)
  have total_distance := speed_train_mps * 32.99736021118311
  have length_of_train := total_distance - 660
  exact sorry

end train_length_correct_l2361_236199


namespace correct_equation_for_tournament_l2361_236122

theorem correct_equation_for_tournament (x : ℕ) (h : x * (x - 1) / 2 = 28) : True :=
sorry

end correct_equation_for_tournament_l2361_236122


namespace arithmetic_sequence_m_value_l2361_236137

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

noncomputable def find_m (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  (a (m + 1) + a (m - 1) - a m ^ 2 = 0) → (S (2 * m - 1) = 38) → m = 10

-- Problem Statement
theorem arithmetic_sequence_m_value :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ),
    arithmetic_sequence a → 
    sum_of_first_n_terms S a → 
    find_m a S m :=
by
  intros a S m ha hs h₁ h₂
  sorry

end arithmetic_sequence_m_value_l2361_236137


namespace max_wickets_bowler_can_take_l2361_236154

noncomputable def max_wickets_per_over : ℕ := 3
noncomputable def overs_bowled : ℕ := 6
noncomputable def max_possible_wickets := max_wickets_per_over * overs_bowled

theorem max_wickets_bowler_can_take : max_possible_wickets = 18 → max_possible_wickets == 10 :=
by
  sorry

end max_wickets_bowler_can_take_l2361_236154


namespace cos_alpha_value_l2361_236161

theorem cos_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (hcos : Real.cos (α + π / 3) = -2 / 3) : Real.cos α = (Real.sqrt 15 - 2) / 6 := 
  by 
  sorry

end cos_alpha_value_l2361_236161


namespace find_y_l2361_236139

-- Define the problem conditions
def avg_condition (y : ℝ) : Prop := (15 + 25 + y) / 3 = 23

-- Prove that the value of 'y' satisfying the condition is 29
theorem find_y (y : ℝ) (h : avg_condition y) : y = 29 :=
sorry

end find_y_l2361_236139


namespace union_of_A_and_B_l2361_236185

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} :=
by sorry

end union_of_A_and_B_l2361_236185


namespace integer_roots_of_quadratic_eq_l2361_236178

theorem integer_roots_of_quadratic_eq (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 + x2 = a ∧ x1 * x2 = 9 * a) ↔
  a = 100 ∨ a = -64 ∨ a = 48 ∨ a = -12 ∨ a = 36 ∨ a = 0 :=
by sorry

end integer_roots_of_quadratic_eq_l2361_236178


namespace find_x_l2361_236183

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 5)
def vec_b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define what it means for two vectors to be parallel
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2)

-- Given condition: vectors a and b are parallel
theorem find_x (x : ℝ) (h : vectors_parallel vec_a (vec_b x)) : x = 5 / 3 :=
by
  sorry

end find_x_l2361_236183


namespace simon_age_l2361_236160

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end simon_age_l2361_236160


namespace percentage_2x_minus_y_of_x_l2361_236156

noncomputable def x_perc_of_2x_minus_y (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) : ℤ :=
  (2 * x - y) * 100 / x

theorem percentage_2x_minus_y_of_x (x y z : ℤ) (h1 : x / y = 4) (h2 : x + y = z) (h3 : z > 0) (h4 : y ≠ 0) :
  x_perc_of_2x_minus_y x y z h1 h2 h3 h4 = 175 :=
  sorry

end percentage_2x_minus_y_of_x_l2361_236156


namespace percent_of_g_is_a_l2361_236140

-- Definitions of the seven consecutive numbers
def consecutive_7_avg_9 (a b c d e f g : ℝ) : Prop :=
  a + b + c + d + e + f + g = 63

def is_median (d : ℝ) : Prop :=
  d = 9

def express_numbers (a b c d e f g : ℝ) : Prop :=
  a = d - 3 ∧ b = d - 2 ∧ c = d - 1 ∧ d = d ∧ e = d + 1 ∧ f = d + 2 ∧ g = d + 3

-- Main statement asserting the percentage relationship
theorem percent_of_g_is_a (a b c d e f g : ℝ) (h_avg : consecutive_7_avg_9 a b c d e f g)
  (h_median : is_median d) (h_express : express_numbers a b c d e f g) :
  (a / g) * 100 = 50 := by
  sorry

end percent_of_g_is_a_l2361_236140


namespace problem_statement_l2361_236124

noncomputable def roots (a b : ℝ) (coef1 coef2 : ℝ) :=
  ∃ x : ℝ, (x = a ∨ x = b) ∧ x^2 + coef1 * x + coef2 = 0

theorem problem_statement
  (a b c d : ℝ)
  (h1 : a + b = -57)
  (h2 : a * b = 1)
  (h3 : c + d = 57)
  (h4 : c * d = 1) :
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := 
by
  sorry

end problem_statement_l2361_236124


namespace sufficient_but_not_necessary_for_circle_l2361_236118

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (m = 0 → ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) ∧ ¬(∀m, ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0 → m = 0) :=
 by
  sorry

end sufficient_but_not_necessary_for_circle_l2361_236118


namespace minimum_transfers_required_l2361_236191

def initial_quantities : List ℕ := [2, 12, 12, 12, 12]
def target_quantity := 10
def min_transfers := 4

theorem minimum_transfers_required :
  ∃ transfers : ℕ, transfers = min_transfers ∧
  ∀ quantities : List ℕ, List.sum initial_quantities = List.sum quantities →
  (∀ q ∈ quantities, q = target_quantity) :=
by
  sorry

end minimum_transfers_required_l2361_236191


namespace expression_equals_negative_two_l2361_236119

def f (x : ℝ) : ℝ := x^3 - x - 1
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem expression_equals_negative_two : 
  f 2023 + f' 2023 + f (-2023) - f' (-2023) = -2 :=
by
  sorry

end expression_equals_negative_two_l2361_236119


namespace edge_length_of_cube_l2361_236112

/-- Define the total paint volume, remaining paint and cube paint volume -/
def total_paint_volume : ℕ := 25 * 40
def remaining_paint : ℕ := 271
def cube_paint_volume : ℕ := total_paint_volume - remaining_paint

/-- Define the volume of the cube and the statement for edge length of the cube -/
theorem edge_length_of_cube (s : ℕ) : s^3 = cube_paint_volume → s = 9 :=
by
  have h1 : cube_paint_volume = 729 := by rfl
  sorry

end edge_length_of_cube_l2361_236112


namespace alien_heads_l2361_236100

theorem alien_heads (l o : ℕ) 
  (h1 : l + o = 60) 
  (h2 : 4 * l + o = 129) : 
  l + 2 * o = 97 := 
by 
  sorry

end alien_heads_l2361_236100


namespace august_8th_is_saturday_l2361_236123

-- Defining the conditions
def august_has_31_days : Prop := true

def august_has_5_mondays : Prop := true

def august_has_4_tuesdays : Prop := true

-- Statement of the theorem
theorem august_8th_is_saturday (h1 : august_has_31_days) (h2 : august_has_5_mondays) (h3 : august_has_4_tuesdays) : ∃ d : ℕ, d = 6 :=
by
  -- Translate the correct answer "August 8th is a Saturday" into the equivalent proposition
  -- Saturday is represented by 6 if we assume 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
  sorry

end august_8th_is_saturday_l2361_236123


namespace find_extrema_l2361_236130

theorem find_extrema (x y : ℝ) (h1 : x < 0) (h2 : -1 < y) (h3 : y < 0) : 
  max (max x (x*y)) (x*y^2) = x*y ∧ min (min x (x*y)) (x*y^2) = x :=
by sorry

end find_extrema_l2361_236130


namespace shortest_player_height_l2361_236126

-- let h_tall be the height of the tallest player
-- let h_short be the height of the shortest player
-- let diff be the height difference between the tallest and the shortest player

variable (h_tall h_short diff : ℝ)

-- conditions given in the problem
axiom tall_player_height : h_tall = 77.75
axiom height_difference : diff = 9.5
axiom height_relationship : h_tall = h_short + diff

-- the statement we need to prove
theorem shortest_player_height : h_short = 68.25 := by
  sorry

end shortest_player_height_l2361_236126


namespace brigade_harvest_time_l2361_236128

theorem brigade_harvest_time (t : ℕ) :
  (t - 5 = (3 * t / 5) + ((t * (t - 8)) / (5 * (t - 4)))) → t = 20 := sorry

end brigade_harvest_time_l2361_236128


namespace simplify_expression_l2361_236132

variable {a : ℝ}

theorem simplify_expression (h₁ : a ≠ 0) (h₂ : a ≠ -1) (h₃ : a ≠ 1) :
  ( ( (a^2 + 1) / a - 2 ) / ( (a^2 - 1) / (a^2 + a) ) ) = a - 1 :=
sorry

end simplify_expression_l2361_236132


namespace range_of_a_l2361_236182

theorem range_of_a : 
  (∃ a : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (x^2 + a ≤ a*x - 3))) ↔ (a ≥ 7) :=
sorry

end range_of_a_l2361_236182


namespace range_of_ab_l2361_236102

-- Given two positive numbers a and b such that ab = a + b + 3, we need to prove ab ≥ 9.

theorem range_of_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b = a + b + 3) : 9 ≤ a * b :=
by
  sorry

end range_of_ab_l2361_236102


namespace sarahs_score_is_140_l2361_236125

theorem sarahs_score_is_140 (g s : ℕ) (h1 : s = g + 60) 
  (h2 : (s + g) / 2 = 110) (h3 : s + g < 450) : s = 140 :=
by
  sorry

end sarahs_score_is_140_l2361_236125


namespace farm_needs_horse_food_per_day_l2361_236193

-- Definition of conditions
def ratio_sheep_to_horses := 4 / 7
def food_per_horse := 230
def number_of_sheep := 32

-- Number of horses based on ratio
def number_of_horses := (number_of_sheep * 7) / 4

-- Proof Statement
theorem farm_needs_horse_food_per_day :
  (number_of_horses * food_per_horse) = 12880 :=
by
  -- skipping the proof steps
  sorry

end farm_needs_horse_food_per_day_l2361_236193
