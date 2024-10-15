import Mathlib

namespace NUMINAMATH_GPT_no_solution_for_x_l1948_194811

theorem no_solution_for_x (x : ℝ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → False :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_x_l1948_194811


namespace NUMINAMATH_GPT_max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l1948_194869

def max_elem_one (c : ℝ) : Prop :=
  max (-2) (max 3 c) = max 3 c

def max_elem_two (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : Prop :=
  max (3 * m) (max ((n + 3) * m) (-m * n)) = - m * n

def min_range_x (x : ℝ) : Prop :=
  min 2 (min (2 * x + 2) (4 - 2 * x)) = 2 → 0 ≤ x ∧ x ≤ 1

def average_min_eq_x : Prop :=
  ∀ (x : ℝ), (2 + (x + 1) + 2 * x) / 3 = min 2 (min (x + 1) (2 * x)) → x = 1

-- Lean 4 statements
theorem max_elem_one_correct (c : ℝ) : max_elem_one c := 
  sorry

theorem max_elem_two_correct {m n : ℝ} (h1 : m < 0) (h2 : n > 0) : max_elem_two m n h1 h2 :=
  sorry

theorem min_range_x_correct (h : min 2 (min (2 * x + 2) (4 - 2 * x)) = 2) : min_range_x x :=
  sorry

theorem average_min_eq_x_correct : average_min_eq_x :=
  sorry

end NUMINAMATH_GPT_max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l1948_194869


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1948_194840

-- Problem Conditions
def inductive_reasoning (s: Sort _) (g: Sort _) : Prop := 
  ∀ (x: s → g), true 

def probabilistic_conclusion : Prop :=
  ∀ (x : Prop), true

def analogical_reasoning (a: Sort _) : Prop := 
  ∀ (x: a), true 

-- The Statements to be Proved
theorem problem1 : ¬ inductive_reasoning Prop Prop = true := 
sorry

theorem problem2 : probabilistic_conclusion = true :=
sorry 

theorem problem3 : ¬ analogical_reasoning Prop = true :=
sorry 

end NUMINAMATH_GPT_problem1_problem2_problem3_l1948_194840


namespace NUMINAMATH_GPT_positive_integer_divisors_of_sum_l1948_194852

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_divisors_of_sum_l1948_194852


namespace NUMINAMATH_GPT_balls_distribution_l1948_194855

theorem balls_distribution : 
  ∃ (n : ℕ), 
    (∀ (b1 b2 : ℕ), ∀ (h : b1 + b2 = 4), b1 ≥ 1 ∧ b2 ≥ 2 → n = 10) :=
sorry

end NUMINAMATH_GPT_balls_distribution_l1948_194855


namespace NUMINAMATH_GPT_simplify_expression_l1948_194885

theorem simplify_expression : 5 * (14 / 3) * (21 / -70) = - 35 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1948_194885


namespace NUMINAMATH_GPT_sally_balloons_l1948_194803

theorem sally_balloons :
  (initial_orange_balloons : ℕ) → (lost_orange_balloons : ℕ) → 
  (remaining_orange_balloons : ℕ) → (doubled_orange_balloons : ℕ) → 
  initial_orange_balloons = 20 → 
  lost_orange_balloons = 5 →
  remaining_orange_balloons = initial_orange_balloons - lost_orange_balloons →
  doubled_orange_balloons = 2 * remaining_orange_balloons → 
  doubled_orange_balloons = 30 :=
by
  intro initial_orange_balloons lost_orange_balloons 
       remaining_orange_balloons doubled_orange_balloons
  intro h1 h2 h3 h4
  rw [h1, h2] at h3
  rw [h3] at h4
  sorry

end NUMINAMATH_GPT_sally_balloons_l1948_194803


namespace NUMINAMATH_GPT_scientific_notation_of_203000_l1948_194851

-- Define the number
def n : ℝ := 203000

-- Define the representation of the number in scientific notation
def scientific_notation (a b : ℝ) : Prop := n = a * 10^b ∧ 1 ≤ a ∧ a < 10

-- The theorem to state 
theorem scientific_notation_of_203000 : ∃ a b : ℝ, scientific_notation a b ∧ a = 2.03 ∧ b = 5 :=
by
  use 2.03
  use 5
  sorry

end NUMINAMATH_GPT_scientific_notation_of_203000_l1948_194851


namespace NUMINAMATH_GPT_arithmetic_sequence_sufficient_but_not_necessary_condition_l1948_194814

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def a_1_a_3_equals_2a_2 (a : ℕ → ℤ) :=
  a 1 + a 3 = 2 * a 2

-- Statement of the mathematical problem
theorem arithmetic_sequence_sufficient_but_not_necessary_condition (a : ℕ → ℤ) :
  is_arithmetic_sequence a → a_1_a_3_equals_2a_2 a ∧ (a_1_a_3_equals_2a_2 a → ¬ is_arithmetic_sequence a) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sufficient_but_not_necessary_condition_l1948_194814


namespace NUMINAMATH_GPT_percentage_error_x_percentage_error_y_l1948_194878

theorem percentage_error_x (x : ℝ) : 
  let correct_result := x * 10
  let erroneous_result := x / 10
  (correct_result - erroneous_result) / correct_result * 100 = 99 :=
by
  sorry

theorem percentage_error_y (y : ℝ) : 
  let correct_result := y + 15
  let erroneous_result := y - 15
  (correct_result - erroneous_result) / correct_result * 100 = (30 / (y + 15)) * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_error_x_percentage_error_y_l1948_194878


namespace NUMINAMATH_GPT_zero_point_of_function_l1948_194807

theorem zero_point_of_function : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_zero_point_of_function_l1948_194807


namespace NUMINAMATH_GPT_max_m_n_squared_l1948_194884

theorem max_m_n_squared (m n : ℤ) 
  (hmn : 1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981)
  (h_eq : (n^2 - m*n - m^2)^2 = 1) : 
  m^2 + n^2 ≤ 3524578 :=
sorry

end NUMINAMATH_GPT_max_m_n_squared_l1948_194884


namespace NUMINAMATH_GPT_family_e_initial_members_l1948_194815

theorem family_e_initial_members 
(a b c d f E : ℕ) 
(h_a : a = 7) 
(h_b : b = 8) 
(h_c : c = 10) 
(h_d : d = 13) 
(h_f : f = 10)
(h_avg : (a - 1 + b - 1 + c - 1 + d - 1 + E - 1 + f - 1) / 6 = 8) : 
E = 6 := 
by 
  sorry

end NUMINAMATH_GPT_family_e_initial_members_l1948_194815


namespace NUMINAMATH_GPT_smallest_positive_perfect_square_div_by_2_3_5_l1948_194875

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end NUMINAMATH_GPT_smallest_positive_perfect_square_div_by_2_3_5_l1948_194875


namespace NUMINAMATH_GPT_oil_bill_for_January_l1948_194894

variables (J F : ℝ)

-- Conditions
def condition1 := F = (5 / 4) * J
def condition2 := (F + 45) / J = 3 / 2

theorem oil_bill_for_January (h1 : condition1 J F) (h2 : condition2 J F) : J = 180 :=
by sorry

end NUMINAMATH_GPT_oil_bill_for_January_l1948_194894


namespace NUMINAMATH_GPT_quadrilateral_midpoints_area_l1948_194872

-- We set up the geometric context and define the problem in Lean 4.

noncomputable def area_of_midpoint_quadrilateral
  (AB CD : ℝ) (AD BC : ℝ)
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) : ℝ :=
  37.5

-- The theorem statement validating the area of the quadrilateral.
theorem quadrilateral_midpoints_area (AB CD AD BC : ℝ) 
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) :
  area_of_midpoint_quadrilateral AB CD AD BC h_AB_CD h_CD_AB h_AD_BC h_BC_AD mid_AB mid_BC mid_CD mid_DA = 37.5 :=
by 
  sorry  -- Proof is omitted.

end NUMINAMATH_GPT_quadrilateral_midpoints_area_l1948_194872


namespace NUMINAMATH_GPT_tangent_lines_through_point_l1948_194889

theorem tangent_lines_through_point {x y : ℝ} (h_circle : (x-1)^2 + (y-1)^2 = 1)
  (h_point : ∀ (x y: ℝ), (x, y) = (2, 4)) :
  (x = 2 ∨ 4 * x - 3 * y + 4 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_lines_through_point_l1948_194889


namespace NUMINAMATH_GPT_milo_dozen_eggs_l1948_194802

theorem milo_dozen_eggs (total_weight_pounds egg_weight_pounds dozen : ℕ) (h1 : total_weight_pounds = 6)
  (h2 : egg_weight_pounds = 1 / 16) (h3 : dozen = 12) :
  total_weight_pounds / egg_weight_pounds / dozen = 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_milo_dozen_eggs_l1948_194802


namespace NUMINAMATH_GPT_reduced_price_l1948_194873

variable (P : ℝ)  -- the original price per kg
variable (reduction_factor : ℝ := 0.5)  -- 50% reduction
variable (extra_kgs : ℝ := 5)  -- 5 kgs more
variable (total_cost : ℝ := 800)  -- Rs. 800

theorem reduced_price :
  total_cost / (P * (1 - reduction_factor)) = total_cost / P + extra_kgs → 
  P / 2 = 80 :=
by
  sorry

end NUMINAMATH_GPT_reduced_price_l1948_194873


namespace NUMINAMATH_GPT_ratio_yuan_david_l1948_194809

-- Definitions
def yuan_age (david_age : ℕ) : ℕ := david_age + 7
def ratio (a b : ℕ) : ℚ := a / b

-- Conditions
variable (david_age : ℕ) (h_david : david_age = 7)

-- Proof Statement
theorem ratio_yuan_david : ratio (yuan_age david_age) david_age = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_yuan_david_l1948_194809


namespace NUMINAMATH_GPT_last_bead_is_black_l1948_194812

-- Definition of the repeating pattern
def pattern := [1, 2, 3, 1, 2]  -- 1: black, 2: white, 3: gray (one full cycle)

-- Given constants
def total_beads : Nat := 91
def pattern_length : Nat := List.length pattern  -- This should be 9

-- Proof statement: The last bead is black
theorem last_bead_is_black : pattern[(total_beads % pattern_length) - 1] = 1 :=
by
  -- The following steps would be the proof which is not required
  sorry

end NUMINAMATH_GPT_last_bead_is_black_l1948_194812


namespace NUMINAMATH_GPT_leaves_falling_every_day_l1948_194830

-- Definitions of the conditions
def roof_capacity := 500 -- in pounds
def leaves_per_pound := 1000 -- number of leaves per pound
def collapse_time := 5000 -- in days

-- Function to calculate the number of leaves falling each day
def leaves_per_day (roof_capacity : Nat) (leaves_per_pound : Nat) (collapse_time : Nat) : Nat :=
  (roof_capacity * leaves_per_pound) / collapse_time

-- Theorem stating the expected result
theorem leaves_falling_every_day :
  leaves_per_day roof_capacity leaves_per_pound collapse_time = 100 :=
by
  sorry

end NUMINAMATH_GPT_leaves_falling_every_day_l1948_194830


namespace NUMINAMATH_GPT_largest_of_consecutive_odds_l1948_194846

-- Defining the six consecutive odd numbers
def consecutive_odd_numbers (a b c d e f : ℕ) : Prop :=
  (a = b + 2) ∧ (b = c + 2) ∧ (c = d + 2) ∧ (d = e + 2) ∧ (e = f + 2)

-- Defining the product condition
def product_of_odds (a b c d e f : ℕ) : Prop :=
  a * b * c * d * e * f = 135135

-- Defining the odd numbers greater than zero
def positive_odds (a b c d e f : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1)

-- Theorem
theorem largest_of_consecutive_odds (a b c d e f : ℕ) 
  (h1 : consecutive_odd_numbers a b c d e f)
  (h2 : product_of_odds a b c d e f)
  (h3 : positive_odds a b c d e f) : 
  a = 13 :=
sorry

end NUMINAMATH_GPT_largest_of_consecutive_odds_l1948_194846


namespace NUMINAMATH_GPT_min_omega_symmetry_l1948_194896

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem min_omega_symmetry :
  ∃ (omega : ℝ), omega > 0 ∧ 
  (∀ x : ℝ, Real.cos (omega * (x - π / 12)) = Real.cos (omega * (2 * (π / 4) - x) - omega * π / 12) ) ∧ 
  (∀ ω_, ω_ > 0 → 
  (∀ x : ℝ, Real.cos (ω_ * (x - π / 12)) = Real.cos (ω_ * (2 * (π / 4) - x) - ω_ * π / 12) → 
  omega ≤ ω_)) ∧ omega = 6 :=
sorry

end NUMINAMATH_GPT_min_omega_symmetry_l1948_194896


namespace NUMINAMATH_GPT_infinite_a_exists_l1948_194877

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ+), ∃ (m : ℕ+), n^6 + 3 * (a : ℕ) = m^3 :=
  sorry

end NUMINAMATH_GPT_infinite_a_exists_l1948_194877


namespace NUMINAMATH_GPT_Robie_boxes_with_him_l1948_194842

-- Definition of the given conditions
def total_cards : Nat := 75
def cards_per_box : Nat := 10
def cards_not_placed : Nat := 5
def boxes_given_away : Nat := 2

-- Definition of the proof that Robie has 5 boxes with him
theorem Robie_boxes_with_him : ((total_cards - cards_not_placed) / cards_per_box) - boxes_given_away = 5 := by
  sorry

end NUMINAMATH_GPT_Robie_boxes_with_him_l1948_194842


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1948_194828

theorem ratio_of_larger_to_smaller (S L k : ℕ) 
  (hS : S = 32)
  (h_sum : S + L = 96)
  (h_multiple : L = k * S) : L / S = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l1948_194828


namespace NUMINAMATH_GPT_tan_theta_expr_l1948_194800

variables {θ x : ℝ}

-- Let θ be an acute angle and let sin(θ/2) = sqrt((x - 2) / (3x)).
theorem tan_theta_expr (h₀ : 0 < θ) (h₁ : θ < (Real.pi / 2)) (h₂ : Real.sin (θ / 2) = Real.sqrt ((x - 2) / (3 * x))) :
  Real.tan θ = (3 * Real.sqrt (7 * x^2 - 8 * x - 16)) / (x + 4) :=
sorry

end NUMINAMATH_GPT_tan_theta_expr_l1948_194800


namespace NUMINAMATH_GPT_ratio_of_cream_max_to_maxine_l1948_194874

def ounces_of_cream_in_max (coffee_sipped : ℕ) (cream_added: ℕ) : ℕ := cream_added

def ounces_of_remaining_cream_in_maxine (initial_coffee : ℚ) (cream_added: ℚ) (sipped : ℚ) : ℚ :=
  let total_mixture := initial_coffee + cream_added
  let remaining_mixture := total_mixture - sipped
  (initial_coffee / total_mixture) * cream_added

theorem ratio_of_cream_max_to_maxine :
  let max_cream := ounces_of_cream_in_max 4 3
  let maxine_cream := ounces_of_remaining_cream_in_maxine 16 3 5
  (max_cream : ℚ) / maxine_cream = 19 / 14 := by 
  sorry

end NUMINAMATH_GPT_ratio_of_cream_max_to_maxine_l1948_194874


namespace NUMINAMATH_GPT_max_area_of_inscribed_equilateral_triangle_l1948_194895

noncomputable def maxInscribedEquilateralTriangleArea : ℝ :=
  let length : ℝ := 12
  let width : ℝ := 15
  let max_area := 369 * Real.sqrt 3 - 540
  max_area

theorem max_area_of_inscribed_equilateral_triangle :
  maxInscribedEquilateralTriangleArea = 369 * Real.sqrt 3 - 540 := 
by
  sorry

end NUMINAMATH_GPT_max_area_of_inscribed_equilateral_triangle_l1948_194895


namespace NUMINAMATH_GPT_computer_additions_per_hour_l1948_194845

theorem computer_additions_per_hour : 
  ∀ (initial_rate : ℕ) (increase_rate: ℚ) (intervals_per_hour : ℕ),
  initial_rate = 12000 → 
  increase_rate = 0.05 → 
  intervals_per_hour = 4 → 
  (12000 * 900) + (12000 * 1.05 * 900) + (12000 * 1.05^2 * 900) + (12000 * 1.05^3 * 900) = 46549350 := 
by
  intros initial_rate increase_rate intervals_per_hour h1 h2 h3
  have h4 : initial_rate = 12000 := h1
  have h5 : increase_rate = 0.05 := h2
  have h6 : intervals_per_hour = 4 := h3
  sorry

end NUMINAMATH_GPT_computer_additions_per_hour_l1948_194845


namespace NUMINAMATH_GPT_fraction_sum_squares_eq_sixteen_l1948_194841

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end NUMINAMATH_GPT_fraction_sum_squares_eq_sixteen_l1948_194841


namespace NUMINAMATH_GPT_fraction_identity_l1948_194866

theorem fraction_identity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 = b^2 + b * c) (h2 : b^2 = c^2 + a * c) : 
  (1 / c) = (1 / a) + (1 / b) :=
by 
  sorry

end NUMINAMATH_GPT_fraction_identity_l1948_194866


namespace NUMINAMATH_GPT_equation_solution_1_equation_solution_2_equation_solution_3_l1948_194865

def system_of_equations (x y : ℝ) : Prop :=
  (x * (x^2 - 3 * y^2) = 16) ∧ (y * (3 * x^2 - y^2) = 88)

theorem equation_solution_1 :
  system_of_equations 4 2 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_2 :
  system_of_equations (-3.7) 2.5 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_3 :
  system_of_equations (-0.3) (-4.5) :=
by
  -- The proof is skipped.
  sorry

end NUMINAMATH_GPT_equation_solution_1_equation_solution_2_equation_solution_3_l1948_194865


namespace NUMINAMATH_GPT_find_positive_x_l1948_194806

theorem find_positive_x :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4))
  ∧ x = 21 + Real.sqrt 449 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_x_l1948_194806


namespace NUMINAMATH_GPT_ratio_mom_pays_to_total_cost_l1948_194887

-- Definitions based on the conditions from the problem
def num_shirts := 4
def num_pants := 2
def num_jackets := 2
def cost_per_shirt := 8
def cost_per_pant := 18
def cost_per_jacket := 60
def amount_carrie_pays := 94

-- Calculate total costs based on given definitions
def cost_shirts := num_shirts * cost_per_shirt
def cost_pants := num_pants * cost_per_pant
def cost_jackets := num_jackets * cost_per_jacket
def total_cost := cost_shirts + cost_pants + cost_jackets

-- Amount Carrie's mom pays
def amount_mom_pays := total_cost - amount_carrie_pays

-- The proving statement
theorem ratio_mom_pays_to_total_cost : (amount_mom_pays : ℝ) / (total_cost : ℝ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_mom_pays_to_total_cost_l1948_194887


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1948_194813

theorem arithmetic_sequence_sum (c d : ℤ) (h₁ : d = 10 - 3)
    (h₂ : c = 17 + (10 - 3)) (h₃ : d = 24 + (10 - 3)) :
    c + d = 55 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1948_194813


namespace NUMINAMATH_GPT_find_angle_A_find_bc_range_l1948_194888

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (c * (a * Real.cos B - (1/2) * b) = a^2 - b^2) ∧ (A = Real.arccos (1/2))

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) :
  A = Real.pi / 3 := 
sorry

theorem find_bc_range (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) (ha : a = Real.sqrt 3) :
  b + c ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3) := 
sorry

end NUMINAMATH_GPT_find_angle_A_find_bc_range_l1948_194888


namespace NUMINAMATH_GPT_ratio_of_40_to_8_l1948_194827

theorem ratio_of_40_to_8 : 40 / 8 = 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_40_to_8_l1948_194827


namespace NUMINAMATH_GPT_closest_ratio_adults_children_l1948_194818

theorem closest_ratio_adults_children (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 2) (h3 : c ≥ 2) : 
  (a : ℚ) / (c : ℚ) = 1 :=
  sorry

end NUMINAMATH_GPT_closest_ratio_adults_children_l1948_194818


namespace NUMINAMATH_GPT_min_value_of_x_sq_plus_6x_l1948_194836

theorem min_value_of_x_sq_plus_6x : ∃ x : ℝ, ∀ y : ℝ, y^2 + 6*y ≥ -9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x_sq_plus_6x_l1948_194836


namespace NUMINAMATH_GPT_cone_angle_l1948_194834

theorem cone_angle (r l : ℝ) (α : ℝ)
  (h1 : 2 * Real.pi * r = Real.pi * l) 
  (h2 : Real.cos α = r / l) : α = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_angle_l1948_194834


namespace NUMINAMATH_GPT_intersecting_lines_l1948_194853

theorem intersecting_lines (m n : ℝ) : 
  (∀ x y : ℝ, y = x / 2 + n → y = mx - 1 → (x = 1 ∧ y = -2)) → 
  m = -1 ∧ n = -5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1948_194853


namespace NUMINAMATH_GPT_price_of_cheaper_feed_l1948_194832

theorem price_of_cheaper_feed 
  (W_total : ℝ) (P_total : ℝ) (E : ℝ) (W_C : ℝ) 
  (H1 : W_total = 27) 
  (H2 : P_total = 0.26)
  (H3 : E = 0.36)
  (H4 : W_C = 14.2105263158) 
  : (W_total * P_total = W_C * C + (W_total - W_C) * E) → 
    (C = 0.17) :=
by {
  sorry
}

end NUMINAMATH_GPT_price_of_cheaper_feed_l1948_194832


namespace NUMINAMATH_GPT_ajax_weight_after_exercise_l1948_194886

theorem ajax_weight_after_exercise :
  ∀ (initial_weight_kg : ℕ) (conversion_rate : ℝ) (daily_exercise_hours : ℕ) (exercise_loss_rate : ℝ) (days_in_week : ℕ) (weeks : ℕ),
    initial_weight_kg = 80 →
    conversion_rate = 2.2 →
    daily_exercise_hours = 2 →
    exercise_loss_rate = 1.5 →
    days_in_week = 7 →
    weeks = 2 →
    initial_weight_kg * conversion_rate - daily_exercise_hours * exercise_loss_rate * (days_in_week * weeks) = 134 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ajax_weight_after_exercise_l1948_194886


namespace NUMINAMATH_GPT_sin_alpha_minus_beta_l1948_194816

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 12 / 13) 
  (h2 : Real.cos β = 4 / 5)
  (hα : π / 2 ≤ α ∧ α ≤ π)
  (hβ : -π / 2 ≤ β ∧ β ≤ 0) :
  Real.sin (α - β) = 33 / 65 := 
sorry

end NUMINAMATH_GPT_sin_alpha_minus_beta_l1948_194816


namespace NUMINAMATH_GPT_lucas_change_l1948_194839

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end NUMINAMATH_GPT_lucas_change_l1948_194839


namespace NUMINAMATH_GPT_perimeter_after_adding_tiles_l1948_194805

theorem perimeter_after_adding_tiles (init_perimeter new_tiles : ℕ) (cond1 : init_perimeter = 14) (cond2 : new_tiles = 2) :
  ∃ new_perimeter : ℕ, new_perimeter = 18 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_after_adding_tiles_l1948_194805


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1948_194867

theorem minimum_value_of_expression 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : 3 * a + 4 * b + 2 * c = 3) : 
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) = 1.5 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1948_194867


namespace NUMINAMATH_GPT_gcd_98_63_l1948_194826

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by
  sorry

end NUMINAMATH_GPT_gcd_98_63_l1948_194826


namespace NUMINAMATH_GPT_polar_equation_l1948_194881

theorem polar_equation (y ρ θ : ℝ) (x : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) 
  (h3 : y^2 = 12 * x) : 
  ρ * (Real.sin θ)^2 = 12 * Real.cos θ := 
by
  sorry

end NUMINAMATH_GPT_polar_equation_l1948_194881


namespace NUMINAMATH_GPT_final_output_value_of_m_l1948_194822

variables (a b m : ℕ)

theorem final_output_value_of_m (h₁ : a = 2) (h₂ : b = 3) (program_logic : (a > b → m = a) ∧ (a ≤ b → m = b)) :
  m = 3 :=
by
  have h₃ : a ≤ b := by
    rw [h₁, h₂]
    exact le_of_lt (by norm_num)
  exact (program_logic.right h₃).trans h₂

end NUMINAMATH_GPT_final_output_value_of_m_l1948_194822


namespace NUMINAMATH_GPT_pairs_m_n_l1948_194833

theorem pairs_m_n (m n : ℤ) : n ^ 2 - 3 * m * n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) :=
by sorry

end NUMINAMATH_GPT_pairs_m_n_l1948_194833


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1948_194857

theorem geometric_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n+1) = a n * q) → -- geometric sequence condition
  a 2 = 6 → -- first condition
  6 * a 1 + a 3 = 30 → -- second condition
  (∀ n, S_n n = (if q = 2 then 3*(2^n - 1) else if q = 3 then 3^n - 1 else 0)) :=
by intros
   sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1948_194857


namespace NUMINAMATH_GPT_product_of_first_two_terms_l1948_194860

-- Given parameters
variables (a d : ℤ) -- a is the first term, d is the common difference

-- Conditions
def fifth_term_condition (a d : ℤ) : Prop := a + 4 * d = 11
def common_difference_condition (d : ℤ) : Prop := d = 1

-- Main statement to prove
theorem product_of_first_two_terms (a d : ℤ) (h1 : fifth_term_condition a d) (h2 : common_difference_condition d) :
  a * (a + d) = 56 :=
by
  sorry

end NUMINAMATH_GPT_product_of_first_two_terms_l1948_194860


namespace NUMINAMATH_GPT_tan_add_pi_over_3_l1948_194893

variable (y : ℝ)

theorem tan_add_pi_over_3 (h : Real.tan y = 3) : 
  Real.tan (y + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := 
by
  sorry

end NUMINAMATH_GPT_tan_add_pi_over_3_l1948_194893


namespace NUMINAMATH_GPT_job_completion_time_l1948_194831

theorem job_completion_time
  (A C : ℝ)
  (A_rate : A = 1 / 6)
  (C_rate : C = 1 / 12)
  (B_share : 390 / 1170 = 1 / 3) :
  ∃ B : ℝ, B = 1 / 8 ∧ (B * 8 = 1) :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_job_completion_time_l1948_194831


namespace NUMINAMATH_GPT_student_made_mistake_l1948_194863

theorem student_made_mistake (AB CD MLNKT : ℕ) (h1 : 10 ≤ AB ∧ AB ≤ 99) (h2 : 10 ≤ CD ∧ CD ≤ 99) (h3 : 10000 ≤ MLNKT ∧ MLNKT < 100000) : AB * CD ≠ MLNKT :=
by {
  sorry
}

end NUMINAMATH_GPT_student_made_mistake_l1948_194863


namespace NUMINAMATH_GPT_compare_f_ln_l1948_194849

variable {f : ℝ → ℝ}

theorem compare_f_ln (h : ∀ x : ℝ, deriv f x > f x) : 3 * f (Real.log 2) < 2 * f (Real.log 3) :=
by
  sorry

end NUMINAMATH_GPT_compare_f_ln_l1948_194849


namespace NUMINAMATH_GPT_num_integer_ks_l1948_194837

theorem num_integer_ks (k : Int) :
  (∃ a b c d : Int, (2*x + a) * (x + b) = 2*x^2 - k*x + 6 ∨
                   (2*x + c) * (x + d) = 2*x^2 - k*x + 6) →
  ∃ ks : Finset Int, ks.card = 6 ∧ k ∈ ks :=
sorry

end NUMINAMATH_GPT_num_integer_ks_l1948_194837


namespace NUMINAMATH_GPT_A_time_to_complete_work_l1948_194864

-- Definitions of work rates for A, B, and C.
variables (A_work B_work C_work : ℚ)

-- Conditions
axiom cond1 : A_work = 3 * B_work
axiom cond2 : B_work = 2 * C_work
axiom cond3 : A_work + B_work + C_work = 1 / 15

-- Proof statement: The time taken by A alone to do the work is 22.5 days.
theorem A_time_to_complete_work : 1 / A_work = 22.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_A_time_to_complete_work_l1948_194864


namespace NUMINAMATH_GPT_temperature_difference_l1948_194819

-- Define variables for the highest and lowest temperatures.
def highest_temp : ℤ := 18
def lowest_temp : ℤ := -2

-- Define the statement for the maximum temperature difference.
theorem temperature_difference : 
  highest_temp - lowest_temp = 20 := 
by 
  sorry

end NUMINAMATH_GPT_temperature_difference_l1948_194819


namespace NUMINAMATH_GPT_range_of_a_fall_within_D_l1948_194862

-- Define the conditions
variable (a : ℝ) (c : ℝ)
axiom A_through : c = 9
axiom D_through : a < 0 ∧ (6, 7) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove the range of a given the conditions
theorem range_of_a : -1/4 < a ∧ a < -1/18 := sorry

-- Define the additional condition for point P
axiom P_through : (2, 8.1) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove that the object can fall within interval D when passing through point P
theorem fall_within_D : a = -9/40 ∧ -1/4 < a ∧ a < -1/18 := sorry

end NUMINAMATH_GPT_range_of_a_fall_within_D_l1948_194862


namespace NUMINAMATH_GPT_packages_per_box_l1948_194848

theorem packages_per_box (P : ℕ) 
  (h1 : 100 * 25 = 2500) 
  (h2 : 2 * P * 250 = 2500) : 
  P = 5 := 
sorry

end NUMINAMATH_GPT_packages_per_box_l1948_194848


namespace NUMINAMATH_GPT_value_of_expression_l1948_194817

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1948_194817


namespace NUMINAMATH_GPT_curve_C2_eq_l1948_194871

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end NUMINAMATH_GPT_curve_C2_eq_l1948_194871


namespace NUMINAMATH_GPT_line_equation_passes_through_l1948_194838

theorem line_equation_passes_through (a b : ℝ) (x y : ℝ) 
  (h_intercept : b = a + 1)
  (h_point : (6 * b) + (-2 * a) = a * b) :
  (x + 2 * y - 2 = 0 ∨ 2 * x + 3 * y - 6 = 0) := 
sorry

end NUMINAMATH_GPT_line_equation_passes_through_l1948_194838


namespace NUMINAMATH_GPT_largest_number_l1948_194897

-- Definitions based on the conditions
def numA := 0.893
def numB := 0.8929
def numC := 0.8931
def numD := 0.839
def numE := 0.8391

-- The statement to be proved 
theorem largest_number : numB = max numA (max numB (max numC (max numD numE))) := by
  sorry

end NUMINAMATH_GPT_largest_number_l1948_194897


namespace NUMINAMATH_GPT_number_of_adults_l1948_194801

theorem number_of_adults (total_bill : ℕ) (cost_per_meal : ℕ) (num_children : ℕ) (total_cost_children : ℕ) 
  (remaining_cost_for_adults : ℕ) (num_adults : ℕ) 
  (H1 : total_bill = 56)
  (H2 : cost_per_meal = 8)
  (H3 : num_children = 5)
  (H4 : total_cost_children = num_children * cost_per_meal)
  (H5 : remaining_cost_for_adults = total_bill - total_cost_children)
  (H6 : num_adults = remaining_cost_for_adults / cost_per_meal) :
  num_adults = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_adults_l1948_194801


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1948_194891

theorem point_in_second_quadrant {x y : ℝ} (hx : x < 0) (hy : y > 0) : 
  ∃ q, q = 2 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1948_194891


namespace NUMINAMATH_GPT_minimum_value_of_f_l1948_194850

noncomputable def f (x y z : ℝ) := (x^2) / (1 + x) + (y^2) / (1 + y) + (z^2) / (1 + z)

theorem minimum_value_of_f (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
  (h7 : b * z + c * y = a) (h8 : a * z + c * x = b) (h9 : a * y + b * x = c) : 
  f x y z ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_f_l1948_194850


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1948_194843

noncomputable def point_on_hyperbola (x y a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focal_length (a b c : ℝ) : Prop :=
  2 * c = 4

noncomputable def eccentricity (e c a : ℝ) : Prop :=
  e = c / a

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_point_on_hyperbola : point_on_hyperbola 2 3 a b h_pos_a h_pos_b)
  (h_focal_length : focal_length a b c)
  : eccentricity e c a :=
sorry -- proof omitted

end NUMINAMATH_GPT_hyperbola_eccentricity_l1948_194843


namespace NUMINAMATH_GPT_extra_bananas_each_child_l1948_194808

theorem extra_bananas_each_child (total_children absent_children planned_bananas_per_child : ℕ) 
    (h1 : total_children = 660) (h2 : absent_children = 330) (h3 : planned_bananas_per_child = 2) : (1320 / (total_children - absent_children)) - planned_bananas_per_child = 2 := by
  sorry

end NUMINAMATH_GPT_extra_bananas_each_child_l1948_194808


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_value_l1948_194882

open Nat

theorem arithmetic_sequence_a2_value (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 + a 3 = 12) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) : 
  a 2 = 5 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_value_l1948_194882


namespace NUMINAMATH_GPT_maximum_value_of_n_with_positive_sequence_l1948_194804

theorem maximum_value_of_n_with_positive_sequence (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, 0 < a n) 
    (h_arithmetic : ∀ n : ℕ, a (n + 1)^2 - a n^2 = 1) : ∃ n : ℕ, n = 24 ∧ a n < 5 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_n_with_positive_sequence_l1948_194804


namespace NUMINAMATH_GPT_greyhound_catches_hare_l1948_194823

theorem greyhound_catches_hare {a b : ℝ} (h_speed : b < a) : ∃ t : ℝ, ∀ s : ℝ, ∃ n : ℕ, (n * t * (a - b)) > s + t * (a + b) :=
by
  sorry

end NUMINAMATH_GPT_greyhound_catches_hare_l1948_194823


namespace NUMINAMATH_GPT_no_preimage_implies_p_gt_1_l1948_194820

   noncomputable def f (x : ℝ) : ℝ :=
     -x^2 + 2 * x

   theorem no_preimage_implies_p_gt_1 (p : ℝ) (hp : ∀ x : ℝ, f x ≠ p) : p > 1 :=
   sorry
   
end NUMINAMATH_GPT_no_preimage_implies_p_gt_1_l1948_194820


namespace NUMINAMATH_GPT_polar_coordinates_equivalence_l1948_194858

theorem polar_coordinates_equivalence :
  ∀ (ρ θ1 θ2 : ℝ), θ1 = π / 3 ∧ θ2 = -5 * π / 3 →
  (ρ = 5) → 
  (ρ * Real.cos θ1 = ρ * Real.cos θ2 ∧ ρ * Real.sin θ1 = ρ * Real.sin θ2) :=
by
  sorry

end NUMINAMATH_GPT_polar_coordinates_equivalence_l1948_194858


namespace NUMINAMATH_GPT_probability_of_A_losing_l1948_194870

variable (p_win p_draw p_lose : ℝ)

def probability_of_A_winning := p_win = (1/3)
def probability_of_draw := p_draw = (1/2)
def sum_of_probabilities := p_win + p_draw + p_lose = 1

theorem probability_of_A_losing
  (h1: probability_of_A_winning p_win)
  (h2: probability_of_draw p_draw)
  (h3: sum_of_probabilities p_win p_draw p_lose) :
  p_lose = (1/6) :=
sorry

end NUMINAMATH_GPT_probability_of_A_losing_l1948_194870


namespace NUMINAMATH_GPT_triangle_perimeter_l1948_194810

theorem triangle_perimeter (a : ℕ) (h1 : a < 8) (h2 : a > 4) (h3 : a % 2 = 0) : 2 + 6 + a = 14 :=
  by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1948_194810


namespace NUMINAMATH_GPT_sufficient_remedy_l1948_194856

-- Definitions based on conditions
def aspirin_relieves_headache : Prop := true
def aspirin_relieves_knee_rheumatism : Prop := true
def aspirin_causes_heart_pain : Prop := true
def aspirin_causes_stomach_pain : Prop := true

def homeopathic_relieves_heart_issues : Prop := true
def homeopathic_relieves_stomach_issues : Prop := true
def homeopathic_causes_hip_rheumatism : Prop := true

def antibiotics_cure_migraines : Prop := true
def antibiotics_cure_heart_pain : Prop := true
def antibiotics_cause_stomach_pain : Prop := true
def antibiotics_cause_knee_pain : Prop := true
def antibiotics_cause_itching : Prop := true

def cortisone_relieves_itching : Prop := true
def cortisone_relieves_knee_rheumatism : Prop := true
def cortisone_exacerbates_hip_rheumatism : Prop := true

def warm_compress_relieves_itching : Prop := true
def warm_compress_relieves_stomach_pain : Prop := true

def severe_headache_morning : Prop := true
def impaired_ability_to_think : Prop := severe_headache_morning

-- Statement of the proof problem
theorem sufficient_remedy :
  (aspirin_relieves_headache ∧ antibiotics_cure_heart_pain ∧ warm_compress_relieves_itching ∧ warm_compress_relieves_stomach_pain) →
  (impaired_ability_to_think → true) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_remedy_l1948_194856


namespace NUMINAMATH_GPT_point_C_velocity_l1948_194844

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end NUMINAMATH_GPT_point_C_velocity_l1948_194844


namespace NUMINAMATH_GPT_plant_branches_l1948_194868

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 91) : 1 + x + x^2 = 91 :=
by sorry

end NUMINAMATH_GPT_plant_branches_l1948_194868


namespace NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l1948_194892

variables (x y z : ℤ)

-- Define the conditions
def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

-- State the theorem
theorem least_possible_value_of_z_minus_x (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) 
    (hx_even : even x) (hy_odd : odd y) (hz_odd : odd z) : z - x = 7 :=
sorry

end NUMINAMATH_GPT_least_possible_value_of_z_minus_x_l1948_194892


namespace NUMINAMATH_GPT_best_fit_model_l1948_194825

theorem best_fit_model
  (R2_M1 R2_M2 R2_M3 R2_M4 : ℝ)
  (h1 : R2_M1 = 0.78)
  (h2 : R2_M2 = 0.85)
  (h3 : R2_M3 = 0.61)
  (h4 : R2_M4 = 0.31) :
  ∀ i, (i = 2 ∧ R2_M2 ≥ R2_M1 ∧ R2_M2 ≥ R2_M3 ∧ R2_M2 ≥ R2_M4) := 
sorry

end NUMINAMATH_GPT_best_fit_model_l1948_194825


namespace NUMINAMATH_GPT_find_first_term_geometric_series_l1948_194824

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end NUMINAMATH_GPT_find_first_term_geometric_series_l1948_194824


namespace NUMINAMATH_GPT_neg_p_necessary_not_sufficient_neg_q_l1948_194879

def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0
def not_p (x : ℝ) : Prop := ¬ (p x)
def not_q (x : ℝ) : Prop := ¬ (q x)

theorem neg_p_necessary_not_sufficient_neg_q : ∀ (x : ℝ), (not_q x → not_p x) ∧ ¬ (not_p x → not_q x) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_necessary_not_sufficient_neg_q_l1948_194879


namespace NUMINAMATH_GPT_rational_solutions_quadratic_eq_l1948_194859

theorem rational_solutions_quadratic_eq (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) :=
by sorry

end NUMINAMATH_GPT_rational_solutions_quadratic_eq_l1948_194859


namespace NUMINAMATH_GPT_exponentiation_rule_proof_l1948_194861

-- Definitions based on conditions
def x : ℕ := 3
def a : ℕ := 4
def b : ℕ := 2

-- The rule that relates the exponents
def rule (x a b : ℕ) : ℕ := x^(a * b)

-- Proposition that we need to prove
theorem exponentiation_rule_proof : rule x a b = 6561 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end NUMINAMATH_GPT_exponentiation_rule_proof_l1948_194861


namespace NUMINAMATH_GPT_supercomputer_transformation_stops_l1948_194854

def transformation_rule (n : ℕ) : ℕ :=
  let A : ℕ := n / 100
  let B : ℕ := n % 100
  2 * A + 8 * B

theorem supercomputer_transformation_stops (n : ℕ) :
  let start := (10^900 - 1) / 9 -- 111...111 with 900 ones
  (n = start) → (∀ m, transformation_rule m < 100 → false) :=
by
  sorry

end NUMINAMATH_GPT_supercomputer_transformation_stops_l1948_194854


namespace NUMINAMATH_GPT_bob_calories_consumed_l1948_194835

theorem bob_calories_consumed 
  (total_slices : ℕ)
  (half_slices : ℕ)
  (calories_per_slice : ℕ) 
  (H1 : total_slices = 8) 
  (H2 : half_slices = total_slices / 2) 
  (H3 : calories_per_slice = 300) : 
  half_slices * calories_per_slice = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_bob_calories_consumed_l1948_194835


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1948_194847

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true
axiom f_zero_eq : f 0 = 2
axiom f_derivative_ineq : ∀ x : ℝ, f x + (deriv f x) > 1

theorem solution_set_of_inequality : { x : ℝ | e^x * f x > e^x + 1 } = { x | x > 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1948_194847


namespace NUMINAMATH_GPT_range_of_m_l1948_194821

theorem range_of_m (α β m : ℝ)
  (h1 : 0 < α ∧ α < 1)
  (h2 : 1 < β ∧ β < 2)
  (h3 : ∀ x, x^2 - m * x + 1 = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1948_194821


namespace NUMINAMATH_GPT_divide_plane_into_four_quadrants_l1948_194876

-- Definitions based on conditions
def perpendicular_axes (x y : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, x t = t ∨ x t = 0) ∧ (∀ t : ℝ, y t = t ∨ y t = 0) ∧ ∀ t : ℝ, x t ≠ y t

-- The mathematical proof statement
theorem divide_plane_into_four_quadrants (x y : ℝ → ℝ) (hx : perpendicular_axes x y) :
  ∃ quadrants : ℕ, quadrants = 4 :=
by
  sorry

end NUMINAMATH_GPT_divide_plane_into_four_quadrants_l1948_194876


namespace NUMINAMATH_GPT_common_root_unique_solution_l1948_194890

theorem common_root_unique_solution
  (p : ℝ) (h : ∃ x, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) :
  p = 3 :=
by sorry

end NUMINAMATH_GPT_common_root_unique_solution_l1948_194890


namespace NUMINAMATH_GPT_candy_store_total_sales_l1948_194898

def price_per_pound_fudge : ℝ := 2.50
def pounds_fudge : ℕ := 20
def price_per_truffle : ℝ := 1.50
def dozens_truffles : ℕ := 5
def price_per_pretzel : ℝ := 2.00
def dozens_pretzels : ℕ := 3

theorem candy_store_total_sales :
  price_per_pound_fudge * pounds_fudge +
  price_per_truffle * (dozens_truffles * 12) +
  price_per_pretzel * (dozens_pretzels * 12) = 212.00 := by
  sorry

end NUMINAMATH_GPT_candy_store_total_sales_l1948_194898


namespace NUMINAMATH_GPT_ways_to_go_from_first_to_fifth_l1948_194899

theorem ways_to_go_from_first_to_fifth (floors : ℕ) (staircases_per_floor : ℕ) (total_ways : ℕ) 
    (h1 : floors = 5) (h2 : staircases_per_floor = 2) (h3 : total_ways = 2^4) : total_ways = 16 :=
by
  sorry

end NUMINAMATH_GPT_ways_to_go_from_first_to_fifth_l1948_194899


namespace NUMINAMATH_GPT_complex_inequality_l1948_194880

theorem complex_inequality (m : ℝ) : 
  (m - 3 ≥ 0 ∧ m^2 - 9 = 0) → m = 3 := 
by
  sorry

end NUMINAMATH_GPT_complex_inequality_l1948_194880


namespace NUMINAMATH_GPT_min_value_expression_l1948_194829

theorem min_value_expression (x y k : ℝ) (hk : 1 < k) (hx : k < x) (hy : k < y) : 
  (∀ x y, x > k → y > k → (∃ m, (m ≤ (x^2 / (y - k) + y^2 / (x - k)))) ∧ (m = 8 * k)) := sorry

end NUMINAMATH_GPT_min_value_expression_l1948_194829


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1948_194883

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1948_194883
