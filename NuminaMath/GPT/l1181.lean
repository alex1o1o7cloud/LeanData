import Mathlib

namespace NUMINAMATH_GPT_quadratic_coeff_b_is_4_sqrt_15_l1181_118154

theorem quadratic_coeff_b_is_4_sqrt_15 :
  ∃ m b : ℝ, (x^2 + bx + 72 = (x + m)^2 + 12) → (m = 2 * Real.sqrt 15) → (b = 4 * Real.sqrt 15) ∧ b > 0 :=
by
  -- Note: Proof not included as per the instruction.
  sorry

end NUMINAMATH_GPT_quadratic_coeff_b_is_4_sqrt_15_l1181_118154


namespace NUMINAMATH_GPT_intersection_sets_l1181_118199

noncomputable def set1 (x : ℝ) : Prop := (x - 2) / (x + 1) ≤ 0
noncomputable def set2 (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0

theorem intersection_sets :
  { x : ℝ | set1 x } ∩ { x : ℝ | set2 x } = { x | (-1 : ℝ) < x ∧ x ≤ 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l1181_118199


namespace NUMINAMATH_GPT_fraction_eq_zero_iff_x_eq_6_l1181_118132

theorem fraction_eq_zero_iff_x_eq_6 (x : ℝ) : (x - 6) / (5 * x) = 0 ↔ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_eq_zero_iff_x_eq_6_l1181_118132


namespace NUMINAMATH_GPT_balloons_lost_l1181_118144

-- Definitions corresponding to the conditions
def initial_balloons : ℕ := 7
def current_balloons : ℕ := 4

-- The mathematically equivalent proof problem
theorem balloons_lost : initial_balloons - current_balloons = 3 := by
  -- proof steps would go here, but we use sorry to skip them 
  sorry

end NUMINAMATH_GPT_balloons_lost_l1181_118144


namespace NUMINAMATH_GPT_max_integer_is_110003_l1181_118140

def greatest_integer : Prop :=
  let a := 100004
  let b := 110003
  let c := 102002
  let d := 100301
  let e := 100041
  b > a ∧ b > c ∧ b > d ∧ b > e

theorem max_integer_is_110003 : greatest_integer :=
by
  sorry

end NUMINAMATH_GPT_max_integer_is_110003_l1181_118140


namespace NUMINAMATH_GPT_slopes_angle_l1181_118167

theorem slopes_angle (k_1 k_2 : ℝ) (θ : ℝ) 
  (h1 : 6 * k_1^2 + k_1 - 1 = 0)
  (h2 : 6 * k_2^2 + k_2 - 1 = 0) :
  θ = π / 4 ∨ θ = 3 * π / 4 := 
by sorry

end NUMINAMATH_GPT_slopes_angle_l1181_118167


namespace NUMINAMATH_GPT_mushroom_collectors_l1181_118190

theorem mushroom_collectors :
  ∃ (n m : ℕ), 13 * n - 10 * m = 2 ∧ 9 ≤ n ∧ n ≤ 15 ∧ 11 ≤ m ∧ m ≤ 20 ∧ n = 14 ∧ m = 18 := by sorry

end NUMINAMATH_GPT_mushroom_collectors_l1181_118190


namespace NUMINAMATH_GPT_part1_part2_l1181_118191

-- Part 1: Definition of "consecutive roots quadratic equation"
def consecutive_roots (a b : ℤ) : Prop := a = b + 1 ∨ b = a + 1

-- Statement that for some k and constant term, the roots of the quadratic form consecutive roots
theorem part1 (k : ℤ) : consecutive_roots 7 8 → k = -15 → (∀ x : ℤ, x^2 + k * x + 56 = 0 → x = 7 ∨ x = 8) :=
by
  sorry

-- Part 2: Generalizing to the nth equation
theorem part2 (n : ℕ) : 
  (∀ x : ℤ, x^2 - (2 * n - 1) * x + n * (n - 1) = 0 → x = n ∨ x = n - 1) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1181_118191


namespace NUMINAMATH_GPT_evaluate_expression_l1181_118156

theorem evaluate_expression (x : ℝ) (h : x = Real.sqrt 3) : 
  ( (x^2 - 2*x + 1) / (x^2 - x) / (x - 1) ) = Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1181_118156


namespace NUMINAMATH_GPT_marble_selection_l1181_118181

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def special_marbles : ℕ := 4
def other_marbles : ℕ := total_marbles - special_marbles

-- Define combination function for ease of use in the theorem
def combination (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Statement of the theorem based on the question and the correct answer
theorem marble_selection : combination other_marbles 4 * special_marbles = 1320 := by
  -- Define specific values based on the problem
  have other_marbles_val : other_marbles = 11 := rfl
  have comb_11_4 : combination 11 4 = 330 := by
    rw [combination]
    rfl
  rw [other_marbles_val, comb_11_4]
  norm_num
  sorry

end NUMINAMATH_GPT_marble_selection_l1181_118181


namespace NUMINAMATH_GPT_twelve_months_game_probability_l1181_118164

/-- The card game "Twelve Months" involves turning over cards according to a set of rules.
Given the rules, we are asked to find the probability that all 12 columns of cards can be fully turned over. -/
def twelve_months_probability : ℚ :=
  1 / 12

theorem twelve_months_game_probability :
  twelve_months_probability = 1 / 12 :=
by
  -- The conditions and their representations are predefined.
  sorry

end NUMINAMATH_GPT_twelve_months_game_probability_l1181_118164


namespace NUMINAMATH_GPT_length_inequality_l1181_118152

noncomputable def l_a (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_b (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_c (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def perimeter (A B C : ℝ) : ℝ :=
  A + B + C

theorem length_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  (l_a A B C * l_b A B C * l_c A B C) / (perimeter A B C)^3 ≤ 1 / 64 :=
by
  sorry

end NUMINAMATH_GPT_length_inequality_l1181_118152


namespace NUMINAMATH_GPT_find_radius_l1181_118122

-- Define the given conditions as variables
variables (l A r : ℝ)

-- Conditions from the problem
-- 1. The arc length of the sector is 2 cm
def arc_length_eq : Prop := l = 2

-- 2. The area of the sector is 2 cm²
def area_eq : Prop := A = 2

-- Formula for the area of the sector
def sector_area (l r : ℝ) : ℝ := 0.5 * l * r

-- Define the goal to prove the radius is 2 cm
theorem find_radius (h₁ : arc_length_eq l) (h₂ : area_eq A) : r = 2 :=
by {
  sorry -- proof omitted
}

end NUMINAMATH_GPT_find_radius_l1181_118122


namespace NUMINAMATH_GPT_determine_a_l1181_118183

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

theorem determine_a (a : ℝ) (A_union_B_eq_A : A a ∪ B a = A a) : a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_GPT_determine_a_l1181_118183


namespace NUMINAMATH_GPT_gamma_max_success_ratio_l1181_118130

theorem gamma_max_success_ratio (x y z w : ℕ) (h_yw : y + w = 500)
    (h_gamma_first_day : 0 < x ∧ x < 170 * y / 280)
    (h_gamma_second_day : 0 < z ∧ z < 150 * w / 220)
    (h_less_than_500 : (28 * x + 22 * z) / 17 < 500) :
    (x + z) ≤ 170 := 
sorry

end NUMINAMATH_GPT_gamma_max_success_ratio_l1181_118130


namespace NUMINAMATH_GPT_three_digit_divisible_by_11_l1181_118116

theorem three_digit_divisible_by_11 {x y z : ℕ} 
  (h1 : 0 ≤ x ∧ x < 10) 
  (h2 : 0 ≤ y ∧ y < 10) 
  (h3 : 0 ≤ z ∧ z < 10) 
  (h4 : x + z = y) : 
  (100 * x + 10 * y + z) % 11 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_three_digit_divisible_by_11_l1181_118116


namespace NUMINAMATH_GPT_apples_count_l1181_118145

def total_apples (mike_apples nancy_apples keith_apples : Nat) : Nat :=
  mike_apples + nancy_apples + keith_apples

theorem apples_count :
  total_apples 7 3 6 = 16 :=
by
  rfl

end NUMINAMATH_GPT_apples_count_l1181_118145


namespace NUMINAMATH_GPT_common_root_conds_l1181_118111

theorem common_root_conds (α a b c d : ℝ) (h₁ : a ≠ c)
  (h₂ : α^2 + a * α + b = 0)
  (h₃ : α^2 + c * α + d = 0) :
  α = (d - b) / (a - c) :=
by 
  sorry

end NUMINAMATH_GPT_common_root_conds_l1181_118111


namespace NUMINAMATH_GPT_find_x_l1181_118187

theorem find_x (x : ℝ) (h : 0.25 * x = 0.12 * 1500 - 15) : x = 660 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_x_l1181_118187


namespace NUMINAMATH_GPT_base_number_l1181_118138

theorem base_number (a x : ℕ) (h1 : a ^ x - a ^ (x - 2) = 3 * 2 ^ 11) (h2 : x = 13) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_base_number_l1181_118138


namespace NUMINAMATH_GPT_concert_ticket_cost_l1181_118172

theorem concert_ticket_cost :
  ∀ (x : ℝ), 
    (12 * x - 2 * 0.05 * x = 476) → 
    x = 40 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_concert_ticket_cost_l1181_118172


namespace NUMINAMATH_GPT_relatively_prime_dates_in_september_l1181_118143

-- Define a condition to check if two numbers are relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the number of days in September
def days_in_september := 30

-- Define the month of September as the 9th month
def month_of_september := 9

-- Define the proposition that the number of relatively prime dates in September is 20
theorem relatively_prime_dates_in_september : 
  ∃ count, (count = 20 ∧ ∀ day, day ∈ Finset.range (days_in_september + 1) → relatively_prime month_of_september day → count = 20) := sorry

end NUMINAMATH_GPT_relatively_prime_dates_in_september_l1181_118143


namespace NUMINAMATH_GPT_value_of_n_l1181_118178

def is_3_digit_integer (n : ℕ) : Prop := (100 ≤ n) ∧ (n < 1000)

def not_divisible_by (n k : ℕ) : Prop := ¬ (k ∣ n)

def least_common_multiple (a b c : ℕ) : Prop := Nat.lcm a b = c

theorem value_of_n (d n : ℕ) (h1 : least_common_multiple d n 690) 
  (h2 : not_divisible_by n 3) (h3 : not_divisible_by d 2) (h4 : is_3_digit_integer n) : n = 230 :=
by
  sorry

end NUMINAMATH_GPT_value_of_n_l1181_118178


namespace NUMINAMATH_GPT_sum_of_dice_not_in_set_l1181_118179

theorem sum_of_dice_not_in_set (a b c : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) (h₃ : 1 ≤ c ∧ c ≤ 6) 
  (h₄ : a * b * c = 72) (h₅ : a = 4 ∨ b = 4 ∨ c = 4) :
  a + b + c ≠ 12 ∧ a + b + c ≠ 14 ∧ a + b + c ≠ 15 ∧ a + b + c ≠ 16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_dice_not_in_set_l1181_118179


namespace NUMINAMATH_GPT_largest_of_a_b_c_l1181_118193

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.sin (Real.pi / 8)

theorem largest_of_a_b_c : b = max (max a b) c :=
by
  have ha : a = 1 / 2 := rfl
  have hb : b = Real.log 3 / Real.log 4 := rfl
  have hc : c = Real.sin (Real.pi / 8) := rfl
  sorry

end NUMINAMATH_GPT_largest_of_a_b_c_l1181_118193


namespace NUMINAMATH_GPT_solve_for_q_l1181_118109

theorem solve_for_q (m n q : ℕ) (h1 : 7/8 = m/96) (h2 : 7/8 = (n + m)/112) (h3 : 7/8 = (q - m)/144) :
  q = 210 :=
sorry

end NUMINAMATH_GPT_solve_for_q_l1181_118109


namespace NUMINAMATH_GPT_num_lines_in_grid_l1181_118184

theorem num_lines_in_grid (columns rows : ℕ) (H1 : columns = 4) (H2 : rows = 3) 
    (total_points : ℕ) (H3 : total_points = columns * rows) :
    ∃ lines, lines = 40 :=
by
  sorry

end NUMINAMATH_GPT_num_lines_in_grid_l1181_118184


namespace NUMINAMATH_GPT_total_feathers_needed_l1181_118188

theorem total_feathers_needed
  (animals_first_group : ℕ := 934)
  (feathers_first_group : ℕ := 7)
  (animals_second_group : ℕ := 425)
  (colored_feathers_second_group : ℕ := 7)
  (golden_feathers_second_group : ℕ := 5)
  (animals_third_group : ℕ := 289)
  (colored_feathers_third_group : ℕ := 4)
  (golden_feathers_third_group : ℕ := 10) :
  (animals_first_group * feathers_first_group) +
  (animals_second_group * (colored_feathers_second_group + golden_feathers_second_group)) +
  (animals_third_group * (colored_feathers_third_group + golden_feathers_third_group)) = 15684 := by
  sorry

end NUMINAMATH_GPT_total_feathers_needed_l1181_118188


namespace NUMINAMATH_GPT_find_x_positive_integers_l1181_118106

theorem find_x_positive_integers (a b c x : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c = x * a * b * c) → (x = 1 ∧ a = 1 ∧ b = 2 ∧ c = 3) ∨
  (x = 2 ∧ a = 1 ∧ b = 1 ∧ c = 2) ∨
  (x = 3 ∧ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end NUMINAMATH_GPT_find_x_positive_integers_l1181_118106


namespace NUMINAMATH_GPT_min_baseball_cards_divisible_by_15_l1181_118103

theorem min_baseball_cards_divisible_by_15 :
  ∀ (j m c e t : ℕ),
    j = m →
    m = c - 6 →
    c = 20 →
    e = 2 * (j + m) →
    t = c + m + j + e →
    t ≥ 104 →
    ∃ k : ℕ, t = 15 * k ∧ t = 105 :=
by
  intros j m c e t h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_min_baseball_cards_divisible_by_15_l1181_118103


namespace NUMINAMATH_GPT_circle_area_irrational_of_rational_radius_l1181_118131

theorem circle_area_irrational_of_rational_radius (r : ℚ) : ¬ ∃ A : ℚ, A = π * (r:ℝ) * (r:ℝ) :=
by sorry

end NUMINAMATH_GPT_circle_area_irrational_of_rational_radius_l1181_118131


namespace NUMINAMATH_GPT_eval_expression_l1181_118119

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l1181_118119


namespace NUMINAMATH_GPT_min_employees_needed_l1181_118135

-- Define the conditions
variable (W A : Finset ℕ)
variable (n_W n_A n_WA : ℕ)

-- Assume the given condition values
def sizeW := 95
def sizeA := 80
def sizeWA := 30

-- Define the proof problem
theorem min_employees_needed :
  (sizeW + sizeA - sizeWA) = 145 :=
by sorry

end NUMINAMATH_GPT_min_employees_needed_l1181_118135


namespace NUMINAMATH_GPT_factorize_expression_l1181_118180

variable {a b : ℝ} -- define a and b as real numbers

theorem factorize_expression : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1181_118180


namespace NUMINAMATH_GPT_walking_area_calculation_l1181_118113

noncomputable def walking_area_of_park (park_length park_width fountain_radius : ℝ) : ℝ :=
  let park_area := park_length * park_width
  let fountain_area := Real.pi * fountain_radius^2
  park_area - fountain_area

theorem walking_area_calculation :
  walking_area_of_park 50 30 5 = 1500 - 25 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_walking_area_calculation_l1181_118113


namespace NUMINAMATH_GPT_area_triangle_parabola_l1181_118151

noncomputable def area_of_triangle_ABC (d : ℝ) (x : ℝ) : ℝ :=
  let A := (x, x^2)
  let B := (x + d, (x + d)^2)
  let C := (x + 2 * d, (x + 2 * d)^2)
  1 / 2 * abs (x * ((x + 2 * d)^2 - (x + d)^2) + (x + d) * ((x + 2 * d)^2 - x^2) + (x + 2 * d) * (x^2 - (x + d)^2))

theorem area_triangle_parabola (d : ℝ) (h_d : 0 < d) (x : ℝ) : 
  area_of_triangle_ABC d x = d^2 := sorry

end NUMINAMATH_GPT_area_triangle_parabola_l1181_118151


namespace NUMINAMATH_GPT_parallel_edges_octahedron_l1181_118153

-- Definition of a regular octahedron's properties
structure regular_octahedron : Type :=
  (edges : ℕ) -- Number of edges in the octahedron

-- Constant to represent the regular octahedron with 12 edges.
def octahedron : regular_octahedron := { edges := 12 }

-- Definition to count unique pairs of parallel edges
def count_parallel_edge_pairs (o : regular_octahedron) : ℕ :=
  if o.edges = 12 then 12 else 0

-- Theorem to assert the number of pairs of parallel edges in a regular octahedron is 12
theorem parallel_edges_octahedron : count_parallel_edge_pairs octahedron = 12 :=
by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_parallel_edges_octahedron_l1181_118153


namespace NUMINAMATH_GPT_herder_bulls_l1181_118163

theorem herder_bulls (total_bulls : ℕ) (herder_fraction : ℚ) (claims : total_bulls = 70) (fraction_claim : herder_fraction = (2/3) * (1/3)) : herder_fraction * (total_bulls : ℚ) = 315 :=
by sorry

end NUMINAMATH_GPT_herder_bulls_l1181_118163


namespace NUMINAMATH_GPT_find_extrema_of_f_l1181_118137

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^2 + 5) / (x^2 + 1)^2

theorem find_extrema_of_f :
  (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) ∧ (∀ x : ℝ, f x ≥ 0.95) ∧ (∃ x : ℝ, f x = 0.95) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_extrema_of_f_l1181_118137


namespace NUMINAMATH_GPT_greatest_radius_l1181_118141

theorem greatest_radius (A : ℝ) (hA : A < 60 * Real.pi) : ∃ r : ℕ, r = 7 ∧ (r : ℝ) * (r : ℝ) < 60 :=
by
  sorry

end NUMINAMATH_GPT_greatest_radius_l1181_118141


namespace NUMINAMATH_GPT_max_f_of_sin_bounded_l1181_118134

theorem max_f_of_sin_bounded (x : ℝ) : (∀ y, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1) → ∃ m, (∀ z, (1 + 2 * Real.sin z) ≤ m) ∧ (∀ n, (∀ z, (1 + 2 * Real.sin z) ≤ n) → m ≤ n) :=
by
  sorry

end NUMINAMATH_GPT_max_f_of_sin_bounded_l1181_118134


namespace NUMINAMATH_GPT_max_possible_x_l1181_118177

noncomputable section

def tan_deg (x : ℕ) : ℝ := Real.tan (x * Real.pi / 180)

theorem max_possible_x (x y : ℕ) (h₁ : tan_deg x - tan_deg y = 1 + tan_deg x * tan_deg y)
  (h₂ : tan_deg x * tan_deg y = 1) (h₃ : x = 98721) : x = 98721 := sorry

end NUMINAMATH_GPT_max_possible_x_l1181_118177


namespace NUMINAMATH_GPT_sum_base8_l1181_118176

theorem sum_base8 (a b c : ℕ) (h₁ : a = 7*8^2 + 7*8 + 7)
                           (h₂ : b = 7*8 + 7)
                           (h₃ : c = 7) :
  a + b + c = 1*8^3 + 1*8^2 + 0*8 + 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_base8_l1181_118176


namespace NUMINAMATH_GPT_find_value_l1181_118168

variable {x y : ℝ}

theorem find_value (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y + x * y = 0) : y / x + x / y = -2 := 
sorry

end NUMINAMATH_GPT_find_value_l1181_118168


namespace NUMINAMATH_GPT_sector_angle_l1181_118118

theorem sector_angle (l S : ℝ) (r α : ℝ) 
  (h_arc_length : l = 6)
  (h_area : S = 6)
  (h_area_formula : S = 1/2 * l * r)
  (h_arc_formula : l = r * α) : 
  α = 3 :=
by
  sorry

end NUMINAMATH_GPT_sector_angle_l1181_118118


namespace NUMINAMATH_GPT_convert_base_5_to_decimal_l1181_118105

-- Define the base-5 number 44 and its decimal equivalent
def base_5_number : ℕ := 4 * 5^1 + 4 * 5^0

-- Prove that the base-5 number 44 equals 24 in decimal
theorem convert_base_5_to_decimal : base_5_number = 24 := by
  sorry

end NUMINAMATH_GPT_convert_base_5_to_decimal_l1181_118105


namespace NUMINAMATH_GPT_bisection_method_root_interval_l1181_118192

noncomputable def f (x : ℝ) : ℝ := 2^x + 3 * x - 7

theorem bisection_method_root_interval :
  f 1 < 0 → f 2 > 0 → f 3 > 0 → ∃ (c : ℝ), 1 < c ∧ c < 2 ∧ f c = 0 :=
by
  sorry

end NUMINAMATH_GPT_bisection_method_root_interval_l1181_118192


namespace NUMINAMATH_GPT_roots_quadratic_l1181_118195

theorem roots_quadratic (a b c d : ℝ) :
  (a + b = 3 * c / 2 ∧ a * b = 4 * d ∧ c + d = 3 * a / 2 ∧ c * d = 4 * b)
  ↔ ( (a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
      (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
      (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4) ) :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_l1181_118195


namespace NUMINAMATH_GPT_min_value_of_expression_l1181_118174

theorem min_value_of_expression
  (a b : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hlines : (∀ x y : ℝ, x + (a-4) * y + 1 = 0) ∧ (∀ x y : ℝ, 2 * b * x + y - 2 = 0) ∧ (∀ x y : ℝ, (x + (a-4) * y + 1 = 0) ∧ (2 * b * x + y - 2 = 0) → -1 * 1 / (a-4) * -2 * b = 1)) :
  ∃ (min_val : ℝ), min_val = (9/5) ∧ min_val = (a + 2)/(a + 1) + 1/(2 * b) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1181_118174


namespace NUMINAMATH_GPT_solution_correct_l1181_118136

def mascot_options := ["A Xiang", "A He", "A Ru", "A Yi", "Le Yangyang"]

def volunteer_options := ["A", "B", "C", "D", "E"]

noncomputable def count_valid_assignments (mascots : List String) (volunteers : List String) : Nat :=
  let all_assignments := mascots.permutations
  let valid_assignments := all_assignments.filter (λ p =>
    (p.get! 0 = "A Xiang" ∨ p.get! 1 = "A Xiang") ∧ p.get! 2 ≠ "Le Yangyang")
  valid_assignments.length

theorem solution_correct :
  count_valid_assignments mascot_options volunteer_options = 36 :=
by
  sorry

end NUMINAMATH_GPT_solution_correct_l1181_118136


namespace NUMINAMATH_GPT_inscribed_circle_radius_isosceles_triangle_l1181_118133

noncomputable def isosceles_triangle_base : ℝ := 30 -- base AC
noncomputable def isosceles_triangle_equal_side : ℝ := 39 -- equal sides AB and BC

theorem inscribed_circle_radius_isosceles_triangle :
  ∀ (AC AB BC: ℝ), 
  AC = isosceles_triangle_base → 
  AB = isosceles_triangle_equal_side →
  BC = isosceles_triangle_equal_side →
  ∃ r : ℝ, r = 10 := 
by
  intros AC AB BC hAC hAB hBC
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_isosceles_triangle_l1181_118133


namespace NUMINAMATH_GPT_Teena_speed_is_55_l1181_118121

def Teena_speed (Roe_speed T : ℝ) (initial_gap final_gap time : ℝ) : Prop :=
  Roe_speed * time + initial_gap + final_gap = T * time

theorem Teena_speed_is_55 :
  Teena_speed 40 55 7.5 15 1.5 :=
by 
  sorry

end NUMINAMATH_GPT_Teena_speed_is_55_l1181_118121


namespace NUMINAMATH_GPT_unique_c1_c2_exists_l1181_118117

theorem unique_c1_c2_exists (a_0 a_1 x_1 x_2 : ℝ) (h_distinct : x_1 ≠ x_2) : 
  ∃! (c_1 c_2 : ℝ), ∀ n : ℕ, a_n = c_1 * x_1^n + c_2 * x_2^n :=
sorry

end NUMINAMATH_GPT_unique_c1_c2_exists_l1181_118117


namespace NUMINAMATH_GPT_distance_city_A_B_l1181_118185

theorem distance_city_A_B (D : ℝ) : 
  (3 : ℝ) + (2.5 : ℝ) = 5.5 → 
  ∃ T_saved, T_saved = 1 →
  80 = (2 * D) / (5.5 - T_saved) →
  D = 180 :=
by
  intros
  sorry

end NUMINAMATH_GPT_distance_city_A_B_l1181_118185


namespace NUMINAMATH_GPT_sasha_remainder_is_20_l1181_118110

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_sasha_remainder_is_20_l1181_118110


namespace NUMINAMATH_GPT_classroom_gpa_l1181_118170

theorem classroom_gpa (n : ℕ) (h1 : 1 ≤ n) : 
  (1/3 : ℝ) * 30 + (2/3 : ℝ) * 33 = 32 :=
by sorry

end NUMINAMATH_GPT_classroom_gpa_l1181_118170


namespace NUMINAMATH_GPT_least_positive_integer_mod_conditions_l1181_118175

theorem least_positive_integer_mod_conditions :
  ∃ N : ℕ, (N % 4 = 3) ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 11 = 10) ∧ N = 4619 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_mod_conditions_l1181_118175


namespace NUMINAMATH_GPT_age_difference_l1181_118128

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : c = a - 10 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1181_118128


namespace NUMINAMATH_GPT_perpendicular_slope_l1181_118108

theorem perpendicular_slope (k : ℝ) : (∀ x, y = k*x) ∧ (∀ x, y = 2*x + 1) → k = -1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l1181_118108


namespace NUMINAMATH_GPT_sum_geq_4k_l1181_118139

theorem sum_geq_4k (a b k : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_k : k > 1)
  (h_lcm_gcd : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : a + b ≥ 4 * k := 
by 
  sorry

end NUMINAMATH_GPT_sum_geq_4k_l1181_118139


namespace NUMINAMATH_GPT_remainder_sum_of_first_six_primes_div_seventh_prime_l1181_118171

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_of_first_six_primes_div_seventh_prime_l1181_118171


namespace NUMINAMATH_GPT_sum_first_5_terms_arithmetic_l1181_118157

variable {a : ℕ → ℝ} -- Defining a sequence

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_eq_1 : a 2 = 1
axiom a4_eq_5 : a 4 = 5

-- Theorem statement
theorem sum_first_5_terms_arithmetic (h_arith : is_arithmetic_sequence a) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end NUMINAMATH_GPT_sum_first_5_terms_arithmetic_l1181_118157


namespace NUMINAMATH_GPT_distance_between_stripes_l1181_118186

/-- Given a crosswalk parallelogram with curbs 60 feet apart, a base of 20 feet, 
and each stripe of length 50 feet, show that the distance between the stripes is 24 feet. -/
theorem distance_between_stripes (h : Real) (b : Real) (s : Real) : h = 60 ∧ b = 20 ∧ s = 50 → (b * h) / s = 24 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stripes_l1181_118186


namespace NUMINAMATH_GPT_minimum_n_minus_m_l1181_118125

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_n_minus_m_l1181_118125


namespace NUMINAMATH_GPT_find_b_l1181_118173

theorem find_b
  (a b c d : ℝ)
  (h₁ : -a + b - c + d = 0)
  (h₂ : a + b + c + d = 0)
  (h₃ : d = 2) :
  b = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_b_l1181_118173


namespace NUMINAMATH_GPT_det_B_squared_sub_3B_eq_10_l1181_118196

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![2, 3], ![2, 2]]

theorem det_B_squared_sub_3B_eq_10 : 
  Matrix.det (B * B - 3 • B) = 10 := by
  sorry

end NUMINAMATH_GPT_det_B_squared_sub_3B_eq_10_l1181_118196


namespace NUMINAMATH_GPT_carl_typing_speed_l1181_118129

theorem carl_typing_speed (words_per_day: ℕ) (minutes_per_day: ℕ) (total_words: ℕ) (days: ℕ) : 
  words_per_day = total_words / days ∧ 
  minutes_per_day = 4 * 60 ∧ 
  (words_per_day / minutes_per_day) = 50 :=
by 
  sorry

end NUMINAMATH_GPT_carl_typing_speed_l1181_118129


namespace NUMINAMATH_GPT_RandomEvent_Proof_l1181_118162

-- Define the events and conditions
def EventA : Prop := ∀ (θ₁ θ₂ θ₃ : ℝ), θ₁ + θ₂ + θ₃ = 360 → False
def EventB : Prop := ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 6) → n < 7
def EventC : Prop := ∃ (factors : ℕ → ℝ), (∃ (uncertainty : ℕ → ℝ), True)
def EventD : Prop := ∀ (balls : ℕ), (balls = 0 ∨ balls ≠ 0) → False

-- The theorem represents the proof problem
theorem RandomEvent_Proof : EventC :=
by
  sorry

end NUMINAMATH_GPT_RandomEvent_Proof_l1181_118162


namespace NUMINAMATH_GPT_measure_of_B_l1181_118150

theorem measure_of_B (a b : ℝ) (A B : ℝ) (angleA_nonneg : 0 < A ∧ A < 180) (angleB_nonneg : 0 < B ∧ B < 180)
    (a_eq : a = 1) (b_eq : b = Real.sqrt 3) (A_eq : A = 30) :
    B = 60 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_B_l1181_118150


namespace NUMINAMATH_GPT_find_c_l1181_118166

-- Define the necessary conditions for the circle equation and the radius
variable (c : ℝ)

-- The given conditions
def circle_eq := ∀ (x y : ℝ), x^2 + 8*x + y^2 - 6*y + c = 0
def radius_five := (∀ (h k r : ℝ), r = 5 → ∃ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2)

theorem find_c (h k r : ℝ) (r_eq : r = 5) : c = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_c_l1181_118166


namespace NUMINAMATH_GPT_slices_per_person_is_correct_l1181_118101

-- Conditions
def slices_per_tomato : Nat := 8
def total_tomatoes : Nat := 20
def people_for_meal : Nat := 8

-- Calculate number of slices for a single person
def slices_needed_for_single_person (slices_per_tomato : Nat) (total_tomatoes : Nat) (people_for_meal : Nat) : Nat :=
  (slices_per_tomato * total_tomatoes) / people_for_meal

-- The statement to be proved
theorem slices_per_person_is_correct : slices_needed_for_single_person slices_per_tomato total_tomatoes people_for_meal = 20 :=
by
  sorry

end NUMINAMATH_GPT_slices_per_person_is_correct_l1181_118101


namespace NUMINAMATH_GPT_total_stickers_l1181_118160

def stickers_in_first_box : ℕ := 23
def stickers_in_second_box : ℕ := stickers_in_first_box + 12

theorem total_stickers :
  stickers_in_first_box + stickers_in_second_box = 58 := 
by
  sorry

end NUMINAMATH_GPT_total_stickers_l1181_118160


namespace NUMINAMATH_GPT_b_2056_l1181_118115

noncomputable def b (n : ℕ) : ℝ := sorry

-- Conditions
axiom h1 : b 1 = 2 + Real.sqrt 8
axiom h2 : b 2023 = 15 + Real.sqrt 8
axiom recurrence : ∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)

-- Problem statement to prove
theorem b_2056 : b 2056 = (2 + Real.sqrt 8)^2 / (15 + Real.sqrt 8) :=
sorry

end NUMINAMATH_GPT_b_2056_l1181_118115


namespace NUMINAMATH_GPT_complete_remaining_parts_l1181_118189

-- Define the main conditions and the proof goal in Lean 4
theorem complete_remaining_parts :
  ∀ (total_parts processed_parts workers days_off remaining_parts_per_day),
  total_parts = 735 →
  processed_parts = 135 →
  workers = 5 →
  days_off = 1 →
  remaining_parts_per_day = total_parts - processed_parts →
  (workers * 2 - days_off) * 15 = processed_parts →
  remaining_parts_per_day / (workers * 15) = 8 :=
by
  -- Starting the proof
  intros total_parts processed_parts workers days_off remaining_parts_per_day
  intros h_total_parts h_processed_parts h_workers h_days_off h_remaining_parts_per_day h_productivity
  -- Replace given variables with their values
  sorry

end NUMINAMATH_GPT_complete_remaining_parts_l1181_118189


namespace NUMINAMATH_GPT_garage_has_18_wheels_l1181_118197

namespace Garage

def bike_wheels_per_bike : ℕ := 2
def bikes_assembled : ℕ := 9

theorem garage_has_18_wheels
  (b : ℕ := bikes_assembled) 
  (w : ℕ := bike_wheels_per_bike) :
  b * w = 18 :=
by
  sorry

end Garage

end NUMINAMATH_GPT_garage_has_18_wheels_l1181_118197


namespace NUMINAMATH_GPT_find_d_l1181_118161

theorem find_d :
  ∃ d : ℝ, (∀ x y : ℝ, x^2 + 3 * y^2 + 6 * x - 18 * y + d = 0 → x = -3 ∧ y = 3) ↔ d = -27 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_d_l1181_118161


namespace NUMINAMATH_GPT_treaty_signed_on_tuesday_l1181_118165

-- Define a constant for the start date and the number of days
def start_day_of_week : ℕ := 1 -- Monday is represented by 1
def days_until_treaty : ℕ := 1301

-- Function to calculate the resulting day of the week
def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

-- Theorem statement: Prove that 1301 days after Monday is Tuesday
theorem treaty_signed_on_tuesday :
  day_of_week_after_days start_day_of_week days_until_treaty = 2 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_treaty_signed_on_tuesday_l1181_118165


namespace NUMINAMATH_GPT_prob_three_cards_in_sequence_l1181_118148

theorem prob_three_cards_in_sequence : 
  let total_cards := 52
  let spades_count := 13
  let hearts_count := 13
  let sequence_prob := (spades_count / total_cards) * (hearts_count / (total_cards - 1)) * ((spades_count - 1) / (total_cards - 2))
  sequence_prob = (78 / 5100) :=
by
  sorry

end NUMINAMATH_GPT_prob_three_cards_in_sequence_l1181_118148


namespace NUMINAMATH_GPT_map_distance_to_real_distance_l1181_118149

theorem map_distance_to_real_distance (d_map : ℝ) (scale : ℝ) (d_real : ℝ) 
    (h1 : d_map = 7.5) (h2 : scale = 8) : d_real = 60 :=
by
  sorry

end NUMINAMATH_GPT_map_distance_to_real_distance_l1181_118149


namespace NUMINAMATH_GPT_bricks_in_chimney_900_l1181_118126

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end NUMINAMATH_GPT_bricks_in_chimney_900_l1181_118126


namespace NUMINAMATH_GPT_middle_digit_base_7_of_reversed_base_9_l1181_118169

noncomputable def middle_digit_of_number_base_7 (N : ℕ) : ℕ :=
  let x := (N / 81) % 9  -- Extract the first digit in base-9
  let y := (N / 9) % 9   -- Extract the middle digit in base-9
  let z := N % 9         -- Extract the last digit in base-9
  -- Given condition: 81x + 9y + z = 49z + 7y + x
  let eq1 := 81 * x + 9 * y + z
  let eq2 := 49 * z + 7 * y + x
  let condition := eq1 = eq2 ∧ 0 ≤ y ∧ y < 7 -- y is a digit in base-7
  if condition then y else sorry

theorem middle_digit_base_7_of_reversed_base_9 (N : ℕ) :
  (∃ (x y z : ℕ), x < 9 ∧ y < 9 ∧ z < 9 ∧
  N = 81 * x + 9 * y + z ∧ N = 49 * z + 7 * y + x) → middle_digit_of_number_base_7 N = 0 :=
  by sorry

end NUMINAMATH_GPT_middle_digit_base_7_of_reversed_base_9_l1181_118169


namespace NUMINAMATH_GPT_distance_from_apex_l1181_118158

theorem distance_from_apex (a₁ a₂ : ℝ) (d : ℝ)
  (ha₁ : a₁ = 150 * Real.sqrt 3)
  (ha₂ : a₂ = 300 * Real.sqrt 3)
  (hd : d = 10) :
  ∃ h : ℝ, h = 10 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_apex_l1181_118158


namespace NUMINAMATH_GPT_selected_female_athletes_l1181_118124

-- Definitions based on conditions
def total_male_athletes := 56
def total_female_athletes := 42
def selected_male_athletes := 8
def male_to_female_ratio := 4 / 3

-- Problem statement: Prove that the number of selected female athletes is 6
theorem selected_female_athletes :
  selected_male_athletes * (3 / 4) = 6 :=
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_selected_female_athletes_l1181_118124


namespace NUMINAMATH_GPT_find_angle_degree_l1181_118120

-- Define the angle
variable {x : ℝ}

-- Define the conditions
def complement (x : ℝ) : ℝ := 90 - x
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the given condition
def condition (x : ℝ) : Prop := complement x = (1/3) * (supplement x)

-- The theorem statement
theorem find_angle_degree (x : ℝ) (h : condition x) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_degree_l1181_118120


namespace NUMINAMATH_GPT_units_digit_of_5_to_4_l1181_118182

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_5_to_4 : units_digit (5^4) = 5 := by
  -- The definition ensures that 5^4 = 625 and the units digit is 5
  sorry

end NUMINAMATH_GPT_units_digit_of_5_to_4_l1181_118182


namespace NUMINAMATH_GPT_product_divisible_by_14_l1181_118114

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7 * a + 8 * b = 14 * c + 28 * d) : 14 ∣ a * b := 
sorry

end NUMINAMATH_GPT_product_divisible_by_14_l1181_118114


namespace NUMINAMATH_GPT_integer_solutions_count_for_equation_l1181_118142

theorem integer_solutions_count_for_equation :
  (∃ n : ℕ, (∀ x y : ℤ, (1/x + 1/y = 1/7) → (x ≠ 0) → (y ≠ 0) → n = 5 )) :=
sorry

end NUMINAMATH_GPT_integer_solutions_count_for_equation_l1181_118142


namespace NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l1181_118112

variable (a d : ℕ)

def arithmetic_sequence_sum (a d : ℕ) : ℕ :=
  a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d)

theorem ninth_term_arithmetic_sequence (h1 : arithmetic_sequence_sum a d = 21) (h2 : a + 6 * d = 7) : a + 8 * d = 9 :=
by
  sorry

end NUMINAMATH_GPT_ninth_term_arithmetic_sequence_l1181_118112


namespace NUMINAMATH_GPT_distance_between_parallel_lines_l1181_118155

-- Definitions of lines l_1 and l_2
def line_l1 (x y : ℝ) : Prop := 3*x + 4*y - 2 = 0
def line_l2 (x y : ℝ) : Prop := 6*x + 8*y - 5 = 0

-- Proof statement that the distance between the two lines is 1/10
theorem distance_between_parallel_lines (x y : ℝ) :
  ∃ d : ℝ, d = 1/10 ∧ ∀ p : ℝ × ℝ,
  (line_l1 p.1 p.2 ∧ line_l2 p.1 p.2 → p = (x, y)) :=
sorry

end NUMINAMATH_GPT_distance_between_parallel_lines_l1181_118155


namespace NUMINAMATH_GPT_equivalent_terminal_side_l1181_118123

theorem equivalent_terminal_side (k : ℤ) : 
    (∃ k : ℤ, (5 * π / 3 = -π / 3 + 2 * π * k)) :=
sorry

end NUMINAMATH_GPT_equivalent_terminal_side_l1181_118123


namespace NUMINAMATH_GPT_solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l1181_118146

theorem solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1 :
  ∀ x : ℝ, 2 * x ^ 2 + 5 * x - 3 ≠ 0 ∧ 2 * x - 1 ≠ 0 → 
  (5 * x + 1) / (2 * x ^ 2 + 5 * x - 3) = (2 * x) / (2 * x - 1) → 
  x = -1 :=
by
  intro x h_cond h_eq
  sorry

end NUMINAMATH_GPT_solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l1181_118146


namespace NUMINAMATH_GPT_raghu_investment_l1181_118100

theorem raghu_investment
  (R trishul vishal : ℝ)
  (h1 : trishul = 0.90 * R)
  (h2 : vishal = 0.99 * R)
  (h3 : R + trishul + vishal = 6647) :
  R = 2299.65 :=
by
  sorry

end NUMINAMATH_GPT_raghu_investment_l1181_118100


namespace NUMINAMATH_GPT_compute_g_f_1_l1181_118147

def f (x : ℝ) : ℝ := x^3 - 2 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem compute_g_f_1 : g (f 1) = 3 :=
by
  sorry

end NUMINAMATH_GPT_compute_g_f_1_l1181_118147


namespace NUMINAMATH_GPT_no_both_squares_l1181_118107

theorem no_both_squares {x y : ℕ} (hx : x > 0) (hy : y > 0) : ¬ (∃ a b : ℕ, a^2 = x^2 + 2 * y ∧ b^2 = y^2 + 2 * x) :=
by
  sorry

end NUMINAMATH_GPT_no_both_squares_l1181_118107


namespace NUMINAMATH_GPT_coupon1_greater_l1181_118127

variable (x : ℝ)

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 50
def coupon3_discount (x : ℝ) : ℝ := 0.25 * x - 62.5

theorem coupon1_greater (x : ℝ) (hx1 : 333.33 < x ∧ x < 625) : 
  coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end NUMINAMATH_GPT_coupon1_greater_l1181_118127


namespace NUMINAMATH_GPT_volume_of_pool_l1181_118194

theorem volume_of_pool :
  let diameter := 60
  let radius := diameter / 2
  let height_shallow := 3
  let height_deep := 15
  let height_total := height_shallow + height_deep
  let volume_cylinder := π * radius^2 * height_total
  volume_cylinder / 2 = 8100 * π :=
by
  sorry

end NUMINAMATH_GPT_volume_of_pool_l1181_118194


namespace NUMINAMATH_GPT_david_more_pushups_than_zachary_l1181_118159

def zacharyPushUps : ℕ := 59
def davidPushUps : ℕ := 78

theorem david_more_pushups_than_zachary :
  davidPushUps - zacharyPushUps = 19 :=
by
  sorry

end NUMINAMATH_GPT_david_more_pushups_than_zachary_l1181_118159


namespace NUMINAMATH_GPT_range_of_a_l1181_118104

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x : ℝ, Real.exp x ≤ 2 * x + a) : a < 2 - 2 * Real.log 2 := 
  sorry

end NUMINAMATH_GPT_range_of_a_l1181_118104


namespace NUMINAMATH_GPT_rita_remaining_money_l1181_118198

theorem rita_remaining_money :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let jackets_cost := 4 * 30
  let transport_cost := 5
  let total_expenses := dresses_cost + pants_cost + jackets_cost + transport_cost
  let initial_money := 400
  let remaining_money := initial_money - total_expenses
  remaining_money = 139 := 
by
  sorry

end NUMINAMATH_GPT_rita_remaining_money_l1181_118198


namespace NUMINAMATH_GPT_round_robin_teams_l1181_118102

theorem round_robin_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 := 
by
  sorry

end NUMINAMATH_GPT_round_robin_teams_l1181_118102
