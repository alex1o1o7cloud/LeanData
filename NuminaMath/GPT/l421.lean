import Mathlib

namespace find_FC_l421_42153

theorem find_FC 
  (DC CB AD: ℝ)
  (h1 : DC = 9)
  (h2 : CB = 6)
  (h3 : AB = (1 / 3) * AD)
  (h4 : ED = (2 / 3) * AD) :
  FC = 9 :=
sorry

end find_FC_l421_42153


namespace train_speed_l421_42189

theorem train_speed (train_length platform_length total_time : ℕ) 
  (h_train_length : train_length = 150) 
  (h_platform_length : platform_length = 250) 
  (h_total_time : total_time = 8) : 
  (train_length + platform_length) / total_time = 50 := 
by
  -- Proof goes here
  -- Given: train_length = 150, platform_length = 250, total_time = 8
  -- We need to prove: (train_length + platform_length) / total_time = 50
  -- So we calculate
  --  (150 + 250)/8 = 400/8 = 50
  sorry

end train_speed_l421_42189


namespace gcd_5280_12155_l421_42116

theorem gcd_5280_12155 : Nat.gcd 5280 12155 = 5 :=
by
  sorry

end gcd_5280_12155_l421_42116


namespace cross_product_correct_l421_42100

def v : ℝ × ℝ × ℝ := (-3, 4, 5)
def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1,
 a.2.2 * b.1 - a.1 * b.2.2,
 a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct : cross_product v w = (21, 22, -5) :=
by
  sorry

end cross_product_correct_l421_42100


namespace mens_wages_l421_42136

-- Definitions from the conditions.
variables (men women boys total_earnings : ℕ) (wage : ℚ)
variable (equivalence : 5 * men = 8 * boys)
variable (totalEarnings : total_earnings = 120)

-- The final statement to prove the men's wages.
theorem mens_wages (h_eq : 5 = 5) : wage = 46.15 :=
by
  sorry

end mens_wages_l421_42136


namespace number_of_boys_is_90_l421_42191

-- Define the conditions
variables (B G : ℕ)
axiom sum_condition : B + G = 150
axiom percentage_condition : G = (B / 150) * 100

-- State the theorem
theorem number_of_boys_is_90 : B = 90 :=
by
  -- We can skip the proof for now using sorry
  sorry

end number_of_boys_is_90_l421_42191


namespace remainder_when_divided_by_x_minus_2_l421_42150

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 10*x^3 + 20*x^2 - 5*x - 21

theorem remainder_when_divided_by_x_minus_2 :
  f 2 = 33 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l421_42150


namespace pencil_length_total_l421_42144

theorem pencil_length_total :
  (1.5 + 0.5 + 2 + 1.25 + 0.75 + 1.8 + 2.5 = 10.3) :=
by
  sorry

end pencil_length_total_l421_42144


namespace hexagon_side_length_l421_42177

-- Define the conditions for the side length of a hexagon where the area equals the perimeter
theorem hexagon_side_length (s : ℝ) (h1 : (3 * Real.sqrt 3 / 2) * s^2 = 6 * s) :
  s = 4 * Real.sqrt 3 / 3 :=
sorry

end hexagon_side_length_l421_42177


namespace intersection_of_M_and_N_l421_42171

open Set

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}
def I : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = I := by
  sorry

end intersection_of_M_and_N_l421_42171


namespace geometric_prog_common_ratio_one_l421_42130

variable {x y z : ℝ}
variable {r : ℝ}

theorem geometric_prog_common_ratio_one
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (hgeom : ∃ a : ℝ, a = x * (y - z) ∧ a * r = y * (z - x) ∧ a * r^2 = z * (x - y))
  (hprod : (x * (y - z)) * (y * (z - x)) * (z * (x - y)) * r^3 = (y * (z - x))^2) : 
  r = 1 := sorry

end geometric_prog_common_ratio_one_l421_42130


namespace line_intersects_y_axis_at_origin_l421_42164

theorem line_intersects_y_axis_at_origin 
  (x₁ y₁ x₂ y₂ : ℤ) 
  (h₁ : (x₁, y₁) = (3, 9)) 
  (h₂ : (x₂, y₂) = (-7, -21)) 
  : 
  ∃ y : ℤ, (0, y) = (0, 0) := by
  sorry

end line_intersects_y_axis_at_origin_l421_42164


namespace find_larger_number_l421_42186

variable (x y : ℝ)
axiom h1 : x + y = 27
axiom h2 : x - y = 5

theorem find_larger_number : x = 16 :=
by
  sorry

end find_larger_number_l421_42186


namespace anoop_joined_after_6_months_l421_42188

/- Conditions -/
def arjun_investment : ℕ := 20000
def arjun_months : ℕ := 12
def anoop_investment : ℕ := 40000

/- Main theorem -/
theorem anoop_joined_after_6_months (x : ℕ) (h : arjun_investment * arjun_months = anoop_investment * (arjun_months - x)) : 
  x = 6 :=
sorry

end anoop_joined_after_6_months_l421_42188


namespace total_white_balls_l421_42145

theorem total_white_balls : ∃ W R B : ℕ,
  W + R = 300 ∧ B = 100 ∧
  ∃ (bw1 bw2 rw3 rw W3 : ℕ),
  bw1 = 27 ∧
  rw3 + rw = 42 ∧
  W3 = rw ∧
  B = bw1 + W3 + rw3 + bw2 ∧
  W = bw1 + 2 * bw2 + 3 * W3 ∧
  R = 3 * rw3 + rw ∧
  W = 158 :=
by
  sorry

end total_white_balls_l421_42145


namespace train_speed_l421_42155

def distance := 11.67 -- distance in km
def time := 10.0 / 60.0 -- time in hours (10 minutes is 10/60 hours)

theorem train_speed : (distance / time) = 70.02 := by
  sorry

end train_speed_l421_42155


namespace combination_problem_l421_42184

noncomputable def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.choose n k

theorem combination_problem (x : ℕ) (h : combination 25 (2 * x) = combination 25 (x + 4)) : x = 4 ∨ x = 7 :=
by {
  sorry
}

end combination_problem_l421_42184


namespace wharf_length_l421_42163

-- Define the constants
def avg_speed := 2 -- average speed in m/s
def travel_time := 16 -- travel time in seconds

-- Define the formula to calculate length of the wharf
def length_of_wharf := 2 * avg_speed * travel_time

-- The goal is to prove that length_of_wharf equals 64
theorem wharf_length : length_of_wharf = 64 :=
by
  -- Proof would be here
  sorry

end wharf_length_l421_42163


namespace arithmetic_geom_seq_l421_42122

variable {a_n : ℕ → ℝ}
variable {d a_1 : ℝ}
variable (h_seq : ∀ n, a_n n = a_1 + (n-1) * d)
variable (d_ne_zero : d ≠ 0)
variable (a_1_ne_zero : a_1 ≠ 0)
variable (geo_seq : (a_1 + d)^2 = a_1 * (a_1 + 3 * d))

theorem arithmetic_geom_seq :
  (a_1 + a_n 14) / a_n 3 = 5 := by
  sorry

end arithmetic_geom_seq_l421_42122


namespace solve_students_in_fifth_grade_class_l421_42148

noncomputable def number_of_students_in_each_fifth_grade_class 
    (third_grade_classes : ℕ) 
    (third_grade_students_per_class : ℕ)
    (fourth_grade_classes : ℕ) 
    (fourth_grade_students_per_class : ℕ) 
    (fifth_grade_classes : ℕ)
    (total_lunch_cost : ℝ)
    (hamburger_cost : ℝ)
    (carrot_cost : ℝ)
    (cookie_cost : ℝ) : ℝ :=
  
  let total_students_third := third_grade_classes * third_grade_students_per_class
  let total_students_fourth := fourth_grade_classes * fourth_grade_students_per_class
  let lunch_cost_per_student := hamburger_cost + carrot_cost + cookie_cost
  let total_students := total_students_third + total_students_fourth
  let total_cost_third_fourth := total_students * lunch_cost_per_student
  let total_cost_fifth := total_lunch_cost - total_cost_third_fourth
  let fifth_grade_students := total_cost_fifth / lunch_cost_per_student
  let students_per_fifth_class := fifth_grade_students / fifth_grade_classes
  students_per_fifth_class

theorem solve_students_in_fifth_grade_class : 
    number_of_students_in_each_fifth_grade_class 5 30 4 28 4 1036 2.10 0.50 0.20 = 27 := 
by 
  sorry

end solve_students_in_fifth_grade_class_l421_42148


namespace graph_of_equation_l421_42110

theorem graph_of_equation (x y : ℝ) : (x + y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := 
by
  sorry

end graph_of_equation_l421_42110


namespace vector_a_properties_l421_42183

-- Definitions of the points in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction to find the vector between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Definition of dot product for vectors
def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Definition of vector magnitude squared for vectors
def magnitude_squared (v : Point3D) : ℝ :=
  v.x * v.x + v.y * v.y + v.z * v.z

-- Main theorem statement
theorem vector_a_properties :
  let A := {x := 0, y := 2, z := 3}
  let B := {x := -2, y := 1, z := 6}
  let C := {x := 1, y := -1, z := 5}
  let AB := vector_sub A B
  let AC := vector_sub A C
  ∀ (a : Point3D), 
    (magnitude_squared a = 3) → 
    (dot_product a AB = 0) → 
    (dot_product a AC = 0) → 
    (a = {x := 1, y := 1, z := 1} ∨ a = {x := -1, y := -1, z := -1}) := 
by
  intros A B C AB AC a ha_magnitude ha_perpendicular_AB ha_perpendicular_AC
  sorry

end vector_a_properties_l421_42183


namespace sector_area_l421_42178

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 4 := 
by sorry

end sector_area_l421_42178


namespace earl_stuff_rate_l421_42192

variable (E L : ℕ)

-- Conditions
def ellen_rate : Prop := L = (2 * E) / 3
def combined_rate : Prop := E + L = 60

-- Main statement
theorem earl_stuff_rate (h1 : ellen_rate E L) (h2 : combined_rate E L) : E = 36 := by
  sorry

end earl_stuff_rate_l421_42192


namespace sum_is_correct_l421_42105

def number : ℕ := 81
def added_number : ℕ := 15
def sum_value (x : ℕ) (y : ℕ) : ℕ := x + y

theorem sum_is_correct : sum_value number added_number = 96 := 
by 
  sorry

end sum_is_correct_l421_42105


namespace min_max_abs_poly_eq_zero_l421_42167

theorem min_max_abs_poly_eq_zero :
  ∃ y : ℝ, (∀ x : ℝ, 0 ≤ x → x ≤ 1 → |x^2 - x^3 * y| ≤ 0) :=
sorry

end min_max_abs_poly_eq_zero_l421_42167


namespace smallest_positive_x_for_maximum_l421_42165

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.cos (x / 9)

theorem smallest_positive_x_for_maximum (x : ℝ) :
  (∀ k m : ℤ, x = 360 * (1 + k) ∧ x = 3600 * m ∧ 0 < x → x = 3600) :=
by
  sorry

end smallest_positive_x_for_maximum_l421_42165


namespace evaluate_expression_l421_42134

theorem evaluate_expression : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end evaluate_expression_l421_42134


namespace conic_is_pair_of_lines_l421_42179

-- Define the specific conic section equation
def conic_eq (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

-- State the theorem
theorem conic_is_pair_of_lines : ∀ x y : ℝ, conic_eq x y ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  -- Sorry is placed to denote that proof steps are omitted in this statement
  sorry

end conic_is_pair_of_lines_l421_42179


namespace smallest_b_for_perfect_square_l421_42114

theorem smallest_b_for_perfect_square : ∃ (b : ℤ), b > 4 ∧ ∃ (n : ℤ), 3 * b + 4 = n * n ∧ b = 7 := by
  sorry

end smallest_b_for_perfect_square_l421_42114


namespace smallest_quotient_is_1_9_l421_42198

def is_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n <= 99

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  let x := n / 10
  let y := n % 10
  x + y

noncomputable def quotient (n : ℕ) : ℚ :=
  n / (sum_of_digits n)

theorem smallest_quotient_is_1_9 :
  ∃ n, is_two_digit_number n ∧ (∃ x y, n = 10 * x + y ∧ x ≠ y ∧ 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9) ∧ quotient n = 1.9 := 
sorry

end smallest_quotient_is_1_9_l421_42198


namespace equilateral_triangle_l421_42139

theorem equilateral_triangle {a b c : ℝ} (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c :=
by {
  sorry
}

end equilateral_triangle_l421_42139


namespace find_x_squared_l421_42101

theorem find_x_squared :
  ∃ x : ℕ, (x^2 >= 2525 * 10^8) ∧ (x^2 < 2526 * 10^8) ∧ (x % 100 = 17 ∨ x % 100 = 33 ∨ x % 100 = 67 ∨ x % 100 = 83) ∧
    (x = 502517 ∨ x = 502533 ∨ x = 502567 ∨ x = 502583) :=
sorry

end find_x_squared_l421_42101


namespace john_tax_rate_l421_42108

theorem john_tax_rate { P: Real → Real → Real → Real → Prop }:
  ∀ (cNikes cBoots totalPaid taxRate: ℝ), 
  cNikes = 150 →
  cBoots = 120 →
  totalPaid = 297 →
  taxRate = ((totalPaid - (cNikes + cBoots)) / (cNikes + cBoots)) * 100 →
  taxRate = 10 :=
by
  intros cNikes cBoots totalPaid taxRate HcNikes HcBoots HtotalPaid HtaxRate
  sorry

end john_tax_rate_l421_42108


namespace emily_disproved_jacob_by_turnover_5_and_7_l421_42162

def is_vowel (c : Char) : Prop :=
  c = 'A'

def is_consonant (c : Char) : Prop :=
  ¬ is_vowel c

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def card_A_is_vowel : Prop := is_vowel 'A'
def card_1_is_odd : Prop := ¬ is_even 1 ∧ ¬ is_prime 1
def card_8_is_even : Prop := is_even 8 ∧ ¬ is_prime 8
def card_R_is_consonant : Prop := is_consonant 'R'
def card_S_is_consonant : Prop := is_consonant 'S'
def card_5_conditions : Prop := ¬ is_even 5 ∧ is_prime 5
def card_7_conditions : Prop := ¬ is_even 7 ∧ is_prime 7

theorem emily_disproved_jacob_by_turnover_5_and_7 :
  card_5_conditions ∧ card_7_conditions →
  (∃ (c : Char), (is_prime 5 ∧ is_consonant c)) ∨
  (∃ (c : Char), (is_prime 7 ∧ is_consonant c)) :=
by sorry

end emily_disproved_jacob_by_turnover_5_and_7_l421_42162


namespace bigger_wheel_roll_distance_l421_42135

/-- The circumference of the bigger wheel is 12 meters -/
def bigger_wheel_circumference : ℕ := 12

/-- The circumference of the smaller wheel is 8 meters -/
def smaller_wheel_circumference : ℕ := 8

/-- The distance the bigger wheel must roll for the points P1 and P2 to coincide again -/
theorem bigger_wheel_roll_distance : Nat.lcm bigger_wheel_circumference smaller_wheel_circumference = 24 :=
by
  -- Proof is omitted
  sorry

end bigger_wheel_roll_distance_l421_42135


namespace solve_for_a_l421_42141

noncomputable def line_slope_parallels (a : ℝ) : Prop :=
  (a^2 - a) = 6

theorem solve_for_a : { a : ℝ // line_slope_parallels a } → (a = -2 ∨ a = 3) := by
  sorry

end solve_for_a_l421_42141


namespace negation_of_P_l421_42161

def P : Prop := ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 ≤ 0

theorem negation_of_P : ¬ P ↔ ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by sorry

end negation_of_P_l421_42161


namespace function_neither_even_nor_odd_l421_42124

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 - x

theorem function_neither_even_nor_odd : ¬is_even_function f ∧ ¬is_odd_function f := by
  sorry

end function_neither_even_nor_odd_l421_42124


namespace hyperbola_focal_length_l421_42129

theorem hyperbola_focal_length (m : ℝ) 
  (h0 : (∀ x y, x^2 / 16 - y^2 / m = 1)) 
  (h1 : (2 * Real.sqrt (16 + m) = 4 * Real.sqrt 5)) : 
  m = 4 := 
by sorry

end hyperbola_focal_length_l421_42129


namespace total_weight_CaBr2_l421_42169

-- Definitions derived from conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_Br : ℝ := 79.904
def mol_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br
def moles_CaBr2 : ℝ := 4

-- Theorem statement based on the problem and correct answer
theorem total_weight_CaBr2 : moles_CaBr2 * mol_weight_CaBr2 = 799.552 :=
by
  -- Prove the theorem step-by-step
  -- substitute the definition of mol_weight_CaBr2
  -- show lhs = rhs
  sorry

end total_weight_CaBr2_l421_42169


namespace solve_inequality_l421_42123

theorem solve_inequality (a b x : ℝ) (h : a ≠ b) :
  a^2 * x + b^2 * (1 - x) ≥ (a * x + b * (1 - x))^2 ↔ 0 ≤ x ∧ x ≤ 1 :=
sorry

end solve_inequality_l421_42123


namespace cubic_roots_real_parts_neg_l421_42117

variable {a0 a1 a2 a3 : ℝ}

theorem cubic_roots_real_parts_neg (h_same_signs : (a0 > 0 ∧ a1 > 0 ∧ a2 > 0 ∧ a3 > 0) ∨ (a0 < 0 ∧ a1 < 0 ∧ a2 < 0 ∧ a3 < 0)) 
  (h_root_condition : a1 * a2 - a0 * a3 > 0) : 
    ∀ (x : ℝ), (a0 * x^3 + a1 * x^2 + a2 * x + a3 = 0 → x < 0 ∨ (∃ (z : ℂ), z.re < 0 ∧ z.im ≠ 0 ∧ z^2 = x)) :=
sorry

end cubic_roots_real_parts_neg_l421_42117


namespace sin_three_pi_div_two_l421_42107

theorem sin_three_pi_div_two : Real.sin (3 * Real.pi / 2) = -1 := 
by
  sorry

end sin_three_pi_div_two_l421_42107


namespace complex_numbers_satisfying_conditions_l421_42185

theorem complex_numbers_satisfying_conditions (x y z : ℂ) 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : x = 1 ∧ y = 1 ∧ z = 1 := 
by sorry

end complex_numbers_satisfying_conditions_l421_42185


namespace set_aside_bars_each_day_l421_42143

-- Definitions for the conditions
def total_bars : Int := 20
def bars_traded : Int := 3
def bars_per_sister : Int := 5
def number_of_sisters : Int := 2
def days_in_week : Int := 7

-- Our goal is to prove that Greg set aside 1 bar per day
theorem set_aside_bars_each_day
  (h1 : 20 - 3 = 17)
  (h2 : 5 * 2 = 10)
  (h3 : 17 - 10 = 7)
  (h4 : 7 / 7 = 1) :
  (total_bars - bars_traded - (bars_per_sister * number_of_sisters)) / days_in_week = 1 := by
  sorry

end set_aside_bars_each_day_l421_42143


namespace prime_factorization_2020_prime_factorization_2021_l421_42156

theorem prime_factorization_2020 : 2020 = 2^2 * 5 * 101 := by
  sorry

theorem prime_factorization_2021 : 2021 = 43 * 47 := by
  sorry

end prime_factorization_2020_prime_factorization_2021_l421_42156


namespace proof_statement_B_proof_statement_D_proof_statement_E_l421_42170

def statement_B (x : ℝ) : Prop := x^2 = 0 → x = 0

def statement_D (x : ℝ) : Prop := x^2 < 2 * x → x > 0

def statement_E (x : ℝ) : Prop := x > 2 → x^2 > x

theorem proof_statement_B (x : ℝ) : statement_B x := sorry

theorem proof_statement_D (x : ℝ) : statement_D x := sorry

theorem proof_statement_E (x : ℝ) : statement_E x := sorry

end proof_statement_B_proof_statement_D_proof_statement_E_l421_42170


namespace vasya_wins_game_l421_42151

/- Define the conditions of the problem -/

def grid_size : Nat := 9
def total_matchsticks : Nat := 2 * grid_size * (grid_size + 1)

/-- Given a game on a 9x9 matchstick grid with Petya going first, 
    Prove that Vasya can always win by ensuring that no whole 1x1 
    squares remain in the end. -/
theorem vasya_wins_game : 
  ∃ strategy_for_vasya : Nat → Nat → Prop, -- Define a strategy for Vasya
  ∀ (matchsticks_left : Nat),
  matchsticks_left % 2 = 1 →     -- Petya makes a move and the remaining matchsticks are odd
  strategy_for_vasya matchsticks_left total_matchsticks :=
sorry

end vasya_wins_game_l421_42151


namespace enclosed_area_abs_x_abs_3y_eq_12_l421_42182

theorem enclosed_area_abs_x_abs_3y_eq_12 : 
  let f (x y : ℝ) := |x| + |3 * y|
  ∃ (A : ℝ), ∀ (x y : ℝ), f x y = 12 → A = 96 := 
sorry

end enclosed_area_abs_x_abs_3y_eq_12_l421_42182


namespace solution_set_l421_42146

-- Define the conditions
variable (f : ℝ → ℝ)
variable (odd_func : ∀ x : ℝ, f (-x) = -f x)
variable (increasing_pos : ∀ a b : ℝ, 0 < a → 0 < b → a < b → f a < f b)
variable (f_neg3_zero : f (-3) = 0)

-- State the theorem
theorem solution_set (x : ℝ) : x * f x < 0 ↔ (-3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3) :=
sorry

end solution_set_l421_42146


namespace ladybugs_without_spots_l421_42194

-- Defining the conditions given in the problem
def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170

-- Proving the number of ladybugs without spots
theorem ladybugs_without_spots : total_ladybugs - ladybugs_with_spots = 54912 := by
  sorry

end ladybugs_without_spots_l421_42194


namespace find_x_value_l421_42103

open Real

theorem find_x_value (a b c : ℤ) (x : ℝ) (h : 5 / (a^2 + b * log x) = c) : 
  x = 10^((5 / c - a^2) / b) := 
by 
  sorry

end find_x_value_l421_42103


namespace first_term_geometric_sequence_l421_42180

theorem first_term_geometric_sequence (a r : ℕ) (h₁ : a * r^5 = 32) (h₂ : r = 2) : a = 1 := by
  sorry

end first_term_geometric_sequence_l421_42180


namespace sally_seashells_l421_42111

theorem sally_seashells (T S: ℕ) (hT : T = 37) (h_total : T + S = 50) : S = 13 := by
  -- Skip the proof
  sorry

end sally_seashells_l421_42111


namespace min_expression_min_expression_achieve_l421_42199

theorem min_expression (x : ℝ) (hx : 0 < x) : 
  (x^2 + 8 * x + 64 / x^3) ≥ 28 :=
sorry

theorem min_expression_achieve (x : ℝ) (hx : x = 2): 
  (x^2 + 8 * x + 64 / x^3) = 28 :=
sorry

end min_expression_min_expression_achieve_l421_42199


namespace double_burger_cost_l421_42102

theorem double_burger_cost (D : ℝ) : 
  let single_burger_cost := 1.00
  let total_burgers := 50
  let double_burgers := 37
  let total_cost := 68.50
  let single_burgers := total_burgers - double_burgers
  let singles_cost := single_burgers * single_burger_cost
  let doubles_cost := total_cost - singles_cost
  let burger_cost := doubles_cost / double_burgers
  burger_cost = D := 
by 
  sorry

end double_burger_cost_l421_42102


namespace blue_face_probability_l421_42133

def sides : ℕ := 12
def green_faces : ℕ := 5
def blue_faces : ℕ := 4
def red_faces : ℕ := 3

theorem blue_face_probability : 
  (blue_faces : ℚ) / sides = 1 / 3 :=
by
  sorry

end blue_face_probability_l421_42133


namespace lois_final_books_l421_42118

-- Definitions for the conditions given in the problem.
def initial_books : ℕ := 40
def books_given_to_nephew (b : ℕ) : ℕ := b / 4
def books_remaining_after_giving (b_given : ℕ) (b : ℕ) : ℕ := b - b_given
def books_donated_to_library (b_remaining : ℕ) : ℕ := b_remaining / 3
def books_remaining_after_donating (b_donated : ℕ) (b_remaining : ℕ) : ℕ := b_remaining - b_donated
def books_purchased : ℕ := 3
def total_books (b_final_remaining : ℕ) (b_purchased : ℕ) : ℕ := b_final_remaining + b_purchased

-- Theorem stating: Given the initial conditions, Lois should have 23 books in the end.
theorem lois_final_books : 
  total_books 
    (books_remaining_after_donating (books_donated_to_library (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books)) 
    (books_remaining_after_giving (books_given_to_nephew initial_books) initial_books))
    books_purchased = 23 :=
  by
    sorry  -- Proof omitted as per instructions.

end lois_final_books_l421_42118


namespace a_value_l421_42120

-- Definition of the operation
def star (x y : ℝ) : ℝ := x + y - x * y

-- Main theorem to prove
theorem a_value :
  let a := star 1 (star 0 1)
  a = 1 :=
by
  sorry

end a_value_l421_42120


namespace solve_inequality_l421_42112

theorem solve_inequality : 
  {x : ℝ | (x^3 - x^2 - 6 * x) / (x^2 - 3 * x + 2) > 0} = 
  {x : ℝ | (-2 < x ∧ x < 0) ∨ (1 < x ∧ x < 2) ∨ (3 < x)} :=
sorry

end solve_inequality_l421_42112


namespace all_statements_true_l421_42127

theorem all_statements_true (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  (a^2 + b^2 < (a + b)^2) ∧ 
  (ab > 0) ∧ 
  (a > b) ∧ 
  (a > 0) ∧
  (b > 0) :=
by
  sorry

end all_statements_true_l421_42127


namespace systematic_sampling_distance_l421_42158

-- Conditions
def total_students : ℕ := 1200
def sample_size : ℕ := 30

-- Problem: Compute sampling distance
def sampling_distance (n : ℕ) (m : ℕ) : ℕ := n / m

-- The formal proof statement
theorem systematic_sampling_distance :
  sampling_distance total_students sample_size = 40 := by
  sorry

end systematic_sampling_distance_l421_42158


namespace split_fraction_l421_42147

theorem split_fraction (n d a b x y : ℤ) (h_d : d = a * b) (h_ad : a.gcd b = 1) (h_frac : (n:ℚ) / (d:ℚ) = 58 / 77) (h_eq : 11 * x + 7 * y = 58) : 
  (58:ℚ) / 77 = (4:ℚ) / 7 + (2:ℚ) / 11 :=
by
  sorry

end split_fraction_l421_42147


namespace trapezoid_area_l421_42168

theorem trapezoid_area 
  (h : ℝ) (BM CM : ℝ) 
  (height_cond : h = 12) 
  (BM_cond : BM = 15) 
  (CM_cond : CM = 13) 
  (angle_bisectors_intersect : ∃ M : ℝ, (BM^2 - h^2) = 9^2 ∧ (CM^2 - h^2) = 5^2) : 
  ∃ (S : ℝ), S = 260.4 :=
by
  -- Skipping the proof part by using sorry
  sorry

end trapezoid_area_l421_42168


namespace find_A_in_triangle_l421_42172

theorem find_A_in_triangle
  (a b : ℝ) (B A : ℝ)
  (h₀ : a = Real.sqrt 3)
  (h₁ : b = Real.sqrt 2)
  (h₂ : B = Real.pi / 4)
  (h₃ : a / Real.sin A = b / Real.sin B) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end find_A_in_triangle_l421_42172


namespace length_of_AB_l421_42109

theorem length_of_AB (x1 y1 x2 y2 : ℝ) 
  (h_parabola_A : y1^2 = 8 * x1) 
  (h_focus_line_A : y1 = 2 * (x1 - 2)) 
  (h_parabola_B : y2^2 = 8 * x2) 
  (h_focus_line_B : y2 = 2 * (x2 - 2)) 
  (h_sum_x : x1 + x2 = 6) : 
  |x1 - x2| = 10 :=
sorry

end length_of_AB_l421_42109


namespace sum_of_reciprocals_negative_l421_42132

theorem sum_of_reciprocals_negative {a b c : ℝ} (h₁ : a + b + c = 0) (h₂ : a * b * c > 0) :
  1/a + 1/b + 1/c < 0 :=
sorry

end sum_of_reciprocals_negative_l421_42132


namespace quadratic_inequality_solution_l421_42142

theorem quadratic_inequality_solution:
  ∃ P q : ℝ,
  (1 / P < 0) ∧
  (-P * q = 6) ∧
  (P^2 = 8) ∧
  (P = -2 * Real.sqrt 2) ∧
  (q = 3 / 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_inequality_solution_l421_42142


namespace find_valid_pairs_l421_42131

-- Decalred the main definition for the problem.
def valid_pairs (x y : ℕ) : Prop :=
  (10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99) ∧ ((x + y)^2 = 100 * x + y)

-- Stating the theorem without the proof.
theorem find_valid_pairs :
  valid_pairs 20 25 ∧ valid_pairs 30 25 :=
sorry

end find_valid_pairs_l421_42131


namespace find_x_value_l421_42157

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) :
  (Real.tan (150 - x * Real.pi / 180) = 
   (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
   (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180))) → 
  x = 110 := 
by 
  sorry

end find_x_value_l421_42157


namespace even_function_phi_l421_42176

noncomputable def phi := (3 * Real.pi) / 2

theorem even_function_phi (phi_val : Real) (hphi : 0 ≤ phi_val ∧ phi_val ≤ 2 * Real.pi) :
  (∀ x, Real.sin ((x + phi) / 3) = Real.sin ((-x + phi) / 3)) ↔ phi_val = phi := by
  sorry

end even_function_phi_l421_42176


namespace find_tangent_equal_l421_42121

theorem find_tangent_equal (n : ℤ) (hn : -90 < n ∧ n < 90) (htan : Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180)) : n = 75 :=
sorry

end find_tangent_equal_l421_42121


namespace loss_percentage_l421_42154

variable (CP SP : ℕ) -- declare the variables for cost price and selling price

theorem loss_percentage (hCP : CP = 1400) (hSP : SP = 1190) : 
  ((CP - SP) / CP * 100) = 15 := by
sorry

end loss_percentage_l421_42154


namespace problem_relation_l421_42175

-- Definitions indicating relationships.
def related₁ : Prop := ∀ (s : ℝ), (s ≥ 0) → (∃ a p : ℝ, a = s^2 ∧ p = 4 * s)
def related₂ : Prop := ∀ (d t : ℝ), (t > 0) → (∃ v : ℝ, d = v * t)
def related₃ : Prop := ∃ (h w : ℝ) (f : ℝ → ℝ), w = f h
def related₄ : Prop := ∀ (h : ℝ) (v : ℝ), False

-- The theorem stating that A, B, and C are related.
theorem problem_relation : 
  related₁ ∧ related₂ ∧ related₃ ∧ ¬ related₄ :=
by sorry

end problem_relation_l421_42175


namespace meaningful_expression_l421_42149

theorem meaningful_expression (x : ℝ) : (1 / (x - 2) ≠ 0) ↔ (x ≠ 2) :=
by
  sorry

end meaningful_expression_l421_42149


namespace average_age_of_first_and_fifth_fastest_dogs_l421_42181

-- Definitions based on the conditions
def first_dog_age := 10
def second_dog_age := first_dog_age - 2
def third_dog_age := second_dog_age + 4
def fourth_dog_age := third_dog_age / 2
def fifth_dog_age := fourth_dog_age + 20

-- Statement to prove
theorem average_age_of_first_and_fifth_fastest_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 := by
  -- Add your proof here
  sorry

end average_age_of_first_and_fifth_fastest_dogs_l421_42181


namespace domain_of_f_symmetry_of_f_l421_42115

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (4 * x^2 - x^4)) / (abs (x - 2) - 2)

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

theorem symmetry_of_f :
  ∀ x : ℝ, f (x + 1) + 1 = f (-(x + 1)) + 1 :=
by
  sorry

end domain_of_f_symmetry_of_f_l421_42115


namespace no_infinite_sequence_of_positive_integers_l421_42137

theorem no_infinite_sequence_of_positive_integers (a : ℕ → ℕ) (H : ∀ n, a n > 0) :
  ¬(∀ n, (a (n+1))^2 ≥ 2 * (a n) * (a (n+2))) :=
sorry

end no_infinite_sequence_of_positive_integers_l421_42137


namespace number_of_real_roots_l421_42174

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem number_of_real_roots (a : ℝ) :
    ((|a| < (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ x₃ : ℝ, f x₁ = a ∧ f x₂ = a ∧ f x₃ = a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)) ∧
    ((|a| > (2 * Real.sqrt 3) / 9) → (∃ x : ℝ, f x = a ∧ ∀ y : ℝ, f y = a → y = x)) ∧
    ((|a| = (2 * Real.sqrt 3) / 9) → (∃ x₁ x₂ : ℝ, f x₁ = a ∧ f x₂ = a ∧ x₁ ≠ x₂ ∧ ∀ y : ℝ, (f y = a → (y = x₁ ∨ y = x₂)) ∧ (x₁ = x₂ ∨ ∀ z : ℝ, (f z = a → z = x₁ ∨ z = x₂)))) := sorry

end number_of_real_roots_l421_42174


namespace trench_dig_time_l421_42197

theorem trench_dig_time (a b c d : ℝ) (h1 : a + b + c + d = 1/6)
  (h2 : 2 * a + (1 / 2) * b + c + d = 1 / 6)
  (h3 : (1 / 2) * a + 2 * b + c + d = 1 / 4) :
  a + b + c = 1 / 6 := sorry

end trench_dig_time_l421_42197


namespace maximum_volume_l421_42138

noncomputable def volume (x : ℝ) : ℝ :=
  (48 - 2*x)^2 * x

theorem maximum_volume :
  (∀ x : ℝ, (0 < x) ∧ (x < 24) → volume x ≤ volume 8) ∧ (volume 8 = 8192) :=
by
  sorry

end maximum_volume_l421_42138


namespace admission_price_for_adults_l421_42119

-- Constants and assumptions
def children_ticket_price : ℕ := 25
def total_persons : ℕ := 280
def total_collected_dollars : ℕ := 140
def total_collected_cents : ℕ := total_collected_dollars * 100
def children_attended : ℕ := 80

-- Definitions based on the conditions
def adults_attended : ℕ := total_persons - children_attended
def total_amount_from_children : ℕ := children_attended * children_ticket_price
def total_amount_from_adults (A : ℕ) : ℕ := total_collected_cents - total_amount_from_children
def adult_ticket_price := (total_collected_cents - total_amount_from_children) / adults_attended

-- Theorem statement to be proved
theorem admission_price_for_adults : adult_ticket_price = 60 := by
  sorry

end admission_price_for_adults_l421_42119


namespace problem_l421_42152

variable (x y z : ℚ)

-- Conditions as definitions
def cond1 : Prop := x / y = 3
def cond2 : Prop := y / z = 5 / 2

-- Theorem statement with the final proof goal
theorem problem (h1 : cond1 x y) (h2 : cond2 y z) : z / x = 2 / 15 := 
by 
  sorry

end problem_l421_42152


namespace sum_of_a_and_b_l421_42159

theorem sum_of_a_and_b (a b : ℕ) (h1 : a > 0) (h2 : b > 1) (h3 : a^b < 500) (h_max : ∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a'^b' ≤ a^b) : a + b = 24 :=
sorry

end sum_of_a_and_b_l421_42159


namespace necessary_and_sufficient_l421_42190

theorem necessary_and_sufficient (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  ¬ ((a > 0 ∧ b > 0 → ab < (a + b) / 2 ^ 2) 
  ∧ (ab < (a + b) / 2 ^ 2 → a > 0 ∧ b > 0)) := 
sorry

end necessary_and_sufficient_l421_42190


namespace smaller_solution_of_quadratic_l421_42195

theorem smaller_solution_of_quadratic :
  (∃ x y : ℝ, x ≠ y ∧ (x^2 - 13 * x + 36 = 0) ∧ (y^2 - 13 * y + 36 = 0) ∧ min x y = 4) :=
sorry

end smaller_solution_of_quadratic_l421_42195


namespace value_of_a_minus_b_l421_42113

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the invertible function f

theorem value_of_a_minus_b (a b : ℝ) (hf_inv : Function.Injective f)
  (hfa : f a = b) (hfb : f b = 6) (ha1 : f 3 = 1) (hb1 : f 1 = 6) : a - b = 2 :=
sorry

end value_of_a_minus_b_l421_42113


namespace minimum_f_value_l421_42126

noncomputable def f (x y : ℝ) : ℝ :=
  y / x + 16 * x / (2 * x + y)

theorem minimum_f_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, (∀ x y, f x y ≥ t) ∧ t = 6 := sorry

end minimum_f_value_l421_42126


namespace valid_divisors_of_196_l421_42125

theorem valid_divisors_of_196 : 
  ∃ d : Finset Nat, (∀ x ∈ d, 1 < x ∧ x < 196 ∧ 196 % x = 0) ∧ d.card = 7 := by
  sorry

end valid_divisors_of_196_l421_42125


namespace no_convex_quad_with_given_areas_l421_42187

theorem no_convex_quad_with_given_areas :
  ¬ ∃ (A B C D M : Type) 
    (T_MAB T_MBC T_MDA T_MDC : ℕ) 
    (H1 : T_MAB = 1) 
    (H2 : T_MBC = 2)
    (H3 : T_MDA = 3) 
    (H4 : T_MDC = 4),
    true :=
by {
  sorry
}

end no_convex_quad_with_given_areas_l421_42187


namespace farm_own_more_horses_than_cows_after_transaction_l421_42140

theorem farm_own_more_horses_than_cows_after_transaction :
  ∀ (x : Nat), 
    3 * (3 * x - 15) = 5 * (x + 15) →
    75 - 45 = 30 :=
by
  intro x h
  -- This is a placeholder for the proof steps which we skip.
  sorry

end farm_own_more_horses_than_cows_after_transaction_l421_42140


namespace impossible_partition_10x10_square_l421_42160

theorem impossible_partition_10x10_square :
  ¬ ∃ (x y : ℝ), (x - y = 1) ∧ (x * y = 1) ∧ (∃ (n m : ℕ), 10 = n * x + m * y ∧ n + m = 100) :=
by
  sorry

end impossible_partition_10x10_square_l421_42160


namespace vector_calculation_l421_42196

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (1, -1)

def vector_operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2)

theorem vector_calculation : vector_operation vector_a vector_b = (1, 5) :=
by sorry

end vector_calculation_l421_42196


namespace play_children_count_l421_42173

theorem play_children_count (cost_adult_ticket cost_children_ticket total_receipts total_attendance adult_count children_count : ℕ) :
  cost_adult_ticket = 25 →
  cost_children_ticket = 15 →
  total_receipts = 7200 →
  total_attendance = 400 →
  adult_count = 280 →
  25 * adult_count + 15 * children_count = total_receipts →
  adult_count + children_count = total_attendance →
  children_count = 120 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end play_children_count_l421_42173


namespace min_sum_squares_roots_l421_42104

theorem min_sum_squares_roots (m : ℝ) :
  (∃ (α β : ℝ), 2 * α^2 - 3 * α + m = 0 ∧ 2 * β^2 - 3 * β + m = 0 ∧ α ≠ β) → 
  (9 - 8 * m ≥ 0) →
  (α^2 + β^2 = (3/2)^2 - 2 * (m/2)) →
  (α^2 + β^2 = 9/8) ↔ m = 9/8 :=
by
  sorry

end min_sum_squares_roots_l421_42104


namespace abs_two_minus_sqrt_five_l421_42193

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l421_42193


namespace range_of_f_l421_42106

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x ^ 2 else Real.cos x

theorem range_of_f : Set.range f = Set.Ici (-1) := 
by
  sorry

end range_of_f_l421_42106


namespace log_inequality_l421_42166

theorem log_inequality (n : ℕ) (h1 : n > 1) : 
  (1 : ℝ) / (n : ℝ) > Real.log ((n + 1 : ℝ) / n) ∧ 
  Real.log ((n + 1 : ℝ) / n) > (1 : ℝ) / (n + 1) := 
by
  sorry

end log_inequality_l421_42166


namespace range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l421_42128

variable (m : ℝ)

def set_A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def set_B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Problem 1: Prove the range of m if B ⊆ A is (-∞, 3]
theorem range_m_if_B_subset_A : (set_B m ⊆ set_A) ↔ m ≤ 3 := sorry

-- Problem 2: Prove the range of m if A ∩ B = ∅ is m < 2 or m > 4
theorem range_m_if_A_inter_B_empty : (set_A ∩ set_B m = ∅) ↔ m < 2 ∨ m > 4 := sorry

end range_m_if_B_subset_A_range_m_if_A_inter_B_empty_l421_42128
