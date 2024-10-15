import Mathlib

namespace NUMINAMATH_GPT_total_pots_needed_l1749_174912

theorem total_pots_needed
    (p : ℕ) (s : ℕ) (h : ℕ)
    (hp : p = 5)
    (hs : s = 3)
    (hh : h = 4) :
    p * s * h = 60 := by
  sorry

end NUMINAMATH_GPT_total_pots_needed_l1749_174912


namespace NUMINAMATH_GPT_compute_j_in_polynomial_arithmetic_progression_l1749_174944

theorem compute_j_in_polynomial_arithmetic_progression 
  (P : Polynomial ℝ)
  (roots : Fin 4 → ℝ)
  (hP : P = Polynomial.C 400 + Polynomial.X * (Polynomial.C k + Polynomial.X * (Polynomial.C j + Polynomial.X * (Polynomial.C 0 + Polynomial.X))))
  (arithmetic_progression : ∃ b d : ℝ, roots 0 = b ∧ roots 1 = b + d ∧ roots 2 = b + 2 * d ∧ roots 3 = b + 3 * d ∧ Polynomial.degree P = 4) :
  j = -200 :=
by
  sorry

end NUMINAMATH_GPT_compute_j_in_polynomial_arithmetic_progression_l1749_174944


namespace NUMINAMATH_GPT_ratio_of_areas_l1749_174956

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1749_174956


namespace NUMINAMATH_GPT_lowest_two_digit_number_whose_digits_product_is_12_l1749_174950

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n < 100 ∧ ∃ d1 d2 : ℕ, 1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ n = 10 * d1 + d2 ∧ d1 * d2 = 12

theorem lowest_two_digit_number_whose_digits_product_is_12 :
  ∃ n : ℕ, is_valid_two_digit_number n ∧ ∀ m : ℕ, is_valid_two_digit_number m → n ≤ m ∧ n = 26 :=
sorry

end NUMINAMATH_GPT_lowest_two_digit_number_whose_digits_product_is_12_l1749_174950


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_l1749_174965

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase : {x : ℝ | -1 ≤ x} ⊆ {x : ℝ | 0 < deriv f x} :=
by
  sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_l1749_174965


namespace NUMINAMATH_GPT_max_b_c_l1749_174948

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end NUMINAMATH_GPT_max_b_c_l1749_174948


namespace NUMINAMATH_GPT_unicorn_journey_length_l1749_174957

theorem unicorn_journey_length (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_flowers : ℕ) (step_length_meters : ℕ) : (num_unicorns = 6) → (flowers_per_step = 4) → (total_flowers = 72000) → (step_length_meters = 3) → 
(total_flowers / flowers_per_step / num_unicorns * step_length_meters / 1000 = 9) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_unicorn_journey_length_l1749_174957


namespace NUMINAMATH_GPT_dresser_clothing_capacity_l1749_174938

theorem dresser_clothing_capacity (pieces_per_drawer : ℕ) (number_of_drawers : ℕ) (total_pieces : ℕ) 
  (h1 : pieces_per_drawer = 5)
  (h2 : number_of_drawers = 8)
  (h3 : total_pieces = 40) :
  pieces_per_drawer * number_of_drawers = total_pieces :=
by {
  sorry
}

end NUMINAMATH_GPT_dresser_clothing_capacity_l1749_174938


namespace NUMINAMATH_GPT_stock_worth_l1749_174997

theorem stock_worth (profit_part loss_part total_loss : ℝ) 
  (h1 : profit_part = 0.10) 
  (h2 : loss_part = 0.90) 
  (h3 : total_loss = 400) 
  (profit_rate : ℝ := 0.20) 
  (loss_rate : ℝ := 0.05)
  (profit_value := profit_rate * profit_part)
  (loss_value := loss_rate * loss_part)
  (overall_loss := total_loss)
  (h4 : loss_value - profit_value = overall_loss) :
  ∃ X : ℝ, X = 16000 :=
by
  sorry

end NUMINAMATH_GPT_stock_worth_l1749_174997


namespace NUMINAMATH_GPT_largest_y_coordinate_ellipse_l1749_174919

theorem largest_y_coordinate_ellipse (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 := 
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_largest_y_coordinate_ellipse_l1749_174919


namespace NUMINAMATH_GPT_sum_of_integer_pair_l1749_174908

theorem sum_of_integer_pair (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 10) (h3 : 1 ≤ b) (h4 : b ≤ 10) (h5 : a * b = 14) : a + b = 9 := 
sorry

end NUMINAMATH_GPT_sum_of_integer_pair_l1749_174908


namespace NUMINAMATH_GPT_coefficient_x_is_five_l1749_174998

theorem coefficient_x_is_five (x y a : ℤ) (h1 : a * x + y = 19) (h2 : x + 3 * y = 1) (h3 : 3 * x + 2 * y = 10) : a = 5 :=
by sorry

end NUMINAMATH_GPT_coefficient_x_is_five_l1749_174998


namespace NUMINAMATH_GPT_sin_gt_sub_cubed_l1749_174929

theorem sin_gt_sub_cubed (x : ℝ) (h₀ : 0 < x) (h₁ : x < Real.pi / 2) : 
  Real.sin x > x - x^3 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_sin_gt_sub_cubed_l1749_174929


namespace NUMINAMATH_GPT_average_ABC_eq_2A_plus_3_l1749_174963

theorem average_ABC_eq_2A_plus_3 (A B C : ℝ) 
  (h1 : 2023 * C - 4046 * A = 8092) 
  (h2 : 2023 * B - 6069 * A = 10115) : 
  (A + B + C) / 3 = 2 * A + 3 :=
sorry

end NUMINAMATH_GPT_average_ABC_eq_2A_plus_3_l1749_174963


namespace NUMINAMATH_GPT_find_sum_x1_x2_l1749_174976

-- Define sets A and B with given properties
def set_A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def set_B (x1 x2 : ℝ) : Set ℝ := {x | x1 ≤ x ∧ x ≤ x2}

-- Conditions of union and intersection
def union_condition (x1 x2 : ℝ) : Prop := set_A ∪ set_B x1 x2 = {x | x > -2}
def intersection_condition (x1 x2 : ℝ) : Prop := set_A ∩ set_B x1 x2 = {x | 1 < x ∧ x ≤ 3}

-- Main theorem to prove
theorem find_sum_x1_x2 (x1 x2 : ℝ) (h_union : union_condition x1 x2) (h_intersect : intersection_condition x1 x2) :
  x1 + x2 = 2 :=
sorry

end NUMINAMATH_GPT_find_sum_x1_x2_l1749_174976


namespace NUMINAMATH_GPT_omega_range_l1749_174911

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end NUMINAMATH_GPT_omega_range_l1749_174911


namespace NUMINAMATH_GPT_correct_operation_l1749_174978

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end NUMINAMATH_GPT_correct_operation_l1749_174978


namespace NUMINAMATH_GPT_fractional_equation_solution_l1749_174921

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end NUMINAMATH_GPT_fractional_equation_solution_l1749_174921


namespace NUMINAMATH_GPT_chord_length_cube_l1749_174927

noncomputable def diameter : ℝ := 1
noncomputable def AC (a : ℝ) : ℝ := a
noncomputable def AD (b : ℝ) : ℝ := b
noncomputable def AE (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / 2
noncomputable def AF (b : ℝ) : ℝ := b^2

theorem chord_length_cube (a b : ℝ) (h : AE a b = b^2) : a = b^3 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_cube_l1749_174927


namespace NUMINAMATH_GPT_inequality_check_l1749_174909

theorem inequality_check : (-1 : ℝ) / 3 < -1 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_check_l1749_174909


namespace NUMINAMATH_GPT_regular_polygon_sides_l1749_174988

theorem regular_polygon_sides (exterior_angle : ℕ) (h : exterior_angle = 30) : (360 / exterior_angle) = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1749_174988


namespace NUMINAMATH_GPT_remainder_3_pow_2023_mod_5_l1749_174900

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_2023_mod_5_l1749_174900


namespace NUMINAMATH_GPT_total_players_on_ground_l1749_174980

theorem total_players_on_ground :
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  cricket_players + hockey_players + football_players + softball_players +
  basketball_players + volleyball_players + netball_players + rugby_players = 263 := 
by 
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  sorry

end NUMINAMATH_GPT_total_players_on_ground_l1749_174980


namespace NUMINAMATH_GPT_tutors_next_together_in_360_days_l1749_174943

open Nat

-- Define the intervals for each tutor
def evan_interval := 5
def fiona_interval := 6
def george_interval := 9
def hannah_interval := 8
def ian_interval := 10

-- Statement to prove
theorem tutors_next_together_in_360_days :
  Nat.lcm (Nat.lcm evan_interval fiona_interval) (Nat.lcm george_interval (Nat.lcm hannah_interval ian_interval)) = 360 :=
by
  sorry

end NUMINAMATH_GPT_tutors_next_together_in_360_days_l1749_174943


namespace NUMINAMATH_GPT_exists_triangle_sides_l1749_174935

theorem exists_triangle_sides (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c ≤ 1 / 4)
  (h2 : 1 / (a^2) + 1 / (b^2) + 1 / (c^2) < 9) : 
  a + b > c ∧ b + c > a ∧ c + a > b := 
by
  sorry

end NUMINAMATH_GPT_exists_triangle_sides_l1749_174935


namespace NUMINAMATH_GPT_sheilas_family_contribution_l1749_174937

theorem sheilas_family_contribution :
  let initial_amount := 3000
  let monthly_savings := 276
  let duration_years := 4
  let total_after_duration := 23248
  let months_in_year := 12
  let total_months := duration_years * months_in_year
  let savings_over_duration := monthly_savings * total_months
  let sheilas_total_savings := initial_amount + savings_over_duration
  let family_contribution := total_after_duration - sheilas_total_savings
  family_contribution = 7000 :=
by
  sorry

end NUMINAMATH_GPT_sheilas_family_contribution_l1749_174937


namespace NUMINAMATH_GPT_add_complex_eq_required_complex_addition_l1749_174918

theorem add_complex_eq (a b c d : ℝ) (i : ℂ) (h : i ^ 2 = -1) :
  (a + b * i) + (c + d * i) = (a + c) + (b + d) * i :=
by sorry

theorem required_complex_addition :
  let a : ℂ := 5 - 3 * i
  let b : ℂ := 2 + 12 * i
  a + b = 7 + 9 * i := 
by sorry

end NUMINAMATH_GPT_add_complex_eq_required_complex_addition_l1749_174918


namespace NUMINAMATH_GPT_problem_statement_l1749_174973

variables {A B C O D : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (a b c o d : A)

-- Define the geometric conditions
axiom condition1 : a + 2 • b + 3 • c = 0
axiom condition2 : ∃ (D: A), (∃ (k : ℝ), a = k • d ∧ k ≠ 0) ∧ (∃ (u v : ℝ),  u • b + v • c = d ∧ u + v = 1)

-- Define points
def OA : A := a - o
def OB : A := b - o
def OC : A := c - o
def OD : A := d - o

-- The main statement to prove
theorem problem_statement : 2 • (b - d) + 3 • (c - d) = (0 : A) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1749_174973


namespace NUMINAMATH_GPT_contrapositive_statement_l1749_174969

-- Definitions derived from conditions
def Triangle (ABC : Type) : Prop := 
  ∃ a b c : ABC, true

def IsIsosceles (ABC : Type) : Prop :=
  ∃ a b c : ABC, a = b ∨ b = c ∨ a = c

def InteriorAnglesNotEqual (ABC : Type) : Prop :=
  ∀ a b : ABC, a ≠ b

-- The contrapositive implication we need to prove
theorem contrapositive_statement (ABC : Type) (h : Triangle ABC) 
  (h_not_isosceles_implies_not_equal : ¬IsIsosceles ABC → InteriorAnglesNotEqual ABC) :
  (∃ a b c : ABC, a = b → IsIsosceles ABC) := 
sorry

end NUMINAMATH_GPT_contrapositive_statement_l1749_174969


namespace NUMINAMATH_GPT_required_fencing_l1749_174968

-- Define constants given in the problem
def L : ℕ := 20
def A : ℕ := 720

-- Define the width W based on the area and the given length L
def W : ℕ := A / L

-- Define the total amount of fencing required
def F : ℕ := 2 * W + L

-- State the theorem that this amount of fencing is equal to 92
theorem required_fencing : F = 92 := by
  sorry

end NUMINAMATH_GPT_required_fencing_l1749_174968


namespace NUMINAMATH_GPT_range_of_a_l1749_174942

variable (a x : ℝ)

-- Condition p: ∀ x ∈ [1, 2], x^2 - a ≥ 0
def p : Prop := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition q: ∃ x ∈ ℝ, x^2 + 2 * a * x + 2 - a = 0
def q : Prop := ∃ x, x^2 + 2 * a * x + 2 - a = 0

-- The proof goal given p ∧ q: a ≤ -2 or a = 1
theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := sorry

end NUMINAMATH_GPT_range_of_a_l1749_174942


namespace NUMINAMATH_GPT_simplify_fraction_l1749_174924

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1749_174924


namespace NUMINAMATH_GPT_sum_of_products_eq_131_l1749_174945

theorem sum_of_products_eq_131 (a b c : ℝ) 
    (h1 : a^2 + b^2 + c^2 = 222)
    (h2 : a + b + c = 22) : 
    a * b + b * c + c * a = 131 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_products_eq_131_l1749_174945


namespace NUMINAMATH_GPT_ratio_of_x_and_y_l1749_174983

theorem ratio_of_x_and_y (x y : ℤ) (h : (3 * x - 2 * y) * 4 = 3 * (2 * x + y)) : (x : ℚ) / y = 11 / 6 :=
  sorry

end NUMINAMATH_GPT_ratio_of_x_and_y_l1749_174983


namespace NUMINAMATH_GPT_calculate_expression_l1749_174928

theorem calculate_expression :
  3^(1+2+3) - (3^1 + 3^2 + 3^3) = 690 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1749_174928


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l1749_174949

/-- Given an arithmetic sequence {a_n}, where S₁₀ = 60 and a₇ = 7, prove that a₄ = 5. -/
theorem arithmetic_sequence_a4 (a₁ d : ℝ) 
  (h1 : 10 * a₁ + 45 * d = 60) 
  (h2 : a₁ + 6 * d = 7) : 
  a₁ + 3 * d = 5 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_l1749_174949


namespace NUMINAMATH_GPT_fraction_subtraction_l1749_174930

theorem fraction_subtraction : (5 / 6 + 1 / 4 - 2 / 3) = (5 / 12) := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l1749_174930


namespace NUMINAMATH_GPT_johnny_age_multiple_l1749_174955

theorem johnny_age_multiple
  (current_age : ℕ)
  (age_in_2_years : ℕ)
  (age_3_years_ago : ℕ)
  (k : ℕ)
  (h1 : current_age = 8)
  (h2 : age_in_2_years = current_age + 2)
  (h3 : age_3_years_ago = current_age - 3)
  (h4 : age_in_2_years = k * age_3_years_ago) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_johnny_age_multiple_l1749_174955


namespace NUMINAMATH_GPT_inverse_h_l1749_174986

-- Definitions from the problem conditions
def f (x : ℝ) : ℝ := 4 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 5
def h (x : ℝ) : ℝ := f (g x)

-- Statement of the theorem for the inverse of h
theorem inverse_h : ∀ x : ℝ, h⁻¹ x = (x + 18) / 12 :=
sorry

end NUMINAMATH_GPT_inverse_h_l1749_174986


namespace NUMINAMATH_GPT_problem_statement_l1749_174962

def a : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (2, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement : dot_product (vector_sub a b) a = 0 := 
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_problem_statement_l1749_174962


namespace NUMINAMATH_GPT_Jake_has_8_peaches_l1749_174971

variable (Jake Steven Jill : ℕ)

theorem Jake_has_8_peaches
  (h_steven_peaches : Steven = 15)
  (h_steven_jill : Steven = Jill + 14)
  (h_jake_steven : Jake = Steven - 7) :
  Jake = 8 := by
  sorry

end NUMINAMATH_GPT_Jake_has_8_peaches_l1749_174971


namespace NUMINAMATH_GPT_monica_usd_start_amount_l1749_174925

theorem monica_usd_start_amount (x : ℕ) (H : ∃ (y : ℕ), y = 40 ∧ (8 : ℚ) / 5 * x - y = x) :
  (x / 100) + (x % 100 / 10) + (x % 10) = 2 := 
by
  sorry

end NUMINAMATH_GPT_monica_usd_start_amount_l1749_174925


namespace NUMINAMATH_GPT_elapsed_time_l1749_174979

theorem elapsed_time (x : ℕ) (h1 : 99 > 0) (h2 : (2 : ℚ) / (3 : ℚ) * x = (4 : ℚ) / (5 : ℚ) * (99 - x)) : x = 54 := by
  sorry

end NUMINAMATH_GPT_elapsed_time_l1749_174979


namespace NUMINAMATH_GPT_solve_inequality_l1749_174964

theorem solve_inequality :
  {x : ℝ | x^2 - 9 * x + 14 < 0} = {x : ℝ | 2 < x ∧ x < 7} := sorry

end NUMINAMATH_GPT_solve_inequality_l1749_174964


namespace NUMINAMATH_GPT_scout_hours_worked_l1749_174995

variable (h : ℕ) -- number of hours worked on Saturday
variable (base_pay : ℕ) -- base pay per hour
variable (tip_per_customer : ℕ) -- tip per customer
variable (saturday_customers : ℕ) -- customers served on Saturday
variable (sunday_hours : ℕ) -- hours worked on Sunday
variable (sunday_customers : ℕ) -- customers served on Sunday
variable (total_earnings : ℕ) -- total earnings over the weekend

theorem scout_hours_worked {h : ℕ} (base_pay : ℕ) (tip_per_customer : ℕ) (saturday_customers : ℕ) (sunday_hours : ℕ) (sunday_customers : ℕ) (total_earnings : ℕ) :
  base_pay = 10 → 
  tip_per_customer = 5 → 
  saturday_customers = 5 → 
  sunday_hours = 5 → 
  sunday_customers = 8 → 
  total_earnings = 155 → 
  10 * h + 5 * 5 + 10 * 5 + 5 * 8 = 155 → 
  h = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_scout_hours_worked_l1749_174995


namespace NUMINAMATH_GPT_solve_for_x_l1749_174951

theorem solve_for_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : 
  x = 2 := 
by sorry

end NUMINAMATH_GPT_solve_for_x_l1749_174951


namespace NUMINAMATH_GPT_quincy_monthly_payment_l1749_174993

-- Definitions based on the conditions:
def car_price : ℕ := 20000
def down_payment : ℕ := 5000
def loan_years : ℕ := 5
def months_in_year : ℕ := 12

-- The mathematical problem to be proven:
theorem quincy_monthly_payment :
  let amount_to_finance := car_price - down_payment
  let total_months := loan_years * months_in_year
  amount_to_finance / total_months = 250 := by
  sorry

end NUMINAMATH_GPT_quincy_monthly_payment_l1749_174993


namespace NUMINAMATH_GPT_remainder_division_P_by_D_l1749_174916

def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15
def D (x : ℝ) := 4 * x - 8

theorem remainder_division_P_by_D :
  let remainder := P 2 % D 2
  remainder = 31 :=
by
  -- Proof will be inserted here, but currently skipped
  sorry

end NUMINAMATH_GPT_remainder_division_P_by_D_l1749_174916


namespace NUMINAMATH_GPT_ellipse_hyperbola_foci_l1749_174987

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), b^2 - a^2 = 25 ∧ a^2 + b^2 = 64) : 
  |a * b| = (Real.sqrt 3471) / 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_hyperbola_foci_l1749_174987


namespace NUMINAMATH_GPT_smallest_nat_number_l1749_174958

theorem smallest_nat_number (n : ℕ) (h1 : ∃ a, 0 ≤ a ∧ a < 20 ∧ n % 20 = a ∧ n % 21 = a + 1) (h2 : n % 22 = 2) : n = 838 := by 
  sorry

end NUMINAMATH_GPT_smallest_nat_number_l1749_174958


namespace NUMINAMATH_GPT_monthly_cost_per_person_is_1000_l1749_174960

noncomputable def john_pays : ℝ := 32000
noncomputable def initial_fee_per_person : ℝ := 4000
noncomputable def total_people : ℝ := 4
noncomputable def john_pays_half : Prop := true

theorem monthly_cost_per_person_is_1000 :
  john_pays_half →
  (john_pays * 2 - (initial_fee_per_person * total_people)) / (total_people * 12) = 1000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_monthly_cost_per_person_is_1000_l1749_174960


namespace NUMINAMATH_GPT_waste_scientific_notation_correct_l1749_174904

def total_waste_in_scientific : ℕ := 500000000000

theorem waste_scientific_notation_correct :
  total_waste_in_scientific = 5 * 10^10 :=
by
  sorry

end NUMINAMATH_GPT_waste_scientific_notation_correct_l1749_174904


namespace NUMINAMATH_GPT_k_plus_alpha_is_one_l1749_174954

variable (f : ℝ → ℝ) (k α : ℝ)

-- Conditions from part a)
def power_function := ∀ x : ℝ, f x = k * x ^ α
def passes_through_point := f (1 / 2) = 2

-- Statement to be proven
theorem k_plus_alpha_is_one (h1 : power_function f k α) (h2 : passes_through_point f) : k + α = 1 :=
sorry

end NUMINAMATH_GPT_k_plus_alpha_is_one_l1749_174954


namespace NUMINAMATH_GPT_find_n_l1749_174936

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end NUMINAMATH_GPT_find_n_l1749_174936


namespace NUMINAMATH_GPT_product_greater_than_sum_l1749_174989

variable {a b : ℝ}

theorem product_greater_than_sum (ha : a > 2) (hb : b > 2) : a * b > a + b := 
  sorry

end NUMINAMATH_GPT_product_greater_than_sum_l1749_174989


namespace NUMINAMATH_GPT_calculate_initial_money_l1749_174992

noncomputable def initial_money (remaining_money: ℝ) (spent_percent: ℝ) : ℝ :=
  remaining_money / (1 - spent_percent)

theorem calculate_initial_money :
  initial_money 3500 0.30 = 5000 := 
by
  rw [initial_money]
  sorry

end NUMINAMATH_GPT_calculate_initial_money_l1749_174992


namespace NUMINAMATH_GPT_quadratic_min_value_max_l1749_174914

theorem quadratic_min_value_max (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 - 4 * a * c ≥ 0) :
    (min (min ((b + c) / a) ((c + a) / b)) ((a + b) / c)) ≤ (5 / 4) :=
sorry

end NUMINAMATH_GPT_quadratic_min_value_max_l1749_174914


namespace NUMINAMATH_GPT_rachel_picked_apples_l1749_174903

-- Defining the conditions
def original_apples : ℕ := 11
def grown_apples : ℕ := 2
def apples_left : ℕ := 6

-- Defining the equation
def equation (x : ℕ) : Prop :=
  original_apples - x + grown_apples = apples_left

-- Stating the theorem
theorem rachel_picked_apples : ∃ x : ℕ, equation x ∧ x = 7 :=
by 
  -- proof skipped 
  sorry

end NUMINAMATH_GPT_rachel_picked_apples_l1749_174903


namespace NUMINAMATH_GPT_remaining_lawn_area_l1749_174923

theorem remaining_lawn_area (lawn_length lawn_width path_width : ℕ) 
  (h_lawn_length : lawn_length = 10) 
  (h_lawn_width : lawn_width = 5) 
  (h_path_width : path_width = 1) : 
  (lawn_length * lawn_width - lawn_length * path_width) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_lawn_area_l1749_174923


namespace NUMINAMATH_GPT_total_number_of_animals_l1749_174906

-- Define the problem conditions
def number_of_cats : ℕ := 645
def number_of_dogs : ℕ := 567

-- State the theorem to be proved
theorem total_number_of_animals : number_of_cats + number_of_dogs = 1212 := by
  sorry

end NUMINAMATH_GPT_total_number_of_animals_l1749_174906


namespace NUMINAMATH_GPT_metal_waste_l1749_174953

theorem metal_waste (l w : ℝ) (h : l > w) :
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  wasted_metal = l * w - w ^ 2 / 2 :=
by
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  sorry

end NUMINAMATH_GPT_metal_waste_l1749_174953


namespace NUMINAMATH_GPT_number_of_moles_of_water_formed_l1749_174970

def balanced_combustion_equation : Prop :=
  ∀ (CH₄ O₂ CO₂ H₂O : ℕ), (CH₄ + 2 * O₂ = CO₂ + 2 * H₂O)

theorem number_of_moles_of_water_formed
  (CH₄_initial moles_of_CH₄ O₂_initial moles_of_O₂ : ℕ)
  (h_CH₄_initial : CH₄_initial = 3)
  (h_O₂_initial : O₂_initial = 6)
  (h_moles_of_H₂O : moles_of_CH₄ * 2 = 2 * moles_of_H₂O) :
  moles_of_H₂O = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_moles_of_water_formed_l1749_174970


namespace NUMINAMATH_GPT_least_prime_factor_five_power_difference_l1749_174915

theorem least_prime_factor_five_power_difference : 
  ∃ p : ℕ, (Nat.Prime p ∧ p ∣ (5^4 - 5^3)) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (5^4 - 5^3) → p ≤ q) := 
sorry

end NUMINAMATH_GPT_least_prime_factor_five_power_difference_l1749_174915


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l1749_174946

theorem hyperbola_eccentricity (m : ℝ) (h1: ∃ x y : ℝ, (x^2 / 3) - (y^2 / m) = 1) (h2: ∀ a b : ℝ, a^2 = 3 ∧ b^2 = m ∧ (2 = Real.sqrt (1 + b^2 / a^2))) : m = -9 := 
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l1749_174946


namespace NUMINAMATH_GPT_books_arrangement_l1749_174922

/-
  Theorem:
  If there are 4 distinct math books, 6 distinct English books, and 3 distinct science books,
  and each category of books must stay together, then the number of ways to arrange
  these books on a shelf is 622080.
-/

def num_math_books : ℕ := 4
def num_english_books : ℕ := 6
def num_science_books : ℕ := 3

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_arrangements :=
  factorial 3 * factorial num_math_books * factorial num_english_books * factorial num_science_books

theorem books_arrangement : num_arrangements = 622080 := by
  sorry

end NUMINAMATH_GPT_books_arrangement_l1749_174922


namespace NUMINAMATH_GPT_total_rainbow_nerds_is_36_l1749_174984

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end NUMINAMATH_GPT_total_rainbow_nerds_is_36_l1749_174984


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l1749_174934

theorem distance_between_foci_of_hyperbola (a b c : ℝ) : (x^2 - y^2 = 4) → (a = 2) → (b = 0) → (c = Real.sqrt (4 + 0)) → 
    dist (2, 0) (-2, 0) = 4 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l1749_174934


namespace NUMINAMATH_GPT_no_real_solution_l1749_174975

noncomputable def quadratic_eq (x : ℝ) : ℝ := (2*x^2 - 3*x + 5)

theorem no_real_solution : 
  ∀ x : ℝ, quadratic_eq x ^ 2 + 1 ≠ 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solution_l1749_174975


namespace NUMINAMATH_GPT_impossible_event_D_l1749_174967

-- Event definitions
def event_A : Prop := true -- This event is not impossible
def event_B : Prop := true -- This event is not impossible
def event_C : Prop := true -- This event is not impossible
def event_D (bag : Finset String) : Prop :=
  if "red" ∈ bag then false else true -- This event is impossible if there are no red balls

-- Bag condition
def bag : Finset String := {"white", "white", "white", "white", "white", "white", "white", "white"}

-- Proof statement
theorem impossible_event_D : event_D bag = true :=
by
  -- The bag contains only white balls, so drawing a red ball is impossible.
  rw [event_D, if_neg]
  sorry

end NUMINAMATH_GPT_impossible_event_D_l1749_174967


namespace NUMINAMATH_GPT_unique_solution_system_eqns_l1749_174972

theorem unique_solution_system_eqns :
  ∃ (x y : ℝ), (2 * x - 3 * |y| = 1 ∧ |x| + 2 * y = 4 ∧ x = 2 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_unique_solution_system_eqns_l1749_174972


namespace NUMINAMATH_GPT_solve_ineq_system_l1749_174947

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_GPT_solve_ineq_system_l1749_174947


namespace NUMINAMATH_GPT_eval_expression_l1749_174926

theorem eval_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 2) * Real.cos (Real.pi + 2)) = Real.sin 2 - Real.cos 2 :=
sorry

end NUMINAMATH_GPT_eval_expression_l1749_174926


namespace NUMINAMATH_GPT_grasshoppers_total_l1749_174917

theorem grasshoppers_total (grasshoppers_on_plant : ℕ) (dozens_of_baby_grasshoppers : ℕ) (dozen_value : ℕ) : 
  grasshoppers_on_plant = 7 → dozens_of_baby_grasshoppers = 2 → dozen_value = 12 → 
  grasshoppers_on_plant + dozens_of_baby_grasshoppers * dozen_value = 31 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_grasshoppers_total_l1749_174917


namespace NUMINAMATH_GPT_difference_in_speed_l1749_174961

theorem difference_in_speed (d : ℕ) (tA tE : ℕ) (vA vE : ℕ) (h1 : d = 300) (h2 : tA = tE - 3) 
    (h3 : vE = 20) (h4 : vE = d / tE) (h5 : vA = d / tA) : vA - vE = 5 := 
    sorry

end NUMINAMATH_GPT_difference_in_speed_l1749_174961


namespace NUMINAMATH_GPT_max_number_of_different_ages_l1749_174933

theorem max_number_of_different_ages
  (a : ℤ) (s : ℤ)
  (h1 : a = 31)
  (h2 : s = 5) :
  ∃ n : ℕ, n = (36 - 26 + 1) :=
by sorry

end NUMINAMATH_GPT_max_number_of_different_ages_l1749_174933


namespace NUMINAMATH_GPT_train_crossing_time_l1749_174939

variable (length_train : ℝ) (time_pole : ℝ) (length_platform : ℝ) (time_platform : ℝ)

-- Given conditions
def train_conditions := 
  length_train = 300 ∧
  time_pole = 14 ∧
  length_platform = 535.7142857142857

-- Theorem statement
theorem train_crossing_time (h : train_conditions length_train time_pole length_platform) :
  time_platform = 39 := sorry

end NUMINAMATH_GPT_train_crossing_time_l1749_174939


namespace NUMINAMATH_GPT_sequence_equality_l1749_174920

theorem sequence_equality (a : Fin 1973 → ℝ) (hpos : ∀ n, a n > 0)
  (heq : a 0 ^ a 0 = a 1 ^ a 2 ∧ a 1 ^ a 2 = a 2 ^ a 3 ∧ 
         a 2 ^ a 3 = a 3 ^ a 4 ∧ 
         -- etc., continued for all indices, 
         -- ensuring last index correctly refers back to a 0
         a 1971 ^ a 1972 = a 1972 ^ a 0) :
  a 0 = a 1972 :=
sorry

end NUMINAMATH_GPT_sequence_equality_l1749_174920


namespace NUMINAMATH_GPT_arithmetic_expression_l1749_174990

theorem arithmetic_expression : (4 + 6 + 4) / 3 - 4 / 3 = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l1749_174990


namespace NUMINAMATH_GPT_solve_equation_l1749_174902

theorem solve_equation : ∀ x : ℝ, -2 * x + 11 = 0 → x = 11 / 2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l1749_174902


namespace NUMINAMATH_GPT_sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l1749_174907

open Real

theorem sqrt_1_plus_inv_squares_4_5 :
  sqrt (1 + 1/4^2 + 1/5^2) = 1 + 1/20 :=
by
  sorry

theorem sqrt_1_plus_inv_squares_general (n : ℕ) (h : 0 < n) :
  sqrt (1 + 1/n^2 + 1/(n+1)^2) = 1 + 1/(n * (n + 1)) :=
by
  sorry

theorem sqrt_101_100_plus_1_121 :
  sqrt (101/100 + 1/121) = 1 + 1/110 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l1749_174907


namespace NUMINAMATH_GPT_product_units_tens_not_divisible_by_5_l1749_174932

-- Define the list of four-digit numbers
def numbers : List ℕ := [4750, 4760, 4775, 4785, 4790]

-- Define a function to check if a number is divisible by 5
def divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

-- Define a function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Define a function to extract the tens digit of a number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Statement: The product of the units digit and the tens digit of the number
-- that is not divisible by 5 in the list is 0
theorem product_units_tens_not_divisible_by_5 : 
  ∃ n ∈ numbers, ¬divisible_by_5 n ∧ (units_digit n * tens_digit n = 0) :=
by sorry

end NUMINAMATH_GPT_product_units_tens_not_divisible_by_5_l1749_174932


namespace NUMINAMATH_GPT_distance_and_speed_l1749_174913

-- Define the conditions given in the problem
def first_car_speed (y : ℕ) := y + 4
def second_car_speed (y : ℕ) := y
def third_car_speed (y : ℕ) := y - 6

def time_relation1 (x : ℕ) (y : ℕ) :=
  x / (first_car_speed y) = x / (second_car_speed y) - 3 / 60

def time_relation2 (x : ℕ) (y : ℕ) :=
  x / (second_car_speed y) = x / (third_car_speed y) - 5 / 60 

-- State the theorem to prove both the distance and the speed of the second car
theorem distance_and_speed : ∃ (x y : ℕ), 
  time_relation1 x y ∧ 
  time_relation2 x y ∧ 
  x = 120 ∧ 
  y = 96 :=
by
  sorry

end NUMINAMATH_GPT_distance_and_speed_l1749_174913


namespace NUMINAMATH_GPT_interest_rate_second_share_l1749_174981

variable (T : ℝ) (r1 : ℝ) (I2 : ℝ) (T_i : ℝ)

theorem interest_rate_second_share 
  (h1 : T = 100000)
  (h2 : r1 = 0.09)
  (h3 : I2 = 24999.999999999996)
  (h4 : T_i = 0.095 * T) : 
  (2750 / I2) * 100 = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_interest_rate_second_share_l1749_174981


namespace NUMINAMATH_GPT_person_died_at_33_l1749_174982

-- Define the conditions and constants
def start_age : ℕ := 25
def insurance_payment : ℕ := 10000
def premium : ℕ := 450
def loss : ℕ := 1000
def annual_interest_rate : ℝ := 0.05
def half_year_factor : ℝ := 1.025 -- half-yearly compounded interest factor

-- Calculate the number of premium periods (as an integer)
def n := 16 -- (derived from the calculations in the given solution)

-- Define the final age based on the number of premium periods
def final_age : ℕ := start_age + (n / 2)

-- The proof statement
theorem person_died_at_33 : final_age = 33 := by
  sorry

end NUMINAMATH_GPT_person_died_at_33_l1749_174982


namespace NUMINAMATH_GPT_ball_box_arrangement_l1749_174999

-- Given n distinguishable balls and m distinguishable boxes,
-- prove that the number of ways to place the n balls into the m boxes is m^n.
-- Specifically for n = 6 and m = 3.

theorem ball_box_arrangement : (3^6 = 729) :=
by
  sorry

end NUMINAMATH_GPT_ball_box_arrangement_l1749_174999


namespace NUMINAMATH_GPT_orange_juice_fraction_l1749_174952

def capacity_small_pitcher := 500 -- mL
def orange_juice_fraction_small := 1 / 4
def capacity_large_pitcher := 800 -- mL
def orange_juice_fraction_large := 1 / 2

def total_orange_juice_volume := 
  (capacity_small_pitcher * orange_juice_fraction_small) + 
  (capacity_large_pitcher * orange_juice_fraction_large)
def total_volume := capacity_small_pitcher + capacity_large_pitcher

theorem orange_juice_fraction :
  (total_orange_juice_volume / total_volume) = (21 / 52) := 
by 
  sorry

end NUMINAMATH_GPT_orange_juice_fraction_l1749_174952


namespace NUMINAMATH_GPT_prove_A_annual_savings_l1749_174905

noncomputable def employee_A_annual_savings
  (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02) : ℝ :=
  let total_deductions := tax_rate + pension_rate + healthcare_rate
  let Income_after_deductions := A_income * (1 - total_deductions)
  let annual_savings := 12 * Income_after_deductions
  annual_savings

theorem prove_A_annual_savings : 
  ∀ (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02),
  employee_A_annual_savings A_income B_income C_income D_income C_income_val income_ratio tax_rate pension_rate healthcare_rate tax_rate_val pension_rate_val healthcare_rate_val = 232400.16 :=
by
  sorry

end NUMINAMATH_GPT_prove_A_annual_savings_l1749_174905


namespace NUMINAMATH_GPT_ratio_expression_value_l1749_174994

variable {A B C : ℚ}

theorem ratio_expression_value (h : A / B = 3 / 2 ∧ A / C = 3 / 6) : (4 * A - 3 * B) / (5 * C + 2 * A) = 1 / 4 := 
sorry

end NUMINAMATH_GPT_ratio_expression_value_l1749_174994


namespace NUMINAMATH_GPT_train_speed_l1749_174940

theorem train_speed (d t : ℝ) (h1 : d = 500) (h2 : t = 3) : d / t = 166.67 := by
  sorry

end NUMINAMATH_GPT_train_speed_l1749_174940


namespace NUMINAMATH_GPT_emily_final_score_l1749_174966

theorem emily_final_score :
  16 + 33 - 48 = 1 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_emily_final_score_l1749_174966


namespace NUMINAMATH_GPT_solve_for_x_l1749_174941

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1749_174941


namespace NUMINAMATH_GPT_apple_cost_price_orange_cost_price_banana_cost_price_l1749_174977

theorem apple_cost_price (A : ℚ) : 15 = A - (1/6 * A) → A = 18 := by
  intro h
  sorry

theorem orange_cost_price (O : ℚ) : 20 = O + (1/5 * O) → O = 100/6 := by
  intro h
  sorry

theorem banana_cost_price (B : ℚ) : 10 = B → B = 10 := by
  intro h
  sorry

end NUMINAMATH_GPT_apple_cost_price_orange_cost_price_banana_cost_price_l1749_174977


namespace NUMINAMATH_GPT_pizza_pieces_per_person_l1749_174901

theorem pizza_pieces_per_person (total_people : ℕ) (fraction_eat : ℚ) (total_pizza : ℕ) (remaining_pizza : ℕ)
  (H1 : total_people = 15) (H2 : fraction_eat = 3/5) (H3 : total_pizza = 50) (H4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (fraction_eat * total_people) = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_pizza_pieces_per_person_l1749_174901


namespace NUMINAMATH_GPT_sofa_price_is_correct_l1749_174996

def price_sofa (invoice_total armchair_price table_price : ℕ) (armchair_count : ℕ) : ℕ :=
  invoice_total - (armchair_price * armchair_count + table_price)

theorem sofa_price_is_correct
  (invoice_total : ℕ)
  (armchair_price : ℕ)
  (table_price : ℕ)
  (armchair_count : ℕ)
  (sofa_price : ℕ)
  (h_invoice : invoice_total = 2430)
  (h_armchair_price : armchair_price = 425)
  (h_table_price : table_price = 330)
  (h_armchair_count : armchair_count = 2)
  (h_sofa_price : sofa_price = 1250) :
  price_sofa invoice_total armchair_price table_price armchair_count = sofa_price :=
by
  sorry

end NUMINAMATH_GPT_sofa_price_is_correct_l1749_174996


namespace NUMINAMATH_GPT_cubic_polynomial_root_sum_cube_value_l1749_174910

noncomputable def α : ℝ := (17 : ℝ)^(1 / 3)
noncomputable def β : ℝ := (67 : ℝ)^(1 / 3)
noncomputable def γ : ℝ := (137 : ℝ)^(1 / 3)

theorem cubic_polynomial_root_sum_cube_value
    (p q r : ℝ)
    (h1 : (p - α) * (p - β) * (p - γ) = 1)
    (h2 : (q - α) * (q - β) * (q - γ) = 1)
    (h3 : (r - α) * (r - β) * (r - γ) = 1) :
    p^3 + q^3 + r^3 = 218 := 
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_root_sum_cube_value_l1749_174910


namespace NUMINAMATH_GPT_min_fencing_cost_l1749_174931

theorem min_fencing_cost {A B C : ℕ} (h1 : A = 25) (h2 : B = 35) (h3 : C = 40)
  (h_ratio : ∃ (x : ℕ), 3 * x * 4 * x = 8748) : 
  ∃ (total_cost : ℝ), total_cost = 87.75 :=
by
  sorry

end NUMINAMATH_GPT_min_fencing_cost_l1749_174931


namespace NUMINAMATH_GPT_jewelry_store_gross_profit_l1749_174991

theorem jewelry_store_gross_profit (purchase_price selling_price new_selling_price gross_profit : ℝ)
    (h1 : purchase_price = 240)
    (h2 : markup = 0.25 * selling_price)
    (h3 : selling_price = purchase_price + markup)
    (h4 : decrease = 0.20 * selling_price)
    (h5 : new_selling_price = selling_price - decrease)
    (h6 : gross_profit = new_selling_price - purchase_price) :
    gross_profit = 16 :=
by
    sorry

end NUMINAMATH_GPT_jewelry_store_gross_profit_l1749_174991


namespace NUMINAMATH_GPT_not_kth_power_l1749_174974

theorem not_kth_power (m k : ℕ) (hk : k > 1) : ¬ ∃ a : ℤ, m * (m + 1) = a^k :=
by
  sorry

end NUMINAMATH_GPT_not_kth_power_l1749_174974


namespace NUMINAMATH_GPT_like_terms_exponents_l1749_174959

theorem like_terms_exponents (m n : ℤ) (h1 : 2 * n - 1 = m) (h2 : m = 3) : m = 3 ∧ n = 2 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_exponents_l1749_174959


namespace NUMINAMATH_GPT_p_work_alone_time_l1749_174985

variable (Wp Wq : ℝ)
variable (x : ℝ)

-- Conditions
axiom h1 : Wp = 1.5 * Wq
axiom h2 : (1 / x) + (Wq / Wp) * (1 / x) = 1 / 15

-- Proof of the question (p alone can complete the work in x days)
theorem p_work_alone_time : x = 25 :=
by
  -- Add your proof here
  sorry

end NUMINAMATH_GPT_p_work_alone_time_l1749_174985
