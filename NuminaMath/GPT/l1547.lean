import Mathlib

namespace NUMINAMATH_GPT_inning_count_l1547_154724

-- Definition of the conditions
variables {n T H L : ℕ}
variables (avg_total : ℕ) (avg_excl : ℕ) (diff : ℕ) (high_score : ℕ)

-- Define the conditions
def conditions :=
  avg_total = 62 ∧
  high_score = 225 ∧
  diff = 150 ∧
  avg_excl = 58

-- Proving the main theorem
theorem inning_count (avg_total := 62) (high_score := 225) (diff := 150) (avg_excl := 58) :
   conditions avg_total avg_excl diff high_score →
   n = 104 :=
sorry

end NUMINAMATH_GPT_inning_count_l1547_154724


namespace NUMINAMATH_GPT_quadratic_roots_l1547_154782

theorem quadratic_roots (a : ℝ) (k c : ℝ) : 
    (∀ x : ℝ, 2 * x^2 + k * x + c = 0 ↔ (x = 7 ∨ x = a)) →
    k = -2 * a - 14 ∧ c = 14 * a :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1547_154782


namespace NUMINAMATH_GPT_groupC_is_all_polyhedra_l1547_154730

inductive GeometricBody
| TriangularPrism : GeometricBody
| QuadrangularPyramid : GeometricBody
| Sphere : GeometricBody
| Cone : GeometricBody
| Cube : GeometricBody
| TruncatedCone : GeometricBody
| HexagonalPyramid : GeometricBody
| Hemisphere : GeometricBody

def isPolyhedron : GeometricBody → Prop
| GeometricBody.TriangularPrism => true
| GeometricBody.QuadrangularPyramid => true
| GeometricBody.Sphere => false
| GeometricBody.Cone => false
| GeometricBody.Cube => true
| GeometricBody.TruncatedCone => false
| GeometricBody.HexagonalPyramid => true
| GeometricBody.Hemisphere => false

def groupA := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Sphere, GeometricBody.Cone]
def groupB := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.TruncatedCone]
def groupC := [GeometricBody.TriangularPrism, GeometricBody.QuadrangularPyramid, GeometricBody.Cube, GeometricBody.HexagonalPyramid]
def groupD := [GeometricBody.Cone, GeometricBody.TruncatedCone, GeometricBody.Sphere, GeometricBody.Hemisphere]

def allPolyhedra (group : List GeometricBody) : Prop :=
  ∀ b, b ∈ group → isPolyhedron b

theorem groupC_is_all_polyhedra : 
  allPolyhedra groupC ∧
  ¬ allPolyhedra groupA ∧
  ¬ allPolyhedra groupB ∧
  ¬ allPolyhedra groupD :=
by
  sorry

end NUMINAMATH_GPT_groupC_is_all_polyhedra_l1547_154730


namespace NUMINAMATH_GPT_car_a_speed_l1547_154745

theorem car_a_speed (d_gap : ℕ) (v_B : ℕ) (t : ℕ) (d_ahead : ℕ) (v_A : ℕ) 
  (h1 : d_gap = 24) (h2 : v_B = 50) (h3 : t = 4) (h4 : d_ahead = 8)
  (h5 : v_A = (d_gap + v_B * t + d_ahead) / t) : v_A = 58 :=
by {
  exact (sorry : v_A = 58)
}

end NUMINAMATH_GPT_car_a_speed_l1547_154745


namespace NUMINAMATH_GPT_find_C_in_terms_of_D_l1547_154792

noncomputable def h (C D x : ℝ) : ℝ := C * x - 3 * D ^ 2
noncomputable def k (D x : ℝ) : ℝ := D * x + 1

theorem find_C_in_terms_of_D (C D : ℝ) (h_eq : h C D (k D 2) = 0) (h_def : ∀ x, h C D x = C * x - 3 * D ^ 2) (k_def : ∀ x, k D x = D * x + 1) (D_ne_neg1 : D ≠ -1) : 
C = (3 * D ^ 2) / (2 * D + 1) := 
by 
  sorry

end NUMINAMATH_GPT_find_C_in_terms_of_D_l1547_154792


namespace NUMINAMATH_GPT_length_of_courtyard_l1547_154736

-- Given conditions

def width_of_courtyard : ℝ := 14
def brick_length : ℝ := 0.25
def brick_width : ℝ := 0.15
def total_bricks : ℝ := 8960

-- To be proven
theorem length_of_courtyard : brick_length * brick_width * total_bricks / width_of_courtyard = 24 := 
by sorry

end NUMINAMATH_GPT_length_of_courtyard_l1547_154736


namespace NUMINAMATH_GPT_temperature_at_midnight_l1547_154711

theorem temperature_at_midnight 
  (morning_temp : ℝ) 
  (afternoon_rise : ℝ) 
  (midnight_drop : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_rise = 1)
  (h3 : midnight_drop = 7) 
  : morning_temp + afternoon_rise - midnight_drop = 24 :=
by
  -- Convert all conditions into the correct forms
  rw [h1, h2, h3]
  -- Perform the arithmetic operations
  norm_num

end NUMINAMATH_GPT_temperature_at_midnight_l1547_154711


namespace NUMINAMATH_GPT_polygon_num_sides_and_exterior_angle_l1547_154786

theorem polygon_num_sides_and_exterior_angle 
  (n : ℕ) (x : ℕ) 
  (h : (n - 2) * 180 + x = 1350) 
  (hx : 0 < x ∧ x < 180) 
  : (n = 9) ∧ (x = 90) := 
by 
  sorry

end NUMINAMATH_GPT_polygon_num_sides_and_exterior_angle_l1547_154786


namespace NUMINAMATH_GPT_angle_through_point_l1547_154753

theorem angle_through_point : 
  (∃ θ : ℝ, ∃ k : ℤ, θ = 2 * k * Real.pi + 5 * Real.pi / 6 ∧ 
                      ∃ x y : ℝ, x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 ∧ 
                                    y / x = Real.tan θ) := 
sorry

end NUMINAMATH_GPT_angle_through_point_l1547_154753


namespace NUMINAMATH_GPT_arctan_sum_eq_pi_div_4_l1547_154781

noncomputable def n : ℤ := 27

theorem arctan_sum_eq_pi_div_4 :
  (Real.arctan (1 / 2) + Real.arctan (1 / 4) + Real.arctan (1 / 5) + Real.arctan (1 / n) = Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_arctan_sum_eq_pi_div_4_l1547_154781


namespace NUMINAMATH_GPT_cars_gain_one_passenger_each_l1547_154774

-- Conditions
def initial_people_per_car : ℕ := 3 -- 2 passengers + 1 driver
def total_cars : ℕ := 20
def total_people_at_end : ℕ := 80

-- Question (equivalent to "answer")
theorem cars_gain_one_passenger_each :
  (total_people_at_end = total_cars * initial_people_per_car + total_cars) →
  total_people_at_end - total_cars * initial_people_per_car = total_cars :=
by sorry

end NUMINAMATH_GPT_cars_gain_one_passenger_each_l1547_154774


namespace NUMINAMATH_GPT_sqrt_square_l1547_154797

theorem sqrt_square (x : ℝ) (h_nonneg : 0 ≤ x) : (Real.sqrt x)^2 = x :=
by
  sorry

example : (Real.sqrt 25)^2 = 25 :=
by
  exact sqrt_square 25 (by norm_num)

end NUMINAMATH_GPT_sqrt_square_l1547_154797


namespace NUMINAMATH_GPT_infinite_solutions_eq_l1547_154727

/-
Proving that the equation x - y + z = 1 has infinite solutions under the conditions:
1. x, y, z are distinct positive integers.
2. The product of any two numbers is divisible by the third one.
-/
theorem infinite_solutions_eq (x y z : ℕ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) 
(h4 : ∃ m n k : ℕ, x = m * n ∧ y = n * k ∧ z = m * k)
(h5 : (x*y) % z = 0) (h6 : (y*z) % x = 0) (h7 : (z*x) % y = 0) : 
∃ (m : ℕ), x - y + z = 1 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by sorry

end NUMINAMATH_GPT_infinite_solutions_eq_l1547_154727


namespace NUMINAMATH_GPT_polygon_sides_eq_six_l1547_154700

theorem polygon_sides_eq_six (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_six_l1547_154700


namespace NUMINAMATH_GPT_exists_k_l1547_154751

-- Definitions of the conditions
def sequence_def (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a (n+1) = Nat.lcm (a n) (a (n-1)) - Nat.lcm (a (n-1)) (a (n-2))

theorem exists_k (a : ℕ → ℕ) (a₁ a₂ a₃ : ℕ) (h₁ : a 1 = a₁) (h₂ : a 2 = a₂) (h₃ : a 3 = a₃)
  (h_seq : sequence_def a) : ∃ k : ℕ, k ≤ a₃ + 4 ∧ a k = 0 := 
sorry

end NUMINAMATH_GPT_exists_k_l1547_154751


namespace NUMINAMATH_GPT_inequality_proof_l1547_154742

variable (a b c : ℝ)
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a + b + c + a * b + b * c + c * a + a * b * c = 7)

theorem inequality_proof : 
  (Real.sqrt (a ^ 2 + b ^ 2 + 2) + Real.sqrt (b ^ 2 + c ^ 2 + 2) + Real.sqrt (c ^ 2 + a ^ 2 + 2)) ≥ 6 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1547_154742


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1547_154731

theorem tan_alpha_minus_pi_over_4 (α : Real) (h1 : 0 < α) (h2 : α < π) (h3 : Real.sin α = 3 / 5) :
  (Real.tan (α - π / 4) = -1/7) ∨ (Real.tan (α - π / 4) = -7) :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1547_154731


namespace NUMINAMATH_GPT_sum_of_cosines_dihedral_angles_l1547_154750

-- Define the conditions of the problem
def sum_of_plane_angles_trihederal (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Define the problem statement
theorem sum_of_cosines_dihedral_angles (α β γ : ℝ) (d1 d2 d3 : ℝ)
  (h : sum_of_plane_angles_trihederal α β γ) : 
  d1 + d2 + d3 = 1 :=
  sorry

end NUMINAMATH_GPT_sum_of_cosines_dihedral_angles_l1547_154750


namespace NUMINAMATH_GPT_passengers_on_ship_l1547_154795

theorem passengers_on_ship :
  (∀ P : ℕ, 
    (P / 12) + (P / 8) + (P / 3) + (P / 6) + 35 = P) → P = 120 :=
by 
  sorry

end NUMINAMATH_GPT_passengers_on_ship_l1547_154795


namespace NUMINAMATH_GPT_race_dead_heat_l1547_154775

variable (v_B v_A L x : ℝ)

theorem race_dead_heat (h : v_A = 17 / 14 * v_B) : x = 3 / 17 * L :=
by
  sorry

end NUMINAMATH_GPT_race_dead_heat_l1547_154775


namespace NUMINAMATH_GPT_balloons_in_each_bag_of_round_balloons_l1547_154726

variable (x : ℕ)

-- Definitions based on the problem's conditions
def totalRoundBalloonsBought := 5 * x
def totalLongBalloonsBought := 4 * 30
def remainingRoundBalloons := totalRoundBalloonsBought x - 5
def totalRemainingBalloons := remainingRoundBalloons x + totalLongBalloonsBought

-- Theorem statement based on the question and derived from the conditions and correct answer
theorem balloons_in_each_bag_of_round_balloons : totalRemainingBalloons x = 215 → x = 20 := by
  -- We acknowledge that the proof steps will follow here (omitted as per instructions)
  sorry

end NUMINAMATH_GPT_balloons_in_each_bag_of_round_balloons_l1547_154726


namespace NUMINAMATH_GPT_min_value_abs_expr_l1547_154719

noncomputable def minExpr (a b : ℝ) : ℝ :=
  |a + b| + |(1 / (a + 1)) - b|

theorem min_value_abs_expr (a b : ℝ) (h₁ : a ≠ -1) : minExpr a b ≥ 1 ∧ (minExpr a b = 1 ↔ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_min_value_abs_expr_l1547_154719


namespace NUMINAMATH_GPT_trig_inequality_l1547_154723

open Real

theorem trig_inequality (a b c : ℝ) (h₁ : a = sin (2 * π / 7))
  (h₂ : b = cos (2 * π / 7)) (h₃ : c = tan (2 * π / 7)) :
  c > a ∧ a > b :=
by 
  sorry

end NUMINAMATH_GPT_trig_inequality_l1547_154723


namespace NUMINAMATH_GPT_solution_of_inequalities_l1547_154716

theorem solution_of_inequalities (x : ℝ) :
  (2 * x / 5 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) ↔ (-5 ≤ x ∧ x < -3 / 2) := by
  sorry

end NUMINAMATH_GPT_solution_of_inequalities_l1547_154716


namespace NUMINAMATH_GPT_cost_per_book_l1547_154714

theorem cost_per_book
  (books_sold_each_time : ℕ)
  (people_bought : ℕ)
  (income_per_book : ℕ)
  (profit : ℕ)
  (total_income : ℕ := books_sold_each_time * people_bought * income_per_book)
  (total_cost : ℕ := total_income - profit)
  (total_books : ℕ := books_sold_each_time * people_bought)
  (cost_per_book : ℕ := total_cost / total_books) :
  books_sold_each_time = 2 ->
  people_bought = 4 ->
  income_per_book = 20 ->
  profit = 120 ->
  cost_per_book = 5 :=
  by intros; sorry

end NUMINAMATH_GPT_cost_per_book_l1547_154714


namespace NUMINAMATH_GPT_extreme_values_l1547_154755

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem extreme_values :
  (∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 5 / 3 ∧ f x1 = -2 ∧ f x2 = -58 / 27) ∧ 
  (∃ (a b : ℝ), a = 2 ∧ b = f 2 ∧ (∀ (x : ℝ), (a, b) = (x, f x) → (∀ y : ℝ, y = x - 4))) :=
by
  sorry

end NUMINAMATH_GPT_extreme_values_l1547_154755


namespace NUMINAMATH_GPT_A_takes_4_hours_l1547_154799

variables (A B C : ℝ)

-- Given conditions
axiom h1 : 1 / B + 1 / C = 1 / 2
axiom h2 : 1 / A + 1 / C = 1 / 2
axiom h3 : B = 4

-- What we need to prove: A = 4
theorem A_takes_4_hours :
  A = 4 := by
  sorry

end NUMINAMATH_GPT_A_takes_4_hours_l1547_154799


namespace NUMINAMATH_GPT_arc_length_sector_l1547_154729

theorem arc_length_sector (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 150 * Real.pi / 180) :
  θ * r = 5 * Real.pi / 2 :=
by
  rw [h_r, h_θ]
  sorry

end NUMINAMATH_GPT_arc_length_sector_l1547_154729


namespace NUMINAMATH_GPT_isosceles_base_length_l1547_154784

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end NUMINAMATH_GPT_isosceles_base_length_l1547_154784


namespace NUMINAMATH_GPT_sum_first_3m_terms_l1547_154717

variable (m : ℕ) (a₁ d : ℕ)

def S (n : ℕ) := n * a₁ + (n * (n - 1)) / 2 * d

-- Given conditions
axiom sum_first_m_terms : S m = 0
axiom sum_first_2m_terms : S (2 * m) = 0

-- Theorem to be proved
theorem sum_first_3m_terms : S (3 * m) = 210 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_3m_terms_l1547_154717


namespace NUMINAMATH_GPT_positive_difference_of_solutions_l1547_154743

theorem positive_difference_of_solutions:
  ∀ (s : ℝ), s ≠ -3 → (s^2 - 5*s - 24) / (s + 3) = 3*s + 10 →
  abs (-1 - (-27)) = 26 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_solutions_l1547_154743


namespace NUMINAMATH_GPT_points_within_distance_5_l1547_154773

noncomputable def distance (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

def within_distance (x y z : ℝ) (d : ℝ) : Prop := distance x y z ≤ d

def A := (1, 1, 1)
def B := (1, 2, 2)
def C := (2, -3, 5)
def D := (3, 0, 4)

theorem points_within_distance_5 :
  within_distance 1 1 1 5 ∧
  within_distance 1 2 2 5 ∧
  ¬ within_distance 2 (-3) 5 5 ∧
  within_distance 3 0 4 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_points_within_distance_5_l1547_154773


namespace NUMINAMATH_GPT_point_not_on_line_l1547_154760

theorem point_not_on_line (m b : ℝ) (h1 : m > 2) (h2 : m * b > 0) : ¬ (b = -2023) :=
by
  sorry

end NUMINAMATH_GPT_point_not_on_line_l1547_154760


namespace NUMINAMATH_GPT_matrix_satisfies_conditions_l1547_154783

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def matrix : List (List ℕ) :=
  [[6, 8, 9], [1, 7, 3], [4, 2, 5]]

noncomputable def sum_list (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def valid_matrix (matrix : List (List ℕ)) : Prop :=
  ∀ row_sum col_sum : ℕ, 
    (row_sum ∈ (matrix.map sum_list) ∧ is_prime row_sum) ∧
    (col_sum ∈ (List.transpose matrix).map sum_list ∧ is_prime col_sum)

theorem matrix_satisfies_conditions : valid_matrix matrix :=
by
  sorry

end NUMINAMATH_GPT_matrix_satisfies_conditions_l1547_154783


namespace NUMINAMATH_GPT_triangle_ratio_l1547_154709

-- Given conditions:
-- a: one side of the triangle
-- h_a: height corresponding to side a
-- r: inradius of the triangle
-- p: semiperimeter of the triangle

theorem triangle_ratio (a h_a r p : ℝ) (area_formula_1 : p * r = 1 / 2 * a * h_a) :
  (2 * p) / a = h_a / r :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_ratio_l1547_154709


namespace NUMINAMATH_GPT_binom_10_3_eq_120_l1547_154791

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end NUMINAMATH_GPT_binom_10_3_eq_120_l1547_154791


namespace NUMINAMATH_GPT_repeating_decimal_exceeds_decimal_l1547_154705

noncomputable def repeating_decimal_to_fraction : ℚ := 9 / 11
noncomputable def decimal_to_fraction : ℚ := 3 / 4

theorem repeating_decimal_exceeds_decimal :
  repeating_decimal_to_fraction - decimal_to_fraction = 3 / 44 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_exceeds_decimal_l1547_154705


namespace NUMINAMATH_GPT_radius_large_circle_l1547_154752

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end NUMINAMATH_GPT_radius_large_circle_l1547_154752


namespace NUMINAMATH_GPT_worm_length_l1547_154738

theorem worm_length (l1 l2 : ℝ) (h1 : l1 = 0.8) (h2 : l2 = l1 + 0.7) : l1 = 0.8 :=
by
  exact h1

end NUMINAMATH_GPT_worm_length_l1547_154738


namespace NUMINAMATH_GPT_remainders_equal_if_difference_divisible_l1547_154778

theorem remainders_equal_if_difference_divisible (a b k : ℤ) (h : k ∣ (a - b)) : 
  a % k = b % k :=
sorry

end NUMINAMATH_GPT_remainders_equal_if_difference_divisible_l1547_154778


namespace NUMINAMATH_GPT_additional_savings_l1547_154720

def window_price : ℕ := 100

def special_offer (windows_purchased : ℕ) : ℕ :=
  windows_purchased + windows_purchased / 6 * 2

def dave_windows : ℕ := 10

def doug_windows : ℕ := 12

def total_windows := dave_windows + doug_windows

def calculate_windows_cost (windows_needed : ℕ) : ℕ :=
  if windows_needed % 8 = 0 then (windows_needed / 8) * 6 * window_price
  else ((windows_needed / 8) * 6 + (windows_needed % 8)) * window_price

def separate_savings : ℕ :=
  window_price * (dave_windows + doug_windows) - (calculate_windows_cost dave_windows + calculate_windows_cost doug_windows)

def combined_savings : ℕ :=
  window_price * total_windows - calculate_windows_cost total_windows

theorem additional_savings :
  separate_savings + 200 = combined_savings :=
sorry

end NUMINAMATH_GPT_additional_savings_l1547_154720


namespace NUMINAMATH_GPT_find_n_divides_polynomial_l1547_154777

theorem find_n_divides_polynomial :
  ∀ (n : ℕ), 0 < n → (n + 2) ∣ (n^3 + 3 * n + 29) ↔ (n = 1 ∨ n = 3 ∨ n = 13) :=
by
  sorry

end NUMINAMATH_GPT_find_n_divides_polynomial_l1547_154777


namespace NUMINAMATH_GPT_triangle_in_and_circumcircle_radius_l1547_154759

noncomputable def radius_of_incircle (AC : ℝ) (BC : ℝ) (AB : ℝ) (Area : ℝ) (s : ℝ) : ℝ :=
  Area / s

noncomputable def radius_of_circumcircle (AB : ℝ) : ℝ :=
  AB / 2

theorem triangle_in_and_circumcircle_radius :
  ∀ (A B C : ℝ × ℝ) (AC : ℝ) (BC : ℝ) (AB : ℝ)
    (AngleA : ℝ) (AngleC : ℝ),
  AngleC = 90 ∧ AngleA = 60 ∧ AC = 6 ∧
  BC = AC * Real.sqrt 3 ∧ AB = 2 * AC
  → radius_of_incircle AC BC AB (18 * Real.sqrt 3) ((AC + BC + AB) / 2) = 6 * (Real.sqrt 3 - 1) / 13 ∧
    radius_of_circumcircle AB = 6 := by
  intros A B C AC BC AB AngleA AngleC h
  sorry

end NUMINAMATH_GPT_triangle_in_and_circumcircle_radius_l1547_154759


namespace NUMINAMATH_GPT_curve_defined_by_r_eq_4_is_circle_l1547_154779

theorem curve_defined_by_r_eq_4_is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = 4 → ∀ θ : ℝ, r = 4 :=
by
  sorry

end NUMINAMATH_GPT_curve_defined_by_r_eq_4_is_circle_l1547_154779


namespace NUMINAMATH_GPT_marble_problem_l1547_154728

-- Define the given conditions
def ratio (red blue green : ℕ) : Prop := red * 3 * 4 = blue * 2 * 4 ∧ blue * 2 * 4 = green * 2 * 3

-- The total number of marbles
def total_marbles (red blue green : ℕ) : ℕ := red + blue + green

-- The number of green marbles is given
def green_marbles : ℕ := 36

-- Proving the number of marbles and number of red marbles
theorem marble_problem
  (red blue green : ℕ)
  (h_ratio : ratio red blue green)
  (h_green : green = green_marbles) :
  total_marbles red blue green = 81 ∧ red = 18 :=
by
  sorry

end NUMINAMATH_GPT_marble_problem_l1547_154728


namespace NUMINAMATH_GPT_fourth_term_geometric_sequence_l1547_154771

theorem fourth_term_geometric_sequence :
  let a := (6: ℝ)^(1/2)
  let b := (6: ℝ)^(1/6)
  let c := (6: ℝ)^(1/12)
  b = a * r ∧ c = a * r^2 → (a * r^3) = 1 := 
by
  sorry

end NUMINAMATH_GPT_fourth_term_geometric_sequence_l1547_154771


namespace NUMINAMATH_GPT_range_of_a_l1547_154733

theorem range_of_a (a : ℝ) : 
  (∃ n : ℕ, (∀ x : ℕ, 1 ≤ x → x ≤ 5 → x < a) ∧ (∀ y : ℕ, x ≥ 1 → y ≥ 6 → y ≥ a)) ↔ (5 < a ∧ a < 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1547_154733


namespace NUMINAMATH_GPT_quadratic_has_exactly_one_root_l1547_154732

noncomputable def discriminant (b c : ℝ) : ℝ :=
b^2 - 4 * c

noncomputable def f (x b c : ℝ) : ℝ :=
x^2 + b * x + c

noncomputable def transformed_f (x b c : ℝ) : ℝ :=
(x - 2020)^2 + b * (x - 2020) + c

theorem quadratic_has_exactly_one_root (b c : ℝ)
  (h_discriminant : discriminant b c = 2020) :
  ∃! x : ℝ, f (x - 2020) b c + f x b c = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_has_exactly_one_root_l1547_154732


namespace NUMINAMATH_GPT_correct_calculation_l1547_154796

-- Definitions for conditions
def cond_A (x y : ℝ) : Prop := 3 * x + 4 * y = 7 * x * y
def cond_B (x : ℝ) : Prop := 5 * x - 2 * x = 3 * x ^ 2
def cond_C (y : ℝ) : Prop := 7 * y ^ 2 - 5 * y ^ 2 = 2
def cond_D (a b : ℝ) : Prop := 6 * a ^ 2 * b - b * a ^ 2 = 5 * a ^ 2 * b

-- Proof statement using conditions
theorem correct_calculation (a b : ℝ) : cond_D a b :=
by
  unfold cond_D
  sorry

end NUMINAMATH_GPT_correct_calculation_l1547_154796


namespace NUMINAMATH_GPT_sam_new_books_not_signed_l1547_154768

noncomputable def num_books_adventure := 13
noncomputable def num_books_mystery := 17
noncomputable def num_books_scifi := 25
noncomputable def num_books_nonfiction := 10
noncomputable def num_books_comics := 5
noncomputable def num_books_total := num_books_adventure + num_books_mystery + num_books_scifi + num_books_nonfiction + num_books_comics

noncomputable def num_books_used := 42
noncomputable def num_books_signed := 10
noncomputable def num_books_borrowed := 3
noncomputable def num_books_lost := 4

noncomputable def num_books_new := num_books_total - num_books_used
noncomputable def num_books_new_not_signed := num_books_new - num_books_signed
noncomputable def num_books_final := num_books_new_not_signed - num_books_lost

theorem sam_new_books_not_signed : num_books_final = 14 :=
by
  sorry

end NUMINAMATH_GPT_sam_new_books_not_signed_l1547_154768


namespace NUMINAMATH_GPT_length_decreased_by_l1547_154769

noncomputable def length_decrease_proof : Prop :=
  let length := 33.333333333333336
  let breadth := length / 2
  let new_length := length - 2.833333333333336
  let new_breadth := breadth + 4
  let original_area := length * breadth
  let new_area := new_length * new_breadth
  (new_area = original_area + 75) ↔ (new_length = length - 2.833333333333336)

theorem length_decreased_by : length_decrease_proof := sorry

end NUMINAMATH_GPT_length_decreased_by_l1547_154769


namespace NUMINAMATH_GPT_seats_with_middle_empty_l1547_154706

-- Define the parameters
def chairs := 5
def people := 4
def middle_empty := 3

-- Define the function to calculate seating arrangements
def number_of_ways (people : ℕ) (chairs : ℕ) (middle_empty : ℕ) : ℕ := 
  if chairs < people + 1 then 0
  else (chairs - 1) * (chairs - 2) * (chairs - 3) * (chairs - 4)

-- The theorem to prove the number of ways given the conditions
theorem seats_with_middle_empty : number_of_ways 4 5 3 = 24 := by
  sorry

end NUMINAMATH_GPT_seats_with_middle_empty_l1547_154706


namespace NUMINAMATH_GPT_triangle_def_ef_value_l1547_154708

theorem triangle_def_ef_value (E F D : ℝ) (DE DF EF : ℝ) (h1 : E = 45)
  (h2 : DE = 100) (h3 : DF = 100 * Real.sqrt 2) :
  EF = Real.sqrt (30000 + 5000*(Real.sqrt 6 - Real.sqrt 2)) := 
sorry 

end NUMINAMATH_GPT_triangle_def_ef_value_l1547_154708


namespace NUMINAMATH_GPT_two_pow_n_minus_one_prime_imp_n_prime_l1547_154770

theorem two_pow_n_minus_one_prime_imp_n_prime (n : ℕ) (h : Nat.Prime (2^n - 1)) : Nat.Prime n := 
sorry

end NUMINAMATH_GPT_two_pow_n_minus_one_prime_imp_n_prime_l1547_154770


namespace NUMINAMATH_GPT_boss_total_amount_l1547_154790

def number_of_staff : ℕ := 20
def rate_per_day : ℕ := 100
def number_of_days : ℕ := 30
def petty_cash_amount : ℕ := 1000

theorem boss_total_amount (number_of_staff : ℕ) (rate_per_day : ℕ) (number_of_days : ℕ) (petty_cash_amount : ℕ) :
  let total_allowance_one_staff := rate_per_day * number_of_days
  let total_allowance_all_staff := total_allowance_one_staff * number_of_staff
  total_allowance_all_staff + petty_cash_amount = 61000 := by
  sorry

end NUMINAMATH_GPT_boss_total_amount_l1547_154790


namespace NUMINAMATH_GPT_find_a7_l1547_154746

-- Define the arithmetic sequence
def a (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions
def S5 : ℤ := 25
def a2 : ℤ := 3

-- Main Goal: Find a_7
theorem find_a7 (a1 d : ℤ) (h1 : sum_first_n_terms a1 d 5 = S5)
                     (h2 : a a1 d 2 = a2) :
  a a1 d 7 = 13 := 
sorry

end NUMINAMATH_GPT_find_a7_l1547_154746


namespace NUMINAMATH_GPT_unique_solution_value_l1547_154721

theorem unique_solution_value (k : ℝ) :
  (∃ x : ℝ, x^2 = 2 * x + k ∧ ∀ y : ℝ, y^2 = 2 * y + k → y = x) ↔ k = -1 := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_value_l1547_154721


namespace NUMINAMATH_GPT_simplify_fraction_l1547_154715

theorem simplify_fraction :
  (3 - 6 + 12 - 24 + 48 - 96) / (6 - 12 + 24 - 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1547_154715


namespace NUMINAMATH_GPT_tangent_sum_formula_application_l1547_154713

-- Define the problem's parameters and statement
noncomputable def thirty_three_degrees_radian := Real.pi * 33 / 180
noncomputable def seventeen_degrees_radian := Real.pi * 17 / 180
noncomputable def twenty_eight_degrees_radian := Real.pi * 28 / 180

theorem tangent_sum_formula_application :
  Real.tan seventeen_degrees_radian + Real.tan twenty_eight_degrees_radian + Real.tan seventeen_degrees_radian * Real.tan twenty_eight_degrees_radian = 1 := 
sorry

end NUMINAMATH_GPT_tangent_sum_formula_application_l1547_154713


namespace NUMINAMATH_GPT_jordan_running_time_l1547_154735

-- Define the conditions given in the problem
variables (time_steve : ℕ) (distance_steve distance_jordan_1 distance_jordan_2 distance_jordan_3 : ℕ)

-- Assign the known values
axiom time_steve_def : time_steve = 24
axiom distance_steve_def : distance_steve = 3
axiom distance_jordan_1_def : distance_jordan_1 = 2
axiom distance_jordan_2_def : distance_jordan_2 = 1
axiom distance_jordan_3_def : distance_jordan_3 = 5

axiom half_time_condition : ∀ t_2, t_2 = time_steve / 2

-- The proof problem
theorem jordan_running_time : ∀ t_j1 t_j2 t_j3, 
  (t_j1 = time_steve / 2 ∧ 
   t_j2 = t_j1 / 2 ∧ 
   t_j3 = t_j2 * 5) →
  t_j3 = 30 := 
by
  intros t_j1 t_j2 t_j3 h
  sorry

end NUMINAMATH_GPT_jordan_running_time_l1547_154735


namespace NUMINAMATH_GPT_distinct_real_roots_of_quadratic_find_m_and_other_root_l1547_154794

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end NUMINAMATH_GPT_distinct_real_roots_of_quadratic_find_m_and_other_root_l1547_154794


namespace NUMINAMATH_GPT_rectangle_length_reduction_l1547_154701

theorem rectangle_length_reduction:
  ∀ (L W : ℝ) (X : ℝ),
  W > 0 →
  L > 0 →
  (L * (1 - X / 100) * (4 / 3)) * W = L * W →
  X = 25 :=
by
  intros L W X hW hL hEq
  sorry

end NUMINAMATH_GPT_rectangle_length_reduction_l1547_154701


namespace NUMINAMATH_GPT_find_c_for_root_ratio_l1547_154707

theorem find_c_for_root_ratio :
  ∃ c : ℝ, (∀ x1 x2 : ℝ, (4 * x1^2 - 5 * x1 + c = 0) ∧ (x1 / x2 = -3 / 4)) → c = -75 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_c_for_root_ratio_l1547_154707


namespace NUMINAMATH_GPT_time_to_walk_without_walkway_l1547_154739

theorem time_to_walk_without_walkway 
  (vp vw : ℝ) 
  (h1 : (vp + vw) * 40 = 80) 
  (h2 : (vp - vw) * 120 = 80) : 
  80 / vp = 60 :=
by
  sorry

end NUMINAMATH_GPT_time_to_walk_without_walkway_l1547_154739


namespace NUMINAMATH_GPT_ten_row_triangle_total_l1547_154749

theorem ten_row_triangle_total:
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  rods + connectors = 231 :=
by
  let rods := 3 * (Finset.range 10).sum id
  let connectors := (Finset.range 11).sum (fun n => n + 1)
  sorry

end NUMINAMATH_GPT_ten_row_triangle_total_l1547_154749


namespace NUMINAMATH_GPT_fractions_correct_l1547_154785
-- Broader import to ensure all necessary libraries are included.

-- Definitions of the conditions
def batman_homes_termite_ridden : ℚ := 1/3
def batman_homes_collapsing : ℚ := 7/10 * batman_homes_termite_ridden
def robin_homes_termite_ridden : ℚ := 3/7
def robin_homes_collapsing : ℚ := 4/5 * robin_homes_termite_ridden
def joker_homes_termite_ridden : ℚ := 1/2
def joker_homes_collapsing : ℚ := 3/8 * joker_homes_termite_ridden

-- Definitions of the fractions of homes that are termite-ridden but not collapsing
def batman_non_collapsing_fraction : ℚ := batman_homes_termite_ridden - batman_homes_collapsing
def robin_non_collapsing_fraction : ℚ := robin_homes_termite_ridden - robin_homes_collapsing
def joker_non_collapsing_fraction : ℚ := joker_homes_termite_ridden - joker_homes_collapsing

-- Proof statement
theorem fractions_correct :
  batman_non_collapsing_fraction = 1/10 ∧
  robin_non_collapsing_fraction = 3/35 ∧
  joker_non_collapsing_fraction = 5/16 :=
sorry

end NUMINAMATH_GPT_fractions_correct_l1547_154785


namespace NUMINAMATH_GPT_find_salary_l1547_154758

variable (S : ℝ)
variable (house_rent_percentage : ℝ) (education_percentage : ℝ) (clothes_percentage : ℝ)
variable (remaining_amount : ℝ)

theorem find_salary (h1 : house_rent_percentage = 0.20)
                    (h2 : education_percentage = 0.10)
                    (h3 : clothes_percentage = 0.10)
                    (h4 : remaining_amount = 1377)
                    (h5 : (1 - clothes_percentage) * (1 - education_percentage) * (1 - house_rent_percentage) * S = remaining_amount) :
                    S = 2125 := 
sorry

end NUMINAMATH_GPT_find_salary_l1547_154758


namespace NUMINAMATH_GPT_problem_statement_l1547_154798

theorem problem_statement (x y z : ℝ) :
    2 * x > y^2 + z^2 →
    2 * y > x^2 + z^2 →
    2 * z > y^2 + x^2 →
    x * y * z < 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1547_154798


namespace NUMINAMATH_GPT_roger_trips_required_l1547_154718

variable (carry_trays_per_trip total_trays : ℕ)

theorem roger_trips_required (h1 : carry_trays_per_trip = 4) (h2 : total_trays = 12) : total_trays / carry_trays_per_trip = 3 :=
by
  -- proof follows
  sorry

end NUMINAMATH_GPT_roger_trips_required_l1547_154718


namespace NUMINAMATH_GPT_percent_problem_l1547_154761

theorem percent_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 :=
sorry

end NUMINAMATH_GPT_percent_problem_l1547_154761


namespace NUMINAMATH_GPT_mike_changed_64_tires_l1547_154722

def total_tires_mike_changed (motorcycles : ℕ) (cars : ℕ) (tires_per_motorcycle : ℕ) (tires_per_car : ℕ) : ℕ :=
  motorcycles * tires_per_motorcycle + cars * tires_per_car

theorem mike_changed_64_tires :
  total_tires_mike_changed 12 10 2 4 = 64 :=
by
  sorry

end NUMINAMATH_GPT_mike_changed_64_tires_l1547_154722


namespace NUMINAMATH_GPT_find_k_l1547_154765

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1547_154765


namespace NUMINAMATH_GPT_judy_shopping_trip_l1547_154748

-- Define the quantities and prices of the items
def num_carrots : ℕ := 5
def price_carrot : ℕ := 1
def num_milk : ℕ := 4
def price_milk : ℕ := 3
def num_pineapples : ℕ := 2
def price_pineapple : ℕ := 4
def num_flour : ℕ := 2
def price_flour : ℕ := 5
def price_ice_cream : ℕ := 7

-- Define the promotion conditions
def pineapple_promotion : ℕ := num_pineapples / 2

-- Define the coupon condition
def coupon_threshold : ℕ := 40
def coupon_value : ℕ := 10

-- Define the total cost without coupon
def total_cost : ℕ := 
  (num_carrots * price_carrot) + 
  (num_milk * price_milk) +
  (pineapple_promotion * price_pineapple) +
  (num_flour * price_flour) +
  price_ice_cream

-- Define the final cost considering the coupon condition
def final_cost : ℕ :=
  if total_cost < coupon_threshold then total_cost else total_cost - coupon_value

-- The theorem to be proven
theorem judy_shopping_trip : final_cost = 38 := by
  sorry

end NUMINAMATH_GPT_judy_shopping_trip_l1547_154748


namespace NUMINAMATH_GPT_tan_sixty_eq_sqrt_three_l1547_154704

theorem tan_sixty_eq_sqrt_three : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_sixty_eq_sqrt_three_l1547_154704


namespace NUMINAMATH_GPT_find_a2014_l1547_154776

open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧
  (∀ n, a (n + 1) = (a n - 2) / (5 * a n / 4 - 2))

theorem find_a2014 (a : ℕ → ℚ) (h : seq a) : a 2014 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a2014_l1547_154776


namespace NUMINAMATH_GPT_value_of_x_l1547_154725

variable {x y z : ℤ}

theorem value_of_x
  (h1 : x + y = 31)
  (h2 : y + z = 47)
  (h3 : x + z = 52)
  (h4 : y + z = x + 16) :
  x = 31 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l1547_154725


namespace NUMINAMATH_GPT_graph_passes_through_point_l1547_154766

theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 ∧ a ≠ 1 → (∃ x y, (x, y) = (0, 2) ∧ y = a^x + 1) :=
by
  intros a ha
  use 0
  use 2
  obtain ⟨ha1, ha2⟩ := ha
  have h : a^0 = 1 := by simp
  simp [h]
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1547_154766


namespace NUMINAMATH_GPT_correct_calc_value_l1547_154793

theorem correct_calc_value (x : ℕ) (h : 2 * (3 * x + 14) = 946) : 2 * (x / 3 + 14) = 130 := 
by
  sorry

end NUMINAMATH_GPT_correct_calc_value_l1547_154793


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1547_154756

theorem geometric_sequence_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2)
  (h2 : a 5 = 4)
  (h3 : ∀ n, a n = a 1 * q^(n - 1)) : 
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1547_154756


namespace NUMINAMATH_GPT_odd_function_value_at_neg_two_l1547_154763

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then 2 * x - 3 else - (2 * (-x) - 3)

theorem odd_function_value_at_neg_two :
  (∀ x, f (-x) = -f x) → f (-2) = -1 :=
by
  intro odd_f
  sorry

end NUMINAMATH_GPT_odd_function_value_at_neg_two_l1547_154763


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_ellipse_l1547_154737

theorem sufficient_but_not_necessary_condition_ellipse (a : ℝ) :
  (a^2 > 1 → ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1)) ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → (a^2 > 1 ∨ 0 < a^2 ∧ a^2 < 1)) → ¬ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 → a^2 > 1))) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_ellipse_l1547_154737


namespace NUMINAMATH_GPT_locus_of_Q_max_area_of_triangle_OPQ_l1547_154702

open Real

theorem locus_of_Q (x y : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x = 3 * x_0 ∧ y = 4 * y_0 →
  (x / 6)^2 + (y / 4)^2 = 1 :=
sorry

theorem max_area_of_triangle_OPQ (S : ℝ) (x_0 y_0 : ℝ) :
  (x_0 / 4)^2 + (y_0 / 3)^2 = 1 ∧
  x_0 > 0 ∧ y_0 > 0 →
  S <= sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_locus_of_Q_max_area_of_triangle_OPQ_l1547_154702


namespace NUMINAMATH_GPT_circle_radius_zero_l1547_154744

theorem circle_radius_zero (x y : ℝ) : 2*x^2 - 8*x + 2*y^2 + 4*y + 10 = 0 → (x - 2)^2 + (y + 1)^2 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_circle_radius_zero_l1547_154744


namespace NUMINAMATH_GPT_pricePerRedStamp_l1547_154710

namespace StampCollection

-- Definitions for the conditions
def totalRedStamps : ℕ := 20
def soldRedStamps : ℕ := 20
def totalBlueStamps : ℕ := 80
def soldBlueStamps : ℕ := 80
def pricePerBlueStamp : ℝ := 0.8
def totalYellowStamps : ℕ := 7
def pricePerYellowStamp : ℝ := 2
def totalTargetEarnings : ℝ := 100

-- Derived definitions from conditions
def earningsFromBlueStamps : ℝ := soldBlueStamps * pricePerBlueStamp
def earningsFromYellowStamps : ℝ := totalYellowStamps * pricePerYellowStamp
def earningsRequiredFromRedStamps : ℝ := totalTargetEarnings - (earningsFromBlueStamps + earningsFromYellowStamps)

-- The statement asserting the main proof obligation
theorem pricePerRedStamp :
  (earningsRequiredFromRedStamps / soldRedStamps) = 1.10 :=
sorry

end StampCollection

end NUMINAMATH_GPT_pricePerRedStamp_l1547_154710


namespace NUMINAMATH_GPT_mn_value_l1547_154703
open Real

-- Define the conditions
def L_1_scenario_1 (m n : ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), θ₁ = 2 * θ₂ ∧ m = tan θ₁ ∧ n = tan θ₂ ∧ m = 4 * n

-- State the theorem
theorem mn_value (m n : ℝ) (hL1 : L_1_scenario_1 m n) (hm : m ≠ 0) : m * n = 2 :=
  sorry

end NUMINAMATH_GPT_mn_value_l1547_154703


namespace NUMINAMATH_GPT_donation_fifth_sixth_l1547_154740

-- Conditions definitions
def total_donation := 10000
def first_home := 2750
def second_home := 1945
def third_home := 1275
def fourth_home := 1890

-- Proof statement
theorem donation_fifth_sixth : 
  (total_donation - (first_home + second_home + third_home + fourth_home)) = 2140 := by
  sorry

end NUMINAMATH_GPT_donation_fifth_sixth_l1547_154740


namespace NUMINAMATH_GPT_proposition_p_is_false_iff_l1547_154764

def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 3)

def p (a : ℝ) : Prop := ∃ x : ℝ, f x < a

theorem proposition_p_is_false_iff (a : ℝ) : (¬p a) ↔ (a < 5) :=
by sorry

end NUMINAMATH_GPT_proposition_p_is_false_iff_l1547_154764


namespace NUMINAMATH_GPT_situps_difference_l1547_154780

def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2
def emma_situps : ℕ := bob_situps / 3

theorem situps_difference : 
  (nathan_situps + bob_situps + emma_situps) - ken_situps = 60 := by
  sorry

end NUMINAMATH_GPT_situps_difference_l1547_154780


namespace NUMINAMATH_GPT_fraction_subtraction_equals_one_l1547_154712

theorem fraction_subtraction_equals_one (x : ℝ) (h : x ≠ 1) : (x / (x - 1)) - (1 / (x - 1)) = 1 := 
by sorry

end NUMINAMATH_GPT_fraction_subtraction_equals_one_l1547_154712


namespace NUMINAMATH_GPT_remainder_of_n_l1547_154788

theorem remainder_of_n {n : ℕ} (h1 : n^2 ≡ 4 [MOD 7]) (h2 : n^3 ≡ 6 [MOD 7]): 
  n ≡ 5 [MOD 7] :=
sorry

end NUMINAMATH_GPT_remainder_of_n_l1547_154788


namespace NUMINAMATH_GPT_solve_proof_problem_l1547_154789

variables (a b c d : ℝ)

noncomputable def proof_problem : Prop :=
  a = 3 * b ∧ b = 3 * c ∧ c = 5 * d → (a * c) / (b * d) = 15

theorem solve_proof_problem : proof_problem a b c d :=
by
  sorry

end NUMINAMATH_GPT_solve_proof_problem_l1547_154789


namespace NUMINAMATH_GPT_sum_difference_of_consecutive_integers_l1547_154762

theorem sum_difference_of_consecutive_integers (n : ℤ) :
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  S2 - S1 = 28 :=
by
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  have hS1 : S1 = (n-3) + (n-2) + (n-1) + n + (n+1) + (n+2) + (n+3) := by sorry
  have hS2 : S2 = (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) := by sorry
  have h_diff : S2 - S1 = 28 := by sorry
  exact h_diff

end NUMINAMATH_GPT_sum_difference_of_consecutive_integers_l1547_154762


namespace NUMINAMATH_GPT_angela_sleep_difference_l1547_154767

theorem angela_sleep_difference :
  let december_sleep_hours := 6.5
  let january_sleep_hours := 8.5
  let december_days := 31
  let january_days := 31
  (january_sleep_hours * january_days) - (december_sleep_hours * december_days) = 62 :=
by
  sorry

end NUMINAMATH_GPT_angela_sleep_difference_l1547_154767


namespace NUMINAMATH_GPT_arlo_stationery_count_l1547_154772

theorem arlo_stationery_count (books pens : ℕ) (ratio_books_pens : ℕ × ℕ) (total_books : ℕ)
  (h_ratio : ratio_books_pens = (7, 3)) (h_books : total_books = 280) :
  books + pens = 400 :=
by
  sorry

end NUMINAMATH_GPT_arlo_stationery_count_l1547_154772


namespace NUMINAMATH_GPT_students_enrolled_in_all_three_l1547_154787

variables {total_students at_least_one robotics_students dance_students music_students at_least_two_students all_three_students : ℕ}

-- Given conditions
axiom H1 : total_students = 25
axiom H2 : at_least_one = total_students
axiom H3 : robotics_students = 15
axiom H4 : dance_students = 12
axiom H5 : music_students = 10
axiom H6 : at_least_two_students = 11

-- We need to prove the number of students enrolled in all three workshops is 1
theorem students_enrolled_in_all_three : all_three_students = 1 :=
sorry

end NUMINAMATH_GPT_students_enrolled_in_all_three_l1547_154787


namespace NUMINAMATH_GPT_jane_started_babysitting_at_age_18_l1547_154734

-- Define the age Jane started babysitting
def jane_starting_age := 18

-- State Jane's current age
def jane_current_age : ℕ := 34

-- State the years since Jane stopped babysitting
def years_since_jane_stopped := 12

-- Calculate Jane's age when she stopped babysitting
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped

-- State the current age of the oldest person she could have babysat
def current_oldest_child_age : ℕ := 25

-- Calculate the age of the oldest child when Jane stopped babysitting
def age_oldest_child_when_stopped : ℕ := current_oldest_child_age - years_since_jane_stopped

-- State the condition that the child was no more than half her age at the time
def child_age_condition (jane_age : ℕ) (child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- The theorem to prove the age Jane started babysitting
theorem jane_started_babysitting_at_age_18
  (jane_current : jane_current_age = 34)
  (years_stopped : years_since_jane_stopped = 12)
  (current_oldest : current_oldest_child_age = 25)
  (age_when_stopped : jane_age_when_stopped = 22)
  (child_when_stopped : age_oldest_child_when_stopped = 13)
  (child_condition : ∀ {j : ℕ}, child_age_condition j age_oldest_child_when_stopped → False) :
  jane_starting_age = 18 :=
sorry

end NUMINAMATH_GPT_jane_started_babysitting_at_age_18_l1547_154734


namespace NUMINAMATH_GPT_vector_dot_product_problem_l1547_154741

variables {a b : ℝ}

theorem vector_dot_product_problem (h1 : a + 2 * b = 0) (h2 : (a + b) * a = 2) : a * b = -2 :=
sorry

end NUMINAMATH_GPT_vector_dot_product_problem_l1547_154741


namespace NUMINAMATH_GPT_sqrt3_op_sqrt3_l1547_154754

def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3 : custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 :=
  sorry

end NUMINAMATH_GPT_sqrt3_op_sqrt3_l1547_154754


namespace NUMINAMATH_GPT_difference_of_squares_550_450_l1547_154747

theorem difference_of_squares_550_450 : (550 ^ 2 - 450 ^ 2) = 100000 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_550_450_l1547_154747


namespace NUMINAMATH_GPT_find_m_l1547_154757

theorem find_m (θ₁ θ₂ : ℝ) (l : ℝ → ℝ) (m : ℕ) 
  (hθ₁ : θ₁ = Real.pi / 100) 
  (hθ₂ : θ₂ = Real.pi / 75)
  (hl : ∀ x, l x = x / 4) 
  (R : ((ℝ → ℝ) → (ℝ → ℝ)))
  (H_R : ∀ l, R l = (sorry : ℝ → ℝ)) 
  (R_n : ℕ → (ℝ → ℝ) → (ℝ → ℝ)) 
  (H_R1 : R_n 1 l = R l) 
  (H_Rn : ∀ n, R_n (n + 1) l = R (R_n n l)) :
  m = 1500 :=
sorry

end NUMINAMATH_GPT_find_m_l1547_154757
