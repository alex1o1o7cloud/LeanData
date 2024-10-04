import Mathlib

namespace proving_b_and_extreme_value_of_f_l130_130261

noncomputable def f (x : ℝ) (b : ℝ) := x^2 + b * Real.log x
noncomputable def g (x : ℝ) := (x - 10) / (x - 4)

theorem proving_b_and_extreme_value_of_f 
  (h_parallel: (deriv (λ x, f x b) 5 = deriv g 5))
  (b_value: b = -20) :
  b = -20 ∧ (∃ x_extreme : ℝ, x_extreme = Real.sqrt 10 ∧ f x_extreme (-20) = 10 - 10 * Real.log 10) :=
by
  have h_b : b = -20 := by sorry
  have h_extreme : ∃ x_extreme : ℝ, x_extreme = Real.sqrt 10 ∧ f x_extreme (-20) = 10 - 10 * Real.log 10 := by sorry
  exact ⟨h_b, h_extreme⟩

end proving_b_and_extreme_value_of_f_l130_130261


namespace unit_digit_3_pow_2012_sub_1_l130_130398

theorem unit_digit_3_pow_2012_sub_1 :
  (3 ^ 2012 - 1) % 10 = 0 :=
sorry

end unit_digit_3_pow_2012_sub_1_l130_130398


namespace sum_of_center_coords_l130_130243

theorem sum_of_center_coords (x y : ℝ) :
  (∃ k : ℝ, (x + 2)^2 + (y + 3)^2 = k ∧ (x^2 + y^2 = -4 * x - 6 * y + 5)) -> x + y = -5 :=
by
sorry

end sum_of_center_coords_l130_130243


namespace octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l130_130693

-- Part (a) - Octahedron
/- 
A connected graph representing an octahedron. 
Each vertex has a degree of 4, making the graph Eulerian.
-/
theorem octahedron_has_eulerian_circuit : 
  ∃ circuit : List (ℕ × ℕ), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

-- Part (b) - Cube
/- 
A connected graph representing a cube.
Each vertex has a degree of 3, making it impossible for the graph to be Eulerian.
-/
theorem cube_has_no_eulerian_circuit : 
  ¬ ∃ (circuit : List (ℕ × ℕ)), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

end octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l130_130693


namespace exists_same_color_rectangle_l130_130717

open Finset

-- Define the grid size
def gridSize : ℕ := 12

-- Define the type of colors
inductive Color
| red
| white
| blue

-- Define a point in the grid
structure Point :=
(x : ℕ)
(y : ℕ)
(hx : x ≥ 1 ∧ x ≤ gridSize)
(hy : y ≥ 1 ∧ y ≤ gridSize)

-- Assume a coloring function
def color (p : Point) : Color := sorry

-- The theorem statement
theorem exists_same_color_rectangle :
  ∃ (p1 p2 p3 p4 : Point),
    p1.x = p2.x ∧ p3.x = p4.x ∧
    p1.y = p3.y ∧ p2.y = p4.y ∧
    color p1 = color p2 ∧
    color p1 = color p3 ∧
    color p1 = color p4 :=
sorry

end exists_same_color_rectangle_l130_130717


namespace inequalities_true_l130_130599

theorem inequalities_true (a b : ℝ) (h : a > b) : a^3 > b^3 ∧ 3 ^ a > 3 ^ b := 
by {
  have h1 : a^3 > b^3 := sorry,
  have h2 : 3 ^ a > 3 ^ b := sorry,
  exact ⟨h1, h2⟩,
}

end inequalities_true_l130_130599


namespace pastries_and_cost_correct_l130_130623

def num_pastries_lola := 13 + 10 + 8 + 6
def cost_lola := 13 * 0.50 + 10 * 1.00 + 8 * 3.00 + 6 * 2.00

def num_pastries_lulu := 16 + 12 + 14 + 9
def cost_lulu := 16 * 0.50 + 12 * 1.00 + 14 * 3.00 + 9 * 2.00

def num_pastries_lila := 22 + 15 + 10 + 12
def cost_lila := 22 * 0.50 + 15 * 1.00 + 10 * 3.00 + 12 * 2.00

def num_pastries_luka := 18 + 20 + 7 + 14 + 25
def cost_luka := 18 * 0.50 + 20 * 1.00 + 7 * 3.00 + 14 * 2.00 + 25 * 1.50

def total_pastries := num_pastries_lola + num_pastries_lulu + num_pastries_lila + num_pastries_luka
def total_cost := cost_lola + cost_lulu + cost_lila + cost_luka

theorem pastries_and_cost_correct :
  total_pastries = 231 ∧ total_cost = 328.00 :=
by
  sorry

end pastries_and_cost_correct_l130_130623


namespace uneaten_chips_correct_l130_130404

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l130_130404


namespace additional_money_needed_for_free_shipping_l130_130090

-- Define the prices of the books
def price_book1 : ℝ := 13.00
def price_book2 : ℝ := 15.00
def price_book3 : ℝ := 10.00
def price_book4 : ℝ := 10.00

-- Define the discount rate
def discount_rate : ℝ := 0.25

-- Calculate the discounted prices
def discounted_price_book1 : ℝ := price_book1 * (1 - discount_rate)
def discounted_price_book2 : ℝ := price_book2 * (1 - discount_rate)

-- Sum of discounted prices of books
def total_cost : ℝ := discounted_price_book1 + discounted_price_book2 + price_book3 + price_book4

-- Free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- The proof statement
theorem additional_money_needed_for_free_shipping : additional_amount = 9.00 := by
  -- calculation steps omitted
  sorry

end additional_money_needed_for_free_shipping_l130_130090


namespace necessary_and_sufficient_condition_l130_130150

theorem necessary_and_sufficient_condition (t : ℝ) :
  ((t + 1) * (1 - |t|) > 0) ↔ (t < 1 ∧ t ≠ -1) :=
by
  sorry

end necessary_and_sufficient_condition_l130_130150


namespace probability_same_color_white_l130_130828

/--
Given a box with 6 white balls and 5 black balls, if 3 balls are drawn such that all drawn balls have the same color,
prove that the probability that these balls are white is 2/3.
-/
theorem probability_same_color_white :
  (∃ (n_white n_black drawn_white drawn_black total_same_color : ℕ),
    n_white = 6 ∧ n_black = 5 ∧
    drawn_white = Nat.choose n_white 3 ∧ drawn_black = Nat.choose n_black 3 ∧
    total_same_color = drawn_white + drawn_black ∧
    (drawn_white:ℚ) / total_same_color = 2 / 3) :=
sorry

end probability_same_color_white_l130_130828


namespace length_of_integer_eq_24_l130_130733

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l130_130733


namespace line_through_origin_and_conditions_l130_130208

-- Definitions:
def system_defines_line (m n p x y z : ℝ) : Prop :=
  (x / m = y / n) ∧ (y / n = z / p)

def lies_in_coordinate_plane (m n p : ℝ) : Prop :=
  (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)

def coincides_with_coordinate_axis (m n p : ℝ) : Prop :=
  (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)

-- Theorem statement:
theorem line_through_origin_and_conditions (m n p x y z : ℝ) :
  system_defines_line m n p x y z →
  (∀ m n p, lies_in_coordinate_plane m n p ↔ (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)) ∧
  (∀ m n p, coincides_with_coordinate_axis m n p ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)) :=
by
  sorry

end line_through_origin_and_conditions_l130_130208


namespace max_min_difference_l130_130048

def y (x : ℝ) : ℝ := x * abs (3 - x) - (x - 3) * abs x

theorem max_min_difference : (0 : ℝ) ≤ x → (x < 3 → y x ≤ y (3 / 4)) ∧ (x < 0 → y x = 0) ∧ (x ≥ 3 → y x = 0) → 
  (y (3 / 4) - (min (y 0) (min (y (-1)) (y 3)))) = 1.125 :=
by
  sorry

end max_min_difference_l130_130048


namespace intersection_sum_l130_130138

-- Definitions for points A, B, C, and midpoints D, E
def A : ℝ × ℝ := (0, 8)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def D : ℝ × ℝ := midpoint A B
def E : ℝ × ℝ := midpoint B C

-- Equations of the lines AE and CD
def line (p1 p2 : ℝ × ℝ) (x : ℝ) : ℝ :=
  let slope := (p2.2 - p1.2) / (p2.1 - p1.1)
  p1.2 + slope * (x - p1.1)

def line_AE (x : ℝ) : ℝ := line A E x
def line_CD (x : ℝ) : ℝ := line C D x

-- Intersection point F of the lines AE and CD
def F : ℝ × ℝ :=
  let x := ((line C D 0) - (line A E 0)) / ((- (A.2 - E.2)/(A.1 - E.1)) - (- (C.2 - D.2)/(C.1 - D.1)))
  let y := line_AE x
  (x, y)

-- The proof problem statement
theorem intersection_sum : F.1 + F.2 = 6 :=
  sorry

end intersection_sum_l130_130138


namespace angle_in_second_quadrant_l130_130117

theorem angle_in_second_quadrant (x : ℝ) (hx1 : Real.tan x < 0) (hx2 : Real.sin x - Real.cos x > 0) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2 ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2) :=
sorry

end angle_in_second_quadrant_l130_130117


namespace range_of_m_l130_130915

variable {R : Type} [LinearOrderedField R]

def discriminant (a b c : R) : R := b^2 - 4 * a * c

theorem range_of_m (m : R) :
  (discriminant (1:R) m (m + 3) > 0) ↔ (m < -2 ∨ m > 6) :=
by
  sorry

end range_of_m_l130_130915


namespace unit_vector_perpendicular_to_a_l130_130597

-- Definitions of a vector and the properties of unit and perpendicular vectors
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def is_unit_vector (v : Vector2D) : Prop :=
  v.x ^ 2 + v.y ^ 2 = 1

def is_perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

-- Given vector a
def a : Vector2D := ⟨3, 4⟩

-- Coordinates of the unit vector that is perpendicular to a
theorem unit_vector_perpendicular_to_a :
  ∃ (b : Vector2D), is_unit_vector b ∧ is_perpendicular a b ∧
  (b = ⟨-4 / 5, 3 / 5⟩ ∨ b = ⟨4 / 5, -3 / 5⟩) :=
sorry

end unit_vector_perpendicular_to_a_l130_130597


namespace smallest_number_divisible_by_11_and_remainder_1_l130_130837

theorem smallest_number_divisible_by_11_and_remainder_1 {n : ℕ} :
  (n % 2 = 1) ∧ 
  (n % 3 = 1) ∧ 
  (n % 4 = 1) ∧ 
  (n % 5 = 1) ∧ 
  (n % 11 = 0) -> n = 121 :=
sorry

end smallest_number_divisible_by_11_and_remainder_1_l130_130837


namespace journey_speed_l130_130695

theorem journey_speed (v : ℚ) 
  (equal_distance : ∀ {d}, (d = 0.22) → ((0.66 / 3) = d))
  (total_distance : ∀ {d}, (d = 660 / 1000) → (660 / 1000 = 0.66))
  (total_time : ∀ {t} , (t = 11 / 60) → (11 / 60 = t)): 
  (0.22 / 2 + 0.22 / v + 0.22 / 6 = 11 / 60) → v = 1.2 := 
by 
  sorry

end journey_speed_l130_130695


namespace mari_vs_kendra_l130_130959

-- Variable Definitions
variables (K M S : ℕ)  -- Number of buttons Kendra, Mari, and Sue made
variables (h1: 2*S = K) -- Sue made half as many as Kendra
variables (h2: S = 6)   -- Sue made 6 buttons
variables (h3: M = 64)  -- Mari made 64 buttons

-- Theorem Statement
theorem mari_vs_kendra (K M S : ℕ) (h1 : 2 * S = K) (h2 : S = 6) (h3 : M = 64) :
  M = 5 * K + 4 :=
sorry

end mari_vs_kendra_l130_130959


namespace cost_per_use_l130_130928

def cost : ℕ := 30
def uses_in_a_week : ℕ := 3
def weeks : ℕ := 2
def total_uses : ℕ := uses_in_a_week * weeks

theorem cost_per_use : cost / total_uses = 5 := by
  sorry

end cost_per_use_l130_130928


namespace distance_from_focus_to_line_l130_130646

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130646


namespace four_digit_numbers_with_property_l130_130277

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l130_130277


namespace linear_equation_in_options_l130_130351

def is_linear_equation_with_one_variable (eqn : String) : Prop :=
  eqn = "3 - 2x = 5"

theorem linear_equation_in_options :
  is_linear_equation_with_one_variable "3 - 2x = 5" :=
by
  sorry

end linear_equation_in_options_l130_130351


namespace problem_solution_l130_130248

def count_valid_n : ℕ :=
  let count_mult_3 := (3000 / 3)
  let count_mult_6 := (3000 / 6)
  count_mult_3 - count_mult_6

theorem problem_solution : count_valid_n = 500 := 
sorry

end problem_solution_l130_130248


namespace gcd_12569_36975_l130_130566

-- Define the integers for which we need to find the gcd
def num1 : ℕ := 12569
def num2 : ℕ := 36975

-- The statement that the gcd of these two numbers is 1
theorem gcd_12569_36975 : Nat.gcd num1 num2 = 1 := by
  sorry

end gcd_12569_36975_l130_130566


namespace second_spray_kill_percent_l130_130053

-- Conditions
def first_spray_kill_percent : ℝ := 50
def both_spray_kill_percent : ℝ := 5
def germs_left_after_both : ℝ := 30

-- Lean 4 statement
theorem second_spray_kill_percent (x : ℝ) 
  (H : 100 - (first_spray_kill_percent + x - both_spray_kill_percent) = germs_left_after_both) :
  x = 15 :=
by
  sorry

end second_spray_kill_percent_l130_130053


namespace positive_difference_l130_130223

theorem positive_difference:
  let a := (7^3 + 7^3) / 7
  let b := (7^3)^2 / 7
  b - a = 16709 :=
by
  sorry

end positive_difference_l130_130223


namespace cubes_of_roots_l130_130716

theorem cubes_of_roots (a b c : ℝ) (h1 : a + b + c = 2) (h2 : ab + ac + bc = 2) (h3 : abc = 3) : 
  a^3 + b^3 + c^3 = 9 :=
by
  sorry

end cubes_of_roots_l130_130716


namespace three_cards_different_suits_probability_l130_130636

-- Define the conditions and problem
noncomputable def prob_three_cards_diff_suits : ℚ :=
  have first_card_options := 52
  have second_card_options := 39
  have third_card_options := 26
  have total_ways_to_pick := (52 : ℕ) * (51 : ℕ) * (50 : ℕ)
  (39 / 51) * (26 / 50)

-- State our proof problem
theorem three_cards_different_suits_probability :
  prob_three_cards_diff_suits = 169 / 425 :=
sorry

end three_cards_different_suits_probability_l130_130636


namespace inequality_proof_l130_130939

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a / (a + 2 * b)^(1/3) + b / (b + 2 * c)^(1/3) + c / (c + 2 * a)^(1/3)) ≥ 1 := 
by
  sorry

end inequality_proof_l130_130939


namespace expected_dietary_restriction_l130_130315

theorem expected_dietary_restriction (n : ℕ) (p : ℚ) (sample_size : ℕ) (expected : ℕ) :
  p = 1 / 4 ∧ sample_size = 300 ∧ expected = sample_size * p → expected = 75 := by
  sorry

end expected_dietary_restriction_l130_130315


namespace taxi_fare_l130_130508

theorem taxi_fare (x : ℝ) (h : 3.00 + 0.25 * ((x - 0.75) / 0.1) = 12) : x = 4.35 :=
  sorry

end taxi_fare_l130_130508


namespace remainder_of_3_pow_101_plus_4_mod_5_l130_130193

theorem remainder_of_3_pow_101_plus_4_mod_5 :
  (3^101 + 4) % 5 = 2 :=
by
  have h1 : 3 % 5 = 3 := by sorry
  have h2 : (3^2) % 5 = 4 := by sorry
  have h3 : (3^3) % 5 = 2 := by sorry
  have h4 : (3^4) % 5 = 1 := by sorry
  -- more steps to show the pattern and use it to prove the final statement
  sorry

end remainder_of_3_pow_101_plus_4_mod_5_l130_130193


namespace solve_equation_l130_130639

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^2 - (x + 3) * (x - 3) = 4 * x - 1 ∧ x = 7 / 4 := 
by
  sorry

end solve_equation_l130_130639


namespace investment_interest_rate_calculation_l130_130535

theorem investment_interest_rate_calculation :
  let initial_investment : ℝ := 15000
  let first_year_rate : ℝ := 0.08
  let first_year_investment : ℝ := initial_investment * (1 + first_year_rate)
  let second_year_investment : ℝ := 17160
  ∃ (s : ℝ), (first_year_investment * (1 + s / 100) = second_year_investment) → s = 6 :=
by
  sorry

end investment_interest_rate_calculation_l130_130535


namespace boards_per_package_calculation_l130_130534

-- Defining the conditions
def total_boards : ℕ := 154
def num_packages : ℕ := 52

-- Defining the division of total_boards by num_packages within rationals
def boards_per_package : ℚ := total_boards / num_packages

-- Prove that the boards per package is mathematically equal to the total boards divided by the number of packages
theorem boards_per_package_calculation :
  boards_per_package = 154 / 52 := by
  sorry

end boards_per_package_calculation_l130_130534


namespace range_of_a_l130_130817

noncomputable def f (a : ℝ) : ℝ → ℝ
| x => if x < 1 then a^x else (a - 3) * x + 4 * a

theorem range_of_a (a : ℝ) (h : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 0 < a ∧ a ≤ 3 / 4 :=
sorry

end range_of_a_l130_130817


namespace point_P_position_l130_130419

variable {a b c d : ℝ}
variable (h1: a ≠ b) (h2: a ≠ c) (h3: a ≠ d) (h4: b ≠ c) (h5: b ≠ d) (h6: c ≠ d)

theorem point_P_position (P : ℝ) (hP: b < P ∧ P < c) (hRatio: (|a - P| / |P - d|) = (|b - P| / |P - c|)) : 
  P = (a * c - b * d) / (a - b + c - d) := 
by
  sorry

end point_P_position_l130_130419


namespace sophia_ate_pie_l130_130170

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l130_130170


namespace triangle_right_angle_solution_l130_130612

def is_right_angle (a b : ℝ × ℝ) : Prop := (a.1 * b.1 + a.2 * b.2 = 0)

def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

theorem triangle_right_angle_solution (x : ℝ) (h1 : (2, -1) = (2, -1)) (h2 : (x, 3) = (x, 3)) : 
  is_right_angle (2, -1) (x, 3) ∨ 
  is_right_angle (2, -1) (vector_sub (x, 3) (2, -1)) ∨ 
  is_right_angle (x, 3) (vector_sub (x, 3) (2, -1)) → 
  x = 3 / 2 ∨ x = 4 :=
sorry

end triangle_right_angle_solution_l130_130612


namespace time_after_1876_minutes_l130_130179

-- Define the structure for Time
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
noncomputable def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.minute + m
  let additional_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let new_hour := (t.hour + additional_hours) % 24
  { hour := new_hour, minute := remaining_minutes }

-- Definition of the starting time
def three_pm : Time := { hour := 15, minute := 0 }

-- The main theorem statement
theorem time_after_1876_minutes : add_minutes three_pm 1876 = { hour := 10, minute := 16 } :=
  sorry

end time_after_1876_minutes_l130_130179


namespace playground_area_22500_l130_130334

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l130_130334


namespace number_satisfies_equation_l130_130871

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end number_satisfies_equation_l130_130871


namespace average_of_list_l130_130323

theorem average_of_list (n : ℕ) (h : (2 + 9 + 4 + n + 2 * n) / 5 = 6) : n = 5 := 
by
  sorry

end average_of_list_l130_130323


namespace sum_of_common_ratios_l130_130619

noncomputable def geometric_sequence (m x : ℝ) : ℝ × ℝ × ℝ := (m, m * x, m * x^2)

theorem sum_of_common_ratios
  (m x y : ℝ)
  (h1 : x ≠ y)
  (h2 : m ≠ 0)
  (h3 : ∃ c3 c2 d3 d2 : ℝ, geometric_sequence m x = (m, c2, c3) ∧ geometric_sequence m y = (m, d2, d3) ∧ c3 - d3 = 3 * (c2 - d2)) :
  x + y = 3 := by
  sorry

end sum_of_common_ratios_l130_130619


namespace distance_from_focus_to_line_l130_130662

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130662


namespace henry_correct_answers_l130_130608

theorem henry_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 :=
by
  sorry

end henry_correct_answers_l130_130608


namespace uneaten_chips_l130_130403

theorem uneaten_chips :
  ∀ (chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips : ℕ),
    (chips_per_cookie = 7) →
    (cookies_total = 12 * 4) →
    (half_cookies = cookies_total / 2) →
    (uneaten_cookies = cookies_total - half_cookies) →
    (uneaten_chips = uneaten_cookies * chips_per_cookie) →
    uneaten_chips = 168 :=
by
  intros chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips
  intros chips_per_cookie_eq cookies_total_eq half_cookies_eq uneaten_cookies_eq uneaten_chips_eq
  rw [chips_per_cookie_eq, cookies_total_eq, half_cookies_eq, uneaten_cookies_eq, uneaten_chips_eq]
  norm_num
  sorry

end uneaten_chips_l130_130403


namespace no_solution_absval_equation_l130_130729

theorem no_solution_absval_equation (x : ℝ) : ¬ (|2*x - 5| = 3*x + 1) :=
by
  sorry

end no_solution_absval_equation_l130_130729


namespace remainder_when_101_divided_by_7_is_3_l130_130800

theorem remainder_when_101_divided_by_7_is_3
    (A : ℤ)
    (h : 9 * A + 1 = 10 * A - 100) : A % 7 = 3 := by
  -- Mathematical steps are omitted as instructed
  sorry

end remainder_when_101_divided_by_7_is_3_l130_130800


namespace find_f_condition_l130_130471

theorem find_f_condition {f : ℂ → ℂ} (h : ∀ z : ℂ, f z + z * f (1 - z) = 1 + z) :
  ∀ z : ℂ, f z = 1 :=
by
  sorry

end find_f_condition_l130_130471


namespace find_alpha_l130_130747

noncomputable def curve_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

noncomputable def line_eq_passes_point (alpha t : ℝ) : ℝ × ℝ :=
  (t * Real.cos alpha, 1 + t * Real.sin alpha)

theorem find_alpha (alpha : ℝ) (x y : ℝ → ℝ) (PA PB: ℝ → ℝ → ℝ) : 
  (curve_eq x y) ∧
  (∀ t, line_eq_passes_point alpha t = (x t, y t)) ∧ 
  (∀ A B, PA x y = sqrt 5 ∧ PB x y = sqrt 5) → 
  (alpha = π / 3 ∨ alpha = 2 * π / 3) :=
sorry

end find_alpha_l130_130747


namespace solve_quadratic_l130_130321

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, 
  (-6) * x1^2 + 11 * x1 - 3 = 0 ∧ (-6) * x2^2 + 11 * x2 - 3 = 0 ∧ x1 = 1.5 ∧ x2 = 1 / 3 :=
by
  sorry

end solve_quadratic_l130_130321


namespace solve_for_q_l130_130489

theorem solve_for_q
  (n m q : ℚ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m - n) / 66)
  (h3 : 5 / 6 = (q - m) / 150) :
  q = 230 :=
by
  sorry

end solve_for_q_l130_130489


namespace log_add_property_l130_130399

theorem log_add_property (log : ℝ → ℝ) (h1 : ∀ a b : ℝ, 0 < a → 0 < b → log a + log b = log (a * b)) (h2 : log 10 = 1) :
  log 5 + log 2 = 1 :=
by
  sorry

end log_add_property_l130_130399


namespace positive_integers_are_N_star_l130_130841

def Q := { x : ℚ | true } -- The set of rational numbers
def N := { x : ℕ | true } -- The set of natural numbers
def N_star := { x : ℕ | x > 0 } -- The set of positive integers
def Z := { x : ℤ | true } -- The set of integers

theorem positive_integers_are_N_star : 
  ∀ x : ℕ, (x ∈ N_star) ↔ (x > 0) := 
sorry

end positive_integers_are_N_star_l130_130841


namespace pure_imaginary_a_zero_l130_130456

theorem pure_imaginary_a_zero (a : ℝ) (i : ℂ) (hi : i^2 = -1) :
  (z = (1 - (a:ℝ)^2 * i) / i) ∧ (∀ (z : ℂ), z.re = 0 → z = (0 : ℂ)) → a = 0 :=
by
  sorry

end pure_imaginary_a_zero_l130_130456


namespace farmer_crops_remaining_l130_130367

theorem farmer_crops_remaining
  (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (half_destroyed : ℚ) :
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  half_destroyed = 1 / 2 →
  (corn_rows * corn_per_row + potato_rows * potatoes_per_row) * (1 - half_destroyed) = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end farmer_crops_remaining_l130_130367


namespace cyclic_sum_ineq_l130_130891

theorem cyclic_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) 
  ≥ (1 / 3) * (a + b + c) :=
by
  sorry

end cyclic_sum_ineq_l130_130891


namespace students_taking_neither_l130_130768

theorem students_taking_neither (total_students : ℕ)
    (students_math : ℕ) (students_physics : ℕ) (students_chemistry : ℕ)
    (students_math_physics : ℕ) (students_physics_chemistry : ℕ) (students_math_chemistry : ℕ)
    (students_all_three : ℕ) :
    total_students = 60 →
    students_math = 40 →
    students_physics = 30 →
    students_chemistry = 25 →
    students_math_physics = 18 →
    students_physics_chemistry = 10 →
    students_math_chemistry = 12 →
    students_all_three = 5 →
    (total_students - (students_math + students_physics + students_chemistry - students_math_physics - students_physics_chemistry - students_math_chemistry + students_all_three)) = 5 :=
by
  intros
  sorry

end students_taking_neither_l130_130768


namespace solve_for_a_l130_130940

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l130_130940


namespace bells_toll_together_l130_130860

theorem bells_toll_together (a b c d : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 11) (h4 : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 :=
by
  rw [h1, h2, h3, h4]
  sorry

end bells_toll_together_l130_130860


namespace problem_part1_problem_part2_l130_130790

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l130_130790


namespace circumcircle_tangent_to_AB_l130_130857

theorem circumcircle_tangent_to_AB (ABC : Triangle)
  (hABC : ABC.is_acute)
  (AB_lt_AC : ABC.side_length AB < ABC.side_length AC)
  (Ω : Circle)
  (h_inscribed : ABC.inscribed_in Ω)
  (M : Point)
  (hM : M = ABC.centroid)
  (AH : Line)
  (hAH : AH = ABC.altitude_from A)
  (A' : Point)
  (hA' : Line.ray M H ∩ Ω = A') :
  ABC.circumcircle_tangent A' H B AB :=
  sorry

end circumcircle_tangent_to_AB_l130_130857


namespace length_AB_of_parallelogram_l130_130826

theorem length_AB_of_parallelogram
  (AD BC : ℝ) (AB CD : ℝ) 
  (h1 : AD = 5) 
  (h2 : BC = 5) 
  (h3 : AB = CD)
  (h4 : AD + BC + AB + CD = 14) : 
  AB = 2 :=
by
  sorry

end length_AB_of_parallelogram_l130_130826


namespace feathers_to_cars_ratio_l130_130643

theorem feathers_to_cars_ratio (initial_feathers : ℕ) (final_feathers : ℕ) (cars_dodged : ℕ)
  (h₁ : initial_feathers = 5263) (h₂ : final_feathers = 5217) (h₃ : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 :=
by
  sorry

end feathers_to_cars_ratio_l130_130643


namespace integral_cos_square_div_one_plus_cos_minus_sin_squared_l130_130047

theorem integral_cos_square_div_one_plus_cos_minus_sin_squared:
  ∫ x in (-2 * Real.pi / 3 : Real)..0, (Real.cos x)^2 / (1 + Real.cos x - Real.sin x)^2 = (Real.sqrt 3) / 2 - Real.log 2 := 
by
  sorry

end integral_cos_square_div_one_plus_cos_minus_sin_squared_l130_130047


namespace next_term_in_geometric_sequence_l130_130679

theorem next_term_in_geometric_sequence (y : ℝ) : 
  let a := 3
  let r := 4*y 
  let t4 := 192*y^3 
  r * t4 = 768*y^4 :=
by
  sorry

end next_term_in_geometric_sequence_l130_130679


namespace contrapositive_prop_l130_130175

theorem contrapositive_prop {α : Type} [Mul α] [Zero α] (a b : α) : 
  (a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0) :=
by sorry

end contrapositive_prop_l130_130175


namespace weight_loss_percentage_l130_130199

theorem weight_loss_percentage {W : ℝ} (hW : 0 < W) :
  (((W - ((1 - 0.13 + 0.02 * (1 - 0.13)) * W)) / W) * 100) = 11.26 :=
by
  sorry

end weight_loss_percentage_l130_130199


namespace cone_filled_with_water_to_2_3_height_l130_130062

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l130_130062


namespace library_visitor_ratio_l130_130469

theorem library_visitor_ratio (T : ℕ) (h1 : 50 + T + 20 * 4 = 250) : T / 50 = 2 :=
by
  sorry

end library_visitor_ratio_l130_130469


namespace greatest_integer_x_l130_130835

theorem greatest_integer_x (x : ℤ) (h : (5 : ℚ) / 8 > (x : ℚ) / 17) : x ≤ 10 :=
by
  sorry

end greatest_integer_x_l130_130835


namespace george_monthly_income_l130_130113

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end george_monthly_income_l130_130113


namespace find_a_l130_130942

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l130_130942


namespace percent_volume_filled_with_water_l130_130065

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l130_130065


namespace isosceles_triangle_third_side_l130_130299

theorem isosceles_triangle_third_side (a b : ℝ) (h₁ : a = 4) (h₂ : b = 9) (h₃ : a = b ∨ ∃ c, c = 9 ∧ (a = c ∨ b = c) ∧ (a + b > c ∧ a + c > b ∧ b + c > a)) :
  a = 9 ∨ b = 9 :=
by
  sorry

end isosceles_triangle_third_side_l130_130299


namespace solve_for_x_l130_130454

theorem solve_for_x (x : ℝ) (hp : 0 < x) (h : 4 * x^2 = 1024) : x = 16 :=
sorry

end solve_for_x_l130_130454


namespace sum_geometric_series_is_correct_l130_130015

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l130_130015


namespace julias_change_l130_130298

theorem julias_change :
  let snickers := 2
  let mms := 3
  let cost_snickers := 1.5
  let cost_mms := 2 * cost_snickers
  let money_given := 2 * 10
  let total_cost := snickers * cost_snickers + mms * cost_mms
  let change := money_given - total_cost
  change = 8 :=
by
  sorry

end julias_change_l130_130298


namespace find_a_l130_130955

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l130_130955


namespace kids_waiting_for_swings_l130_130182

theorem kids_waiting_for_swings (x : ℕ) (h1 : 2 * 60 = 120) 
  (h2 : ∀ y, y = 2 → (y * x = 2 * x)) 
  (h3 : 15 * (2 * x) = 30 * x)
  (h4 : 120 * x - 30 * x = 270) : x = 3 :=
sorry

end kids_waiting_for_swings_l130_130182


namespace parabola_points_l130_130343

theorem parabola_points :
  {p : ℝ × ℝ | p.2 = p.1^2 - 1 ∧ p.2 = 3} = {(-2, 3), (2, 3)} :=
by
  sorry

end parabola_points_l130_130343


namespace interest_earned_is_correct_l130_130642

-- Define the principal amount, interest rate, and duration
def principal : ℝ := 2000
def rate : ℝ := 0.02
def duration : ℕ := 3

-- The compound interest formula to calculate the future value
def future_value (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r) ^ n

-- Calculate the interest earned
def interest (P : ℝ) (A : ℝ) : ℝ := A - P

-- Theorem statement: The interest Bart earns after 3 years is 122 dollars
theorem interest_earned_is_correct : interest principal (future_value principal rate duration) = 122 :=
by
  sorry

end interest_earned_is_correct_l130_130642


namespace four_digit_num_condition_l130_130285

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l130_130285


namespace minimum_rubles_to_reverse_order_of_chips_100_l130_130301

noncomputable def minimum_rubles_to_reverse_order_of_chips (n : ℕ) : ℕ :=
if n = 100 then 61 else 0

theorem minimum_rubles_to_reverse_order_of_chips_100 :
  minimum_rubles_to_reverse_order_of_chips 100 = 61 :=
by sorry

end minimum_rubles_to_reverse_order_of_chips_100_l130_130301


namespace total_prairie_area_l130_130373

theorem total_prairie_area (A B C : ℕ) (Z1 Z2 Z3 : ℚ) (unaffected : ℕ) (total_area : ℕ) : 
  A = 55000 →
  B = 35000 →
  C = 45000 →
  Z1 = 0.80 →
  Z2 = 0.60 →
  Z3 = 0.95 →
  unaffected = 1500 →
  total_area = Z1 * A + Z2 * B + Z3 * C + unaffected →
  total_area = 109250 := sorry

end total_prairie_area_l130_130373


namespace books_bought_l130_130782

noncomputable def totalCost : ℤ :=
  let numFilms := 9
  let costFilm := 5
  let numCDs := 6
  let costCD := 3
  let costBook := 4
  let totalSpent := 79
  totalSpent - (numFilms * costFilm + numCDs * costCD)

theorem books_bought : ∃ B : ℤ, B * 4 = totalCost := by
  sorry

end books_bought_l130_130782


namespace avg_speed_3x_km_l130_130844

-- Definitions based on the conditions
def distance1 (x : ℕ) : ℕ := x
def speed1 : ℕ := 90
def distance2 (x : ℕ) : ℕ := 2 * x
def speed2 : ℕ := 20

-- The total distance covered
def total_distance (x : ℕ) : ℕ := distance1 x + distance2 x

-- The time taken for each part of the journey
def time1 (x : ℕ) : ℚ := distance1 x / speed1
def time2 (x : ℕ) : ℚ := distance2 x / speed2

-- The total time taken
def total_time (x : ℕ) : ℚ := time1 x + time2 x

-- The average speed
def average_speed (x : ℕ) : ℚ := total_distance x / total_time x

-- The theorem we want to prove
theorem avg_speed_3x_km (x : ℕ) : average_speed x = 27 := by
  sorry

end avg_speed_3x_km_l130_130844


namespace chess_tournament_max_N_l130_130606

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l130_130606


namespace pairings_equal_l130_130300

-- Definitions for City A
def A_girls (n : ℕ) : Type := Fin n
def A_boys (n : ℕ) : Type := Fin n
def A_knows (n : ℕ) (g : A_girls n) (b : A_boys n) : Prop := True

-- Definitions for City B
def B_girls (n : ℕ) : Type := Fin n
def B_boys (n : ℕ) : Type := Fin (2 * n - 1)
def B_knows (n : ℕ) (i : Fin n) (j : Fin (2 * n - 1)) : Prop :=
  j.val < 2 * (i.val + 1)

-- Function to count the number of ways to pair r girls and r boys in city A
noncomputable def A (n r : ℕ) : ℕ := 
  if h : r ≤ n then 
    Nat.choose n r * Nat.choose n r * (r.factorial)
  else 0

-- Recurrence relation for city B
noncomputable def B (n r : ℕ) : ℕ :=
  if r = 0 then 1 else if r > n then 0 else
  if n < 2 then if r = 1 then (2 - 1) * 2 else 0 else
  B (n - 1) r + (2 * n - r) * B (n - 1) (r - 1)

-- We want to prove that number of pairings in city A equals number of pairings in city B for any r <= n
theorem pairings_equal (n r : ℕ) (h : r ≤ n) : A n r = B n r := sorry

end pairings_equal_l130_130300


namespace abs_pos_of_ne_zero_l130_130032

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l130_130032


namespace metallic_weight_problem_l130_130541

variables {m1 m2 m3 m4 : ℝ}

theorem metallic_weight_problem
  (h_total : m1 + m2 + m3 + m4 = 35)
  (h1 : m1 = 1.5 * m2)
  (h2 : m2 = (3/4) * m3)
  (h3 : m3 = (5/6) * m4) :
  m4 = 105 / 13 :=
sorry

end metallic_weight_problem_l130_130541


namespace largest_common_multiple_of_7_8_l130_130101

noncomputable def largest_common_multiple_of_7_8_sub_2 (n : ℕ) : ℕ :=
  if n <= 100 then n else 0

theorem largest_common_multiple_of_7_8 :
  ∃ x : ℕ, x <= 100 ∧ (x - 2) % Nat.lcm 7 8 = 0 ∧ x = 58 :=
by
  let x := 58
  use x
  have h1 : x <= 100 := by norm_num
  have h2 : (x - 2) % Nat.lcm 7 8 = 0 := by norm_num
  have h3 : x = 58 := by norm_num
  exact ⟨h1, h2, h3⟩

end largest_common_multiple_of_7_8_l130_130101


namespace iron_weight_l130_130634

theorem iron_weight 
  (A : ℝ) (hA : A = 0.83) 
  (I : ℝ) (hI : I = A + 10.33) : 
  I = 11.16 := 
by 
  sorry

end iron_weight_l130_130634


namespace largest_of_five_consecutive_ints_15120_l130_130571

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l130_130571


namespace beef_original_weight_l130_130218

theorem beef_original_weight (W : ℝ) (h : 0.65 * W = 546): W = 840 :=
sorry

end beef_original_weight_l130_130218


namespace pairs_of_integers_l130_130097

-- The main theorem to prove:
theorem pairs_of_integers (x y : ℤ) :
  y ^ 2 = x ^ 3 + 16 ↔ (x = 0 ∧ (y = 4 ∨ y = -4)) :=
by sorry

end pairs_of_integers_l130_130097


namespace A_work_days_l130_130211

theorem A_work_days (x : ℝ) (H : 3 * (1 / x + 1 / 20) = 0.35) : x = 15 := 
by
  sorry

end A_work_days_l130_130211


namespace point_on_line_l130_130441

theorem point_on_line (m : ℝ) : (2 = m - 1) → (m = 3) :=
by sorry

end point_on_line_l130_130441


namespace smallest_positive_integer_l130_130195

def is_prime_gt_60 (n : ℕ) : Prop :=
  n > 60 ∧ Prime n

def smallest_integer_condition (k : ℕ) : Prop :=
  ¬ Prime k ∧ ¬ (∃ m : ℕ, m * m = k) ∧ 
  ∀ p : ℕ, Prime p → p ∣ k → p > 60

theorem smallest_positive_integer : ∃ k : ℕ, k = 4087 ∧ smallest_integer_condition k := by
  sorry

end smallest_positive_integer_l130_130195


namespace domain_of_func_l130_130815

noncomputable def func (x : ℝ) : ℝ := 1 / (2 * x - 1)

theorem domain_of_func :
  ∀ x : ℝ, x ≠ 1 / 2 ↔ ∃ y : ℝ, y = func x := sorry

end domain_of_func_l130_130815


namespace koala_fiber_intake_l130_130144

theorem koala_fiber_intake (x : ℝ) (h : 0.30 * x = 12) : x = 40 := 
sorry

end koala_fiber_intake_l130_130144


namespace father_l130_130355

theorem father's_age (M F : ℕ) 
  (h1 : M = (2 / 5 : ℝ) * F)
  (h2 : M + 14 = (1 / 2 : ℝ) * (F + 14)) : 
  F = 70 := 
  sorry

end father_l130_130355


namespace equation_of_parallel_line_l130_130816

theorem equation_of_parallel_line (x y : ℝ) :
  (∀ b : ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → b = 0) →
  (∀ x y b: ℝ, 2 * x + y + b = 0 → x = -1 → y = 2 → 2 * x + y = 0) :=
by
  sorry

end equation_of_parallel_line_l130_130816


namespace line_through_points_has_sum_m_b_3_l130_130495

-- Define the structure that two points are given
structure LineThroughPoints (P1 P2 : ℝ × ℝ) : Prop :=
  (slope_intercept_form : ∃ m b, (P1.snd = m * P1.fst + b) ∧ (P2.snd = b)) 

-- Define the particular points
def point1 : ℝ × ℝ := (-2, 0)
def point2 : ℝ × ℝ := (0, 2)

-- The theorem statement
theorem line_through_points_has_sum_m_b_3 
  (h : LineThroughPoints point1 point2) : 
  ∃ m b, (point1.snd = m * point1.fst + b) ∧ (point2.snd = b) ∧ (m + b = 3) :=
by
  sorry

end line_through_points_has_sum_m_b_3_l130_130495


namespace amy_final_money_l130_130378

theorem amy_final_money :
  let initial_money := 2
  let chore_payment := 5 * 13
  let birthday_gift := 3
  let toy_cost := 12
  let remaining_money := initial_money + chore_payment + birthday_gift - toy_cost
  let grandparents_reward := 2 * remaining_money
  remaining_money + grandparents_reward = 174 := 
by
  sorry

end amy_final_money_l130_130378


namespace find_A_l130_130437

variable {A B C : ℚ}

theorem find_A (h1 : A = 1/2 * B) (h2 : B = 3/4 * C) (h3 : A + C = 55) : A = 15 :=
by
  sorry

end find_A_l130_130437


namespace find_a_from_perpendicular_lines_l130_130898

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l130_130898


namespace part_I_part_II_l130_130906

open Set

variable (a b : ℝ)

theorem part_I (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hB_sub_A : B ⊆ A) : a = -1 :=
  sorry

theorem part_II (A : Set ℝ) (B : Set ℝ) (hA_def : A = { x | a * x^2 + b * x + 1 = 0 })
  (hB_def : B = { -1, 1 }) (hA_inter_B_nonempty : A ∩ B ≠ ∅) : a^2 - b^2 + 2 * a = -1 :=
  sorry

end part_I_part_II_l130_130906


namespace boy_work_completion_days_l130_130082

theorem boy_work_completion_days (M W B : ℚ) (D : ℚ)
  (h1 : M + W + B = 1 / 4)
  (h2 : M = 1 / 6)
  (h3 : W = 1 / 36)
  (h4 : B = 1 / D) :
  D = 18 := by
  sorry

end boy_work_completion_days_l130_130082


namespace cos_A_side_c_l130_130532

-- helper theorem for cosine rule usage
theorem cos_A (a b c : ℝ) (cosA cosB cosC : ℝ) (h : 3 * a * cosA = c * cosB + b * cosC) : cosA = 1 / 3 :=
by
  sorry

-- main statement combining conditions 1 and 2 with side value results
theorem side_c (a b c : ℝ) (cosA cosB cosC : ℝ) (h1 : 3 * a * cosA = c * cosB + b * cosC) (h2 : cosB + cosC = 0) (h3 : a = 1) : c = 2 :=
by
  have h_cosA : cosA = 1 / 3 := cos_A a b c cosA cosB cosC h1
  sorry

end cos_A_side_c_l130_130532


namespace ellipse_eccentricity_l130_130953

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l130_130953


namespace ellipse_eccentricity_l130_130952

theorem ellipse_eccentricity (a1 a2 b1 b2 c1 c2 e1 e2 : ℝ)
  (h1 : a1 > 1)
  (h2 : 4 * (a1^2 - 1) = a1^2)
  (h3 : a2 = 2)
  (h4 : b2 = 1)
  (h5 : c2 = Real.sqrt (a2^2 - b2^2))
  (h6 : e2 = c2 / a2)
  (h7 : e2 = Real.sqrt 3 * e1)
  (h8 : e1 = c1 / a1)
  (h9 : c1 = a1 / 2):
  a1 = 2 * Real.sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_l130_130952


namespace range_of_a_l130_130588

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) 
    (h_deriv : ∀ x ∈ set.Ioi 0, f' x = f'(x))
    (h_diff_eq : ∀ x ∈ set.Ioi 0, x * (f' x) - x * (f x) = exp x)
    (h_f1 : f 1 = 2 * exp 1)
    (h_f_constraint : f (1 - 1/(2*a)) ≤ exp (1/exp 1)) :
  (1/2 : ℝ) < a ∧ a ≤ exp 1 / (2 * (exp 1 - 1)) :=
sorry

end range_of_a_l130_130588


namespace calculate_value_of_expression_l130_130224

theorem calculate_value_of_expression :
  3.5 * 7.2 * (6.3 - 1.4) = 122.5 :=
  by
  sorry

end calculate_value_of_expression_l130_130224


namespace new_perimeter_is_20_l130_130324

/-
Ten 1x1 square tiles are arranged to form a figure whose outside edges form a polygon with a perimeter of 16 units.
Four additional tiles of the same size are added to the figure so that each new tile shares at least one side with 
one of the squares in the original figure. Prove that the new perimeter of the figure could be 20 units.
-/

theorem new_perimeter_is_20 (initial_perimeter : ℕ) (num_initial_tiles : ℕ) 
                            (num_new_tiles : ℕ) (shared_sides : ℕ) 
                            (total_tiles : ℕ) : 
  initial_perimeter = 16 → num_initial_tiles = 10 → num_new_tiles = 4 → 
  shared_sides ≤ 8 → total_tiles = 14 → (initial_perimeter + 2 * (num_new_tiles - shared_sides)) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end new_perimeter_is_20_l130_130324


namespace imaginary_part_of_z_l130_130958

-- Define the complex number z
def z : ℂ :=
  3 - 2 * Complex.I

-- Lean theorem statement to prove the imaginary part of z is -2
theorem imaginary_part_of_z :
  Complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l130_130958


namespace both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l130_130135

variables (p1 p2 : Prop)

theorem both_shots_hit (p1 p2 : Prop) : (p1 ∧ p2) ↔ (p1 ∧ p2) :=
by sorry

theorem both_shots_missed (p1 p2 : Prop) : (¬p1 ∧ ¬p2) ↔ (¬p1 ∧ ¬p2) :=
by sorry

theorem exactly_one_shot_hit (p1 p2 : Prop) : ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) ↔ ((p1 ∧ ¬p2) ∨ (p2 ∧ ¬p1)) :=
by sorry

theorem at_least_one_shot_hit (p1 p2 : Prop) : (p1 ∨ p2) ↔ (p1 ∨ p2) :=
by sorry

end both_shots_hit_both_shots_missed_exactly_one_shot_hit_at_least_one_shot_hit_l130_130135


namespace find_R_l130_130452

theorem find_R (a b Q R : ℕ) (ha_prime : Prime a) (hb_prime : Prime b) (h_distinct : a ≠ b)
  (h1 : a^2 - a * Q + R = 0) (h2 : b^2 - b * Q + R = 0) : R = 6 :=
sorry

end find_R_l130_130452


namespace russell_oranges_taken_l130_130752

-- Conditions
def initial_oranges : ℕ := 60
def oranges_left : ℕ := 25

-- Statement to prove
theorem russell_oranges_taken : ℕ :=
  initial_oranges - oranges_left = 35

end russell_oranges_taken_l130_130752


namespace common_tangent_curves_l130_130132

theorem common_tangent_curves (s t a : ℝ) (e : ℝ) (he : e > 0) :
  (t = (1 / (2 * e)) * s^2) →
  (t = a * Real.log s) →
  (s / e = a / s) →
  a = 1 :=
by
  intro h1 h2 h3
  sorry

end common_tangent_curves_l130_130132


namespace turtle_finishes_in_10_minutes_l130_130374

def skunk_time : ℕ := 6
def rabbit_speed_ratio : ℕ := 3
def turtle_speed_ratio : ℕ := 5
def rabbit_time := skunk_time / rabbit_speed_ratio
def turtle_time := turtle_speed_ratio * rabbit_time

theorem turtle_finishes_in_10_minutes : turtle_time = 10 := by
  sorry

end turtle_finishes_in_10_minutes_l130_130374


namespace xiao_wang_program_output_l130_130039

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l130_130039


namespace find_angle_C_l130_130764

variable {A B C a b c : ℝ}

theorem find_angle_C (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π) (h8 : a > 0) (h9 : b > 0) (h10 : c > 0) 
  (h11 : (a + b - c) * (a + b + c) = a * b) : C = 2 * π / 3 :=
by
  sorry

end find_angle_C_l130_130764


namespace expected_value_girls_left_of_boys_l130_130708


noncomputable def expected_girls_to_left_of_all_boys (n m : ℕ) [fact (0 < n)] [fact (0 < m)] : ℝ :=
  let total := n + m
  (m : ℝ) / (total + 1)

theorem expected_value_girls_left_of_boys : 
  let n := 10
  let m := 7
  expected_girls_to_left_of_all_boys n m = 7 / 11 :=
by
  -- This is the key part of the theorem as described in the task
  sorry

end expected_value_girls_left_of_boys_l130_130708


namespace number_of_trips_l130_130832

theorem number_of_trips (bags_per_trip : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ)
  (h1 : bags_per_trip = 10)
  (h2 : weight_per_bag = 50)
  (h3 : total_weight = 10000) : 
  total_weight / (bags_per_trip * weight_per_bag) = 20 :=
by
  sorry

end number_of_trips_l130_130832


namespace functional_equation_solution_exists_l130_130562

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution_exists (f : ℝ → ℝ) (h : ∀ x y, 0 < x → 0 < y → f x * f y = 2 * f (x + y * f x)) :
  ∃ c : ℝ, ∀ x, 0 < x → f x = x + c := 
sorry

end functional_equation_solution_exists_l130_130562


namespace seahawks_field_goals_l130_130830

-- Defining the conditions as hypotheses
def final_score_seahawks : ℕ := 37
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3
def touchdowns_seahawks : ℕ := 4

-- Stating the goal to prove
theorem seahawks_field_goals : 
  (final_score_seahawks - touchdowns_seahawks * points_per_touchdown) / points_per_fieldgoal = 3 := 
by 
  sorry

end seahawks_field_goals_l130_130830


namespace correlation_coefficients_l130_130690

-- Definition of the variables and constants
def relative_risks_starting_age : List (ℕ × ℝ) := [(16, 15.10), (18, 12.81), (20, 9.72), (22, 3.21)]
def relative_risks_cigarettes_per_day : List (ℕ × ℝ) := [(10, 7.5), (20, 9.5), (30, 16.6)]

def r1 : ℝ := -- The correlation coefficient between starting age and relative risk
  sorry

def r2 : ℝ := -- The correlation coefficient between number of cigarettes per day and relative risk
  sorry

theorem correlation_coefficients :
  r1 < 0 ∧ 0 < r2 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end correlation_coefficients_l130_130690


namespace four_digit_numbers_property_l130_130282

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l130_130282


namespace walk_to_school_l130_130295

theorem walk_to_school (W P : ℕ) (h1 : W + P = 41) (h2 : W = P + 3) : W = 22 :=
by 
  sorry

end walk_to_school_l130_130295


namespace part1_part2_l130_130786

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l130_130786


namespace hcf_of_two_numbers_l130_130667

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end hcf_of_two_numbers_l130_130667


namespace each_regular_tire_distance_used_l130_130848

-- Define the conditions of the problem
def total_distance_traveled : ℕ := 50000
def spare_tire_distance : ℕ := 2000
def regular_tires_count : ℕ := 4

-- Using these conditions, we will state the problem as a theorem
theorem each_regular_tire_distance_used : 
  (total_distance_traveled - spare_tire_distance) / regular_tires_count = 12000 :=
by
  sorry

end each_regular_tire_distance_used_l130_130848


namespace total_flour_amount_l130_130209

-- Define the initial amount of flour in the bowl
def initial_flour : ℝ := 2.75

-- Define the amount of flour added by the baker
def added_flour : ℝ := 0.45

-- Prove that the total amount of flour is 3.20 kilograms
theorem total_flour_amount : initial_flour + added_flour = 3.20 :=
by
  sorry

end total_flour_amount_l130_130209


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l130_130789

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l130_130789


namespace smallest_number_of_small_bottles_l130_130540

def minimum_bottles_needed (large_bottle_capacity : ℕ) (small_bottle1 : ℕ) (small_bottle2 : ℕ) : ℕ :=
  if large_bottle_capacity = 720 ∧ small_bottle1 = 40 ∧ small_bottle2 = 45 then 16 else 0

theorem smallest_number_of_small_bottles :
  minimum_bottles_needed 720 40 45 = 16 := by
  sorry

end smallest_number_of_small_bottles_l130_130540


namespace part1_part2_l130_130470

def f (x : ℝ) : ℝ :=
  abs (x - 1) + 2 * abs (x + 5)

theorem part1 : ∀ x, f x < 10 ↔ (x > -19 / 3 ∧ x ≤ -5) ∨ (-5 < x ∧ x < -1) :=
  sorry

theorem part2 (a b x : ℝ) (ha : abs a < 3) (hb : abs b < 3) :
  abs (a + b) + abs (a - b) < f x :=
  sorry

end part1_part2_l130_130470


namespace necessary_and_sufficient_condition_l130_130582

theorem necessary_and_sufficient_condition (x : ℝ) :
  (x - 2) * (x - 3) ≤ 0 ↔ |x - 2| + |x - 3| = 1 := sorry

end necessary_and_sufficient_condition_l130_130582


namespace geometric_series_sum_l130_130026

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l130_130026


namespace sum_first_n_abs_terms_arithmetic_seq_l130_130434

noncomputable def sum_abs_arithmetic_sequence (n : ℕ) (h : n ≥ 3) : ℚ :=
  if n = 1 ∨ n = 2 then (n * (4 + 7 - 3 * n)) / 2
  else (3 * n^2 - 11 * n + 20) / 2

theorem sum_first_n_abs_terms_arithmetic_seq (n : ℕ) (h : n ≥ 3) :
  sum_abs_arithmetic_sequence n h = (3 * n^2) / 2 - (11 * n) / 2 + 10 :=
sorry

end sum_first_n_abs_terms_arithmetic_seq_l130_130434


namespace cylinder_volume_l130_130676

-- Definitions based on conditions
def lateral_surface_to_rectangle (generatrix_a generatrix_b : ℝ) (volume : ℝ) :=
  -- Condition: Rectangle with sides 8π and 4π
  (generatrix_a = 8 * Real.pi ∧ volume = 32 * Real.pi^2) ∨
  (generatrix_a = 4 * Real.pi ∧ volume = 64 * Real.pi^2)

-- Statement
theorem cylinder_volume (generatrix_a generatrix_b : ℝ)
  (h : (generatrix_a = 8 * Real.pi ∨ generatrix_b = 4 * Real.pi) ∧ (generatrix_b = 4 * Real.pi ∨ generatrix_b = 8 * Real.pi)) :
  ∃ (volume : ℝ), lateral_surface_to_rectangle generatrix_a generatrix_b volume :=
sorry

end cylinder_volume_l130_130676


namespace average_ABC_l130_130229

/-- Given three numbers A, B, and C such that 1503C - 3006A = 6012 and 1503B + 4509A = 7509,
their average is 3  -/
theorem average_ABC (A B C : ℚ) 
  (h1 : 1503 * C - 3006 * A = 6012) 
  (h2 : 1503 * B + 4509 * A = 7509) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_l130_130229


namespace runner_time_difference_l130_130698

theorem runner_time_difference (v : ℝ) (h1 : 0 < v) (h2 : 0 < 20 / v) (h3 : 8 = 40 / v) :
  8 - (20 / v) = 4 := by
  sorry

end runner_time_difference_l130_130698


namespace values_of_a_and_b_l130_130884

def is_root (a b x : ℝ) : Prop := x^2 - 2*a*x + b = 0

noncomputable def A : Set ℝ := {-1, 1}
noncomputable def B (a b : ℝ) : Set ℝ := {x | is_root a b x}

theorem values_of_a_and_b (a b : ℝ) (h_nonempty : Set.Nonempty (B a b)) (h_union : A ∪ B a b = A) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1) :=
sorry

end values_of_a_and_b_l130_130884


namespace four_digit_numbers_with_property_l130_130267

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l130_130267


namespace problem_one_problem_two_l130_130137

variables (a₁ a₂ a₃ : ℤ) (n : ℕ)
def arith_sequence : Prop :=
  a₁ + a₂ + a₃ = 21 ∧ a₁ * a₂ * a₃ = 231

theorem problem_one (h : arith_sequence a₁ a₂ a₃) : a₂ = 7 :=
sorry

theorem problem_two (h : arith_sequence a₁ a₂ a₃) :
  (∃ d : ℤ, (d = -4 ∨ d = 4) ∧ (a_n = a₁ + (n - 1) * d ∨ a_n = a₃ + (n - 1) * d)) :=
sorry

end problem_one_problem_two_l130_130137


namespace customers_per_table_l130_130549

theorem customers_per_table (total_tables : ℝ) (left_tables : ℝ) (total_customers : ℕ)
  (h1 : total_tables = 44.0)
  (h2 : left_tables = 12.0)
  (h3 : total_customers = 256) :
  total_customers / (total_tables - left_tables) = 8 :=
by {
  sorry
}

end customers_per_table_l130_130549


namespace sugar_concentration_after_adding_water_l130_130550

def initial_mass_of_sugar_water : ℝ := 90
def initial_sugar_concentration : ℝ := 0.10
def final_sugar_concentration : ℝ := 0.08
def mass_of_water_added : ℝ := 22.5

theorem sugar_concentration_after_adding_water 
  (m_sugar_water : ℝ := initial_mass_of_sugar_water)
  (c_initial : ℝ := initial_sugar_concentration)
  (c_final : ℝ := final_sugar_concentration)
  (m_water_added : ℝ := mass_of_water_added) :
  (m_sugar_water * c_initial = (m_sugar_water + m_water_added) * c_final) := 
sorry

end sugar_concentration_after_adding_water_l130_130550


namespace rahim_average_price_l130_130806

/-- 
Rahim bought 40 books for Rs. 600 from one shop and 20 books for Rs. 240 from another.
What is the average price he paid per book?
-/
def books1 : ℕ := 40
def cost1 : ℕ := 600
def books2 : ℕ := 20
def cost2 : ℕ := 240
def totalBooks : ℕ := books1 + books2
def totalCost : ℕ := cost1 + cost2
def averagePricePerBook : ℕ := totalCost / totalBooks

theorem rahim_average_price :
  averagePricePerBook = 14 :=
by
  sorry

end rahim_average_price_l130_130806


namespace solve_equation_l130_130099

open Real

theorem solve_equation :
  ∀ x : ℝ, (
    (1 / ((x - 2) * (x - 3))) +
    (1 / ((x - 3) * (x - 4))) +
    (1 / ((x - 4) * (x - 5))) = (1 / 12)
  ) ↔ (x = 5 + sqrt 19 ∨ x = 5 - sqrt 19) := 
by 
  sorry

end solve_equation_l130_130099


namespace average_charge_proof_l130_130515

noncomputable def averageChargePerPerson
  (chargeFirstDay : ℝ)
  (chargeSecondDay : ℝ)
  (chargeThirdDay : ℝ)
  (chargeFourthDay : ℝ)
  (ratioFirstDay : ℝ)
  (ratioSecondDay : ℝ)
  (ratioThirdDay : ℝ)
  (ratioFourthDay : ℝ)
  : ℝ :=
  let totalRevenue := ratioFirstDay * chargeFirstDay + ratioSecondDay * chargeSecondDay + ratioThirdDay * chargeThirdDay + ratioFourthDay * chargeFourthDay
  let totalVisitors := ratioFirstDay + ratioSecondDay + ratioThirdDay + ratioFourthDay
  totalRevenue / totalVisitors

theorem average_charge_proof :
  averageChargePerPerson 25 15 7.5 2.5 3 7 11 19 = 7.75 := by
  simp [averageChargePerPerson]
  sorry

end average_charge_proof_l130_130515


namespace appropriate_mass_units_l130_130723

def unit_of_mass_basket_of_eggs : String :=
  if 5 = 5 then "kilograms" else "unknown"

def unit_of_mass_honeybee : String :=
  if 5 = 5 then "grams" else "unknown"

def unit_of_mass_tank : String :=
  if 6 = 6 then "tons" else "unknown"

theorem appropriate_mass_units :
  unit_of_mass_basket_of_eggs = "kilograms" ∧
  unit_of_mass_honeybee = "grams" ∧
  unit_of_mass_tank = "tons" :=
by {
  -- skip the proof
  sorry
}

end appropriate_mass_units_l130_130723


namespace james_collected_on_first_day_l130_130779

-- Conditions
variables (x : ℕ) -- the number of tins collected on the first day
variable (h1 : 500 = x + 3 * x + (3 * x - 50) + 4 * 50) -- total number of tins collected

-- Theorem to be proved
theorem james_collected_on_first_day : x = 50 :=
by
  sorry

end james_collected_on_first_day_l130_130779


namespace fred_total_earnings_l130_130737

def fred_earnings (earnings_per_hour hours_worked : ℝ) : ℝ := earnings_per_hour * hours_worked

theorem fred_total_earnings :
  fred_earnings 12.5 8 = 100 := by
sorry

end fred_total_earnings_l130_130737


namespace range_of_k_for_domain_real_l130_130444

theorem range_of_k_for_domain_real (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 6 * k * x + (k + 8) ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end range_of_k_for_domain_real_l130_130444


namespace ratio_child_to_jane_babysit_l130_130311

-- Definitions of the conditions
def jane_current_age : ℕ := 32
def years_since_jane_stopped_babysitting : ℕ := 10
def oldest_person_current_age : ℕ := 24

-- Derived definitions
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped_babysitting
def oldest_person_age_when_jane_stopped : ℕ := oldest_person_current_age - years_since_jane_stopped_babysitting

-- Statement of the problem to be proven in Lean 4
theorem ratio_child_to_jane_babysit :
  (oldest_person_age_when_jane_stopped : ℚ) / (jane_age_when_stopped : ℚ) = 7 / 11 :=
by
  sorry

end ratio_child_to_jane_babysit_l130_130311


namespace smallest_possible_value_abs_sum_l130_130519

theorem smallest_possible_value_abs_sum : 
  ∀ (x : ℝ), 
    (|x + 3| + |x + 6| + |x + 7| + 2) ≥ 8 :=
by
  sorry

end smallest_possible_value_abs_sum_l130_130519


namespace christmas_distribution_l130_130812

theorem christmas_distribution :
  ∃ (n x : ℕ), 
    (240 + 120 + 1 = 361) ∧
    (n * x = 361) ∧
    (n = 19) ∧
    (x = 19) ∧
    ∃ (a b : ℕ), (a + b = 19) ∧ (a * 5 + b * 6 = 100) :=
by
  sorry

end christmas_distribution_l130_130812


namespace playground_area_22500_l130_130333

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l130_130333


namespace count_possible_values_of_x_l130_130264

theorem count_possible_values_of_x :
  let n := (set.count {x : ℕ | 25 ≤ x ∧ x ≤ 33 ∧ ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c * x < 100 ∧ 3 ≤ b ≤ 100/x}) in
  n = 9 :=
by
  -- Here we must prove the statement by the provided conditions
  sorry

end count_possible_values_of_x_l130_130264


namespace geom_series_sum_l130_130023

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l130_130023


namespace geometric_series_sum_l130_130018

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l130_130018


namespace gcd_of_462_and_330_l130_130190

theorem gcd_of_462_and_330 :
  Nat.gcd 462 330 = 66 :=
sorry

end gcd_of_462_and_330_l130_130190


namespace medians_formula_l130_130881

noncomputable def ma (a b c : ℝ) : ℝ := (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2))
noncomputable def mb (a b c : ℝ) : ℝ := (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2))
noncomputable def mc (a b c : ℝ) : ℝ := (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2))

theorem medians_formula (a b c : ℝ) :
  ma a b c = (1 / 2) * ((2 * b^2 + 2 * c^2 - a^2) ^ (1 / 2)) ∧
  mb a b c = (1 / 2) * ((2 * c^2 + 2 * a^2 - b^2) ^ (1 / 2)) ∧
  mc a b c = (1 / 2) * ((2 * a^2 + 2 * b^2 - c^2) ^ (1 / 2)) :=
by sorry

end medians_formula_l130_130881


namespace discriminant_of_quadratic_l130_130730

def a := 5
def b := 5 + 1/5
def c := 1/5
def discriminant (a b c : ℚ) := b^2 - 4 * a * c

theorem discriminant_of_quadratic :
  discriminant a b c = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l130_130730


namespace exists_quadratic_function_l130_130308

theorem exists_quadratic_function :
  (∃ (a b c : ℝ), ∀ (k : ℕ), k > 0 → (a * (5 / 9 * (10^k - 1))^2 + b * (5 / 9 * (10^k - 1)) + c = 5/9 * (10^(2*k) - 1))) :=
by
  have a := 9 / 5
  have b := 2
  have c := 0
  use a, b, c
  intros k hk
  sorry

end exists_quadratic_function_l130_130308


namespace prime_eq_solution_l130_130472

theorem prime_eq_solution (a b : ℕ) (h1 : Nat.Prime a) (h2 : b > 0)
  (h3 : 9 * (2 * a + b) ^ 2 = 509 * (4 * a + 511 * b)) : 
  (a = 251 ∧ b = 7) :=
sorry

end prime_eq_solution_l130_130472


namespace cone_water_fill_percentage_l130_130074

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l130_130074


namespace distance_from_focus_to_line_l130_130649

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130649


namespace angle_set_equality_l130_130096

open Real

theorem angle_set_equality (α : ℝ) :
  ({sin α, sin (2 * α), sin (3 * α)} = {cos α, cos (2 * α), cos (3 * α)}) ↔ 
  ∃ (k : ℤ), α = π / 8 + (k : ℝ) * (π / 2) :=
by
  sorry

end angle_set_equality_l130_130096


namespace plate_acceleration_l130_130833

theorem plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) (hR : R = 1.25) (hr : r = 0.75) (hm : m = 100) (hα : α = Real.arccos 0.92) (hg : g = 10) : 
  let a := g * Real.sin(α / 2) in
  a = 2 :=
by
  -- Declaration of given data
  have hR : R = 1.25 := hR,
  have hr : r = 0.75 := hr,
  have hm : m = 100 := hm,
  have hα : α = Real.arccos 0.92 := hα,
  have hg : g = 10 := hg,
  
  -- Calculate acceleration
  
  sorry

end plate_acceleration_l130_130833


namespace inequality_solution_l130_130574

theorem inequality_solution (x : ℝ) : (x^3 - 10 * x^2 > -25 * x) ↔ (0 < x ∧ x < 5) ∨ (5 < x) := 
sorry

end inequality_solution_l130_130574


namespace opposite_of_neg_half_l130_130338

-- Define the opposite of a number
def opposite (x : ℝ) : ℝ := -x

-- The theorem we want to prove
theorem opposite_of_neg_half : opposite (-1/2) = 1/2 :=
by
  -- Proof goes here
  sorry

end opposite_of_neg_half_l130_130338


namespace problem_statement_l130_130387

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l130_130387


namespace tangent_line_through_point_l130_130330

theorem tangent_line_through_point (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) : 
  (∃ k : ℝ, 15 * x - 8 * y - 13 = 0) ∨ x = 3 := sorry

end tangent_line_through_point_l130_130330


namespace expression_value_l130_130028

theorem expression_value :
  ( (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) ) = 1 := by
  sorry

end expression_value_l130_130028


namespace percent_filled_cone_l130_130072

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l130_130072


namespace fraction_equation_solution_l130_130573

theorem fraction_equation_solution (x : ℝ) : (4 + x) / (7 + x) = (2 + x) / (3 + x) ↔ x = -1 := 
by
  sorry

end fraction_equation_solution_l130_130573


namespace crayons_taken_out_l130_130993

-- Define the initial and remaining number of crayons
def initial_crayons : ℕ := 7
def remaining_crayons : ℕ := 4

-- Define the proposition to prove
theorem crayons_taken_out : initial_crayons - remaining_crayons = 3 := by
  sorry

end crayons_taken_out_l130_130993


namespace sequence_properties_l130_130120

noncomputable def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
a1 + d * (n - 1)

theorem sequence_properties (d a1 : ℤ) (h_d_ne_zero : d ≠ 0)
(h1 : arithmetic_sequence a1 d 2 + arithmetic_sequence a1 d 4 = 10)
(h2 : (arithmetic_sequence a1 d 2)^2 = (arithmetic_sequence a1 d 1) * (arithmetic_sequence a1 d 5)) :
a1 = 1 ∧ ∀ n : ℕ, n > 0 → arithmetic_sequence 1 2 n = 2 * n - 1 :=
by
  sorry

end sequence_properties_l130_130120


namespace sin_arithmetic_sequence_l130_130726

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 360) :
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ 
  (a = 30 ∨ a = 150 ∨ a = 210 ∨ a = 330) :=
by
  sorry

end sin_arithmetic_sequence_l130_130726


namespace derivative_of_f_eq_f_deriv_l130_130861

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x - (Real.sin a) ^ x

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x * Real.log (Real.cos a) - (Real.sin a) ^ x * Real.log (Real.sin a)

theorem derivative_of_f_eq_f_deriv (a : ℝ) (h : 0 < a ∧ a < Real.pi / 2) :
  (deriv (f a)) = f_deriv a :=
by
  sorry

end derivative_of_f_eq_f_deriv_l130_130861


namespace triangle_properties_l130_130439

theorem triangle_properties (b c : ℝ) (C : ℝ)
  (hb : b = 10)
  (hc : c = 5 * Real.sqrt 6)
  (hC : C = Real.pi / 3) :
  let R := c / (2 * Real.sin C)
  let B := Real.arcsin (b * Real.sin C / c)
  R = 5 * Real.sqrt 2 ∧ B = Real.pi / 4 :=
by
  sorry

end triangle_properties_l130_130439


namespace smallest_possible_value_other_integer_l130_130498

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l130_130498


namespace max_value_fraction_diff_l130_130587

noncomputable def max_fraction_diff (a b : ℝ) : ℝ :=
  1 / a - 1 / b

theorem max_value_fraction_diff (a b : ℝ) (ha : a > 0) (hb : b > 0) (hc : 4 * a - b ≥ 2) :
  max_fraction_diff a b ≤ 1 / 2 :=
by
  sorry

end max_value_fraction_diff_l130_130587


namespace Gerald_charge_per_chore_l130_130424

noncomputable def charge_per_chore (E SE SP C : ℕ) : ℕ :=
  let total_expenditure := E * SE
  let monthly_saving_goal := total_expenditure / SP
  monthly_saving_goal / C

theorem Gerald_charge_per_chore :
  charge_per_chore 100 4 8 5 = 10 :=
by
  sorry

end Gerald_charge_per_chore_l130_130424


namespace hockey_games_in_season_l130_130185

theorem hockey_games_in_season
  (games_per_month : ℤ)
  (months_in_season : ℤ)
  (h1 : games_per_month = 25)
  (h2 : months_in_season = 18) :
  games_per_month * months_in_season = 450 :=
by
  sorry

end hockey_games_in_season_l130_130185


namespace total_students_accommodated_l130_130086

def num_columns : ℕ := 4
def num_rows : ℕ := 10
def num_buses : ℕ := 6

theorem total_students_accommodated : num_columns * num_rows * num_buses = 240 := by
  sorry

end total_students_accommodated_l130_130086


namespace sufficient_condition_not_necessary_condition_l130_130253

variables (p q : Prop)
def φ := ¬p ∧ ¬q
def ψ := ¬p

theorem sufficient_condition : φ p q → ψ p := 
sorry

theorem not_necessary_condition : ψ p → ¬ (φ p q) :=
sorry

end sufficient_condition_not_necessary_condition_l130_130253


namespace sin_arithmetic_sequence_l130_130727

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l130_130727


namespace x_intercept_rotation_30_degrees_eq_l130_130798

noncomputable def x_intercept_new_line (x0 y0 : ℝ) (θ : ℝ) (a b c : ℝ) : ℝ :=
  let m := a / b
  let m' := (m + θ.tan) / (1 - m * θ.tan)
  let x_intercept := x0 - (y0 * (b - m * c)) / (m' * (b - m * c) - a)
  x_intercept

theorem x_intercept_rotation_30_degrees_eq :
  x_intercept_new_line 7 4 (Real.pi / 6) 4 (-7) 28 = 7 - (4 * (7 * Real.sqrt 3 - 4) / (4 * Real.sqrt 3 + 7)) :=
by 
  -- detailed math proof goes here 
  sorry

end x_intercept_rotation_30_degrees_eq_l130_130798


namespace positive_integer_satisfies_condition_l130_130360

theorem positive_integer_satisfies_condition : 
  ∃ n : ℕ, (12 * n = n^2 + 36) ∧ n = 6 :=
by
  sorry

end positive_integer_satisfies_condition_l130_130360


namespace floor_condition_x_l130_130874

theorem floor_condition_x (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 48) : 8 ≤ x ∧ x < 49 / 6 := 
by 
  sorry

end floor_condition_x_l130_130874


namespace smallest_b_1111_is_square_l130_130417

theorem smallest_b_1111_is_square : 
  ∃ b : ℕ, b > 0 ∧ (∀ n : ℕ, (b^3 + b^2 + b + 1 = n^2 → b = 7)) :=
by
  sorry

end smallest_b_1111_is_square_l130_130417


namespace four_digit_numbers_property_l130_130279

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l130_130279


namespace sum_of_coefficients_l130_130049

theorem sum_of_coefficients (a : ℕ → ℤ) (x : ℂ) :
  (2*x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + 
  a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 →
  a 0 = 1 →
  a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 20 :=
sorry

end sum_of_coefficients_l130_130049


namespace triangle_area_is_24_l130_130011

-- Define the vertices
def vertex1 : ℝ × ℝ := (3, 2)
def vertex2 : ℝ × ℝ := (3, -4)
def vertex3 : ℝ × ℝ := (11, -4)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Prove the area of the triangle with the given vertices is 24 square units
theorem triangle_area_is_24 : triangle_area vertex1 vertex2 vertex3 = 24 := by
  sorry

end triangle_area_is_24_l130_130011


namespace ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l130_130853

section

variable {a b c : ℝ}

-- Statement 1
theorem ac_le_bc_if_a_gt_b_and_c_le_zero (h1 : a > b) (h2 : c ≤ 0) : a * c ≤ b * c := 
  sorry

-- Statement 2
theorem a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero (h1 : a * c ^ 2 > b * c ^ 2) (h2 : b ≥ 0) : a ^ 2 > b ^ 2 := 
  sorry

-- Statement 3
theorem log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1 (h1 : a > b) (h2 : b > -1) : Real.log (a + 1) > Real.log (b + 1) := 
  sorry

-- Statement 4
theorem inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero (h1 : a > b) (h2 : a * b > 0) : 1 / a < 1 / b := 
  sorry

end

end ac_le_bc_if_a_gt_b_and_c_le_zero_a_sq_gt_b_sq_if_ac_sq_gt_bc_sq_and_b_ge_zero_log_a1_gt_log_b1_if_a_gt_b_and_b_gt_neg1_inv_a_lt_inv_b_if_a_gt_b_and_ab_gt_zero_l130_130853


namespace expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l130_130807

section
variables (a b : ℚ)

theorem expansion_of_a_plus_b_pow_4 :
  (a + b) ^ 4 = a ^ 4 + 4 * a ^ 3 * b + 6 * a ^ 2 * b ^ 2 + 4 * a * b ^ 3 + b ^ 4 :=
sorry

theorem expansion_of_a_plus_b_pow_5 :
  (a + b) ^ 5 = a ^ 5 + 5 * a ^ 4 * b + 10 * a ^ 3 * b ^ 2 + 10 * a ^ 2 * b ^ 3 + 5 * a * b ^ 4 + b ^ 5 :=
sorry

theorem computation_of_formula :
  2^4 + 4*2^3*(-1/3) + 6*2^2*(-1/3)^2 + 4*2*(-1/3)^3 + (-1/3)^4 = 625 / 81 :=
sorry
end

end expansion_of_a_plus_b_pow_4_expansion_of_a_plus_b_pow_5_computation_of_formula_l130_130807


namespace proof_of_problem_l130_130114

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, (x + 2) ^ (x + 3) = 1 ↔ (x = -1 ∨ x = -3)

theorem proof_of_problem : proof_problem :=
by
  sorry

end proof_of_problem_l130_130114


namespace gcd_91_72_l130_130237

/-- Prove that the greatest common divisor of 91 and 72 is 1. -/
theorem gcd_91_72 : Nat.gcd 91 72 = 1 :=
by
  sorry

end gcd_91_72_l130_130237


namespace profit_15000_l130_130044

theorem profit_15000
  (P : ℝ)
  (invest_mary : ℝ := 550)
  (invest_mike : ℝ := 450)
  (total_invest := invest_mary + invest_mike)
  (share_ratio_mary := invest_mary / total_invest)
  (share_ratio_mike := invest_mike / total_invest)
  (effort_share := P / 6)
  (invest_share_mary := share_ratio_mary * (2 * P / 3))
  (invest_share_mike := share_ratio_mike * (2 * P / 3))
  (mary_total := effort_share + invest_share_mary)
  (mike_total := effort_share + invest_share_mike)
  (condition : mary_total - mike_total = 1000) :
  P = 15000 :=  
sorry

end profit_15000_l130_130044


namespace volume_of_pyramid_l130_130543

-- Define the conditions
def pyramid_conditions : Prop :=
  ∃ (s h : ℝ),
  s^2 = 256 ∧
  ∃ (h_A h_C h_B : ℝ),
  ((∃ h_A, 128 = 1/2 * s * h_A) ∧
  (∃ h_C, 112 = 1/2 * s * h_C) ∧
  (∃ h_B, 96 = 1/2 * s * h_B)) ∧
  h^2 + (s/2)^2 = h_A^2 ∧
  h^2 = 256 - (s/2)^2 ∧
  h^2 + (s/4)^2 = h_B^2

-- Define the theorem
theorem volume_of_pyramid :
  pyramid_conditions → 
  ∃ V : ℝ, V = 682.67 * Real.sqrt 3 :=
sorry

end volume_of_pyramid_l130_130543


namespace problem_solution_l130_130412

theorem problem_solution :
  0.45 * 0.65 + 0.1 * 0.2 = 0.3125 :=
by
  sorry

end problem_solution_l130_130412


namespace ajays_monthly_income_l130_130685

theorem ajays_monthly_income :
  ∀ (I : ℝ), 
  (0.50 * I) + (0.25 * I) + (0.15 * I) + 9000 = I → I = 90000 :=
by
  sorry

end ajays_monthly_income_l130_130685


namespace yeri_change_l130_130352

theorem yeri_change :
  let cost_candies := 5 * 120
  let cost_chocolates := 3 * 350
  let total_cost := cost_candies + cost_chocolates
  let amount_handed_over := 2500
  amount_handed_over - total_cost = 850 :=
by
  sorry

end yeri_change_l130_130352


namespace sin_arithmetic_sequence_l130_130728

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l130_130728


namespace tangent_line_eq_l130_130414

def perp_eq (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 1

theorem tangent_line_eq (x y : ℝ) (h1 : perp_eq x y) (h2 : y = curve x) : 
  ∃ (m : ℝ), y = -3 * x + m ∧ y = -3 * x - 2 := 
sorry

end tangent_line_eq_l130_130414


namespace percentage_of_respondents_l130_130366

variables {X Y : ℝ}
variable (h₁ : 23 <= 100 - X)

theorem percentage_of_respondents 
  (h₁ : 0 ≤ X) 
  (h₂ : X ≤ 100) 
  (h₃ : 0 ≤ 23) 
  (h₄ : 23 ≤ 23) : 
  Y = 100 - X := 
by
  sorry

end percentage_of_respondents_l130_130366


namespace fraction_identity_l130_130887

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l130_130887


namespace count_valid_numbers_l130_130274
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l130_130274


namespace remainder_div_19_l130_130356

theorem remainder_div_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
sorry

end remainder_div_19_l130_130356


namespace polynomial_value_l130_130118

theorem polynomial_value (a b : ℝ) : 
  (|a - 2| + (b + 1/2)^2 = 0) → (2 * a * b^2 + a^2 * b) - (3 * a * b^2 + a^2 * b - 1) = 1/2 :=
by
  sorry

end polynomial_value_l130_130118


namespace speed_of_current_l130_130372

-- Define the context and variables
variables (m c : ℝ)
-- State the conditions
variables (h1 : m + c = 12) (h2 : m - c = 8)

-- State the goal which is to prove the speed of the current
theorem speed_of_current : c = 2 :=
by
  sorry

end speed_of_current_l130_130372


namespace jackie_sleeping_hours_l130_130926

def hours_in_a_day : ℕ := 24
def work_hours : ℕ := 8
def exercise_hours : ℕ := 3
def free_time_hours : ℕ := 5
def accounted_hours : ℕ := work_hours + exercise_hours + free_time_hours

theorem jackie_sleeping_hours :
  hours_in_a_day - accounted_hours = 8 := by
  sorry

end jackie_sleeping_hours_l130_130926


namespace coefficient_of_x_in_expansion_l130_130174

theorem coefficient_of_x_in_expansion : 
  ∀ (x : ℝ), (∃ (coeff : ℝ), coeff = 10 ∧ 
    (∀ r : ℕ, r = 3 → (binomial 5 r) * x^(10 - 3 * r) = coeff * x)) :=
by
  sorry

end coefficient_of_x_in_expansion_l130_130174


namespace a_b_finish_job_in_15_days_l130_130353

theorem a_b_finish_job_in_15_days (A B C : ℝ) 
  (h1 : A + B + C = 1 / 5)
  (h2 : C = 1 / 7.5) : 
  (1 / (A + B)) = 15 :=
by
  sorry

end a_b_finish_job_in_15_days_l130_130353


namespace distance_hyperbola_focus_to_line_l130_130660

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l130_130660


namespace harry_took_5_eggs_l130_130992

theorem harry_took_5_eggs (initial : ℕ) (left : ℕ) (took : ℕ) 
  (h1 : initial = 47) (h2 : left = 42) (h3 : left = initial - took) : 
  took = 5 :=
sorry

end harry_took_5_eggs_l130_130992


namespace selling_price_ratio_l130_130371

theorem selling_price_ratio (C : ℝ) (hC : C > 0) :
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  S₂ / S₁ = 21 / 8 :=
by
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  sorry

end selling_price_ratio_l130_130371


namespace determine_divisors_l130_130931

theorem determine_divisors (n : ℕ) (h_pos : n > 0) (d : ℕ) (h_div : d ∣ 3 * n^2) (h_exists : ∃ k : ℤ, n^2 + d = k^2) : d = 3 * n^2 := 
sorry

end determine_divisors_l130_130931


namespace probability_both_white_given_same_color_l130_130346

open Finset

def bag := ["white", "white", "white", "black", "black"]

def same_color_pairs := ((choose (bag.filter (λ c, c = "white")) ⟨2, by sorry⟩) ++ (choose (bag.filter (λ c, c = "black")) ⟨2, by sorry⟩))

theorem probability_both_white_given_same_color : 
  (3 : ℚ) / 4 = 3 / 10 := 
sorry

end probability_both_white_given_same_color_l130_130346


namespace correct_sequence_of_linear_regression_analysis_l130_130520

def linear_regression_steps : List ℕ := [2, 4, 3, 1]

theorem correct_sequence_of_linear_regression_analysis :
  linear_regression_steps = [2, 4, 3, 1] :=
by
  sorry

end correct_sequence_of_linear_regression_analysis_l130_130520


namespace largest_of_five_consecutive_ints_15120_l130_130572

theorem largest_of_five_consecutive_ints_15120 :
  ∃ (a b c d e : ℕ), 
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
  a * b * c * d * e = 15120 ∧ 
  e = 10 := 
sorry

end largest_of_five_consecutive_ints_15120_l130_130572


namespace valid_numbers_count_l130_130349

def count_valid_numbers (n : ℕ) : ℕ := 1 / 4 * (5^n + 2 * 3^n + 1)

theorem valid_numbers_count (n : ℕ) : count_valid_numbers n = (1 / 4) * (5^n + 2 * 3^n + 1) :=
by sorry

end valid_numbers_count_l130_130349


namespace chocolate_bar_breaks_l130_130052

-- Definition of the problem as per the conditions
def chocolate_bar (rows : ℕ) (cols : ℕ) : ℕ := rows * cols

-- Statement of the proving problem
theorem chocolate_bar_breaks :
  ∀ (rows cols : ℕ), chocolate_bar rows cols = 40 → rows = 5 → cols = 8 → 
  (rows - 1) + (cols * (rows - 1)) = 39 :=
by
  intros rows cols h_bar h_rows h_cols
  sorry

end chocolate_bar_breaks_l130_130052


namespace abs_pos_of_ne_zero_l130_130033

theorem abs_pos_of_ne_zero (a : ℤ) (h : a ≠ 0) : |a| > 0 := sorry

end abs_pos_of_ne_zero_l130_130033


namespace geometric_series_sum_l130_130020

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l130_130020


namespace tv_purchase_time_l130_130530

-- Define the constants
def monthly_income : ℕ := 30000
def food_expenses : ℕ := 15000
def utilities_expenses : ℕ := 5000
def other_expenses : ℕ := 2500
def current_savings : ℕ := 10000
def tv_cost : ℕ := 25000

-- Define the total expenses
def total_expenses : ℕ := food_expenses + utilities_expenses + other_expenses

-- Define the disposable income
def disposable_income : ℕ := monthly_income - total_expenses

-- Define the amount needed to buy the TV
def amount_needed : ℕ := tv_cost - current_savings

-- Define the number of months needed to save the amount needed
def number_of_months : ℕ := amount_needed / disposable_income

-- The theorem specifying that we need 2 months to save enough money for the TV
theorem tv_purchase_time : number_of_months = 2 := by
  sorry

end tv_purchase_time_l130_130530


namespace anna_prob_at_least_two_correct_l130_130382

open scoped BigOperators

noncomputable def binomial_probability (k n : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

def prob_at_least_two_correct : ℝ :=
  1 - (binomial_probability 0 5 (1/4) + binomial_probability 1 5 (1/4))

theorem anna_prob_at_least_two_correct : 
  prob_at_least_two_correct = 47 / 128 :=
by
  sorry

end anna_prob_at_least_two_correct_l130_130382


namespace largest_consecutive_odd_number_sum_75_l130_130507

theorem largest_consecutive_odd_number_sum_75 (a b c : ℤ) 
    (h1 : a + b + c = 75) 
    (h2 : b = a + 2) 
    (h3 : c = b + 2) : 
    c = 27 :=
by
  sorry

end largest_consecutive_odd_number_sum_75_l130_130507


namespace tan_alpha_of_cos_alpha_l130_130425

theorem tan_alpha_of_cos_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi) (h_cos : Real.cos α = -3/5) :
  Real.tan α = -4/3 :=
sorry

end tan_alpha_of_cos_alpha_l130_130425


namespace find_z_l130_130466

theorem find_z (x y z : ℝ) (h1 : y = 3 * x - 5) (h2 : z = 3 * x + 3) (h3 : y = 1) : z = 9 := 
by
  sorry

end find_z_l130_130466


namespace four_digit_num_condition_l130_130284

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l130_130284


namespace least_positive_integer_divisibility_l130_130678

theorem least_positive_integer_divisibility :
  ∃ n > 1, (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_divisibility_l130_130678


namespace validate_assignment_l130_130551

-- Define the statements as conditions
def S1 := "x = x + 1"
def S2 := "b ="
def S3 := "x = y = 10"
def S4 := "x + y = 10"

-- A function to check if a statement is a valid assignment
def is_valid_assignment (s : String) : Prop :=
  s = S1

-- The theorem statement proving that S1 is the only valid assignment
theorem validate_assignment : is_valid_assignment S1 ∧
                              ¬is_valid_assignment S2 ∧
                              ¬is_valid_assignment S3 ∧
                              ¬is_valid_assignment S4 :=
by
  sorry

end validate_assignment_l130_130551


namespace units_digit_G_1000_l130_130479

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G_1000 : (G 1000) % 10 = 4 :=
  sorry

end units_digit_G_1000_l130_130479


namespace maximize_revenue_l130_130692

theorem maximize_revenue (p : ℝ) (h : p ≤ 30) : 
  (∀ q : ℝ, q ≤ 30 → (150 * 18.75 - 4 * (18.75:ℝ)^2) ≥ (150 * q - 4 * q^2)) ↔ p = 18.75 := 
sorry

end maximize_revenue_l130_130692


namespace sqrt_neg4_squared_l130_130400

theorem sqrt_neg4_squared : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := 
by 
-- add proof here
sorry

end sqrt_neg4_squared_l130_130400


namespace shenille_points_l130_130459

def shenille_total_points (x y : ℕ) : ℝ :=
  0.6 * x + 0.6 * y

theorem shenille_points (x y : ℕ) (h : x + y = 30) : 
  shenille_total_points x y = 18 := by
  sorry

end shenille_points_l130_130459


namespace peter_pizza_fraction_l130_130482

def pizza_slices : ℕ := 16
def peter_slices_alone : ℕ := 2
def shared_slice : ℚ := 1 / 2

theorem peter_pizza_fraction :
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  total_fraction = 5 / 32 :=
by
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  sorry

end peter_pizza_fraction_l130_130482


namespace tangent_product_20_40_60_80_l130_130810

theorem tangent_product_20_40_60_80 :
  Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) * Real.tan (80 * Real.pi / 180) = 3 :=
by
  sorry

end tangent_product_20_40_60_80_l130_130810


namespace Kimiko_age_proof_l130_130783

variables (Kimiko_age Kayla_age : ℕ)
variables (min_driving_age wait_years : ℕ)

def is_half_age (a b : ℕ) : Prop := a = b / 2
def minimum_driving_age (a b : ℕ) : Prop := a + b = 18

theorem Kimiko_age_proof
  (h1 : is_half_age Kayla_age Kimiko_age)
  (h2 : wait_years = 5)
  (h3 : minimum_driving_age Kayla_age wait_years) :
  Kimiko_age = 26 :=
sorry

end Kimiko_age_proof_l130_130783


namespace Uncle_Fyodor_age_l130_130480

variable (age : ℕ)

-- Conditions from the problem
def Sharik_statement : Prop := age > 11
def Matroskin_statement : Prop := age > 10

-- The theorem stating the problem to be proved
theorem Uncle_Fyodor_age
  (H : (Sharik_statement age ∧ ¬Matroskin_statement age) ∨ (¬Sharik_statement age ∧ Matroskin_statement age)) :
  age = 11 :=
by
  sorry

end Uncle_Fyodor_age_l130_130480


namespace num_zeros_in_decimal_representation_l130_130907

theorem num_zeros_in_decimal_representation :
  let denom := 2^3 * 5^10
  let frac := (1 : ℚ) / denom
  ∃ n : ℕ, n = 7 ∧ (∃ (a : ℕ) (b : ℕ), frac = a / 10^b ∧ ∃ (k : ℕ), b = n + k + 3) :=
sorry

end num_zeros_in_decimal_representation_l130_130907


namespace calculate_expr_l130_130393

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l130_130393


namespace cone_volume_percentage_filled_l130_130055

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l130_130055


namespace exists_circle_touching_given_circles_and_line_l130_130001

-- Define the given radii
def r1 := 1
def r2 := 3
def r3 := 4

-- Prove that there exists a circle with a specific radius touching the given circles and line AB
theorem exists_circle_touching_given_circles_and_line (x : ℝ) :
  ∃ (r : ℝ), r > 0 ∧ (r + r1) = x ∧ (r + r2) = x ∧ (r + r3) = x :=
sorry

end exists_circle_touching_given_circles_and_line_l130_130001


namespace afternoon_to_morning_ratio_l130_130545

theorem afternoon_to_morning_ratio (total_kg : ℕ) (afternoon_kg : ℕ) (morning_kg : ℕ) 
  (h1 : total_kg = 390) (h2 : afternoon_kg = 260) (h3 : morning_kg = total_kg - afternoon_kg) :
  afternoon_kg / morning_kg = 2 :=
sorry

end afternoon_to_morning_ratio_l130_130545


namespace cos_alpha_minus_pi_six_l130_130426

theorem cos_alpha_minus_pi_six (α : ℝ) (h : Real.sin (α + Real.pi / 3) = 4 / 5) : 
  Real.cos (α - Real.pi / 6) = 4 / 5 :=
sorry

end cos_alpha_minus_pi_six_l130_130426


namespace range_of_a_l130_130191

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a < |x - 4| + |x + 3|) → a < 7 :=
by
  sorry

end range_of_a_l130_130191


namespace five_digit_numbers_with_2_and_3_not_adjacent_six_digit_numbers_with_123_descending_l130_130423

-- Define the problem settings
def digits : finset ℕ := {0, 1, 2, 3, 4, 5}

-- Question 1: Prove the number of five-digit numbers containing 2 and 3, but not adjacent is 252
theorem five_digit_numbers_with_2_and_3_not_adjacent : 
  (number of five-digit numbers containing 2 and 3 but not adjacent) = 252 := 
sorry

-- Question 2: Prove the number of six-digit numbers where digits 1, 2, 3 are in descending order is 100
theorem six_digit_numbers_with_123_descending : 
  (number of six-digit numbers where digits 1, 2, 3 are in descending order) = 100 := 
sorry

end five_digit_numbers_with_2_and_3_not_adjacent_six_digit_numbers_with_123_descending_l130_130423


namespace abs_quadratic_eq_linear_iff_l130_130875

theorem abs_quadratic_eq_linear_iff (x : ℝ) : 
  (|x^2 - 5*x + 6| = x + 2) ↔ (x = 3 + Real.sqrt 5 ∨ x = 3 - Real.sqrt 5) :=
by
  sorry

end abs_quadratic_eq_linear_iff_l130_130875


namespace correct_percentage_is_500_over_7_l130_130771

-- Given conditions
variable (x : ℕ)
def total_questions : ℕ := 7 * x
def missed_questions : ℕ := 2 * x

-- Definition of the fraction and percentage calculation
def correct_fraction : ℚ := (total_questions x - missed_questions x : ℕ) / total_questions x
def correct_percentage : ℚ := correct_fraction x * 100

-- The theorem to prove
theorem correct_percentage_is_500_over_7 : correct_percentage x = 500 / 7 :=
by
  -- Proof goes here
  sorry

end correct_percentage_is_500_over_7_l130_130771


namespace find_t_from_tan_conditions_l130_130304

theorem find_t_from_tan_conditions 
  (α t : ℝ)
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + Real.pi / 4) = 4 / t)
  (h3 : Real.tan (α + Real.pi / 4) = (Real.tan (Real.pi / 4) + Real.tan α) / (1 - Real.tan (Real.pi / 4) * Real.tan α)) :
  t = 2 := 
  by
  sorry

end find_t_from_tan_conditions_l130_130304


namespace sophia_ate_pie_l130_130169

theorem sophia_ate_pie (weight_fridge weight_total weight_ate : ℕ)
  (h1 : weight_fridge = 1200) 
  (h2 : weight_fridge = 5 * weight_total / 6) :
  weight_ate = weight_total / 6 :=
by
  have weight_total_formula : weight_total = 6 * weight_fridge / 5 := by
    sorry
  have weight_ate_formula : weight_ate = weight_total / 6 := by
    sorry
  sorry

end sophia_ate_pie_l130_130169


namespace distance_from_wall_to_picture_edge_l130_130696

theorem distance_from_wall_to_picture_edge
  (wall_width : ℕ)
  (picture_width : ℕ)
  (centered : Prop)
  (h1 : wall_width = 22)
  (h2 : picture_width = 4)
  (h3 : centered) :
  ∃ x : ℕ, x = 9 :=
by
  sorry

end distance_from_wall_to_picture_edge_l130_130696


namespace number_of_solutions_l130_130755

theorem number_of_solutions :
  {x : ℝ | |x - 1| = |x - 2| + |x - 3|}.finite.to_finset.card = 2 :=
sorry

end number_of_solutions_l130_130755


namespace find_a_l130_130943

-- Definitions for the ellipses
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

-- Define the eccentricities
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Let e1 and e2 be the eccentricities of ellipse1 and ellipse2
def e1 (a : ℝ) : ℝ := eccentricity a 1
def e2 : ℝ := eccentricity 2 1

-- Given relationship
def relationship (a : ℝ) : Prop := e2 = Real.sqrt 3 * e1 a

-- Proof statement
theorem find_a (a : ℝ) (h₁ : ellipse1 a) (h₂ : ellipse2) (h₃ : relationship a) : a = 2 * Real.sqrt 3 / 3 := sorry

end find_a_l130_130943


namespace find_x_l130_130918

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l130_130918


namespace planes_1_and_6_adjacent_prob_l130_130186

noncomputable def probability_planes_adjacent (total_planes: ℕ) : ℚ :=
  if total_planes = 6 then 1/3 else 0

theorem planes_1_and_6_adjacent_prob :
  probability_planes_adjacent 6 = 1/3 := 
by
  sorry

end planes_1_and_6_adjacent_prob_l130_130186


namespace equivalent_expression_l130_130094

theorem equivalent_expression : 8^8 * 4^4 / 2^28 = 16 := by
  -- Here, we're stating the equivalency directly
  sorry

end equivalent_expression_l130_130094


namespace find_m_l130_130901

-- Circle equation: x^2 + y^2 + 2x - 6y + 1 = 0
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 6 * y + 1 = 0

-- Line equation: x + m * y + 4 = 0
def line_eq (x y m : ℝ) : Prop := x + m * y + 4 = 0

-- Prove that the value of m such that the center of the circle lies on the line is -1
theorem find_m (m : ℝ) : 
  (∃ x y : ℝ, circle_eq x y ∧ (x, y) = (-1, 3) ∧ line_eq x y m) → m = -1 :=
by {
  sorry
}

end find_m_l130_130901


namespace molecular_weight_correct_l130_130222

-- Define atomic weights of elements
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_D : ℝ := 2.01

-- Define the number of each type of atom in the compound
def num_Ba : ℕ := 2
def num_O : ℕ := 3
def num_H : ℕ := 4
def num_D : ℕ := 1

-- Define the molecular weight calculation
def molecular_weight : ℝ :=
  (num_Ba * atomic_weight_Ba) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_D * atomic_weight_D)

-- Theorem stating the molecular weight is 328.71 g/mol
theorem molecular_weight_correct :
  molecular_weight = 328.71 :=
by
  -- The proof will go here
  sorry

end molecular_weight_correct_l130_130222


namespace original_number_l130_130239

theorem original_number (n : ℕ) (h1 : 2319 % 21 = 0) (h2 : 2319 = 21 * (n + 1) - 1) : n = 2318 := 
sorry

end original_number_l130_130239


namespace regular_polygons_constructible_l130_130681

-- Define a right triangle where the smaller leg is half the length of the hypotenuse
structure RightTriangle30_60_90 :=
(smaller_leg hypotenuse : ℝ)
(ratio : smaller_leg = hypotenuse / 2)

-- Define the constructibility of polygons
def canConstructPolygon (n: ℕ) : Prop :=
n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 12

theorem regular_polygons_constructible (T : RightTriangle30_60_90) :
  ∀ n : ℕ, canConstructPolygon n :=
by
  intro n
  sorry

end regular_polygons_constructible_l130_130681


namespace elizabeth_wedding_gift_cost_l130_130232

-- Defining the given conditions
def cost_steak_knife_set : ℝ := 80.00
def num_steak_knife_sets : ℝ := 2
def cost_dinnerware_set : ℝ := 200.00
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Calculating total expense
def total_cost (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set : ℝ) : ℝ :=
  (cost_steak_knife_set * num_steak_knife_sets) + cost_dinnerware_set

def discounted_price (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (total_cost * discount_rate)

def final_price (discounted_price sales_tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * sales_tax_rate)

def elizabeth_spends (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate : ℝ) : ℝ :=
  final_price (discounted_price (total_cost cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set) discount_rate) sales_tax_rate

theorem elizabeth_wedding_gift_cost
  (cost_steak_knife_set : ℝ)
  (num_steak_knife_sets : ℝ)
  (cost_dinnerware_set : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ) :
  elizabeth_spends cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate = 340.20 := 
by
  sorry -- Proof is to be completed

end elizabeth_wedding_gift_cost_l130_130232


namespace sum_of_series_eq_l130_130307

open BigOperators

theorem sum_of_series_eq (n : ℕ) (h : 0 < n) : 
  ∑ i in Finset.range n, (i.succ ^ 2) / ((2 * i.succ - 1) * (2 * i.succ + 1)) = (n * n + n) / (4 * n + 2) := by
  sorry

end sum_of_series_eq_l130_130307


namespace fraction_second_year_not_third_year_l130_130774

theorem fraction_second_year_not_third_year (N T S : ℕ) (hN : N = 100) (hT : T = N / 2) (hS : S = N * 3 / 10) :
  (S / (N - T) : ℚ) = 3 / 5 :=
by
  rw [hN, hT, hS]
  norm_num
  sorry

end fraction_second_year_not_third_year_l130_130774


namespace det_example_l130_130247

theorem det_example : (1 * 4 - 2 * 3) = -2 :=
by
  -- Skip the proof with sorry
  sorry

end det_example_l130_130247


namespace cone_volume_filled_88_8900_percent_l130_130064

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l130_130064


namespace seeds_distributed_equally_l130_130233

theorem seeds_distributed_equally (S G n seeds_per_small_garden : ℕ) 
  (hS : S = 42) 
  (hG : G = 36) 
  (hn : n = 3) 
  (h_seeds : seeds_per_small_garden = (S - G) / n) : 
  seeds_per_small_garden = 2 := by
  rw [hS, hG, hn] at h_seeds
  simp at h_seeds
  exact h_seeds

end seeds_distributed_equally_l130_130233


namespace find_f3_l130_130148

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 - c * x + 2

theorem find_f3 (a b c : ℝ)
  (h1 : f a b c (-3) = 9) :
  f a b c 3 = -5 :=
by
  sorry

end find_f3_l130_130148


namespace m_zero_sufficient_but_not_necessary_l130_130131

-- Define the sequence a_n
variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the condition for equal difference of squares sequence
def equal_diff_of_squares_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, (a (n+1))^2 - (a n)^2 = d

-- Define the sequence b_n as an arithmetic sequence with common difference m
variable (b : ℕ → ℝ)
variable (m : ℝ)

def arithmetic_sequence (b : ℕ → ℝ) (m : ℝ) : Prop :=
  ∀ n, b (n+1) - b n = m

-- Prove "m = 0" is a sufficient but not necessary condition for {b_n} to be an equal difference of squares sequence
theorem m_zero_sufficient_but_not_necessary (a b : ℕ → ℝ) (d m : ℝ) :
  equal_diff_of_squares_sequence a d → arithmetic_sequence b m → (m = 0 → equal_diff_of_squares_sequence b d) ∧ (¬(m ≠ 0) → equal_diff_of_squares_sequence b d) :=
sorry


end m_zero_sufficient_but_not_necessary_l130_130131


namespace find_a2_l130_130584

theorem find_a2 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = 2 * (a n - 1))
  (h2 : S 1 = a 1)
  (h3 : S 2 = a 1 + a 2) :
  a 2 = 4 :=
sorry

end find_a2_l130_130584


namespace average_one_eighth_one_sixth_l130_130974

theorem average_one_eighth_one_sixth :
  (1/8 + 1/6) / 2 = 7/48 := 
by
  sorry

end average_one_eighth_one_sixth_l130_130974


namespace right_triangle_area_l130_130460

theorem right_triangle_area (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : (1 / 2 : ℚ) * (a * b) = 864 := 
by 
  sorry

end right_triangle_area_l130_130460


namespace proof_problem_l130_130413

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 / 3 ∧ x ≤ 2

theorem proof_problem (x : ℝ) (h : valid_x x) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 :=
sorry

end proof_problem_l130_130413


namespace liquid_X_percent_in_mixed_solution_l130_130553

theorem liquid_X_percent_in_mixed_solution (wP wQ : ℝ) (xP xQ : ℝ) (mP mQ : ℝ) :
  xP = 0.005 * wP →
  xQ = 0.015 * wQ →
  wP = 200 →
  wQ = 800 →
  13 / 1000 * 100 = 1.3 :=
by
  intros h1 h2 h3 h4
  sorry

end liquid_X_percent_in_mixed_solution_l130_130553


namespace find_a_l130_130578

theorem find_a (a : ℝ) (h : 1 ∈ ({a + 2, (a + 1)^2, a^2 + 3a + 3} : set ℝ)) : a = 0 :=
sorry

end find_a_l130_130578


namespace geometric_series_sum_l130_130027

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l130_130027


namespace oranges_taken_l130_130753

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end oranges_taken_l130_130753


namespace inequality_solution_l130_130166

theorem inequality_solution (x : ℝ) : (x < -4 ∨ x > -4) → (x + 3) / (x + 4) > (2 * x + 7) / (3 * x + 12) :=
by
  intro h
  sorry

end inequality_solution_l130_130166


namespace gcd_lcm_product_l130_130106

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l130_130106


namespace almost_surely_equal_l130_130319

noncomputable theory
open ProbabilityTheory

variables {Ω : Type*} {ξ ζ : Ω → ℝ} {ξ_n : ℕ → Ω → ℝ}

axiom conv_prob_ξ : ∀ ε > 0, lim_sup (λ n, probability (λ ω, abs (ξ_n n ω - ξ ω) ≥ ε)) = 0
axiom conv_prob_ζ : ∀ ε > 0, lim_sup (λ n, probability (λ ω, abs (ξ_n n ω - ζ ω) ≥ ε)) = 0

theorem almost_surely_equal :
  probability (λ ω, ξ ω = ζ ω) = 1 :=
sorry

end almost_surely_equal_l130_130319


namespace number_of_benches_l130_130083

-- Define the conditions
def bench_capacity : ℕ := 4
def people_sitting : ℕ := 80
def available_spaces : ℕ := 120
def total_capacity : ℕ := people_sitting + available_spaces -- this equals 200

-- The theorem to prove the number of benches
theorem number_of_benches (B : ℕ) : bench_capacity * B = total_capacity → B = 50 :=
by
  intro h
  exact sorry

end number_of_benches_l130_130083


namespace simplify_expression_l130_130811

theorem simplify_expression : 4 * (18 / 5) * (25 / -72) = -5 := by
  sorry

end simplify_expression_l130_130811


namespace negation_proof_l130_130504

theorem negation_proof : ∀ x : ℝ, x^2 + 2 * x + 2 > 0 :=
by
  sorry

end negation_proof_l130_130504


namespace geometric_sequence_common_ratio_l130_130641

theorem geometric_sequence_common_ratio (a1 a2 a3 : ℤ) (r : ℤ)
  (h1 : a1 = 9) (h2 : a2 = -18) (h3 : a3 = 36) (h4 : a2 / a1 = r) (h5 : a3 = a2 * r) :
  r = -2 := 
sorry

end geometric_sequence_common_ratio_l130_130641


namespace frost_cakes_total_l130_130554

-- Conditions
def Cagney_time := 60 -- seconds per cake
def Lacey_time := 40  -- seconds per cake
def total_time := 10 * 60 -- 10 minutes in seconds

-- The theorem to prove
theorem frost_cakes_total (Cagney_time Lacey_time total_time : ℕ) (h1 : Cagney_time = 60) (h2 : Lacey_time = 40) (h3 : total_time = 600):
  (total_time / (Cagney_time * Lacey_time / (Cagney_time + Lacey_time))) = 25 :=
by
  -- Proof to be filled in
  sorry

end frost_cakes_total_l130_130554


namespace playground_area_l130_130331

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l130_130331


namespace sum_of_roots_l130_130758

theorem sum_of_roots (x1 x2 : ℝ) (h : x1^2 + 5*x1 - 1 = 0 ∧ x2^2 + 5*x2 - 1 = 0) : x1 + x2 = -5 :=
sorry

end sum_of_roots_l130_130758


namespace proof_m_div_x_plus_y_l130_130474

variables (a b c x y m : ℝ)

-- 1. The ratio of 'a' to 'b' is 4 to 5
axiom h1 : a / b = 4 / 5

-- 2. 'c' is half of 'a'.
axiom h2 : c = a / 2

-- 3. 'x' equals 'a' increased by 27 percent of 'a'.
axiom h3 : x = 1.27 * a

-- 4. 'y' equals 'b' decreased by 16 percent of 'b'.
axiom h4 : y = 0.84 * b

-- 5. 'm' equals 'c' increased by 14 percent of 'c'.
axiom h5 : m = 1.14 * c

theorem proof_m_div_x_plus_y : m / (x + y) = 0.2457 :=
by
  -- Proof goes here
  sorry

end proof_m_div_x_plus_y_l130_130474


namespace simplify_sin_formula_l130_130320

theorem simplify_sin_formula : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := by
  -- Conditions and values used in the proof
  sorry

end simplify_sin_formula_l130_130320


namespace number_of_valid_arrangements_l130_130827

def valid_arrangements : Finset (Finset.Perm (Fin 5) (Fin 5)) :=
  (Finset.Perm (Fin 5)).filter (λ p,
    p 2 = 2 ∧
    ¬((p 0 = 1 ∧ p 1 = 2) ∨ (p 1 = 2 ∧ p 2 = 1) ∨
      (p 2 = 3 ∧ p 3 = 2) ∨ (p 3 = 2 ∧ p 4 = 3) ∨
      (p 4 = 3 ∧ p 3 = 4) ∨ (p 3 = 4 ∧ p 4 = 3)))

theorem number_of_valid_arrangements : valid_arrangements.card = 16 := by
  sorry

end number_of_valid_arrangements_l130_130827


namespace sum_of_volumes_is_correct_l130_130521

-- Define the dimensions of the base of the tank
def tank_base_length : ℝ := 44
def tank_base_width : ℝ := 35

-- Define the increase in water height when the train and the car are submerged
def train_water_height_increase : ℝ := 7
def car_water_height_increase : ℝ := 3

-- Calculate the area of the base of the tank
def base_area : ℝ := tank_base_length * tank_base_width

-- Calculate the volumes of the toy train and the toy car
def volume_train : ℝ := base_area * train_water_height_increase
def volume_car : ℝ := base_area * car_water_height_increase

-- Theorem to prove the sum of the volumes is 15400 cubic centimeters
theorem sum_of_volumes_is_correct : volume_train + volume_car = 15400 := by
  sorry

end sum_of_volumes_is_correct_l130_130521


namespace championship_positions_l130_130139

def positions_valid : Prop :=
  ∃ (pos_A pos_B pos_D pos_E pos_V pos_G : ℕ),
  (pos_A = pos_B + 3) ∧
  (pos_D < pos_E ∧ pos_E < pos_B) ∧
  (pos_V < pos_G) ∧
  (pos_D = 1) ∧
  (pos_E = 2) ∧
  (pos_B = 3) ∧
  (pos_V = 4) ∧
  (pos_G = 5) ∧
  (pos_A = 6)

theorem championship_positions : positions_valid :=
by
  sorry

end championship_positions_l130_130139


namespace evaluate_complex_expression_l130_130869

noncomputable def expression := 
  Complex.mk (-1) (Real.sqrt 3) / 2

noncomputable def conjugate_expression := 
  Complex.mk (-1) (-Real.sqrt 3) / 2

theorem evaluate_complex_expression :
  (expression ^ 12 + conjugate_expression ^ 12) = 2 := by
  sorry

end evaluate_complex_expression_l130_130869


namespace lemon_bag_mass_l130_130214

variable (m : ℝ)  -- mass of one bag of lemons in kg

-- Conditions
def max_load := 900  -- maximum load in kg
def num_bags := 100  -- number of bags
def extra_load := 100  -- additional load in kg

-- Proof statement (target)
theorem lemon_bag_mass : num_bags * m + extra_load = max_load → m = 8 :=
by
  sorry

end lemon_bag_mass_l130_130214


namespace total_sections_after_admissions_l130_130821

theorem total_sections_after_admissions (S : ℕ) (h1 : (S * 24 + 24 = (S + 3) * 21)) :
  (S + 3) = 16 :=
  sorry

end total_sections_after_admissions_l130_130821


namespace ellipse_eccentricity_l130_130948

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l130_130948


namespace earnings_total_l130_130859

-- Define the earnings for each day based on given conditions
def Monday_earnings : ℝ := 0.20 * 10 * 3
def Tuesday_earnings : ℝ := 0.25 * 12 * 4
def Wednesday_earnings : ℝ := 0.10 * 15 * 5
def Thursday_earnings : ℝ := 0.15 * 8 * 6
def Friday_earnings : ℝ := 0.30 * 20 * 2

-- Compute total earnings over the five days
def total_earnings : ℝ :=
  Monday_earnings + Tuesday_earnings + Wednesday_earnings + Thursday_earnings + Friday_earnings

-- Lean statement to prove the total earnings
theorem earnings_total :
  total_earnings = 44.70 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end earnings_total_l130_130859


namespace cost_price_of_computer_table_l130_130339

theorem cost_price_of_computer_table (SP : ℝ) (h1 : SP = 1.15 * CP ∧ SP = 6400) : CP = 5565.22 :=
by
  sorry

end cost_price_of_computer_table_l130_130339


namespace trig_identity_l130_130040

theorem trig_identity (α : ℝ) : 
  (2 * (Real.sin (4 * α))^2 - 1) / 
  (2 * (1 / Real.tan (Real.pi / 4 + 4 * α)) * (Real.cos (5 * Real.pi / 4 - 4 * α))^2) = -1 :=
by
  sorry

end trig_identity_l130_130040


namespace ratio_of_democrats_l130_130994

variable (F M D_F D_M : ℕ)

theorem ratio_of_democrats (h1 : F + M = 750)
    (h2 : D_F = 1 / 2 * F)
    (h3 : D_F = 125)
    (h4 : D_M = 1 / 4 * M) :
    (D_F + D_M) / 750 = 1 / 3 :=
sorry

end ratio_of_democrats_l130_130994


namespace ratio_B_to_A_l130_130714

-- Definitions for conditions
def w_B : ℕ := 275 -- weight of element B in grams
def w_X : ℕ := 330 -- total weight of compound X in grams

-- Statement to prove
theorem ratio_B_to_A : (w_B:ℚ) / (w_X - w_B) = 5 :=
by 
  sorry

end ratio_B_to_A_l130_130714


namespace frac_sum_is_one_l130_130886

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l130_130886


namespace find_q_l130_130000

variable (x : ℝ)

def f (x : ℝ) := (5 * x^4 + 15 * x^3 + 30 * x^2 + 10 * x + 10)
def g (x : ℝ) := (2 * x^6 + 4 * x^4 + 10 * x^2)
def q (x : ℝ) := (-2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)

theorem find_q :
  (∀ x, q x + g x = f x) ↔ (∀ x, q x = -2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)
:= sorry

end find_q_l130_130000


namespace unique_positive_a_for_one_solution_l130_130876

theorem unique_positive_a_for_one_solution :
  ∃ (d : ℝ), d ≠ 0 ∧ (∀ a : ℝ, a > 0 → (∀ x : ℝ, x^2 + (a + 1/a) * x + d = 0 ↔ x^2 + (a + 1/a) * x + d = 0)) ∧ d = 1 := 
by
  sorry

end unique_positive_a_for_one_solution_l130_130876


namespace monthly_average_growth_rate_eq_l130_130051

theorem monthly_average_growth_rate_eq (x : ℝ) :
  16 * (1 + x)^2 = 25 :=
sorry

end monthly_average_growth_rate_eq_l130_130051


namespace gcd_1911_1183_l130_130819

theorem gcd_1911_1183 : gcd 1911 1183 = 91 :=
by sorry

end gcd_1911_1183_l130_130819


namespace inequality1_inequality2_l130_130969

theorem inequality1 (x : ℝ) : 
  x^2 - 2 * x - 1 > 0 -> x > Real.sqrt 2 + 1 ∨ x < -Real.sqrt 2 + 1 := 
by sorry

theorem inequality2 (x : ℝ) : 
  (2 * x - 1) / (x - 3) ≥ 3 -> 3 < x ∧ x <= 8 := 
by sorry

end inequality1_inequality2_l130_130969


namespace ship_speed_l130_130802

theorem ship_speed 
  (D : ℝ)
  (h1 : (D/2) - 200 = D/3)
  (S := (D / 2) / 20):
  S = 30 :=
by
  -- proof here
  sorry

end ship_speed_l130_130802


namespace sin_arithmetic_sequence_l130_130725

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 360) :
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ 
  (a = 30 ∨ a = 150 ∨ a = 210 ∨ a = 330) :=
by
  sorry

end sin_arithmetic_sequence_l130_130725


namespace largest_prime_divisor_13_plus_14_fact_l130_130103

theorem largest_prime_divisor_13_plus_14_fact : 
  ∀ p : ℕ, prime p ∧ p ∣ 13! + 14! → p ≤ 13 := 
sorry

end largest_prime_divisor_13_plus_14_fact_l130_130103


namespace gcd_lcm_product_l130_130105

theorem gcd_lcm_product (a b : ℕ) (h₁ : a = 90) (h₂ : b = 135) : 
  Nat.gcd a b * Nat.lcm a b = 12150 :=
by
  -- Using given assumptions
  rw [h₁, h₂]
  -- Lean's definition of gcd and lcm in Nat
  sorry

end gcd_lcm_product_l130_130105


namespace cone_volume_percentage_filled_l130_130056

theorem cone_volume_percentage_filled (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  ((V_w / V) * 100).truncate = 29.6296 :=
by
  sorry

end cone_volume_percentage_filled_l130_130056


namespace opera_house_earnings_l130_130704

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l130_130704


namespace find_S_9_l130_130475

-- Conditions
def aₙ (n : ℕ) : ℕ := sorry  -- arithmetic sequence

def Sₙ (n : ℕ) : ℕ := sorry  -- sum of the first n terms of the sequence

axiom condition_1 : 2 * aₙ 8 = 6 + aₙ 11

-- Proof goal
theorem find_S_9 : Sₙ 9 = 54 :=
sorry

end find_S_9_l130_130475


namespace find_D_l130_130313

theorem find_D (A B D : ℕ) (h1 : (100 * A + 10 * B + D) * (A + B + D) = 1323) (h2 : A ≥ B) : D = 1 :=
sorry

end find_D_l130_130313


namespace find_rate_of_interest_l130_130370

-- Define the problem conditions
def principal_B : ℝ := 4000
def principal_C : ℝ := 2000
def time_B : ℝ := 2
def time_C : ℝ := 4
def total_interest : ℝ := 2200

-- Define the unknown rate of interest per annum
noncomputable def rate_of_interest (R : ℝ) : Prop :=
  let interest_B := (principal_B * R * time_B) / 100
  let interest_C := (principal_C * R * time_C) / 100
  interest_B + interest_C = total_interest

-- Statement to prove that the rate of interest is 13.75%
theorem find_rate_of_interest : rate_of_interest 13.75 := by
  sorry

end find_rate_of_interest_l130_130370


namespace pump_A_time_to_empty_pool_l130_130198

theorem pump_A_time_to_empty_pool :
  ∃ (A : ℝ), (1/A + 1/9 = 1/3.6) ∧ A = 6 :=
sorry

end pump_A_time_to_empty_pool_l130_130198


namespace find_m_l130_130749

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 25) + (p.2 ^ 2 / 16) = 1}
noncomputable def hyperbola (m : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / m) - (p.2 ^ 2 / 5) = 1}

theorem find_m (m : ℝ) (h1 : ∃ f : ℝ × ℝ, f ∈ ellipse ∧ f ∈ hyperbola m) : m = 4 := by
  sorry

end find_m_l130_130749


namespace adam_tickets_left_l130_130707

-- Define the initial number of tickets, cost per ticket, and total spending on the ferris wheel
def initial_tickets : ℕ := 13
def cost_per_ticket : ℕ := 9
def total_spent : ℕ := 81

-- Define the number of tickets Adam has after riding the ferris wheel
def tickets_left (initial_tickets cost_per_ticket total_spent : ℕ) : ℕ :=
  initial_tickets - (total_spent / cost_per_ticket)

-- Proposition to prove that Adam has 4 tickets left
theorem adam_tickets_left : tickets_left initial_tickets cost_per_ticket total_spent = 4 :=
by
  sorry

end adam_tickets_left_l130_130707


namespace distance_from_hyperbola_focus_to_line_l130_130655

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l130_130655


namespace inlet_pipe_filling_rate_l130_130369

def leak_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def net_emptying_rate (volume : ℕ) (time_hours : ℕ) : ℕ :=
  volume / time_hours

def inlet_rate_per_hour (net_rate : ℕ) (leak_rate : ℕ) : ℕ :=
  leak_rate - net_rate

def convert_to_minutes (rate_per_hour : ℕ) : ℕ :=
  rate_per_hour / 60

theorem inlet_pipe_filling_rate :
  let volume := 4320
  let time_to_empty_with_leak := 6
  let net_time_to_empty := 12
  let leak_rate := leak_rate volume time_to_empty_with_leak
  let net_rate := net_emptying_rate volume net_time_to_empty
  let fill_rate_per_hour := inlet_rate_per_hour net_rate leak_rate
  convert_to_minutes fill_rate_per_hour = 6 := by
    -- Proof ends with a placeholder 'sorry'
    sorry

end inlet_pipe_filling_rate_l130_130369


namespace sum_of_underlined_numbers_non_negative_l130_130691

-- Definitions used in the problem
def is_positive (n : Int) : Prop := n > 0
def underlined (nums : List Int) : List Int := sorry -- Define underlining based on conditions

def sum_of_underlined_numbers (nums : List Int) : Int :=
  (underlined nums).sum

-- The proof problem statement
theorem sum_of_underlined_numbers_non_negative
  (nums : List Int)
  (h_len : nums.length = 100) :
  0 < sum_of_underlined_numbers nums := sorry

end sum_of_underlined_numbers_non_negative_l130_130691


namespace present_worth_of_bill_l130_130340

theorem present_worth_of_bill (P : ℝ) (TD BD : ℝ) 
  (hTD : TD = 36) (hBD : BD = 37.62) 
  (hFormula : BD = (TD * (P + TD)) / P) : P = 800 :=
by
  sorry

end present_worth_of_bill_l130_130340


namespace correct_system_of_equations_l130_130136

theorem correct_system_of_equations (x y : ℝ) :
  (y = x + 4.5 ∧ 0.5 * y = x - 1) ↔
  (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by sorry

end correct_system_of_equations_l130_130136


namespace distance_from_hyperbola_focus_to_line_l130_130656

-- Definitions of the problem conditions
def hyperbola : Prop := ∀ x y : ℝ, (x^2 / 4 - y^2 / 5 = 1)
def line : Prop := ∀ x y : ℝ, (x + 2 * y - 8 = 0)

-- The main theorem we wish to prove
theorem distance_from_hyperbola_focus_to_line : 
  (∀ x y : ℝ, hyperbola) ∧ (∀ x y : ℝ, line) → ∃ d : ℝ, d = Real.sqrt 5 :=
by
  sorry

end distance_from_hyperbola_focus_to_line_l130_130656


namespace average_speed_ratio_l130_130309

theorem average_speed_ratio 
  (jack_marathon_distance : ℕ) (jack_marathon_time : ℕ) 
  (jill_marathon_distance : ℕ) (jill_marathon_time : ℕ)
  (h1 : jack_marathon_distance = 40) (h2 : jack_marathon_time = 45) 
  (h3 : jill_marathon_distance = 40) (h4 : jill_marathon_time = 40) :
  (889 : ℕ) / 1000 = (jack_marathon_distance / jack_marathon_time) / 
                      (jill_marathon_distance / jill_marathon_time) :=
by
  sorry

end average_speed_ratio_l130_130309


namespace middle_part_of_sum_is_120_l130_130449

theorem middle_part_of_sum_is_120 (x : ℚ) (h : 2 * x + x + (1 / 2) * x = 120) : 
  x = 240 / 7 := sorry

end middle_part_of_sum_is_120_l130_130449


namespace tanA_tanB_eq_thirteen_div_four_l130_130822

-- Define the triangle and its properties
variables {A B C : Type}
variables (a b c : ℝ)  -- sides BC, AC, AB
variables (HF HC : ℝ)  -- segments of altitude CF
variables (tanA tanB : ℝ)

-- Given conditions
def orthocenter_divide_altitude (HF HC : ℝ) : Prop :=
  HF = 8 ∧ HC = 18

-- The result we want to prove
theorem tanA_tanB_eq_thirteen_div_four (h : orthocenter_divide_altitude HF HC) : 
  tanA * tanB = 13 / 4 :=
  sorry

end tanA_tanB_eq_thirteen_div_four_l130_130822


namespace race_distance_difference_l130_130203

theorem race_distance_difference
  (d : ℕ) (tA tB : ℕ)
  (h_d: d = 80) 
  (h_tA: tA = 20) 
  (h_tB: tB = 25) :
  (d / tA) * tA = d ∧ (d - (d / tB) * tA) = 16 := 
by
  sorry

end race_distance_difference_l130_130203


namespace four_digit_numbers_with_property_l130_130268

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l130_130268


namespace mark_and_carolyn_total_l130_130960

theorem mark_and_carolyn_total (m c : ℝ) (hm : m = 3 / 4) (hc : c = 3 / 10) :
    m + c = 1.05 :=
by
  sorry

end mark_and_carolyn_total_l130_130960


namespace students_attended_school_l130_130962

-- Definitions based on conditions
def total_students (S : ℕ) : Prop :=
  ∃ (L R : ℕ), 
    (L = S / 2) ∧ 
    (R = L / 4) ∧ 
    (5 = R / 5)

-- Theorem stating the problem
theorem students_attended_school (S : ℕ) : total_students S → S = 200 :=
by
  intro h
  sorry

end students_attended_school_l130_130962


namespace ellipse_a_value_l130_130951

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l130_130951


namespace density_ξ_eta_density_ξ_div_eta_l130_130146

-- Definitions of the densities and their properties
variable {Ξ : Type*} [MeasureSpace Ξ] [HasPDF Ξ]
variable {Η : Type*} [MeasureSpace Η] [HasPDF Η]

-- Define the densities
variable (f_ξ : ℝ → ℝ) (f_η : ℝ → ℝ)
variable (I_01 : ℝ → ℝ) := λ y, if 0 ≤ y ∧ y ≤ 1 then 1 else 0

-- Assumptions
variable (independent : Independent Ξ Η)
variable (ξ_density : ∀ x, f_ξ x ≥ 0)
variable (η_uniform : f_η = I_01)

-- Theorem statements
theorem density_ξ_eta (z : ℝ) :
  ∀ z, (∃ (f_Z : ℝ → ℝ), true) :=
  sorry

theorem density_ξ_div_eta (z : ℝ) :
  ∀ z, (∃ (f_W : ℝ → ℝ), true) :=
  sorry


end density_ξ_eta_density_ξ_div_eta_l130_130146


namespace Nicole_fewer_questions_l130_130314

-- Definitions based on the given conditions
def Nicole_correct : ℕ := 22
def Cherry_correct : ℕ := 17
def Kim_correct : ℕ := Cherry_correct + 8

-- Theorem to prove the number of fewer questions Nicole answered compared to Kim
theorem Nicole_fewer_questions : Kim_correct - Nicole_correct = 3 :=
by
  -- We set up the definitions
  let Nicole_correct := 22
  let Cherry_correct := 17
  let Kim_correct := Cherry_correct + 8
  -- The proof will be filled in here. 
  -- The goal theorem statement is filled with 'sorry' to bypass the actual proof.
  have : Kim_correct - Nicole_correct = 3 := sorry
  exact this

end Nicole_fewer_questions_l130_130314


namespace geometric_seq_fraction_l130_130602

theorem geometric_seq_fraction (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
  (h2 : (a 1 + 3 * a 3) / (a 2 + 3 * a 4) = 1 / 2) : 
  (a 4 * a 6 + a 6 * a 8) / (a 6 * a 8 + a 8 * a 10) = 1 / 16 :=
by
  sorry

end geometric_seq_fraction_l130_130602


namespace ellipse_eccentricity_a_l130_130946

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l130_130946


namespace count_valid_numbers_l130_130271
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l130_130271


namespace bernie_postcards_l130_130384

theorem bernie_postcards :
  let initial_postcards := 18
  let price_sell := 15
  let price_buy := 5
  let sold_postcards := initial_postcards / 2
  let earned_money := sold_postcards * price_sell
  let bought_postcards := earned_money / price_buy
  let remaining_postcards := initial_postcards - sold_postcards
  let final_postcards := remaining_postcards + bought_postcards
  final_postcards = 36 := by sorry

end bernie_postcards_l130_130384


namespace distance_from_right_focus_to_line_l130_130654

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l130_130654


namespace least_value_expression_l130_130453

theorem least_value_expression (x : ℝ) (h : x < -2) :
  2 * x < x ∧ 2 * x < x + 2 ∧ 2 * x < (1 / 2) * x ∧ 2 * x < x - 2 :=
by
  sorry

end least_value_expression_l130_130453


namespace gauss_polynomial_reciprocal_l130_130485

def gauss_polynomial (k l : ℤ) (x : ℝ) : ℝ := sorry -- Placeholder for actual polynomial definition

theorem gauss_polynomial_reciprocal (k l : ℤ) (x : ℝ) : 
  x^(k * l) * gauss_polynomial k l (1 / x) = gauss_polynomial k l x :=
sorry

end gauss_polynomial_reciprocal_l130_130485


namespace tom_watching_days_l130_130675

noncomputable def total_watch_time : ℕ :=
  30 * 22 + 28 * 25 + 27 * 29 + 20 * 31 + 25 * 27 + 20 * 35

noncomputable def daily_watch_time : ℕ := 2 * 60

theorem tom_watching_days : ⌈(total_watch_time / daily_watch_time : ℚ)⌉ = 35 := by
  sorry

end tom_watching_days_l130_130675


namespace combined_cost_is_3490_l130_130251

-- Definitions for the quantities of gold each person has and their respective prices per gram
def Gary_gold_grams : ℕ := 30
def Gary_gold_price_per_gram : ℕ := 15

def Anna_gold_grams : ℕ := 50
def Anna_gold_price_per_gram : ℕ := 20

def Lisa_gold_grams : ℕ := 40
def Lisa_gold_price_per_gram : ℕ := 18

def John_gold_grams : ℕ := 60
def John_gold_price_per_gram : ℕ := 22

-- Combined cost
def combined_cost : ℕ :=
  Gary_gold_grams * Gary_gold_price_per_gram +
  Anna_gold_grams * Anna_gold_price_per_gram +
  Lisa_gold_grams * Lisa_gold_price_per_gram +
  John_gold_grams * John_gold_price_per_gram

-- Proof that the combined cost is equal to $3490
theorem combined_cost_is_3490 : combined_cost = 3490 :=
  by
  -- proof skipped
  sorry

end combined_cost_is_3490_l130_130251


namespace bowling_average_decrease_l130_130081

/-- Represents data about the bowler's performance. -/
structure BowlerPerformance :=
(old_average : ℚ)
(last_match_runs : ℚ)
(last_match_wickets : ℕ)
(previous_wickets : ℕ)

/-- Calculates the new total runs given. -/
def new_total_runs (perf : BowlerPerformance) : ℚ :=
  perf.old_average * ↑perf.previous_wickets + perf.last_match_runs

/-- Calculates the new total number of wickets. -/
def new_total_wickets (perf : BowlerPerformance) : ℕ :=
  perf.previous_wickets + perf.last_match_wickets

/-- Calculates the new bowling average. -/
def new_average (perf : BowlerPerformance) : ℚ :=
  new_total_runs perf / ↑(new_total_wickets perf)

/-- Calculates the decrease in the bowling average. -/
def decrease_in_average (perf : BowlerPerformance) : ℚ :=
  perf.old_average - new_average perf

/-- The proof statement to be verified. -/
theorem bowling_average_decrease :
  ∀ (perf : BowlerPerformance),
    perf.old_average = 12.4 →
    perf.last_match_runs = 26 →
    perf.last_match_wickets = 6 →
    perf.previous_wickets = 115 →
    decrease_in_average perf = 0.4 :=
by
  intros
  sorry

end bowling_average_decrease_l130_130081


namespace distance_to_line_is_sqrt5_l130_130665

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l130_130665


namespace problem_statement_l130_130390

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l130_130390


namespace cylinder_radius_exists_l130_130697

theorem cylinder_radius_exists (r h : ℕ) (pr : r ≥ 1) :
  (π * ↑r ^ 2 * ↑h = 2 * π * ↑r * (↑h + ↑r)) ↔
  (r = 3 ∨ r = 4 ∨ r = 6) :=
by
  sorry

end cylinder_radius_exists_l130_130697


namespace complex_star_angle_sum_correct_l130_130864

-- Definitions corresponding to the conditions
def complex_star_interior_angle_sum (n : ℕ) (h : n ≥ 7) : ℕ :=
  180 * (n - 4)

-- The theorem stating the problem
theorem complex_star_angle_sum_correct (n : ℕ) (h : n ≥ 7) :
  complex_star_interior_angle_sum n h = 180 * (n - 4) :=
sorry

end complex_star_angle_sum_correct_l130_130864


namespace opera_house_earnings_l130_130702

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l130_130702


namespace part1_proof_part2_proof_l130_130787

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l130_130787


namespace max_value_M_l130_130245

def J_k (k : ℕ) : ℕ := 10^(k + 3) + 1600

def M (k : ℕ) : ℕ := (J_k k).factors.count 2

theorem max_value_M : ∃ k > 0, (M k) = 7 ∧ ∀ m > 0, M m ≤ 7 :=
by 
  sorry

end max_value_M_l130_130245


namespace Ms_Hatcher_total_students_l130_130799

noncomputable def number_of_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) : ℕ :=
  third_graders + fourth_graders + fifth_graders + sixth_graders

theorem Ms_Hatcher_total_students (third_graders fourth_graders fifth_graders sixth_graders : ℕ) 
  (h1 : third_graders = 20)
  (h2 : fourth_graders = 2 * third_graders) 
  (h3 : fifth_graders = third_graders / 2) 
  (h4 : sixth_graders = 3 * (third_graders + fourth_graders) / 4) : 
  number_of_students third_graders fourth_graders fifth_graders sixth_graders = 115 :=
by
  sorry

end Ms_Hatcher_total_students_l130_130799


namespace first_fun_friday_is_march_30_l130_130054

def month_days := 31
def start_day := 4 -- 1 for Sunday, 2 for Monday, ..., 7 for Saturday; 4 means Thursday
def first_friday := 2
def fun_friday (n : ℕ) : ℕ := first_friday + (n - 1) * 7

theorem first_fun_friday_is_march_30 (h1 : start_day = 4)
                                    (h2 : month_days = 31) :
                                    fun_friday 5 = 30 :=
by 
  -- Proof is omitted
  sorry

end first_fun_friday_is_march_30_l130_130054


namespace linear_combination_solution_l130_130092

theorem linear_combination_solution :
  ∃ a b c : ℚ, 
    a • (⟨1, -2, 3⟩ : ℚ × ℚ × ℚ) + b • (⟨4, 1, -1⟩ : ℚ × ℚ × ℚ) + c • (⟨-3, 2, 1⟩ : ℚ × ℚ × ℚ) = ⟨0, 1, 4⟩ ∧
    a = -491/342 ∧
    b = 233/342 ∧
    c = 49/38 :=
by
  sorry

end linear_combination_solution_l130_130092


namespace renata_donation_l130_130961

variable (D L : ℝ)

theorem renata_donation : ∃ D : ℝ, 
  (10 - D + 90 - L - 2 + 65 = 94) ↔ D = 4 :=
by
  sorry

end renata_donation_l130_130961


namespace simplify_expression_l130_130638

variable {x y : ℝ}

theorem simplify_expression : (x^5 * x^3 * y^2 * y^4) = (x^8 * y^6) := by
  sorry

end simplify_expression_l130_130638


namespace A_finishes_remaining_work_in_6_days_l130_130842

-- Definitions for conditions
def A_workdays : ℕ := 18
def B_workdays : ℕ := 15
def B_worked_days : ℕ := 10

-- Proof problem statement
theorem A_finishes_remaining_work_in_6_days (A_workdays B_workdays B_worked_days : ℕ) :
  let rate_A := 1 / A_workdays
  let rate_B := 1 / B_workdays
  let work_done_by_B := B_worked_days * rate_B
  let remaining_work := 1 - work_done_by_B
  let days_A_needs := remaining_work / rate_A
  days_A_needs = 6 :=
by
  sorry

end A_finishes_remaining_work_in_6_days_l130_130842


namespace inequality_solutions_l130_130937

theorem inequality_solutions (p p' q q' : ℕ) (hp : p ≠ p') (hq : q ≠ q') (hp_pos : 0 < p) (hp'_pos : 0 < p') (hq_pos : 0 < q) (hq'_pos : 0 < q') :
  (-(q : ℚ) / p > -(q' : ℚ) / p') ↔ (q * p' < p * q') :=
by
  sorry

end inequality_solutions_l130_130937


namespace jack_morning_emails_l130_130778

-- Define the conditions as constants
def totalEmails : ℕ := 10
def emailsAfternoon : ℕ := 3
def emailsEvening : ℕ := 1

-- Problem statement to prove emails in the morning
def emailsMorning : ℕ := totalEmails - (emailsAfternoon + emailsEvening)

-- The theorem to prove
theorem jack_morning_emails : emailsMorning = 6 := by
  sorry

end jack_morning_emails_l130_130778


namespace intersection_A_B_l130_130128

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : 
  A ∩ B = {x | 0 < x ∧ x ≤ 2} :=
  sorry

end intersection_A_B_l130_130128


namespace fractions_ordered_l130_130189

theorem fractions_ordered :
  (2 / 5 : ℚ) < (3 / 5) ∧ (3 / 5) < (4 / 6) ∧ (4 / 6) < (4 / 5) ∧ (4 / 5) < (6 / 5) ∧ (6 / 5) < (4 / 3) :=
by
  sorry

end fractions_ordered_l130_130189


namespace sequence_product_l130_130922

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = q * a n) (h₄ : a 4 = 2) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l130_130922


namespace sum_of_consecutive_integers_420_l130_130287

theorem sum_of_consecutive_integers_420 : 
  ∃ (k n : ℕ) (h1 : k ≥ 2) (h2 : k * n + k * (k - 1) / 2 = 420), 
  ∃ K : Finset ℕ, K.card = 6 ∧ (∀ x ∈ K, k = x) :=
by
  sorry

end sum_of_consecutive_integers_420_l130_130287


namespace infinite_series_k3_over_3k_l130_130410

theorem infinite_series_k3_over_3k :
  ∑' k : ℕ, (k^3 : ℝ) / 3^k = 165 / 16 := 
sorry

end infinite_series_k3_over_3k_l130_130410


namespace green_peaches_eq_three_l130_130297

theorem green_peaches_eq_three (p r g : ℕ) (h1 : p = r + g) (h2 : r + 2 * g = p + 3) : g = 3 := 
by 
  sorry

end green_peaches_eq_three_l130_130297


namespace find_b_l130_130443

theorem find_b
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) -> True)
  (h4 : ∃ e, e = (Real.sqrt 5) / 3)
  (h5 : 2 * a = 12) :
  b = 4 :=
by
  sorry

end find_b_l130_130443


namespace average_of_six_numbers_l130_130673

theorem average_of_six_numbers :
  (∀ a b : ℝ, (a + b) / 2 = 6.2) →
  (∀ c d : ℝ, (c + d) / 2 = 6.1) →
  (∀ e f : ℝ, (e + f) / 2 = 6.9) →
  ((a + b + c + d + e + f) / 6 = 6.4) :=
by
  intros h1 h2 h3
  -- Proof goes here, but will be skipped with sorry.
  sorry

end average_of_six_numbers_l130_130673


namespace num_valid_x_values_l130_130263

noncomputable def count_valid_x : ℕ :=
  ((Finset.range 34).filter (λ x, x ≥ 25 ∧ 3 * x < 100 ∧ 4 * x > 99)).card

theorem num_valid_x_values : count_valid_x = 9 := by
  sorry

end num_valid_x_values_l130_130263


namespace max_yes_men_l130_130159

-- Define types of inhabitants
inductive Inhabitant
| Knight
| Liar
| YesMan

-- Main theorem stating the problem
theorem max_yes_men (total inhabitants: ℕ) (yes_answers: ℕ)
  (K L S: ℕ)
  (Hcondition: K + L + S = 2018)
  (Hyes: yes_answers = 1009)
  (Hbehaviour: ∀ x, (x = Inhabitant.Knight → is_true x) ∧ 
                     (x = Inhabitant.Liar → ¬is_true x) ∧ 
                     (x = Inhabitant.YesMan → (majority_so_far x → is_true x) ∨ (majority_so_far x → ¬is_true x))):
  S ≤ 1009 := sorry

end max_yes_men_l130_130159


namespace part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l130_130740

variable {x a : ℝ}

theorem part1_solution (h1 : a > 1 / 3) (h2 : (a * x - 1) / (x ^ 2 - 1) = 0) : x = 3 := by
  sorry

theorem part2_solution_1 (h1 : -1 < a) (h2 : a < 0) : {x | x < (1 / a) ∨ (-1 < x ∧ x < 1)} := by
  sorry

theorem part2_solution_2 (h1 : a = -1) : {x | x < 1 ∧ x ≠ -1} := by
  sorry

theorem part2_solution_3 (h1 : a < -1) : {x | x < -1 ∨ (1 / a < x ∧ x < 1)} := by
  sorry

end part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l130_130740


namespace volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l130_130929

noncomputable def volume_tetrahedron (V A B C : Point) : ℝ := sorry

def is_interior_point (M V A B C : Point) : Prop := sorry -- Definition of an interior point

def is_barycenter (M V A B C : Point) : Prop := sorry -- Definition of a barycenter

def intersects_lines_planes (M V A B C A1 B1 C1 : Point) : Prop := sorry -- Definition of intersection points

def intersects_lines_sides (V A1 B1 C1 A B C A2 B2 C2 : Point) : Prop := sorry -- Definition of intersection points with sides

theorem volume_le_one_fourth_of_original (V A B C: Point) 
  (M : Point) (A1 B1 C1 A2 B2 C2 : Point) 
  (h_interior : is_interior_point M V A B C) 
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1) 
  (h_intersects_sides : intersects_lines_sides V A1 B1 C1 A B C A2 B2 C2) :
  volume_tetrahedron V A2 B2 C2 ≤ (1/4) * volume_tetrahedron V A B C :=
sorry

theorem volume_of_sub_tetrahedron (V A B C: Point) 
  (M V1 : Point) (A1 B1 C1 : Point)
  (h_barycenter : is_barycenter M V A B C)
  (h_intersects_planes : intersects_lines_planes M V A B C A1 B1 C1)
  (h_point_V1 : intersects_something_to_find_V1) : 
  volume_tetrahedron V1 A1 B1 C1 = (1/4) * volume_tetrahedron V A B C :=
sorry

end volume_le_one_fourth_of_original_volume_of_sub_tetrahedron_l130_130929


namespace positive_integer_solutions_l130_130873

theorem positive_integer_solutions (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) :
  1 + 2^x + 2^(2*x+1) = y^n ↔ 
  (x = 4 ∧ y = 23 ∧ n = 2) ∨ (∃ t : ℕ, 0 < t ∧ x = t ∧ y = 1 + 2^t + 2^(2*t+1) ∧ n = 1) :=
sorry

end positive_integer_solutions_l130_130873


namespace find_number_l130_130668

theorem find_number (a b N : ℕ) (h1 : b = 7) (h2 : b - a = 2) (h3 : a * b = 2 * (a + b) + N) : N = 11 :=
  sorry

end find_number_l130_130668


namespace cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_l130_130302

open Real

namespace CartesianCoordinates

-- Conditions for the problem
def parametric_C1 (φ : ℝ) : ℝ × ℝ := (2 * cos φ, sin φ)
def polar_center_C2 : ℝ × ℝ := (0, 3)

-- Cartesian equation of curve C1
theorem cartesian_eq_C1 (x y : ℝ) (h : ∃ φ : ℝ, x = 2 * cos φ ∧ y = sin φ) :
  x ^ 2 / 4 + y ^ 2 = 1 := sorry

-- Cartesian equation of curve C2
theorem cartesian_eq_C2 (x y : ℝ) :
  (x - polar_center_C2.1) ^ 2 + (y - polar_center_C2.2) ^ 2 = 1 ↔ x ^ 2 + (y - 3) ^ 2 = 1 := sorry

-- Range of values for the distance |MN|
theorem range_distance_MN (φ θ : ℝ) :
  let M := parametric_C1 φ;
  let N := (polar_center_C2.1 + cos θ, polar_center_C2.2 + sin θ);
  let distance := sqrt ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) in
  1 ≤ distance ∧ distance ≤ 5 := sorry

end CartesianCoordinates

end cartesian_eq_C1_cartesian_eq_C2_range_distance_MN_l130_130302


namespace find_a_l130_130575

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l130_130575


namespace distance_from_focus_to_line_l130_130645

-- Define the hyperbola and the line
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 5 = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus (x y : ℝ) : Prop := (x = 3) ∧ (y = 0)

-- Define the distance function from a point to a line
noncomputable def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / real.sqrt (A^2 + B^2)

-- The theorem statement
theorem distance_from_focus_to_line : distance_to_line 3 0 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130645


namespace smallest_integer_l130_130500

theorem smallest_integer (x : ℕ) (n : ℕ) (h_pos : 0 < x)
  (h_gcd : Nat.gcd 30 n = x + 3)
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) : n = 70 :=
begin
  sorry
end

end smallest_integer_l130_130500


namespace complement_union_l130_130446

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ) (hA : A = { x | x < 0 }) (hB : B = { x | x ≥ 2 }) :
  C_U U (A ∪ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end complement_union_l130_130446


namespace remainder_x1002_div_x2_minus_1_mul_x_plus_1_l130_130109

noncomputable def polynomial_div_remainder (a b : Polynomial ℝ) : Polynomial ℝ := sorry

theorem remainder_x1002_div_x2_minus_1_mul_x_plus_1 :
  polynomial_div_remainder (Polynomial.X ^ 1002) ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 1)) = 1 :=
by sorry

end remainder_x1002_div_x2_minus_1_mul_x_plus_1_l130_130109


namespace cost_of_first_variety_l130_130306

theorem cost_of_first_variety (x : ℝ) (cost2 : ℝ) (cost_mix : ℝ) (ratio : ℝ) :
    cost2 = 8.75 →
    cost_mix = 7.50 →
    ratio = 0.625 →
    (x - cost_mix) / (cost2 - cost_mix) = ratio →
    x = 8.28125 := 
by
  intros h1 h2 h3 h4
  sorry

end cost_of_first_variety_l130_130306


namespace problem_1_problem_2_l130_130126

theorem problem_1 {m : ℝ} (h₁ : 0 < m) (h₂ : ∀ x : ℝ, (m - |x + 2| ≥ 0) ↔ (-3 ≤ x ∧ x ≤ -1)) :
  m = 1 :=
sorry

theorem problem_2 {a b c : ℝ} (h₃ : 0 < a ∧ 0 < b ∧ 0 < c) (h₄ : (1 / a) + (1 / (2 * b)) + (1 / (3 * c)) = 1)
  : a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_1_problem_2_l130_130126


namespace find_n_l130_130213

theorem find_n (n : ℕ) (h : (2 * n + 1) / 3 = 2022) : n = 3033 :=
sorry

end find_n_l130_130213


namespace prob_correct_l130_130161

-- Define percentages as ratio values
def prob_beginner_excel : ℝ := 0.35
def prob_intermediate_excel : ℝ := 0.25
def prob_advanced_excel : ℝ := 0.20
def prob_no_excel : ℝ := 0.20

def prob_day_shift : ℝ := 0.70
def prob_night_shift : ℝ := 0.30

def prob_weekend : ℝ := 0.40
def prob_not_weekend : ℝ := 0.60

-- Define the target probability calculation
def prob_intermediate_or_advanced_excel : ℝ := prob_intermediate_excel + prob_advanced_excel
def prob_combined : ℝ := prob_intermediate_or_advanced_excel * prob_night_shift * prob_not_weekend

-- The proof problem statement
theorem prob_correct : prob_combined = 0.081 :=
by
  sorry

end prob_correct_l130_130161


namespace find_A_l130_130219

theorem find_A (A B C : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : B ≠ C) (h4 : A < 10) (h5 : B < 10) (h6 : C < 10) (h7 : 10 * A + B + 10 * B + C = 101 * B + 10 * C) : A = 9 :=
sorry

end find_A_l130_130219


namespace moving_circle_fixed_point_coordinates_l130_130432

theorem moving_circle_fixed_point_coordinates (m x y : Real) :
    (∀ m : ℝ, x^2 + y^2 - 2 * m * x - 4 * m * y + 6 * m - 2 = 0) →
    (x = 1 ∧ y = 1 ∨ x = 1 / 5 ∧ y = 7 / 5) :=
  by
    sorry

end moving_circle_fixed_point_coordinates_l130_130432


namespace sets_equality_l130_130595

variables {α : Type*} (A B C : Set α)

theorem sets_equality (h1 : A ∪ B ⊆ C) (h2 : A ∪ C ⊆ B) (h3 : B ∪ C ⊆ A) : A = B ∧ B = C :=
by
  sorry

end sets_equality_l130_130595


namespace votes_to_win_l130_130357

theorem votes_to_win (total_votes : ℕ) (geoff_votes_percent : ℝ) (additional_votes : ℕ) (x : ℝ) 
(h1 : total_votes = 6000)
(h2 : geoff_votes_percent = 0.5)
(h3 : additional_votes = 3000)
(h4 : x = 50.5) :
  ((geoff_votes_percent / 100 * total_votes) + additional_votes) / total_votes * 100 = x :=
by
  sorry

end votes_to_win_l130_130357


namespace length_of_integer_eq_24_l130_130734

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l130_130734


namespace pencil_length_after_sharpening_l130_130923

def initial_length : ℕ := 50
def monday_sharpen : ℕ := 2
def tuesday_sharpen : ℕ := 3
def wednesday_sharpen : ℕ := 4
def thursday_sharpen : ℕ := 5

def total_sharpened : ℕ := monday_sharpen + tuesday_sharpen + wednesday_sharpen + thursday_sharpen

def final_length : ℕ := initial_length - total_sharpened

theorem pencil_length_after_sharpening : final_length = 36 := by
  -- Here would be the proof body
  sorry

end pencil_length_after_sharpening_l130_130923


namespace distance_from_focus_to_line_l130_130661

-- Define the hyperbola as a predicate
def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

-- Define the line as a predicate
def line (x y : ℝ) : Prop :=
  x + 2 * y - 8 = 0

-- Given: the right focus of the hyperbola is (3, 0)
def right_focus : ℝ × ℝ := (3, 0)

-- Function to calculate the distance from a point (x₀, y₀) to the line Ax + By + C = 0
def distance_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / real.sqrt (A^2 + B^2)

-- Formalize the proof problem
theorem distance_from_focus_to_line :
  distance_to_line right_focus.1 right_focus.2 1 2 (-8) = real.sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130661


namespace Janice_time_left_l130_130615

-- Define the conditions as variables and parameters
def homework_time := 30
def cleaning_time := homework_time / 2
def dog_walking_time := homework_time + 5
def trash_time := homework_time / 6
def total_time_before_movie := 2 * 60

-- Calculation of total time required for all tasks
def total_time_required_for_tasks : Nat :=
  homework_time + cleaning_time + dog_walking_time + trash_time

-- Time left before the movie starts after completing all tasks
def time_left_before_movie : Nat :=
  total_time_before_movie - total_time_required_for_tasks

-- The final statement to prove
theorem Janice_time_left : time_left_before_movie = 35 :=
  by
    -- This will execute automatically to verify the theorem
    sorry

end Janice_time_left_l130_130615


namespace geom_series_sum_l130_130022

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l130_130022


namespace calculate_expr_l130_130392

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l130_130392


namespace largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l130_130836

-- Definitions based on conditions
def isProperFraction (n d : ℕ) : Prop := n < d
def isImproperFraction (n d : ℕ) : Prop := n ≥ d
def isMixedNumber (w n d : ℕ) : Prop := w > 0 ∧ isProperFraction n d

-- Fractional part is 1/9, meaning all fractions considered have part = 1/9
def fractionalPart := 1 / 9

-- Lean 4 statements to verify the correct answers
theorem largest_proper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isProperFraction n d ∧ (n, d) = (8, 9) := sorry

theorem smallest_improper_fraction : ∃ n d : ℕ, fractionalPart = n / d ∧ isImproperFraction n d ∧ (n, d) = (9, 9) := sorry

theorem smallest_mixed_number : ∃ w n d : ℕ, fractionalPart = n / d ∧ isMixedNumber w n d ∧ ((w, n, d) = (1, 1, 9) ∨ (w, n, d) = (10, 9)) := sorry

end largest_proper_fraction_smallest_improper_fraction_smallest_mixed_number_l130_130836


namespace initial_stock_decaf_percentage_l130_130539

-- Definitions as conditions of the problem
def initial_coffee_stock : ℕ := 400
def purchased_coffee_stock : ℕ := 100
def percentage_decaf_purchased : ℕ := 60
def total_percentage_decaf : ℕ := 32

/-- The proof problem statement -/
theorem initial_stock_decaf_percentage : 
  ∃ x : ℕ, x * initial_coffee_stock / 100 + percentage_decaf_purchased * purchased_coffee_stock / 100 = total_percentage_decaf * (initial_coffee_stock + purchased_coffee_stock) / 100 ∧ x = 25 :=
sorry

end initial_stock_decaf_percentage_l130_130539


namespace divide_400_l130_130336

theorem divide_400 (a b c d : ℕ) (h1 : a + b + c + d = 400) 
  (h2 : a + 1 = b - 2) (h3 : a + 1 = 3 * c) (h4 : a + 1 = d / 4) 
  : a = 62 ∧ b = 65 ∧ c = 21 ∧ d = 252 :=
sorry

end divide_400_l130_130336


namespace richmond_tigers_tickets_l130_130493

theorem richmond_tigers_tickets (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) : 
  total_tickets - first_half_tickets = 5703 :=
by
  -- Proof steps would go here
  sorry

end richmond_tigers_tickets_l130_130493


namespace water_filled_percent_l130_130069

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l130_130069


namespace jason_picked_7_pears_l130_130627

def pears_picked_by_jason (total_pears mike_pears : ℕ) : ℕ :=
  total_pears - mike_pears

theorem jason_picked_7_pears :
  pears_picked_by_jason 15 8 = 7 :=
by
  -- Proof is required but we can insert sorry here to skip it for now
  sorry

end jason_picked_7_pears_l130_130627


namespace find_second_dimension_l130_130375

noncomputable def rectangular_tank_second_dimension (cost: ℝ) (cost_per_sqft: ℝ) (length: ℕ) (height: ℕ) : ℝ :=
  let total_surface_area := cost / cost_per_sqft
  let second_dimension := (total_surface_area - 2 * length * height - 2 * length * 2 - 2 * height * 2) / (2 * length + 2 * 2)
  second_dimension

theorem find_second_dimension :
  let length := 3
  let height := 2
  let cost := 1440
  let cost_per_sqft := 20 in
  rectangular_tank_second_dimension cost cost_per_sqft length height = 6 :=
by
  let length := 3 in
  let height := 2 in
  let cost := 1440 in
  let cost_per_sqft := 20 in
  sorry

end find_second_dimension_l130_130375


namespace part_a_part_b_l130_130006

noncomputable def volume_of_prism (V : ℝ) : ℝ :=
  (9 / 250) * V

noncomputable def max_volume_of_prism (V : ℝ) : ℝ :=
  (1 / 12) * V

theorem part_a (V : ℝ) :
  volume_of_prism V = (9 / 250) * V :=
  by sorry

theorem part_b (V : ℝ) :
  max_volume_of_prism V = (1 / 12) * V :=
  by sorry

end part_a_part_b_l130_130006


namespace trapezoid_circle_tangent_ratio_l130_130995

/-- Given trapezoid EFGH with specified side lengths,
    where EF is parallel to GH, and a circle with
    center Q on EF tangent to FG and HE,
    the ratio EQ : QF is 12 : 37. -/
theorem trapezoid_circle_tangent_ratio :
  ∀ (EF FG GH HE : ℝ) (EQ QF : ℝ),
  EF = 40 → FG = 25 → GH = 12 → HE = 35 →
  ∃ (Q : ℝ) (EQ QF : ℝ),
  EQ + QF = EF ∧ EQ / QF = 12 / 37 ∧ gcd 12 37 = 1 :=
by
  sorry

end trapezoid_circle_tangent_ratio_l130_130995


namespace sum_geometric_series_is_correct_l130_130014

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l130_130014


namespace stockholm_uppsala_distance_l130_130644

variable (map_distance : ℝ) (scale_factor : ℝ)

def actual_distance (d : ℝ) (s : ℝ) : ℝ := d * s

theorem stockholm_uppsala_distance :
  actual_distance 65 20 = 1300 := by
  sorry

end stockholm_uppsala_distance_l130_130644


namespace max_voters_is_five_l130_130514

noncomputable def max_voters_after_T (x : ℕ) : ℕ :=
if h : 0 ≤ (x - 11) then x - 11 else 0

theorem max_voters_is_five (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) :
  max_voters_after_T x = 5 :=
by
  sorry

end max_voters_is_five_l130_130514


namespace quadratic_condition_l130_130206

theorem quadratic_condition (p q : ℝ) (x1 x2 : ℝ) (hx : x1 + x2 = -p) (hq : x1 * x2 = q) :
  p + q = 0 := sorry

end quadratic_condition_l130_130206


namespace xy_value_l130_130121

theorem xy_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 3 / x = y + 3 / y) (hxy : x ≠ y) : x * y = 3 :=
sorry

end xy_value_l130_130121


namespace Y_4_3_l130_130450

def Y (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem Y_4_3 : Y 4 3 = -11 :=
by
  -- This line is added to skip the proof and focus on the statement.
  sorry

end Y_4_3_l130_130450


namespace fraction_identity_l130_130888

theorem fraction_identity (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end fraction_identity_l130_130888


namespace max_possible_N_l130_130604

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l130_130604


namespace cos_and_sin_double_angle_l130_130590

variables (θ : ℝ)

-- Conditions
def is_in_fourth_quadrant (θ : ℝ) : Prop :=
  θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi

def sin_theta (θ : ℝ) : Prop :=
  Real.sin θ = -1 / 3

-- Problem statement
theorem cos_and_sin_double_angle (h1 : is_in_fourth_quadrant θ) (h2 : sin_theta θ) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 ∧ Real.sin (2 * θ) = -(4 * Real.sqrt 2 / 9) :=
sorry

end cos_and_sin_double_angle_l130_130590


namespace volume_filled_cone_l130_130067

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l130_130067


namespace area_of_gray_region_l130_130188

theorem area_of_gray_region :
  (radius_smaller = (2 : ℝ) / 2) →
  (radius_larger = 4 * radius_smaller) →
  (gray_area = π * radius_larger ^ 2 - π * radius_smaller ^ 2) →
  gray_area = 15 * π :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  sorry

end area_of_gray_region_l130_130188


namespace equation_has_two_solutions_l130_130754

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end equation_has_two_solutions_l130_130754


namespace students_receiving_B_lee_l130_130457

def num_students_receiving_B (students_kipling: ℕ) (B_kipling: ℕ) (students_lee: ℕ) : ℕ :=
  let ratio := (B_kipling * students_lee) / students_kipling
  ratio

theorem students_receiving_B_lee (students_kipling B_kipling students_lee : ℕ) 
  (h : B_kipling = 8 ∧ students_kipling = 12 ∧ students_lee = 30) :
  num_students_receiving_B students_kipling B_kipling students_lee = 20 :=
by
  sorry

end students_receiving_B_lee_l130_130457


namespace calculate_expr_l130_130391

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l130_130391


namespace moles_of_water_formed_l130_130447

-- Definitions
def moles_of_H2SO4 : Nat := 3
def moles_of_NaOH : Nat := 3
def moles_of_NaHSO4 : Nat := 3
def moles_of_H2O := moles_of_NaHSO4

-- Theorem
theorem moles_of_water_formed :
  moles_of_H2SO4 = 3 →
  moles_of_NaOH = 3 →
  moles_of_NaHSO4 = 3 →
  moles_of_H2O = 3 :=
by
  intros h1 h2 h3
  rw [moles_of_H2O]
  exact h3

end moles_of_water_formed_l130_130447


namespace distance_from_focus_to_line_l130_130650

open Real

-- Define the hyperbola and line equations as conditions
def hyperbola (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 5) = 1
def line (x y : ℝ) : Prop := x + 2 * y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus := (3 : ℝ, 0 : ℝ)

-- Define the Euclidean distance from a point to a line
def distance_point_to_line (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

-- The theorem statement to prove
theorem distance_from_focus_to_line : distance_point_to_line 3 0 1 2 (-8) = sqrt 5 :=
by
  sorry

end distance_from_focus_to_line_l130_130650


namespace ellipse_semimajor_axis_value_l130_130957

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l130_130957


namespace evaluate_expression_l130_130870

theorem evaluate_expression :
  let a := 17
  let b := 19
  let c := 23
  let numerator1 := 136 * (1 / b - 1 / c) + 361 * (1 / c - 1 / a) + 529 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  let numerator2 := 144 * (1 / b - 1 / c) + 400 * (1 / c - 1 / a) + 576 * (1 / a - 1 / b)
  (numerator1 / denominator) * (numerator2 / denominator) = 3481 := by
  sorry

end evaluate_expression_l130_130870


namespace missed_angle_l130_130084

theorem missed_angle (sum_calculated : ℕ) (missed_angle_target : ℕ) 
  (h1 : sum_calculated = 2843) 
  (h2 : missed_angle_target = 37) : 
  ∃ n : ℕ, (sum_calculated + missed_angle_target = n * 180) :=
by {
  sorry
}

end missed_angle_l130_130084


namespace athleteA_time_to_complete_race_l130_130043

theorem athleteA_time_to_complete_race
    (v : ℝ)
    (t : ℝ)
    (h1 : v = 1000 / t)
    (h2 : v = 948 / (t + 18)) :
    t = 18000 / 52 := by
  sorry

end athleteA_time_to_complete_race_l130_130043


namespace lucas_total_pages_l130_130624

-- Define the variables and conditions
def lucas_read_pages : Nat :=
  let pages_first_four_days := 4 * 20
  let pages_break_day := 0
  let pages_next_four_days := 4 * 30
  let pages_last_day := 15
  pages_first_four_days + pages_break_day + pages_next_four_days + pages_last_day

-- State the theorem
theorem lucas_total_pages :
  lucas_read_pages = 215 :=
sorry

end lucas_total_pages_l130_130624


namespace distance_from_right_focus_to_line_l130_130647

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l130_130647


namespace sum_of_cubes_of_roots_l130_130984

theorem sum_of_cubes_of_roots :
  ∀ (x1 x2 : ℝ), (2 * x1^2 - 5 * x1 + 1 = 0) ∧ (2 * x2^2 - 5 * x2 + 1 = 0) →
  (x1 + x2 = 5 / 2) ∧ (x1 * x2 = 1 / 2) →
  (x1^3 + x2^3 = 95 / 8) :=
by
  sorry

end sum_of_cubes_of_roots_l130_130984


namespace distance_between_stations_l130_130513

theorem distance_between_stations
  (v₁ v₂ : ℝ)
  (D₁ D₂ : ℝ)
  (T : ℝ)
  (h₁ : v₁ = 20)
  (h₂ : v₂ = 25)
  (h₃ : D₂ = D₁ + 70)
  (h₄ : D₁ = v₁ * T)
  (h₅ : D₂ = v₂ * T) : 
  D₁ + D₂ = 630 := 
by
  sorry

end distance_between_stations_l130_130513


namespace proj_vector_correct_l130_130442

open Real

noncomputable def vector_proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot := u.1 * v.1 + u.2 * v.2
  let mag_sq := v.1 * v.1 + v.2 * v.2
  (dot / mag_sq) • v

theorem proj_vector_correct :
  vector_proj ⟨3, -1⟩ ⟨4, -6⟩ = ⟨18 / 13, -27 / 13⟩ :=
  sorry

end proj_vector_correct_l130_130442


namespace multiples_of_4_l130_130990

theorem multiples_of_4 (n : ℕ) (h : n + 23 * 4 = 112) : n = 20 :=
by
  sorry

end multiples_of_4_l130_130990


namespace digit_start_l130_130002

theorem digit_start (a n p q : ℕ) (hp : a * 10^p < 2^n) (hq : 2^n < (a + 1) * 10^p)
  (hr : a * 10^q < 5^n) (hs : 5^n < (a + 1) * 10^q) :
  a = 3 :=
by
  -- The proof goes here.
  sorry

end digit_start_l130_130002


namespace find_perpendicular_slope_value_l130_130897

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l130_130897


namespace problem_l130_130451

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem_l130_130451


namespace white_tshirt_cost_l130_130972

-- Define the problem conditions
def total_tshirts : ℕ := 200
def total_minutes : ℕ := 25
def black_tshirt_cost : ℕ := 30
def revenue_per_minute : ℕ := 220

-- Prove the cost of white t-shirts given the conditions
theorem white_tshirt_cost : 
  (total_tshirts / 2) * revenue_per_minute * total_minutes 
  - (total_tshirts / 2) * black_tshirt_cost = 2500
  → 2500 / (total_tshirts / 2) = 25 :=
by
  sorry

end white_tshirt_cost_l130_130972


namespace isabel_money_left_l130_130465

theorem isabel_money_left (initial_amount : ℕ) (half_toy_expense half_book_expense money_left : ℕ) :
  initial_amount = 204 →
  half_toy_expense = initial_amount / 2 →
  half_book_expense = (initial_amount - half_toy_expense) / 2 →
  money_left = initial_amount - half_toy_expense - half_book_expense →
  money_left = 51 :=
by
  intros h1 h2 h3 h4
  sorry

end isabel_money_left_l130_130465


namespace number_of_sides_of_polygon_l130_130458

theorem number_of_sides_of_polygon (exterior_angle : ℝ) (sum_exterior_angles : ℝ) (h1 : exterior_angle = 30) (h2 : sum_exterior_angles = 360) :
  sum_exterior_angles / exterior_angle = 12 := 
by
  sorry

end number_of_sides_of_polygon_l130_130458


namespace geom_seq_a7_a10_sum_l130_130769

theorem geom_seq_a7_a10_sum (a_n : ℕ → ℝ) (q a1 : ℝ)
  (h_seq : ∀ n, a_n (n + 1) = a1 * (q ^ n))
  (h1 : a1 + a1 * q = 2)
  (h2 : a1 * (q ^ 2) + a1 * (q ^ 3) = 4) :
  a_n 7 + a_n 8 + a_n 9 + a_n 10 = 48 := 
sorry

end geom_seq_a7_a10_sum_l130_130769


namespace sum_nine_smallest_even_multiples_of_7_l130_130196

theorem sum_nine_smallest_even_multiples_of_7 : 
  ∑ i in finset.range 9, 14 * (i + 1) = 630 :=
by
  sorry

end sum_nine_smallest_even_multiples_of_7_l130_130196


namespace frac_sum_is_one_l130_130885

theorem frac_sum_is_one (a b c : ℝ) (h : a / 2 = b / 3 ∧ b / 3 = c / 5) : (a + b) / c = 1 :=
by
  sorry

end frac_sum_is_one_l130_130885


namespace opera_house_earnings_l130_130703

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l130_130703


namespace water_filled_percent_l130_130070

-- Definitions and conditions
def cone_volume (R H : ℝ) : ℝ := (1 / 3) * π * R^2 * H
def water_cone_height (H : ℝ) : ℝ := (2 / 3) * H
def water_cone_radius (R : ℝ) : ℝ := (2 / 3) * R

-- The problem statement to be proved
theorem water_filled_percent (R H : ℝ) (hR : 0 < R) (hH : 0 < H) :
  let V := cone_volume R H in
  let v := cone_volume (water_cone_radius R) (water_cone_height H) in
  (v / V * 100) = 88.8889 :=
by 
  sorry

end water_filled_percent_l130_130070


namespace rest_area_location_l130_130347

theorem rest_area_location :
  ∀ (A B : ℝ), A = 50 → B = 230 → (5 / 8 * (B - A) + A = 162.5) :=
by
  intros A B hA hB
  rw [hA, hB]
  -- doing the computation to show the rest area is at 162.5 km
  sorry

end rest_area_location_l130_130347


namespace unique_point_intersection_l130_130503

theorem unique_point_intersection (k : ℝ) :
  (∃ x y, y = k * x + 2 ∧ y ^ 2 = 8 * x) → 
  ((k = 0) ∨ (k = 1)) :=
by {
  sorry
}

end unique_point_intersection_l130_130503


namespace jay_savings_first_week_l130_130780

theorem jay_savings_first_week :
  ∀ (x : ℕ), (x + (x + 10) + (x + 20) + (x + 30) = 60) → x = 0 :=
by
  intro x h
  sorry

end jay_savings_first_week_l130_130780


namespace sum_of_digits_l130_130438

theorem sum_of_digits (a b c d : ℕ) (h_diff : ∀ x y : ℕ, (x = a ∨ x = b ∨ x = c ∨ x = d) → (y = a ∨ y = b ∨ y = c ∨ y = d) → x ≠ y) (h1 : a + c = 10) (h2 : b + c = 8) (h3 : a + d = 11) : 
  a + b + c + d = 18 :=
by
  sorry

end sum_of_digits_l130_130438


namespace y_gt_1_l130_130982

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end y_gt_1_l130_130982


namespace missy_total_patients_l130_130629

theorem missy_total_patients 
  (P : ℕ)
  (h1 : ∀ x, (∃ y, y = ↑(1/3) * ↑x) → ∃ z, z = y * (120/100))
  (h2 : ∀ x, 5 * x = 5 * (x - ↑(1/3) * ↑x) + (120/100) * 5 * (↑(1/3) * ↑x))
  (h3 : 64 = 5 * (2/3) * (P : ℕ) + 6 * (1/3) * (P : ℕ)) :
  P = 12 :=
by
  sorry

end missy_total_patients_l130_130629


namespace y_greater_than_one_l130_130981

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l130_130981


namespace geometric_series_sum_l130_130017

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l130_130017


namespace bella_total_roses_l130_130854

-- Define the constants and conditions
def dozen := 12
def roses_from_parents := 2 * dozen
def friends := 10
def roses_per_friend := 2
def total_roses := roses_from_parents + (roses_per_friend * friends)

-- Prove that the total number of roses Bella received is 44
theorem bella_total_roses : total_roses = 44 := 
by
  sorry

end bella_total_roses_l130_130854


namespace valid_vector_parameterizations_of_line_l130_130335

theorem valid_vector_parameterizations_of_line (t : ℝ) :
  (∃ t : ℝ, (∃ x y : ℝ, (x = 1 + t ∧ y = t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = -t ∧ y = -1 - t ∧ y = x - 1)) ∨
            (∃ x y : ℝ, (x = 2 + 0.5 * t ∧ y = 1 + 0.5 * t ∧ y = x - 1))) :=
by sorry

end valid_vector_parameterizations_of_line_l130_130335


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l130_130569

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l130_130569


namespace shaded_fraction_is_four_fifteenths_l130_130699

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ)
  let r := (1/16 : ℚ)
  a / (1 - r)

theorem shaded_fraction_is_four_fifteenths :
  shaded_fraction = (4 / 15 : ℚ) := sorry

end shaded_fraction_is_four_fifteenths_l130_130699


namespace angle_ACB_ninety_degrees_l130_130904

theorem angle_ACB_ninety_degrees : 
  let line := λ p : ℝ × ℝ, p.1 - p.2 + 2 = 0
  let circle := λ p : ℝ × ℝ, (p.1 - 3)^2 + (p.2 - 3)^2 = 4
  let A B : ℝ × ℝ
  
  (line A) ∧ (circle A) ∧ (line B) ∧ (circle B) →
  ∃ C : ℝ × ℝ, (C = (3, 3)) →
  ∠ACB = 90 :=
by
  sorry

end angle_ACB_ninety_degrees_l130_130904


namespace boat_speed_in_still_water_l130_130462

variable (B S : ℝ)

theorem boat_speed_in_still_water :
  (B + S = 38) ∧ (B - S = 16) → B = 27 :=
by
  sorry

end boat_speed_in_still_water_l130_130462


namespace calculate_expr_l130_130395

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l130_130395


namespace solve_quadratic_l130_130289

theorem solve_quadratic (x : ℝ) : x^2 - 4 * x + 3 = 0 ↔ (x = 1 ∨ x = 3) :=
by
  sorry

end solve_quadratic_l130_130289


namespace ellipse_a_value_l130_130950

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (b^2 / a^2))

theorem ellipse_a_value :
  ∀ (a : ℝ), 
  (1 < a) →
  (C_1 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1) →
  (C_2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1) →
  let e2 := eccentricity 2 1 in
  let e1 := e2 / Real.sqrt 3 in
  e1 = 1 / 2 →
  a = 2 * Real.sqrt 3 / 3 :=
by
  intros a h1 C_1 C_2 e2 e1 he1
  have h2 : e2 = Real.sqrt 3 / 2 := by sorry
  have h3 : e1 = 1 / 2 := by sorry
  have h4 : a = 2 * Real.sqrt 3 / 3 := by sorry
  exact sorry

end ellipse_a_value_l130_130950


namespace total_money_made_l130_130003

-- Define the conditions
def dollars_per_day : Int := 144
def number_of_days : Int := 22

-- State the proof problem
theorem total_money_made : (dollars_per_day * number_of_days = 3168) :=
by
  sorry

end total_money_made_l130_130003


namespace largest_prime_divisor_of_factorial_sum_l130_130102

theorem largest_prime_divisor_of_factorial_sum {n : ℕ} (h1 : n = 13) : 
  Nat.gcd (Nat.factorial 13) 15 = 1 ∧ Nat.gcd (Nat.factorial 13 * 15) 13 = 13 :=
by
  sorry

end largest_prime_divisor_of_factorial_sum_l130_130102


namespace difference_length_breadth_l130_130814

theorem difference_length_breadth (B L A : ℕ) (h1 : B = 11) (h2 : A = 21 * B) (h3 : A = L * B) :
  L - B = 10 :=
by
  sorry

end difference_length_breadth_l130_130814


namespace sandwiches_consumption_difference_l130_130157

theorem sandwiches_consumption_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let combined_monday_tuesday := monday_total + tuesday_total

  combined_monday_tuesday - wednesday_total = -5 :=
by
  sorry

end sandwiches_consumption_difference_l130_130157


namespace calculate_T1_T2_l130_130491

def triangle (a b c : ℤ) : ℤ := a + b - 2 * c

def T1 := triangle 3 4 5
def T2 := triangle 6 8 2

theorem calculate_T1_T2 : 2 * T1 + 3 * T2 = 24 :=
  by
    sorry

end calculate_T1_T2_l130_130491


namespace find_a_l130_130580

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l130_130580


namespace largest_of_five_consecutive_integers_with_product_15120_eq_9_l130_130570

theorem largest_of_five_consecutive_integers_with_product_15120_eq_9 :
  ∃ n : ℕ, (n + 0) * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 15120 ∧ n + 4 = 9 :=
by
  sorry

end largest_of_five_consecutive_integers_with_product_15120_eq_9_l130_130570


namespace infinite_series_computation_l130_130796

noncomputable def infinite_series_sum (a b : ℝ) : ℝ :=
  ∑' n : ℕ, if n = 0 then (0 : ℝ) else
    (1 : ℝ) / ((2 * (n - 1 : ℕ) * a - (n - 2 : ℕ) * b) * (2 * n * a - (n - 1 : ℕ) * b))

theorem infinite_series_computation (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b) :
  infinite_series_sum a b = 1 / ((2 * a - b) * (2 * b)) :=
by
  sorry

end infinite_series_computation_l130_130796


namespace number_of_blue_candles_l130_130713

def total_candles : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def blue_candles : ℕ := total_candles - (yellow_candles + red_candles)

theorem number_of_blue_candles : blue_candles = 38 :=
by
  unfold blue_candles
  unfold total_candles yellow_candles red_candles
  sorry

end number_of_blue_candles_l130_130713


namespace bees_population_reduction_l130_130849

theorem bees_population_reduction :
  ∀ (initial_population loss_per_day : ℕ),
  initial_population = 80000 → 
  loss_per_day = 1200 → 
  ∃ days : ℕ, initial_population - days * loss_per_day = initial_population / 4 ∧ days = 50 :=
by
  intros initial_population loss_per_day h_initial h_loss
  use 50
  sorry

end bees_population_reduction_l130_130849


namespace base_500_in_base_has_six_digits_l130_130531

theorem base_500_in_base_has_six_digits (b : ℕ) : b^5 ≤ 500 ∧ 500 < b^6 ↔ b = 3 := 
by
  sorry

end base_500_in_base_has_six_digits_l130_130531


namespace original_flour_quantity_l130_130625

-- Definitions based on conditions
def flour_called (x : ℝ) : Prop := 
  -- total flour Mary uses is x + extra 2 cups, which equals to 9 cups.
  x + 2 = 9

-- The proof statement we need to show
theorem original_flour_quantity : ∃ x : ℝ, flour_called x ∧ x = 7 := 
  sorry

end original_flour_quantity_l130_130625


namespace fraction_to_decimal_l130_130561

theorem fraction_to_decimal :
  ∀ x : ℚ, x = 52 / 180 → x = 0.1444 := 
sorry

end fraction_to_decimal_l130_130561


namespace together_work_days_l130_130205

/-- 
  X does the work in 10 days and Y does the same work in 15 days.
  Together, they will complete the work in 6 days.
 -/
theorem together_work_days (hx : ℝ) (hy : ℝ) : 
  (hx = 10) → (hy = 15) → (1 / (1 / hx + 1 / hy) = 6) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end together_work_days_l130_130205


namespace number_of_male_students_drawn_l130_130536

theorem number_of_male_students_drawn (total_students : ℕ) (total_male_students : ℕ) (total_female_students : ℕ) (sample_size : ℕ)
    (H1 : total_students = 350)
    (H2 : total_male_students = 70)
    (H3 : total_female_students = 280)
    (H4 : sample_size = 50) :
    total_male_students * sample_size / total_students = 10 :=
by
  sorry

end number_of_male_students_drawn_l130_130536


namespace common_root_unique_solution_l130_130882

theorem common_root_unique_solution
  (p : ℝ) (h : ∃ x, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) :
  p = 3 :=
by sorry

end common_root_unique_solution_l130_130882


namespace arc_length_correct_l130_130397

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ φ in (0 : ℝ)..(5 * Real.pi / 12), (2 : ℝ) * (Real.sqrt (φ ^ 2 + 1))

theorem arc_length_correct :
  arcLengthOfCurve = (65 / 144) + Real.log (3 / 2) := by
  sorry

end arc_length_correct_l130_130397


namespace distance_from_focus_to_line_l130_130657

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l130_130657


namespace total_boys_slide_l130_130996

theorem total_boys_slide (initial_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : additional_boys = 13) :
  initial_boys + additional_boys = 35 :=
by
  sorry

end total_boys_slide_l130_130996


namespace geometric_sequence_term_l130_130610

theorem geometric_sequence_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 8) 
  (h_a5 : a 5 = 64) : 
  a 3 = 16 := 
by 
  sorry

end geometric_sequence_term_l130_130610


namespace minimum_value_fraction_l130_130262

theorem minimum_value_fraction (m n : ℝ) (h_line : 2 * m * 2 + n * 2 - 4 = 0) (h_pos_m : m > 0) (h_pos_n : n > 0) :
  (m + n / 2 = 1) -> ∃ (m n : ℝ), (m > 0 ∧ n > 0) ∧ (3 + 2 * Real.sqrt 2 ≤ (1 / m + 4 / n)) :=
by
  sorry

end minimum_value_fraction_l130_130262


namespace parallelogram_area_correct_l130_130162

noncomputable def parallelogram_area (s1 s2 : ℝ) (a : ℝ) : ℝ :=
s2 * (2 * s2 * Real.sin a)

theorem parallelogram_area_correct (s2 a : ℝ) (h_pos_s2 : 0 < s2) :
  parallelogram_area (2 * s2) s2 a = 2 * s2^2 * Real.sin a :=
by
  unfold parallelogram_area
  sorry

end parallelogram_area_correct_l130_130162


namespace find_locus_of_p_l130_130971

noncomputable def locus_of_point_p (a b : ℝ) : Set (ℝ × ℝ) :=
{p | (p.snd = 0 ∧ -a < p.fst ∧ p.fst < a) ∨ (p.fst = a^2 / Real.sqrt (a^2 + b^2))}

theorem find_locus_of_p (a b : ℝ) (P : ℝ × ℝ) :
  (∃ (x0 y0: ℝ),
      P = (x0, y0) ∧
      ( ∃ (x1 y1 x2 y2 : ℝ),
        (x0 ≠ 0 ∨ y0 ≠ 0) ∧
        (x1 ≠ x2 ∨ y1 ≠ y2) ∧
        (y0 = 0 ∨ (b^2 * x0 = -a^2 * (x0 - Real.sqrt (a^2 + b^2)))) ∧
        ((y0 = 0 ∧ -a < x0 ∧ x0 < a) ∨ x0 = a^2 / Real.sqrt (a^2 + b^2)))) ↔ 
  P ∈ locus_of_point_p a b :=
sorry

end find_locus_of_p_l130_130971


namespace inequality_div_two_l130_130890

theorem inequality_div_two (a b : ℝ) (h : a > b) : (a / 2) > (b / 2) :=
sorry

end inequality_div_two_l130_130890


namespace profit_percentage_l130_130217

theorem profit_percentage (SP CP : ℝ) (H_SP : SP = 1800) (H_CP : CP = 1500) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end profit_percentage_l130_130217


namespace find_k_range_for_two_roots_l130_130445

noncomputable def f (k x : ℝ) : ℝ := (Real.log x / x) - k * x

theorem find_k_range_for_two_roots :
  ∃ k_min k_max : ℝ, k_min = (2 / (Real.exp 4)) ∧ k_max = (1 / (2 * Real.exp 1)) ∧
  ∀ k : ℝ, (k_min ≤ k ∧ k < k_max) ↔
    ∃ x1 x2 : ℝ, 
    (1 / Real.exp 1) ≤ x1 ∧ x1 ≤ Real.exp 2 ∧ 
    (1 / Real.exp 1) ≤ x2 ∧ x2 ≤ Real.exp 2 ∧ 
    f k x1 = 0 ∧ f k x2 = 0 ∧ 
    x1 ≠ x2 :=
sorry

end find_k_range_for_two_roots_l130_130445


namespace scientific_notation_of_0_0000007_l130_130317

theorem scientific_notation_of_0_0000007 :
  0.0000007 = 7 * 10 ^ (-7) :=
  by
  sorry

end scientific_notation_of_0_0000007_l130_130317


namespace find_additional_fuel_per_person_l130_130226

def num_passengers : ℕ := 30
def num_crew : ℕ := 5
def num_people : ℕ := num_passengers + num_crew
def num_bags_per_person : ℕ := 2
def num_bags : ℕ := num_people * num_bags_per_person
def fuel_empty_plane : ℕ := 20
def fuel_per_bag : ℕ := 2
def total_trip_fuel : ℕ := 106000
def trip_distance : ℕ := 400
def fuel_per_mile : ℕ := total_trip_fuel / trip_distance

def additional_fuel_per_person (x : ℕ) : Prop :=
  fuel_empty_plane + num_people * x + num_bags * fuel_per_bag = fuel_per_mile

theorem find_additional_fuel_per_person : additional_fuel_per_person 3 :=
  sorry

end find_additional_fuel_per_person_l130_130226


namespace compute_expression_l130_130718

noncomputable def quadratic_roots (a b c : ℝ) :
  {x : ℝ × ℝ // a * x.fst^2 + b * x.fst + c = 0 ∧ a * x.snd^2 + b * x.snd + c = 0} :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  ⟨(root1, root2), by sorry⟩

theorem compute_expression :
  let roots := quadratic_roots 5 (-3) (-4)
  let x1 := roots.val.fst
  let x2 := roots.val.snd
  2 * x1^2 + 3 * x2^2 = (178 : ℝ) / 25 := by
  sorry

end compute_expression_l130_130718


namespace total_cost_is_53_l130_130808

-- Defining the costs and quantities as constants
def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount : ℕ := 5

-- Get the cost of sandwiches purchased
def cost_of_sandwiches : ℕ := num_sandwiches * sandwich_cost

-- Get the cost of sodas purchased
def cost_of_sodas : ℕ := num_sodas * soda_cost

-- Calculate the total cost before discount
def total_cost_before_discount : ℕ := cost_of_sandwiches + cost_of_sodas

-- Calculate the total cost after discount
def total_cost_after_discount : ℕ := total_cost_before_discount - discount

-- The theorem stating that the total cost is 53 dollars
theorem total_cost_is_53 : total_cost_after_discount = 53 :=
by
  sorry

end total_cost_is_53_l130_130808


namespace time_passed_since_midnight_l130_130839

theorem time_passed_since_midnight (h : ℝ) :
  h = (12 - h) + (2/5) * h → h = 7.5 :=
by
  sorry

end time_passed_since_midnight_l130_130839


namespace chips_left_uneaten_l130_130407

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l130_130407


namespace find_theta_l130_130252

theorem find_theta (θ : Real) (h : abs θ < π / 2) (h_eq : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ)) :
  θ = π / 3 :=
sorry

end find_theta_l130_130252


namespace chess_tournament_max_N_l130_130607

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l130_130607


namespace gcd_1037_425_l130_130236

theorem gcd_1037_425 : Int.gcd 1037 425 = 17 :=
by
  sorry

end gcd_1037_425_l130_130236


namespace B_finishes_in_4_days_l130_130050

theorem B_finishes_in_4_days
  (A_days : ℕ) (B_days : ℕ) (working_days_together : ℕ) 
  (A_rate : ℝ) (B_rate : ℝ) (combined_rate : ℝ) (work_done : ℝ) (remaining_work : ℝ)
  (B_rate_alone : ℝ) (days_B: ℝ) :
  A_days = 5 →
  B_days = 10 →
  working_days_together = 2 →
  A_rate = 1 / A_days →
  B_rate = 1 / B_days →
  combined_rate = A_rate + B_rate →
  work_done = combined_rate * working_days_together →
  remaining_work = 1 - work_done →
  B_rate_alone = 1 / B_days →
  days_B = remaining_work / B_rate_alone →
  days_B = 4 := 
by
  intros
  sorry

end B_finishes_in_4_days_l130_130050


namespace circle_center_coordinates_l130_130494

theorem circle_center_coordinates :
  ∀ (x y : ℝ), x^2 + y^2 - 10 * x + 6 * y + 25 = 0 → (5, -3) = ((-(-10) / 2), (-6 / 2)) :=
by
  intros x y h
  have H : (5, -3) = ((-(-10) / 2), (-6 / 2)) := sorry
  exact H

end circle_center_coordinates_l130_130494


namespace find_intersection_l130_130416

noncomputable def intersection_of_lines : Prop :=
  ∃ (x y : ℚ), (5 * x - 3 * y = 15) ∧ (6 * x + 2 * y = 14) ∧ (x = 11 / 4) ∧ (y = -5 / 4)

theorem find_intersection : intersection_of_lines :=
  sorry

end find_intersection_l130_130416


namespace y_greater_than_one_l130_130980

variable (x y : ℝ)

theorem y_greater_than_one (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
sorry

end y_greater_than_one_l130_130980


namespace simone_finishes_task_at_1115_l130_130487

noncomputable def simone_finish_time
  (start_time: Nat) -- Start time in minutes past midnight
  (task_1_duration: Nat) -- Duration of the first task in minutes
  (task_2_duration: Nat) -- Duration of the second task in minutes
  (break_duration: Nat) -- Duration of the break in minutes
  (task_3_duration: Nat) -- Duration of the third task in minutes
  (end_time: Nat) := -- End time to be proven
  start_time + task_1_duration + task_2_duration + break_duration + task_3_duration = end_time

theorem simone_finishes_task_at_1115 :
  simone_finish_time 480 45 45 15 90 675 := -- 480 minutes is 8:00 AM; 675 minutes is 11:15 AM
  by sorry

end simone_finishes_task_at_1115_l130_130487


namespace gcd_lcm_product_l130_130567

theorem gcd_lcm_product (a b : ℕ) (g : gcd a b) (l : lcm a b) :
  a = 180 → b = 250 →
  g = 2 * 5 →
  l = 2^2 * 3^2 * 5^3 →
  g * l = 45000 := by
  intros
  sorry

end gcd_lcm_product_l130_130567


namespace not_all_on_C_implies_exists_not_on_C_l130_130919

def F (x y : ℝ) : Prop := sorry  -- Define F according to specifics
def on_curve_C (x y : ℝ) : Prop := sorry -- Define what it means to be on curve C according to specifics

theorem not_all_on_C_implies_exists_not_on_C (h : ¬ ∀ x y : ℝ, F x y → on_curve_C x y) :
  ∃ x y : ℝ, F x y ∧ ¬ on_curve_C x y := sorry

end not_all_on_C_implies_exists_not_on_C_l130_130919


namespace least_even_integer_square_l130_130246

theorem least_even_integer_square (E : ℕ) (h_even : E % 2 = 0) (h_square : ∃ (I : ℕ), 300 * E = I^2) : E = 6 ∧ ∃ (I : ℕ), I = 30 ∧ 300 * E = I^2 :=
sorry

end least_even_integer_square_l130_130246


namespace playground_area_l130_130332

theorem playground_area
  (w l : ℕ)
  (h₁ : l = 2 * w + 25)
  (h₂ : 2 * (l + w) = 650) :
  w * l = 22500 := 
sorry

end playground_area_l130_130332


namespace repeating_decimal_to_fraction_l130_130682

theorem repeating_decimal_to_fraction : (0.2727272727 : ℝ) = 3 / 11 := 
sorry

end repeating_decimal_to_fraction_l130_130682


namespace smallest_possible_value_other_integer_l130_130499

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l130_130499


namespace nine_possible_xs_l130_130265

theorem nine_possible_xs :
  ∃! (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 33) (h3 : 25 ≤ x),
    ∀ n, (1 ≤ n ∧ n ≤ 3 → n * x < 100 ∧ (n + 1) * x ≥ 100) :=
sorry

end nine_possible_xs_l130_130265


namespace opera_house_earnings_l130_130705

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l130_130705


namespace evaluate_P_l130_130785

noncomputable def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

theorem evaluate_P (y : ℝ) (z : ℝ) (hz : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P 2 = -22 := by
  sorry

end evaluate_P_l130_130785


namespace notebooks_cost_l130_130328

theorem notebooks_cost 
  (P N : ℝ)
  (h1 : 96 * P + 24 * N = 520)
  (h2 : ∃ x : ℝ, 3 * P + x * N = 60)
  (h3 : P + N = 15.512820512820513) :
  ∃ x : ℕ, x = 4 :=
by
  sorry

end notebooks_cost_l130_130328


namespace four_digit_numbers_property_l130_130280

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l130_130280


namespace soup_options_l130_130973

-- Define the given conditions
variables (lettuce_types tomato_types olive_types total_options : ℕ)
variable (S : ℕ)

-- State the conditions
theorem soup_options :
  lettuce_types = 2 →
  tomato_types = 3 →
  olive_types = 4 →
  total_options = 48 →
  (lettuce_types * tomato_types * olive_types * S = total_options) →
  S = 2 :=
by
  sorry

end soup_options_l130_130973


namespace ellipse_eccentricity_l130_130949

theorem ellipse_eccentricity (a : Real) 
  (h1 : a > 1) 
  (h2 : ∀ x y : Real, x^2 / (a^2) + y^2 = 1) 
  (h3 : ∀ x y : Real, x^2 / 4 + y^2 = 1) 
  (e1 e2 : Real)
  (h4 : e1 = 1 / 2) 
  (h5 : e2 = sqrt 3 * e1) : 
  a = 2 * sqrt 3 / 3 := 
sorry

end ellipse_eccentricity_l130_130949


namespace solve_first_system_solve_second_system_l130_130640

-- Define the first system of equations
def first_system (x y : ℝ) : Prop := (3 * x + 2 * y = 5) ∧ (y = 2 * x - 8)

-- Define the solution to the first system
def solution1 (x y : ℝ) : Prop := (x = 3) ∧ (y = -2)

-- Define the second system of equations
def second_system (x y : ℝ) : Prop := (2 * x - y = 10) ∧ (2 * x + 3 * y = 2)

-- Define the solution to the second system
def solution2 (x y : ℝ) : Prop := (x = 4) ∧ (y = -2)

-- Define the problem statement in Lean
theorem solve_first_system : ∃ x y : ℝ, first_system x y ↔ solution1 x y :=
by
  sorry

theorem solve_second_system : ∃ x y : ℝ, second_system x y ↔ solution2 x y :=
by
  sorry

end solve_first_system_solve_second_system_l130_130640


namespace range_of_a_minus_b_l130_130448

theorem range_of_a_minus_b (a b : ℝ) (h1 : 1 < a ∧ a < 4) (h2 : -2 < b ∧ b < 4) : -3 < a - b ∧ a - b < 6 :=
sorry

end range_of_a_minus_b_l130_130448


namespace largest_prime_divisor_of_factorial_sum_l130_130104

theorem largest_prime_divisor_of_factorial_sum :
  ∀ (n : ℕ), 13 ≤ n → (n ∣ 13! + 14!) → is_prime n → n = 13 :=
by sorry

end largest_prime_divisor_of_factorial_sum_l130_130104


namespace Pyarelal_loss_share_l130_130845

-- Define the conditions
variables (P : ℝ) (A : ℝ) (total_loss : ℝ)

-- Ashok's capital is 1/9 of Pyarelal's capital
axiom Ashok_capital : A = (1 / 9) * P

-- Total loss is Rs 900
axiom total_loss_val : total_loss = 900

-- Prove Pyarelal's share of the loss is Rs 810
theorem Pyarelal_loss_share : (P / (A + P)) * total_loss = 810 :=
by
  sorry

end Pyarelal_loss_share_l130_130845


namespace ball_hits_ground_time_l130_130847

theorem ball_hits_ground_time (t : ℚ) :
  (-4.9 * (t : ℝ)^2 + 5 * (t : ℝ) + 10 = 0) → t = 10 / 7 :=
sorry

end ball_hits_ground_time_l130_130847


namespace point_symmetric_second_quadrant_l130_130762

theorem point_symmetric_second_quadrant (m : ℝ) 
  (symmetry : ∃ x y : ℝ, P = (-m, m-3) ∧ (-x, -y) = (x, y)) 
  (second_quadrant : ∃ x y : ℝ, P = (-m, m-3) ∧ x < 0 ∧ y > 0) : 
  m < 0 := 
sorry

end point_symmetric_second_quadrant_l130_130762


namespace sum_mod_six_l130_130012

theorem sum_mod_six (n : ℤ) : ((10 - 2 * n) + (4 * n + 2)) % 6 = 0 :=
by {
  sorry
}

end sum_mod_six_l130_130012


namespace Asya_Petya_l130_130720

theorem Asya_Petya (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
  (h : 1000 * a + b = 7 * a * b) : a = 143 ∧ b = 143 :=
by
  sorry

end Asya_Petya_l130_130720


namespace circle_radius_d_l130_130420

theorem circle_radius_d (d : ℝ) : ∀ (x y : ℝ), (x^2 + 8 * x + y^2 + 2 * y + d = 0) → (∃ r : ℝ, r = 5) → d = -8 :=
by
  sorry

end circle_radius_d_l130_130420


namespace min_a_for_monotonic_increase_l130_130125

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x ^ 3 + 2 * a * x ^ 2 + 2

theorem min_a_for_monotonic_increase :
  ∀ a : ℝ, (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → x^2 + 4 * a * x ≥ 0) ↔ a ≥ -1/4 := sorry

end min_a_for_monotonic_increase_l130_130125


namespace product_less_than_5_probability_l130_130250

open_locale classical

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4}

-- Define the pairs of different numbers
def pairs : Finset (ℕ × ℕ) := num_set.product num_set.filter (λ (a : ℕ × ℕ), a.1 < a.2)

-- Define the pairs whose product is less than 5
def pairs_less_than_5 : Finset (ℕ × ℕ) := pairs.filter (λ (p : ℕ × ℕ), p.1 * p.2 < 5)

-- Define the probability computation
def probability : ℚ := pairs_less_than_5.card / pairs.card

-- Theorem statement
theorem product_less_than_5_probability : probability = 1 / 2 := by
  sorry

end product_less_than_5_probability_l130_130250


namespace more_customers_left_than_stayed_l130_130548

-- Define the initial number of customers.
def initial_customers : ℕ := 11

-- Define the number of customers who stayed behind.
def customers_stayed : ℕ := 3

-- Define the number of customers who left.
def customers_left : ℕ := initial_customers - customers_stayed

-- Prove that the number of customers who left is 5 more than those who stayed behind.
theorem more_customers_left_than_stayed : customers_left - customers_stayed = 5 := by
  -- Sorry to skip the proof 
  sorry

end more_customers_left_than_stayed_l130_130548


namespace gcd_power_diff_l130_130677

theorem gcd_power_diff (m n : ℕ) (h1 : m = 2^2021 - 1) (h2 : n = 2^2000 - 1) :
  Nat.gcd m n = 2097151 :=
by sorry

end gcd_power_diff_l130_130677


namespace find_x_l130_130510

theorem find_x (x y : ℝ)
  (h1 : 2 * x + (x - 30) = 360)
  (h2 : y = x - 30)
  (h3 : 2 * x = 4 * y) :
  x = 130 := 
sorry

end find_x_l130_130510


namespace total_games_played_l130_130829

theorem total_games_played (months : ℕ) (games_per_month : ℕ) (h1 : months = 17) (h2 : games_per_month = 19) : 
  months * games_per_month = 323 :=
by
  sorry

end total_games_played_l130_130829


namespace andy_coats_l130_130925

-- Define the initial number of minks Andy buys
def initial_minks : ℕ := 30

-- Define the number of babies each mink has
def babies_per_mink : ℕ := 6

-- Define the total initial minks including babies
def total_initial_minks : ℕ := initial_minks * babies_per_mink + initial_minks

-- Define the number of minks set free by activists
def minks_set_free : ℕ := total_initial_minks / 2

-- Define the number of minks remaining after half are set free
def remaining_minks : ℕ := total_initial_minks - minks_set_free

-- Define the number of mink skins needed for one coat
def mink_skins_per_coat : ℕ := 15

-- Define the number of coats Andy can make
def coats_andy_can_make : ℕ := remaining_minks / mink_skins_per_coat

-- The theorem to prove the number of coats Andy can make
theorem andy_coats : coats_andy_can_make = 7 := by
  sorry

end andy_coats_l130_130925


namespace distance_hyperbola_focus_to_line_l130_130659

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l130_130659


namespace molecular_weight_correct_l130_130517

def potassium_weight : ℝ := 39.10
def chromium_weight : ℝ := 51.996
def oxygen_weight : ℝ := 16.00

def num_potassium_atoms : ℕ := 2
def num_chromium_atoms : ℕ := 2
def num_oxygen_atoms : ℕ := 7

def molecular_weight_of_compound : ℝ :=
  (num_potassium_atoms * potassium_weight) +
  (num_chromium_atoms * chromium_weight) +
  (num_oxygen_atoms * oxygen_weight)

theorem molecular_weight_correct :
  molecular_weight_of_compound = 294.192 :=
by
  sorry

end molecular_weight_correct_l130_130517


namespace movie_ticket_ratio_l130_130477

-- Definitions based on the conditions
def monday_cost : ℕ := 5
def wednesday_cost : ℕ := 2 * monday_cost

theorem movie_ticket_ratio (S : ℕ) (h1 : wednesday_cost + S = 35) :
  S / monday_cost = 5 :=
by
  -- Placeholder for proof
  sorry

end movie_ticket_ratio_l130_130477


namespace inscribed_circle_implies_rhombus_l130_130486

theorem inscribed_circle_implies_rhombus (AB : ℝ) (AD : ℝ)
  (h_parallelogram : AB = CD ∧ AD = BC) 
  (h_inscribed : AB + CD = AD + BC) : 
  AB = AD := by
  sorry

end inscribed_circle_implies_rhombus_l130_130486


namespace number_of_ways_to_divide_day_l130_130076

theorem number_of_ways_to_divide_day (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = 1440) : 
  ∃ (pairs : List (ℕ × ℕ)), (pairs.length = 36) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 * p.2 = 1440)) :=
sorry

end number_of_ways_to_divide_day_l130_130076


namespace cone_volume_filled_88_8900_percent_l130_130063

noncomputable def cone_volume_ratio_filled_to_two_thirds_height
  (h r : ℝ) (π : ℝ) : ℝ :=
  let V := (1 / 3) * π * r ^ 2 * h
  let V' := (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)
  (V' / V * 100)

theorem cone_volume_filled_88_8900_percent
  (h r π : ℝ) (V V' : ℝ)
  (V_def : V = (1 / 3) * π * r ^ 2 * h)
  (V'_def : V' = (1 / 3) * π * (2 / 3 * r) ^ 2 * (2 / 3 * h)):
  cone_volume_ratio_filled_to_two_thirds_height h r π = 88.8900 :=
by
  sorry

end cone_volume_filled_88_8900_percent_l130_130063


namespace locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l130_130212

/-- Given a circle with center at point P passes through point A (1,0) 
    and is tangent to the line x = -1, the locus of point P is the parabola C. -/
theorem locus_of_P_is_parabola (P A : ℝ × ℝ) (x y : ℝ):
  (A = (1, 0)) → (P.1 + 1)^2 + P.2^2 = 0 → y^2 = 4 * x := 
sorry

/-- If the line passing through point H(4, 0) intersects the parabola 
    C (denoted by y^2 = 4x) at points M and N, and T is any point on 
    the line x = -4, then the slopes of lines TM, TH, and TN form an 
    arithmetic sequence. -/
theorem slopes_form_arithmetic_sequence (H M N T : ℝ × ℝ) (m n k : ℝ): 
  (H = (4, 0)) → (T.1 = -4) → 
  (M.1, M.2) = (k^2, 4*k) ∧ (N.1, N.2) = (m^2, 4*m) → 
  ((T.2 - M.2) / (T.1 - M.1) + (T.2 - N.2) / (T.1 - N.1)) = 
  2 * (T.2 / -8) := 
sorry

end locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l130_130212


namespace eq_y_as_x_l130_130856

theorem eq_y_as_x (y x : ℝ) : 
  (y = 2*x - 3*y) ∨ (x = 2 - 3*y) ∨ (-y = 2*x - 1) ∨ (y = x) → (y = x) :=
by
  sorry

end eq_y_as_x_l130_130856


namespace proof_problem_l130_130151

noncomputable def f (x : ℝ) := 3 * Real.sin x + 2 * Real.cos x + 1

theorem proof_problem (a b c : ℝ) (h : ∀ x : ℝ, a * f x + b * f (x - c) = 1) :
  (b * Real.cos c / a) = -1 :=
sorry

end proof_problem_l130_130151


namespace sine_of_negative_90_degrees_l130_130863

theorem sine_of_negative_90_degrees : Real.sin (-(Real.pi / 2)) = -1 := 
sorry

end sine_of_negative_90_degrees_l130_130863


namespace sin_transformation_l130_130738

theorem sin_transformation (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 3 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -7 / 25 :=
by
  sorry

end sin_transformation_l130_130738


namespace area_of_rhombus_l130_130473

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 4) (h2 : d2 = 4) :
    (d1 * d2) / 2 = 8 := by
  sorry

end area_of_rhombus_l130_130473


namespace negation_of_universal_proposition_l130_130979

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_proposition_l130_130979


namespace diagonal_splits_odd_vertices_l130_130921

theorem diagonal_splits_odd_vertices (n : ℕ) (H : n^2 ≤ (2 * n + 2) * (2 * n + 1) / 2) :
  ∃ (x y : ℕ), x < y ∧ x ≤ 2 * n + 1 ∧ y ≤ 2 * n + 2 ∧ (y - x) % 2 = 0 :=
sorry

end diagonal_splits_odd_vertices_l130_130921


namespace work_days_in_week_l130_130077

theorem work_days_in_week (total_toys_per_week : ℕ) (toys_produced_each_day : ℕ) (h1 : total_toys_per_week = 6500) (h2 : toys_produced_each_day = 1300) : 
  total_toys_per_week / toys_produced_each_day = 5 :=
by
  sorry

end work_days_in_week_l130_130077


namespace cos_60_eq_half_l130_130988

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l130_130988


namespace lifespan_of_bat_l130_130978

theorem lifespan_of_bat (B : ℕ) (h₁ : ∀ B, B - 6 < B)
    (h₂ : ∀ B, 4 * (B - 6) < 4 * B)
    (h₃ : B + (B - 6) + 4 * (B - 6) = 30) :
    B = 10 := by
  sorry

end lifespan_of_bat_l130_130978


namespace max_value_of_operation_l130_130087

theorem max_value_of_operation : 
  ∃ (n : ℤ), (10 ≤ n ∧ n ≤ 99) ∧ 4 * (300 - n) = 1160 := by
  sorry

end max_value_of_operation_l130_130087


namespace sqrt_difference_eq_neg_six_sqrt_two_l130_130555

theorem sqrt_difference_eq_neg_six_sqrt_two :
  (Real.sqrt ((5 - 3 * Real.sqrt 2)^2)) - (Real.sqrt ((5 + 3 * Real.sqrt 2)^2)) = -6 * Real.sqrt 2 := 
sorry

end sqrt_difference_eq_neg_six_sqrt_two_l130_130555


namespace base_conversion_sum_l130_130235

-- Definition of conversion from base 13 to base 10
def base13_to_base10 (n : ℕ) : ℕ :=
  3 * (13^2) + 4 * (13^1) + 5 * (13^0)

-- Definition of conversion from base 14 to base 10 where C = 12 and D = 13
def base14_to_base10 (m : ℕ) : ℕ :=
  4 * (14^2) + 12 * (14^1) + 13 * (14^0)

theorem base_conversion_sum :
  base13_to_base10 345 + base14_to_base10 (4 * 14^2 + 12 * 14 + 13) = 1529 := 
by
  sorry -- proof to be provided

end base_conversion_sum_l130_130235


namespace total_sections_l130_130688

theorem total_sections (boys girls gcd sections_boys sections_girls : ℕ) 
  (h_boys : boys = 408) 
  (h_girls : girls = 264) 
  (h_gcd: gcd = Nat.gcd boys girls)
  (h_sections_boys : sections_boys = boys / gcd)
  (h_sections_girls : sections_girls = girls / gcd)
  (h_total_sections : sections_boys + sections_girls = 28)
: sections_boys + sections_girls = 28 := by
  sorry

end total_sections_l130_130688


namespace geometric_series_sum_l130_130021

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l130_130021


namespace Richard_Orlando_ratio_l130_130616

def Jenny_cards : ℕ := 6
def Orlando_more_cards : ℕ := 2
def Total_cards : ℕ := 38

theorem Richard_Orlando_ratio :
  let Orlando_cards := Jenny_cards + Orlando_more_cards
  let Richard_cards := Total_cards - (Jenny_cards + Orlando_cards)
  let ratio := Richard_cards / Orlando_cards
  ratio = 3 :=
by
  sorry

end Richard_Orlando_ratio_l130_130616


namespace find_n_l130_130516

theorem find_n (n : ℤ) (h₀ : 0 ≤ n) (h₁ : n < 59) (h₂ : 58 * n ≡ 20 [ZMOD 59]) : n = 39 :=
  sorry

end find_n_l130_130516


namespace prism_distance_to_plane_l130_130122

theorem prism_distance_to_plane
  (side_length : ℝ)
  (volume : ℝ)
  (h : ℝ)
  (base_is_square : side_length = 6)
  (volume_formula : volume = (1 / 3) * h * (side_length ^ 2)) :
  h = 8 := 
  by sorry

end prism_distance_to_plane_l130_130122


namespace total_visitors_400_l130_130204

variables (V E U : ℕ)

def visitors_did_not_enjoy_understand (V : ℕ) := 3 * V / 4 + 100 = V
def visitors_enjoyed_equal_understood (E U : ℕ) := E = U
def total_visitors_satisfy_34 (V E : ℕ) := 3 * V / 4 = E

theorem total_visitors_400
  (h1 : ∀ V, visitors_did_not_enjoy_understand V)
  (h2 : ∀ E U, visitors_enjoyed_equal_understood E U)
  (h3 : ∀ V E, total_visitors_satisfy_34 V E) :
  V = 400 :=
by { sorry }

end total_visitors_400_l130_130204


namespace milton_apple_pie_slices_l130_130221

theorem milton_apple_pie_slices :
  ∀ (A : ℕ),
  (∀ (peach_pie_slices_per : ℕ), peach_pie_slices_per = 6) →
  (∀ (apple_pie_slices_sold : ℕ), apple_pie_slices_sold = 56) →
  (∀ (peach_pie_slices_sold : ℕ), peach_pie_slices_sold = 48) →
  (∀ (total_pies_sold : ℕ), total_pies_sold = 15) →
  (∃ (apple_pie_slices : ℕ), apple_pie_slices = 56 / (total_pies_sold - (peach_pie_slices_sold / peach_pie_slices_per))) → 
  A = 8 :=
by sorry

end milton_apple_pie_slices_l130_130221


namespace count_valid_numbers_l130_130273
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l130_130273


namespace arithmetic_geometric_inequality_l130_130751

variables {a b A1 A2 G1 G2 x y d q : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b)
variables (h₂ : a = x - 3 * d) (h₃ : A1 = x - d) (h₄ : A2 = x + d) (h₅ : b = x + 3 * d)
variables (h₆ : a = y / q^3) (h₇ : G1 = y / q) (h₈ : G2 = y * q) (h₉ : b = y * q^3)
variables (h₁₀ : x - 3 * d = y / q^3) (h₁₁ : x + 3 * d = y * q^3)

theorem arithmetic_geometric_inequality : A1 * A2 ≥ G1 * G2 :=
by {
  sorry
}

end arithmetic_geometric_inequality_l130_130751


namespace prevent_white_cube_n2_prevent_white_cube_n3_l130_130483

def min_faces_to_paint (n : ℕ) : ℕ :=
  if n = 2 then 2 else if n = 3 then 12 else sorry

theorem prevent_white_cube_n2 : min_faces_to_paint 2 = 2 := by
  sorry

theorem prevent_white_cube_n3 : min_faces_to_paint 3 = 12 := by
  sorry

end prevent_white_cube_n2_prevent_white_cube_n3_l130_130483


namespace largest_integer_in_mean_set_l130_130326

theorem largest_integer_in_mean_set :
  ∃ (A B C D : ℕ), 
    A < B ∧ B < C ∧ C < D ∧
    (A + B + C + D) = 4 * 68 ∧
    A ≥ 5 ∧
    D = 254 :=
sorry

end largest_integer_in_mean_set_l130_130326


namespace triangle_min_value_l130_130777

open Real

theorem triangle_min_value
  (A B C : ℝ)
  (h_triangle: A + B + C = π)
  (h_sin: sin (2 * A + B) = 2 * sin B) :
  tan A + tan C + 2 / tan B ≥ 2 :=
sorry

end triangle_min_value_l130_130777


namespace triangle_sides_angles_l130_130791

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l130_130791


namespace coefficient_of_x_in_first_term_l130_130533

variable {a k n : ℝ} (x : ℝ)

theorem coefficient_of_x_in_first_term (h1 : (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
  (h2 : a - n + k = 7) :
  3 = 3 := 
sorry

end coefficient_of_x_in_first_term_l130_130533


namespace tangent_line_equations_l130_130892

theorem tangent_line_equations (k b : ℝ) :
  (∃ l : ℝ → ℝ, (∀ x, l x = k * x + b) ∧
    (∃ x₁, x₁^2 = k * x₁ + b) ∧ -- Tangency condition with C1: y = x²
    (∃ x₂, -(x₂ - 2)^2 = k * x₂ + b)) -- Tangency condition with C2: y = -(x-2)²
  → ((k = 0 ∧ b = 0) ∨ (k = 4 ∧ b = -4)) := sorry

end tangent_line_equations_l130_130892


namespace not_car_probability_l130_130745

-- Defining the probabilities of taking different modes of transportation.
def P_train : ℝ := 0.5
def P_car : ℝ := 0.2
def P_plane : ℝ := 0.3

-- Defining the event that these probabilities are for mutually exclusive events
axiom mutually_exclusive_events : P_train + P_car + P_plane = 1

-- Statement of the theorem to prove
theorem not_car_probability : P_train + P_plane = 0.8 := 
by 
  -- Use the definitions and axiom provided
  sorry

end not_car_probability_l130_130745


namespace bound_c_n_l130_130506

theorem bound_c_n (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, a (n + 1) = a n * (a n - 1)) →
  (∀ n, 2^b n = a n) →
  (∀ n, 2^(n - c n) = b n) →
  ∃ (m M : ℝ), (m = 0) ∧ (M = 1) ∧ ∀ n > 0, m ≤ c n ∧ c n ≤ M :=
by
  intro h1 h2 h3 h4
  use 0
  use 1
  sorry

end bound_c_n_l130_130506


namespace volume_of_solid_l130_130715

def x_y_relation (x y : ℝ) : Prop := x = (y - 2)^(1/3)
def x1 (x : ℝ) : Prop := x = 1
def y1 (y : ℝ) : Prop := y = 1

theorem volume_of_solid :
  ∀ (x y : ℝ),
    (x_y_relation x y ∧ x1 x ∧ y1 y) →
    ∃ V : ℝ, V = (44 / 7) * Real.pi :=
by
  -- Proof will go here
  sorry

end volume_of_solid_l130_130715


namespace count_friend_or_enemy_triples_l130_130770

-- Define a group of 30 people
noncomputable def people : Finset (Fin 30) := Finset.univ

-- Define the property of having exactly 6 enemies
def has_six_enemies (person : Fin 30) (enemies : Finset (Fin 30)) : Prop := 
  enemies.card = 6 ∧ ∀ e ∈ enemies, e ≠ person

-- Statement: There are 1990 ways to choose 3 people such that all are friends or all are enemies
theorem count_friend_or_enemy_triples (enemy_sets : (Fin 30) → Finset (Fin 30)) 
  (h : ∀ person, has_six_enemies person (enemy_sets person)) : 
  ∃ count : ℕ, count = 1990 ∧ count = 
    (people.choose 3).card - Finset.card (Finset.filter 
      (λ grp, ∃ a b c, {a, b, c} = grp ∧ 
        ((enemy_sets a).card = (enemy_sets b).card 
        ∧ (enemy_sets b).card = (enemy_sets c).card)) 
      (people.choose 3)) :=
sorry

end count_friend_or_enemy_triples_l130_130770


namespace equivalent_statements_l130_130029

variables (P Q : Prop)

theorem equivalent_statements (h : P → Q) : 
  ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) ↔ (P → Q) := by
sorry

end equivalent_statements_l130_130029


namespace number_that_multiplies_x_l130_130293

variables (n x y : ℝ)

theorem number_that_multiplies_x :
  n * x = 3 * y → 
  x * y ≠ 0 → 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 →
  n = 5 :=
by
  intros h1 h2 h3
  sorry

end number_that_multiplies_x_l130_130293


namespace ferris_wheel_capacity_l130_130537

theorem ferris_wheel_capacity :
  let num_seats := 4
  let people_per_seat := 4
  num_seats * people_per_seat = 16 := 
by
  let num_seats := 4
  let people_per_seat := 4
  sorry

end ferris_wheel_capacity_l130_130537


namespace distance_from_focus_to_line_l130_130658

theorem distance_from_focus_to_line :
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  distance = sqrt 5 :=
by
  let a := 2
  let b := sqrt (1 + 5 / 4)
  let focus := (a * b, 0)
  let distance := abs (1 * focus.1 + 2 * focus.2 - 8) / sqrt (1^2 + 2^2)
  have : distance = sqrt 5 := sorry
  exact this

end distance_from_focus_to_line_l130_130658


namespace avg_age_of_women_l130_130687

theorem avg_age_of_women (T : ℕ) (W : ℕ) (T_avg : ℕ) (H1 : T_avg = T / 10)
  (H2 : (T_avg + 6) = ((T - 18 - 22 + W) / 10)) : (W / 2) = 50 :=
sorry

end avg_age_of_women_l130_130687


namespace average_gpa_of_whole_class_l130_130846

-- Define the conditions
variables (n : ℕ)
def num_students_in_group1 := n / 3
def num_students_in_group2 := 2 * n / 3

def gpa_group1 := 15
def gpa_group2 := 18

-- Lean statement for the proof problem
theorem average_gpa_of_whole_class (hn_pos : 0 < n):
  ((num_students_in_group1 * gpa_group1) + (num_students_in_group2 * gpa_group2)) / n = 17 :=
sorry

end average_gpa_of_whole_class_l130_130846


namespace distance_from_right_focus_to_line_l130_130653

theorem distance_from_right_focus_to_line :
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  (c = 3) → (d = real.sqrt 5) :=
by
  let c := real.sqrt (4 + 5)
  let focus := (c, 0)
  let d := abs (1 * c + 2 * 0 - 8) / real.sqrt (1^2 + 2^2)
  have h_c : c = real.sqrt 9 := by sorry
  have h_focus : focus = (3, 0) := by sorry
  have h_d : d = real.sqrt 5 := by sorry
  exact h_d

end distance_from_right_focus_to_line_l130_130653


namespace probability_of_valid_triangle_l130_130408

open ProbabilityTheory

noncomputable def stick_lengths : List ℕ := [3, 4, 5, 8, 10, 12, 15, 18]

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ (a ≥ 10 ∨ b ≥ 10 ∨ c ≥ 10)

def count_valid_sets : ℕ :=
  stick_lengths.combinations 3 |>.filter (λ l, match l.sorted with
    | [a, b, c] => valid_triangle a b c
    | _ => false
  end) |>.length

def total_combinations : ℕ := (Nat.choose 8 3)

def probability_valid_triangle : ℚ :=
  count_valid_sets / total_combinations

theorem probability_of_valid_triangle : probability_valid_triangle = 11 / 56 := by
  sorry

end probability_of_valid_triangle_l130_130408


namespace graphs_intersect_exactly_one_point_l130_130422

theorem graphs_intersect_exactly_one_point (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 5 * x + 4 = 2 * x - 6 → x = (7 / (2 * k))) ↔ k = (49 / 40) := 
by
  sorry

end graphs_intersect_exactly_one_point_l130_130422


namespace find_coefficients_sum_l130_130743

theorem find_coefficients_sum :
  let f := (2 * x - 1) ^ 5 + (x + 2) ^ 4
  let a_0 := 15
  let a_1 := 42
  let a_2 := -16
  let a_5 := 32
  (|a_0| + |a_1| + |a_2| + |a_5| = 105) := 
by {
  sorry
}

end find_coefficients_sum_l130_130743


namespace y_gt_1_l130_130983

theorem y_gt_1 (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : y > 1 :=
by sorry

end y_gt_1_l130_130983


namespace sum_of_squares_due_to_regression_eq_72_l130_130294

theorem sum_of_squares_due_to_regression_eq_72
    (total_squared_deviations : ℝ)
    (correlation_coefficient : ℝ)
    (h1 : total_squared_deviations = 120)
    (h2 : correlation_coefficient = 0.6)
    : total_squared_deviations * correlation_coefficient^2 = 72 :=
by
  -- Proof goes here
  sorry

end sum_of_squares_due_to_regression_eq_72_l130_130294


namespace binomial_arithmetic_progression_rational_terms_l130_130902

open Nat

theorem binomial_arithmetic_progression (x : ℝ) (n : ℕ) (h1 : n < 15) (h2 : n > 0)
  (h3 : binomialCoeff n 8 + binomialCoeff n 10 = 2 * binomialCoeff n 9) : n = 14 :=
  sorry

theorem rational_terms (x : ℝ) (n : ℕ) (h : n = 14) :
  (∀ r, r ∈ [0, 6, 12] → (C 14 r * x ^ (7 - r / 6)).isRational) :=
  sorry

end binomial_arithmetic_progression_rational_terms_l130_130902


namespace probability_of_sum_3_or_6_l130_130766

noncomputable def probability_event_sums_3_or_6 : ℝ :=
  let balls := {1, 2, 3, 4, 5}
  let total_outcomes := choose 5 2
  let favorable_outcomes := 3  -- (1,2), (1,5), (2,4)
  favorable_outcomes / total_outcomes

theorem probability_of_sum_3_or_6 :
  probability_event_sums_3_or_6 = 3 / 10 :=
by
  sorry

end probability_of_sum_3_or_6_l130_130766


namespace number_subsets_property_p_l130_130797

def has_property_p (a b : ℕ) : Prop := 17 ∣ (a + b)

noncomputable def num_subsets_with_property_p : ℕ :=
  -- sorry, put computation result here using the steps above but skipping actual computation for brevity
  3928

theorem number_subsets_property_p :
  num_subsets_with_property_p = 3928 := sorry

end number_subsets_property_p_l130_130797


namespace unique_line_through_A_parallel_to_a_l130_130746

variables {Point Line Plane : Type}
variables {α β : Plane}
variables {a l : Line}
variables {A : Point}

-- Definitions are necessary from conditions in step a)
def parallel_to (a b : Line) : Prop := sorry -- Definition that two lines are parallel
def contains (p : Plane) (x : Point) : Prop := sorry -- Definition that a plane contains a point
def line_parallel_to_plane (a : Line) (p : Plane) : Prop := sorry -- Definition that a line is parallel to a plane

-- Given conditions in the proof problem
variable (a_parallel_α : line_parallel_to_plane a α)
variable (A_in_α : contains α A)

-- Statement to be proven: There is only one line that passes through point A and is parallel to line a, and that line is within plane α.
theorem unique_line_through_A_parallel_to_a : 
  ∃! l : Line, contains α A ∧ parallel_to l a := sorry

end unique_line_through_A_parallel_to_a_l130_130746


namespace area_ratio_of_triangles_l130_130290

theorem area_ratio_of_triangles (AC AD : ℝ) (h : ℝ) (hAC : AC = 1) (hAD : AD = 4) :
  (AC * h / 2) / ((AD - AC) * h / 2) = 1 / 3 :=
by
  sorry

end area_ratio_of_triangles_l130_130290


namespace find_star_1993_1935_l130_130721

axiom star (x y : ℕ) : ℕ
axiom star_idempotent (x : ℕ) : star x x = 0
axiom star_assoc (x y z : ℕ) : star x (star y z) = star x y + z

theorem find_star_1993_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_star_1993_1935_l130_130721


namespace determine_a_l130_130231

def quadratic_condition (a : ℝ) (x : ℝ) : Prop := 
  abs (x^2 + 2 * a * x + 3 * a) ≤ 2

theorem determine_a : {a : ℝ | ∃! x : ℝ, quadratic_condition a x} = {1, 2} :=
sorry

end determine_a_l130_130231


namespace four_digit_numbers_with_property_l130_130269

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l130_130269


namespace roots_of_polynomial_l130_130110

theorem roots_of_polynomial : ∀ x : ℝ, (x^2 - 5*x + 6) * (x - 3) * (x + 2) = 0 ↔ x = 2 ∨ x = 3 ∨ x = -2 :=
by sorry

end roots_of_polynomial_l130_130110


namespace num_positive_x_count_num_positive_x_l130_130266

theorem num_positive_x (x : ℕ) : (3 * x < 100) ∧ (4 * x ≥ 100) → x ≥ 25 ∧ x ≤ 33 := by
  sorry

theorem count_num_positive_x : 
  (∃ x : ℕ, (3 * x < 100) ∧ (4 * x ≥ 100)) → 
  (finset.range 34).filter (λ x, (3 * x < 100 ∧ 4 * x ≥ 100)).card = 9 := by
  sorry

end num_positive_x_count_num_positive_x_l130_130266


namespace Jake_weight_is_118_l130_130910

-- Define the current weights of Jake, his sister, and Mark
variable (J S M : ℕ)

-- Define the given conditions
axiom h1 : J - 12 = 2 * (S + 4)
axiom h2 : M = J + S + 50
axiom h3 : J + S + M = 385

theorem Jake_weight_is_118 : J = 118 :=
by
  sorry

end Jake_weight_is_118_l130_130910


namespace vertex_of_parabola_l130_130327

theorem vertex_of_parabola (a b c : ℝ) (h k : ℝ) (x y : ℝ) :
  (∀ x, y = (1/2) * (x - 1)^2 + 2) → (h, k) = (1, 2) :=
by
  intro hy
  exact sorry

end vertex_of_parabola_l130_130327


namespace combined_tax_rate_l130_130041

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 3 * Mork_income)
  (tax_Mork tax_Mindy : ℝ) (h2 : tax_Mork = 0.10 * Mork_income) (h3 : tax_Mindy = 0.20 * Mindy_income)
  : (tax_Mork + tax_Mindy) / (Mork_income + Mindy_income) = 0.175 :=
by
  sorry

end combined_tax_rate_l130_130041


namespace determine_range_of_a_l130_130761

variables {a : ℝ}
def f (x : ℝ) := sin x ^ 3 + a * cos x ^ 2

theorem determine_range_of_a (h : ∃ x ∈ Ioo 0 π, is_min_on f (Ioo 0 π) x) : 0 < a :=
sorry

end determine_range_of_a_l130_130761


namespace proportion_in_triangle_l130_130920

-- Definitions of the variables and conditions
variables {P Q R E : Point}
variables {p q r m n : ℝ}

-- Conditions
def angle_bisector_theorem (h : p = 2 * q) (h1 : m = q + q) (h2 : n = 2 * q) : Prop :=
  ∀ (p q r m n : ℝ), 
  (m / r) = (n / q) ∧ 
  (m + n = p) ∧
  (p = 2 * q)

-- The theorem to be proved
theorem proportion_in_triangle (h : p = 2 * q) (h1 : m / r = n / q) (h2 : m + n = p) : 
  (n / q = 2 * q / (r + q)) :=
by
  sorry

end proportion_in_triangle_l130_130920


namespace pictures_total_l130_130967

theorem pictures_total (peter_pics : ℕ) (quincy_extra_pics : ℕ) (randy_pics : ℕ) (quincy_pics : ℕ) (total_pics : ℕ) 
  (h1 : peter_pics = 8)
  (h2 : quincy_extra_pics = 20)
  (h3 : randy_pics = 5)
  (h4 : quincy_pics = peter_pics + quincy_extra_pics)
  (h5 : total_pics = randy_pics + peter_pics + quincy_pics) :
  total_pics = 41 :=
by sorry

end pictures_total_l130_130967


namespace four_digit_numbers_with_property_l130_130278

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l130_130278


namespace endpoints_undetermined_l130_130152

theorem endpoints_undetermined (m : ℝ → ℝ) :
  (∀ x, m x = x - 2) ∧ (∃ mid : ℝ × ℝ, ∃ (x1 x2 y1 y2 : ℝ), 
    mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m mid.1 = mid.2) → 
  ¬ (∃ (x1 x2 y1 y2 : ℝ), mid = ((x1 + x2) / 2, (y1 + y2) / 2) ∧ 
    m ((x1 + x2) / 2) = (y1 + y2) / 2 ∧
    x1 = the_exact_endpoint ∧ x2 = the_exact_other_endpoint) :=
by sorry

end endpoints_undetermined_l130_130152


namespace find_a5_of_geometric_sequence_l130_130592

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r, ∀ n, a (n + 1) = a n * r

theorem find_a5_of_geometric_sequence (a : ℕ → ℝ) (h : geometric_sequence a)
  (h₀ : a 1 = 1) (h₁ : a 9 = 3) : a 5 = Real.sqrt 3 :=
sorry

end find_a5_of_geometric_sequence_l130_130592


namespace isabel_uploaded_pictures_l130_130141

theorem isabel_uploaded_pictures :
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  total_pictures = 25 :=
by
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  show total_pictures = 25
  sorry

end isabel_uploaded_pictures_l130_130141


namespace ratio_of_apple_to_orange_cost_l130_130552

-- Define the costs of fruits based on the given conditions.
def cost_per_kg_oranges : ℝ := 12
def cost_per_kg_apples : ℝ := 2

-- The theorem to prove.
theorem ratio_of_apple_to_orange_cost : cost_per_kg_apples / cost_per_kg_oranges = 1 / 6 :=
by
  sorry

end ratio_of_apple_to_orange_cost_l130_130552


namespace work_completion_l130_130686

theorem work_completion (Rp Rq Dp W : ℕ) 
  (h1 : Rq = W / 12) 
  (h2 : W = 4*Rp + 6*(Rp + Rq)) 
  (h3 : Rp = W / Dp) 
  : Dp = 20 :=
by
  sorry

end work_completion_l130_130686


namespace find_number_l130_130840

theorem find_number (x : ℤ) (h : 4 * x = 28) : x = 7 :=
sorry

end find_number_l130_130840


namespace find_number_of_observations_l130_130009

theorem find_number_of_observations 
  (n : ℕ) 
  (mean_before_correction : ℝ)
  (incorrect_observation : ℝ)
  (correct_observation : ℝ)
  (mean_after_correction : ℝ) 
  (h0 : mean_before_correction = 36)
  (h1 : incorrect_observation = 23)
  (h2 : correct_observation = 45)
  (h3 : mean_after_correction = 36.5) 
  (h4 : (n * mean_before_correction + (correct_observation - incorrect_observation)) / n = mean_after_correction) : 
  n = 44 := 
by
  sorry

end find_number_of_observations_l130_130009


namespace find_g_l130_130963

theorem find_g (g : ℕ) (h : g > 0) :
  (1 / 3) = ((4 + g * (g - 1)) / ((g + 4) * (g + 3))) → g = 5 :=
by
  intro h_eq
  sorry 

end find_g_l130_130963


namespace shares_difference_l130_130379

theorem shares_difference (x : ℝ) (h_ratio : 2.5 * x + 3.5 * x + 7.5 * x + 9.8 * x = (23.3 * x))
  (h_difference : 7.5 * x - 3.5 * x = 4500) : 9.8 * x - 2.5 * x = 8212.5 :=
by
  sorry

end shares_difference_l130_130379


namespace sphere_volume_of_hexagonal_prism_l130_130080

noncomputable def volume_of_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_volume_of_hexagonal_prism
  (a h : ℝ)
  (volume : ℝ)
  (base_perimeter : ℝ)
  (vertices_on_sphere : ∀ (x y : ℝ) (hx : x^2 + y^2 = a^2) (hy : y = h / 2), x^2 + y^2 = 1) :
  volume = 9 / 8 ∧ base_perimeter = 3 →
  volume_of_sphere 1 = 4 * Real.pi / 3 :=
by
  sorry

end sphere_volume_of_hexagonal_prism_l130_130080


namespace max_x_of_conditions_l130_130149

theorem max_x_of_conditions (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 11) : x ≤ 2 :=
by
  -- Placeholder for the actual proof
  sorry

end max_x_of_conditions_l130_130149


namespace students_playing_both_l130_130383

theorem students_playing_both
    (total_students baseball_team hockey_team : ℕ)
    (h1 : total_students = 36)
    (h2 : baseball_team = 25)
    (h3 : hockey_team = 19)
    (h4 : total_students = baseball_team + hockey_team - students_both) :
    students_both = 8 := by
  sorry

end students_playing_both_l130_130383


namespace wheel_stop_probability_l130_130377

theorem wheel_stop_probability 
  (pD pE pG pF : ℚ) 
  (h1 : pD = 1 / 4) 
  (h2 : pE = 1 / 3) 
  (h3 : pG = 1 / 6) 
  (h4 : pD + pE + pG + pF = 1) : 
  pF = 1 / 4 := 
by 
  sorry

end wheel_stop_probability_l130_130377


namespace loss_percentage_is_75_l130_130358

-- Given conditions
def cost_price_one_book (C : ℝ) : Prop := C > 0
def selling_price_one_book (S : ℝ) : Prop := S > 0
def cost_price_5_equals_selling_price_20 (C S : ℝ) : Prop := 5 * C = 20 * S

-- Proof goal
theorem loss_percentage_is_75 (C S : ℝ) (h1 : cost_price_one_book C) (h2 : selling_price_one_book S) (h3 : cost_price_5_equals_selling_price_20 C S) : 
  ((C - S) / C) * 100 = 75 :=
by
  sorry

end loss_percentage_is_75_l130_130358


namespace problem_l130_130759

def x : ℕ := 660
def percentage_25_of_x : ℝ := 0.25 * x
def percentage_12_of_1500 : ℝ := 0.12 * 1500
def difference_of_percentages : ℝ := percentage_12_of_1500 - percentage_25_of_x

theorem problem : difference_of_percentages = 15 := by
  -- begin proof (content replaced by sorry)
  sorry

end problem_l130_130759


namespace number_of_possible_lists_l130_130361

/-- 
Define the basic conditions: 
- 18 balls, numbered 1 through 18
- Selection process is repeated 4 times 
- Each selection is independent
- After each selection, the ball is replaced 
- We need to prove the total number of possible lists of four numbers 
--/
def number_of_balls : ℕ := 18
def selections : ℕ := 4

theorem number_of_possible_lists : (number_of_balls ^ selections) = 104976 := by
  sorry

end number_of_possible_lists_l130_130361


namespace point_P_distance_to_y_axis_l130_130440

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, 3)

-- The distance from point P to the y-axis
def distance_to_y_axis (pt : ℝ × ℝ) : ℝ :=
  abs pt.1

-- Statement to prove
theorem point_P_distance_to_y_axis :
  distance_to_y_axis point_P = 2 :=
by
  sorry

end point_P_distance_to_y_axis_l130_130440


namespace find_a_l130_130954

-- Definitions given in the problem
def ellipse1 (a : ℝ) : Prop := a > 1 ∧ (∀ x y : ℝ, (x^2 / a^2 + y^2 = 1))
def ellipse2 : Prop := ∀ x y : ℝ, (x^2 / 4 + y^2 = 1)

-- Eccentricities
def eccentricity (c a : ℝ) : ℝ := c / a
def e2 := eccentricity (√3) 2

theorem find_a (a : ℝ) (e1 : ℝ) : 
  ellipse1 a →
  ellipse2 →
  e2 = (√3 * e1) →
  e1 = 1 / 2 →
  a = 2 * √3 / 3 :=
sorry

end find_a_l130_130954


namespace product_of_distinct_roots_l130_130757

theorem product_of_distinct_roots (x1 x2 : ℝ) (hx1 : x1 ^ 2 - 2 * x1 = 1) (hx2 : x2 ^ 2 - 2 * x2 = 1) (h_distinct : x1 ≠ x2) : 
  x1 * x2 = -1 := 
  sorry

end product_of_distinct_roots_l130_130757


namespace ratio_of_money_with_Ram_and_Gopal_l130_130341

noncomputable section

variable (R K G : ℕ)

theorem ratio_of_money_with_Ram_and_Gopal 
  (hR : R = 735) 
  (hK : K = 4335) 
  (hRatio : G * 17 = 7 * K) 
  (hGCD : Nat.gcd 735 1785 = 105) :
  R * 17 = 7 * G := 
by
  sorry

end ratio_of_money_with_Ram_and_Gopal_l130_130341


namespace solution1_solution2_l130_130225

noncomputable def problem1 : ℝ :=
  40 + ((1 / 6) - (2 / 3) + (3 / 4)) * 12

theorem solution1 : problem1 = 43 := by
  sorry

noncomputable def problem2 : ℝ :=
  (-1 : ℝ) ^ 2021 + |(-9 : ℝ)| * (2 / 3) + (-3) / (1 / 5)

theorem solution2 : problem2 = -10 := by
  sorry

end solution1_solution2_l130_130225


namespace find_x_l130_130116

def delta (x : ℝ) : ℝ := 4 * x + 9
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x (x : ℝ) (h : delta (phi x) = 10) : x = -23 / 36 := 
by 
  sorry

end find_x_l130_130116


namespace arithmetic_equation_false_l130_130362

theorem arithmetic_equation_false :
  4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 := by
  sorry

end arithmetic_equation_false_l130_130362


namespace total_distance_covered_l130_130368

theorem total_distance_covered :
  let speed_fox := 50       -- km/h
  let speed_rabbit := 60    -- km/h
  let speed_deer := 80      -- km/h
  let time_hours := 2       -- hours
  let distance_fox := speed_fox * time_hours
  let distance_rabbit := speed_rabbit * time_hours
  let distance_deer := speed_deer * time_hours
  distance_fox + distance_rabbit + distance_deer = 380 := by
sorry

end total_distance_covered_l130_130368


namespace mosquitoes_required_l130_130085

theorem mosquitoes_required
  (blood_loss_to_cause_death : Nat)
  (drops_per_mosquito_A : Nat)
  (drops_per_mosquito_B : Nat)
  (drops_per_mosquito_C : Nat)
  (n : Nat) :
  blood_loss_to_cause_death = 15000 →
  drops_per_mosquito_A = 20 →
  drops_per_mosquito_B = 25 →
  drops_per_mosquito_C = 30 →
  75 * n = blood_loss_to_cause_death →
  n = 200 := by
  sorry

end mosquitoes_required_l130_130085


namespace max_possible_N_l130_130605

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l130_130605


namespace congruence_equivalence_l130_130805

theorem congruence_equivalence (m n a b : ℤ) (h_coprime : Int.gcd m n = 1) :
  a ≡ b [ZMOD m * n] ↔ (a ≡ b [ZMOD m] ∧ a ≡ b [ZMOD n]) :=
sorry

end congruence_equivalence_l130_130805


namespace simplify_absolute_values_l130_130115

theorem simplify_absolute_values (a : ℝ) (h : -2 < a ∧ a < 0) : |a| + |a + 2| = 2 :=
sorry

end simplify_absolute_values_l130_130115


namespace final_apples_count_l130_130176

def initial_apples : ℝ := 5708
def apples_given_away : ℝ := 2347.5
def additional_apples_harvested : ℝ := 1526.75

theorem final_apples_count :
  initial_apples - apples_given_away + additional_apples_harvested = 4887.25 :=
by
  sorry

end final_apples_count_l130_130176


namespace Mark_hours_left_l130_130858

theorem Mark_hours_left (sick_days vacation_days : ℕ) (hours_per_day : ℕ) 
  (h1 : sick_days = 10) (h2 : vacation_days = 10) (h3 : hours_per_day = 8) 
  (used_sick_days : ℕ) (used_vacation_days : ℕ) 
  (h4 : used_sick_days = sick_days / 2) (h5 : used_vacation_days = vacation_days / 2) 
  : (sick_days + vacation_days - used_sick_days - used_vacation_days) * hours_per_day = 80 :=
by
  sorry

end Mark_hours_left_l130_130858


namespace proof_problem_l130_130124

variable {R : Type*} [Real R]

noncomputable def f : R → R := sorry

theorem proof_problem 
  (h_domain : ∀ (x : R), Continuous (f x))
  (h_functional_eq : ∀ (x y : R), f (x * y) = y^2 * f x + x^2 * f y) :
  f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : R, f (-x) = f x) :=
by
  sorry

end proof_problem_l130_130124


namespace sum_of_reciprocals_l130_130005

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : (1 / x + 1 / y) = (1 / 3) :=
by
  sorry

end sum_of_reciprocals_l130_130005


namespace intersecting_chords_length_l130_130598

theorem intersecting_chords_length
  (h1 : ∃ c1 c2 : ℝ, c1 = 8 ∧ c2 = x + 4 * x ∧ x = 2)
  (h2 : ∀ (a b c d : ℝ), a * b = c * d → a = 4 ∧ b = 4 ∧ c = x ∧ d = 4 * x ∧ x = 2):
  (10 : ℝ) = (x + 4 * x) := by
  sorry

end intersecting_chords_length_l130_130598


namespace probability_all_captains_selected_l130_130831

theorem probability_all_captains_selected :
  let teams := [6, 9, 10],
  let captains := 3,
  (1 / 3 : ℚ) * ((6 / (6 * 5 * 4)) + (6 / (9 * 8 * 7)) + (6 / (10 * 9 * 8))) = 177 / 12600 :=
by
  sorry

end probability_all_captains_selected_l130_130831


namespace orchids_initially_l130_130184

-- Definitions and Conditions
def initial_orchids (current_orchids: ℕ) (cut_orchids: ℕ) : ℕ :=
  current_orchids + cut_orchids

-- Proof statement
theorem orchids_initially (current_orchids: ℕ) (cut_orchids: ℕ) : initial_orchids current_orchids cut_orchids = 3 :=
by 
  have h1 : current_orchids = 7 := sorry
  have h2 : cut_orchids = 4 := sorry
  have h3 : initial_orchids current_orchids cut_orchids = 7 + 4 := sorry
  have h4 : initial_orchids current_orchids cut_orchids = 3 := sorry
  sorry

end orchids_initially_l130_130184


namespace george_income_l130_130112

def half (x: ℝ) : ℝ := x / 2

theorem george_income (income : ℝ) (H1 : half income - 20 = 100) : income = 240 := 
by 
  sorry

end george_income_l130_130112


namespace valid_three_digit_numbers_count_l130_130908

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end valid_three_digit_numbers_count_l130_130908


namespace water_volume_percentage_l130_130059

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l130_130059


namespace simplest_common_denominator_l130_130004

theorem simplest_common_denominator (x y : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (d : ℤ), d = x^2 * y^2 ∧ ∀ (a b : ℤ), 
    (∃ (k : ℤ), a = k * (x^2 * y)) ∧ (∃ (m : ℤ), b = m * (x * y^2)) → d = lcm a b :=
by
  sorry

end simplest_common_denominator_l130_130004


namespace inequality_proof_l130_130581

theorem inequality_proof (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c :=
by
  -- Proof will be provided here
  sorry

end inequality_proof_l130_130581


namespace find_C_prove_relation_l130_130793

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l130_130793


namespace sqrt_divisors_l130_130526

theorem sqrt_divisors (n : ℕ) (h1 : n = p ^ 4) (hp : Prime p) : Nat.divisors (Nat.sqrt n) = {1, p, p^2} := by
  sorry

end sqrt_divisors_l130_130526


namespace optimal_tower_configuration_l130_130609

theorem optimal_tower_configuration (x y : ℕ) (h : x + 2 * y = 30) :
    x * y ≤ 112 := by
  sorry

end optimal_tower_configuration_l130_130609


namespace distance_to_line_is_sqrt5_l130_130666

noncomputable def distance_from_right_focus_to_line : Real :=
  let a : ℝ := 2
  let b : ℝ := sqrt 5
  let c : ℝ := sqrt (a^2 + b^2)
  let right_focus : ℝ × ℝ := (c, 0)
  let A : ℝ := 1
  let B : ℝ := 2
  let C : ℝ := -8
  let (x0, y0) := right_focus
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

theorem distance_to_line_is_sqrt5 :
  distance_from_right_focus_to_line = sqrt 5 :=
  sorry

end distance_to_line_is_sqrt5_l130_130666


namespace gcd_lcm_product_l130_130107

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l130_130107


namespace fg_equals_gf_l130_130760

theorem fg_equals_gf (m n p q : ℝ) (h : m + q = n + p) : ∀ x : ℝ, (m * (p * x + q) + n = p * (m * x + n) + q) :=
by sorry

end fg_equals_gf_l130_130760


namespace problem_statement_l130_130388

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l130_130388


namespace f_5_5_l130_130589

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_even (x : ℝ) : f x = f (-x) := sorry

lemma f_recurrence (x : ℝ) : f (x + 2) = - (1 / f x) := sorry

lemma f_interval (x : ℝ) (h : 2 ≤ x ∧ x ≤ 3) : f x = x := sorry

theorem f_5_5 : f 5.5 = 2.5 :=
by
  sorry

end f_5_5_l130_130589


namespace calculate_expr_l130_130394

theorem calculate_expr : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  -- We are skipping the proof part here
  sorry

end calculate_expr_l130_130394


namespace length_four_implies_value_twenty_four_l130_130735

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l130_130735


namespace max_colored_nodes_without_cycle_in_convex_polygon_l130_130119

def convex_polygon (n : ℕ) : Prop := n ≥ 3

def valid_diagonals (n : ℕ) : Prop := n = 2019

def no_three_diagonals_intersect_at_single_point (x : Type*) : Prop :=
  sorry -- You can provide a formal definition here based on combinatorial geometry.

def no_loops (n : ℕ) (k : ℕ) : Prop :=
  k ≤ (n * (n - 3)) / 2 - 1

theorem max_colored_nodes_without_cycle_in_convex_polygon :
  convex_polygon 2019 →
  valid_diagonals 2019 →
  no_three_diagonals_intersect_at_single_point ℝ →
  ∃ k, k = 2035151 ∧ no_loops 2019 k := 
by
  -- The proof would be constructed here.
  sorry

end max_colored_nodes_without_cycle_in_convex_polygon_l130_130119


namespace continuous_at_x1_discontinuous_at_x2_l130_130558

-- Define the function
def f (x : ℝ) : ℝ := 4 ^ (1 / (3 - x))

-- Define the points
def x1 := 1
def x2 := 3

-- Statement to prove continuity at x1
theorem continuous_at_x1 : ContinuousAt f x1 := sorry

-- Statement to prove discontinuity at x2
theorem discontinuous_at_x2 : ¬ ContinuousAt f x2 := sorry

end continuous_at_x1_discontinuous_at_x2_l130_130558


namespace asymptotes_of_C2_l130_130889

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def C1 (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
noncomputable def C2 (x y : ℝ) : Prop := (y^2 / a^2 - x^2 / b^2 = 1)
noncomputable def ecc1 : ℝ := (Real.sqrt (a^2 - b^2)) / a
noncomputable def ecc2 : ℝ := (Real.sqrt (a^2 + b^2)) / a

theorem asymptotes_of_C2 :
  a > b → b > 0 → ecc1 * ecc2 = Real.sqrt 3 / 2 → by exact (∀ x y : ℝ, C2 x y → x = - Real.sqrt 2 * y ∨ x = Real.sqrt 2 * y) :=
sorry

end asymptotes_of_C2_l130_130889


namespace greatest_value_b_l130_130100

-- Define the polynomial and the inequality condition
def polynomial (b : ℝ) : ℝ := -b^2 + 8*b - 12
#check polynomial
-- State the main theorem with the given condition and the result
theorem greatest_value_b (b : ℝ) : -b^2 + 8*b - 12 ≥ 0 → b ≤ 6 :=
sorry

end greatest_value_b_l130_130100


namespace initial_pipes_l130_130913

variables (x : ℕ)

-- Defining the conditions
def one_pipe_time := x -- time for 1 pipe to fill the tank in hours
def eight_pipes_time := 1 / 4 -- 15 minutes = 1/4 hour

-- Proving the number of pipes
theorem initial_pipes (h1 : eight_pipes_time * 8 = one_pipe_time) : x = 2 :=
by
  sorry

end initial_pipes_l130_130913


namespace Sara_snow_volume_l130_130986

theorem Sara_snow_volume :
  let length := 30
  let width := 3
  let first_half_length := length / 2
  let second_half_length := length / 2
  let depth1 := 0.5
  let depth2 := 1.0 / 3.0
  let volume1 := first_half_length * width * depth1
  let volume2 := second_half_length * width * depth2
  volume1 + volume2 = 37.5 :=
by
  sorry

end Sara_snow_volume_l130_130986


namespace matrix_not_invertible_l130_130557

def is_not_invertible_matrix (y : ℝ) : Prop :=
  let a := 2 + y
  let b := 9
  let c := 4 - y
  let d := 10
  a * d - b * c = 0

theorem matrix_not_invertible (y : ℝ) : is_not_invertible_matrix y ↔ y = 16 / 19 :=
  sorry

end matrix_not_invertible_l130_130557


namespace problem1_solution_problem2_solution_l130_130879

-- Problem 1: 
theorem problem1_solution (x : ℝ) (h : 4 * x^2 = 9) : x = 3 / 2 ∨ x = - (3 / 2) := 
by sorry

-- Problem 2: 
theorem problem2_solution (x : ℝ) (h : (1 - 2 * x)^3 = 8) : x = - 1 / 2 := 
by sorry

end problem1_solution_problem2_solution_l130_130879


namespace no_stromino_covering_of_5x5_board_l130_130852

-- Define the conditions
def isStromino (r : ℕ) (c : ℕ) : Prop := 
  (r = 3 ∧ c = 1) ∨ (r = 1 ∧ c = 3)

def is5x5Board (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

-- The main goal is to show this proposition
theorem no_stromino_covering_of_5x5_board : 
  ∀ (board_size : ℕ × ℕ),
    is5x5Board board_size.1 board_size.2 →
    ∀ (stromino_count : ℕ),
      stromino_count = 16 →
      (∀ (stromino : ℕ × ℕ), 
        isStromino stromino.1 stromino.2 →
        ∀ (cover : ℕ), 
          3 = cover) →
      ¬(∃ (cover_fn : ℕ × ℕ → ℕ), 
          (∀ (pos : ℕ × ℕ), pos.fst < 5 ∧ pos.snd < 5 →
            cover_fn pos = 1 ∨ cover_fn pos = 2) ∧
          (∀ (i : ℕ), i < 25 → 
            ∃ (stromino_pos : ℕ × ℕ), 
              stromino_pos.fst < 5 ∧ stromino_pos.snd < 5 ∧ 
              -- Each stromino must cover exactly 3 squares, 
              -- which implies that the covering function must work appropriately.
              (cover_fn (stromino_pos.fst, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 1, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 2, stromino_pos.snd) = 3 ∨
               cover_fn (stromino_pos.fst, stromino_pos.snd + 1) +
               cover_fn (stromino_pos.fst, stromino_pos.snd + 2) = 3))) :=
by sorry

end no_stromino_covering_of_5x5_board_l130_130852


namespace four_digit_numbers_with_property_l130_130276

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l130_130276


namespace probability_of_at_least_two_but_fewer_than_five_heads_l130_130078

open Finset

def favorable_outcomes : ℕ :=
  choose 8 2 + choose 8 3 + choose 8 4

def total_outcomes : ℕ :=
  2 ^ 8

theorem probability_of_at_least_two_but_fewer_than_five_heads :
  let p := favorable_outcomes / total_outcomes in
  p = (77 : ℚ) / 128 :=
by
  sorry

end probability_of_at_least_two_but_fewer_than_five_heads_l130_130078


namespace initial_bananas_l130_130156

theorem initial_bananas (x B : ℕ) (h1 : 840 * x = B) (h2 : 420 * (x + 2) = B) : x = 2 :=
by
  sorry

end initial_bananas_l130_130156


namespace binary_arithmetic_l130_130724

-- Define the binary numbers 11010_2, 11100_2, and 100_2
def x : ℕ := 0b11010 -- base 2 number 11010 in base 10 representation
def y : ℕ := 0b11100 -- base 2 number 11100 in base 10 representation
def d : ℕ := 0b100   -- base 2 number 100 in base 10 representation

-- Define the correct answer
def correct_answer : ℕ := 0b10101101 -- base 2 number 10101101 in base 10 representation

-- The proof problem statement
theorem binary_arithmetic : (x * y) / d = correct_answer := by
  sorry

end binary_arithmetic_l130_130724


namespace chicago_bulls_heat_games_total_l130_130134

-- Statement of the problem in Lean 4
theorem chicago_bulls_heat_games_total :
  ∀ (bulls_games : ℕ) (heat_games : ℕ),
    bulls_games = 70 →
    heat_games = bulls_games + 5 →
    bulls_games + heat_games = 145 :=
by
  intros bulls_games heat_games h_bulls h_heat
  rw [h_bulls, h_heat]
  exact sorry

end chicago_bulls_heat_games_total_l130_130134


namespace question_d_l130_130427

variable {x a : ℝ}

theorem question_d (h1 : x < a) (h2 : a < 0) : x^3 > a * x ∧ a * x < 0 :=
  sorry

end question_d_l130_130427


namespace find_x_l130_130917

theorem find_x (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 :=
sorry

end find_x_l130_130917


namespace option_a_is_correct_l130_130522

theorem option_a_is_correct (a b : ℝ) :
  (a - b) * (-a - b) = b^2 - a^2 :=
sorry

end option_a_is_correct_l130_130522


namespace distance_from_focus_to_line_l130_130651

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l130_130651


namespace sarah_proof_l130_130568

-- Defining cards and conditions
inductive Card
| P : Card
| A : Card
| C5 : Card
| C4 : Card
| C7 : Card

-- Definition of vowel
def is_vowel : Card → Prop
| Card.P => false
| Card.A => true
| _ => false

-- Definition of prime numbers for the sides
def is_prime : Card → Prop
| Card.C5 => true
| Card.C4 => false
| Card.C7 => true
| _ => false

-- Tom's statement
def toms_statement (c : Card) : Prop :=
is_vowel c → is_prime c

-- Sarah shows Tom was wrong by turning over one card
theorem sarah_proof : ∃ c, toms_statement c = false ∧ c = Card.A :=
sorry

end sarah_proof_l130_130568


namespace find_three_numbers_l130_130288

theorem find_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a + b - c = 10) 
  (h3 : a - b + c = 8) : 
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := 
by 
  sorry

end find_three_numbers_l130_130288


namespace uneaten_chips_correct_l130_130405

def cookies_per_dozen : Nat := 12
def dozens : Nat := 4
def chips_per_cookie : Nat := 7

def total_cookies : Nat := dozens * cookies_per_dozen
def total_chips : Nat := total_cookies * chips_per_cookie
def eaten_cookies : Nat := total_cookies / 2
def uneaten_cookies : Nat := total_cookies - eaten_cookies

def uneaten_chips : Nat := uneaten_cookies * chips_per_cookie

theorem uneaten_chips_correct : uneaten_chips = 168 :=
by
  -- Placeholder for the proof
  sorry

end uneaten_chips_correct_l130_130405


namespace max_days_for_C_l130_130669

-- Define the durations of the processes and the total project duration
def A := 2
def B := 5
def D := 4
def T := 9

-- Define the condition to prove the maximum days required for process C
theorem max_days_for_C (x : ℕ) (h : 2 + x + 4 = 9) : x = 3 := by
  sorry

end max_days_for_C_l130_130669


namespace seating_arrangement_count_l130_130776

-- Define the conditions.
def chairs : ℕ := 7
def people : ℕ := 5
def end_chairs : ℕ := 3

-- Define the main theorem to prove the number of arrangements.
theorem seating_arrangement_count :
  (end_chairs * 2) * (6 * 5 * 4 * 3) = 2160 := by
  sorry

end seating_arrangement_count_l130_130776


namespace common_ratio_of_series_l130_130565

theorem common_ratio_of_series (a1 a2 : ℚ) (h1 : a1 = 5/6) (h2 : a2 = -4/9) :
  (a2 / a1) = -8/15 :=
by
  sorry

end common_ratio_of_series_l130_130565


namespace sum_of_v_values_is_zero_l130_130593

def v (x : ℝ) : ℝ := sorry

theorem sum_of_v_values_is_zero
  (h_odd : ∀ x : ℝ, v (-x) = -v x) :
  v (-3.14) + v (-1.57) + v (1.57) + v (3.14) = 0 :=
by
  sorry

end sum_of_v_values_is_zero_l130_130593


namespace geom_series_sum_l130_130024

theorem geom_series_sum : 
  let a₀ := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 5 in
  ∑ i in Finset.range n, a₀ * r ^ i = 341 / 1024 := 
  by
    sorry

end geom_series_sum_l130_130024


namespace tanya_time_proof_l130_130528

noncomputable def time_sakshi : ℝ := 10
noncomputable def efficiency_increase : ℝ := 1.25
noncomputable def time_tanya (time_sakshi : ℝ) (efficiency_increase : ℝ) : ℝ := time_sakshi / efficiency_increase

theorem tanya_time_proof : time_tanya time_sakshi efficiency_increase = 8 := 
by 
  sorry

end tanya_time_proof_l130_130528


namespace weights_problem_l130_130428

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

end weights_problem_l130_130428


namespace drums_needed_for_profit_l130_130617

def cost_to_enter_contest : ℝ := 10
def money_per_drum : ℝ := 0.025
def money_needed_for_profit (drums_hit : ℝ) : Prop :=
  drums_hit * money_per_drum > cost_to_enter_contest

theorem drums_needed_for_profit : ∃ D : ℝ, money_needed_for_profit D ∧ D = 400 :=
  by
  use 400
  sorry

end drums_needed_for_profit_l130_130617


namespace joyce_apples_l130_130143

theorem joyce_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 75 → 
    given_apples = 52 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 23 :=
by 
  intros initial_apples given_apples remaining_apples h_initial h_given h_remaining
  rw [h_initial, h_given] at h_remaining
  exact h_remaining

end joyce_apples_l130_130143


namespace negation_of_all_exp_monotonic_l130_130750

theorem negation_of_all_exp_monotonic :
  ¬ (∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) → (∃ g : ℝ → ℝ, ∃ x y : ℝ, x < y ∧ g x ≥ g y)) :=
sorry

end negation_of_all_exp_monotonic_l130_130750


namespace evaluate_g_at_neg2_l130_130838

def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem evaluate_g_at_neg2 : g (-2) = -16 := by
  sorry

end evaluate_g_at_neg2_l130_130838


namespace simplify_expression_l130_130488

theorem simplify_expression (a : ℝ) : a^2 * (-a)^4 = a^6 := by
  sorry

end simplify_expression_l130_130488


namespace cylindrical_plane_l130_130732

open Set

-- Define a cylindrical coordinate point (r, θ, z)
structure CylindricalCoord where
  r : ℝ
  theta : ℝ
  z : ℝ

-- Condition 1: In cylindrical coordinates, z is the height
def height_in_cylindrical := λ coords : CylindricalCoord => coords.z 

-- Condition 2: z is constant c
variable (c : ℝ)

-- The theorem to be proven
theorem cylindrical_plane (c : ℝ) :
  {p : CylindricalCoord | p.z = c} = {q : CylindricalCoord | q.z = c} :=
by
  sorry

end cylindrical_plane_l130_130732


namespace find_unknown_number_l130_130868

theorem find_unknown_number (x : ℝ) (h : (8 / 100) * x = 96) : x = 1200 :=
by
  sorry

end find_unknown_number_l130_130868


namespace equilateral_triangle_ratio_l130_130632

theorem equilateral_triangle_ratio (A B C X Y Z : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (hABC : is_equilateral_triangle A B C)
  (hX : X ∈ line_segment A B)
  (hY : Y ∈ line_segment B C)
  (hZ : Z ∈ line_segment C A)
  (hRatio : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ AX / XB = x ∧ BY / YC = x ∧ CZ / ZA = x)
  (hArea : ∃ S : ℝ, area_triangle C X A _ + area_triangle B Z _ _ + area_triangle A Y _ B = ¼ * (area_triangle A B C)) :
  ∃ (x : ℝ), x = (3 - real.sqrt 5) / 2 := sorry

end equilateral_triangle_ratio_l130_130632


namespace four_digit_numbers_property_l130_130281

theorem four_digit_numbers_property : 
  ∃ count : ℕ, count = 3 ∧ 
  (∀ N : ℕ, 
    1000 ≤ N ∧ N < 10000 → -- N is a four-digit number
    (∃ a x : ℕ, 
      a ≥ 1 ∧ a < 10 ∧
      100 ≤ x ∧ x < 1000 ∧
      N = 1000 * a + x ∧
      N = 11 * x) → count = 3) :=
sorry

end four_digit_numbers_property_l130_130281


namespace total_number_of_flowers_l130_130181

theorem total_number_of_flowers (pots : ℕ) (flowers_per_pot : ℕ) (h_pots : pots = 544) (h_flowers_per_pot : flowers_per_pot = 32) : 
  pots * flowers_per_pot = 17408 := by
  sorry

end total_number_of_flowers_l130_130181


namespace domain_of_tan_sub_pi_over_4_l130_130329

theorem domain_of_tan_sub_pi_over_4 :
  ∀ x : ℝ, (∃ k : ℤ, x = k * π + 3 * π / 4) ↔ ∃ y : ℝ, y = (x - π / 4) ∧ (∃ k : ℤ, y = (2 * k + 1) * π / 2) := 
sorry

end domain_of_tan_sub_pi_over_4_l130_130329


namespace length_four_implies_value_twenty_four_l130_130736

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l130_130736


namespace part1_part2_l130_130788

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l130_130788


namespace product_of_possible_values_l130_130305

theorem product_of_possible_values : 
  (∀ x : ℝ, |x - 5| - 4 = -1 → (x = 2 ∨ x = 8)) → (2 * 8) = 16 :=
by 
  sorry

end product_of_possible_values_l130_130305


namespace apples_in_basket_l130_130756

-- Definitions based on conditions
def total_apples : ℕ := 138
def apples_per_box : ℕ := 18

-- Problem: prove the number of apples in the basket
theorem apples_in_basket : (total_apples % apples_per_box) = 12 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end apples_in_basket_l130_130756


namespace four_digit_num_condition_l130_130283

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l130_130283


namespace simplify_and_evaluate_expr_l130_130165

theorem simplify_and_evaluate_expr (a : ℝ) (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a = 2) :
  (a - (a^2 / (a^2 - 1))) / (a^2 / (a^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expr_l130_130165


namespace gas_cost_l130_130880

theorem gas_cost (x : ℝ) (h₁ : 5 * (x / 5 - 9) = 8 * (x / 8)) : x = 120 :=
by
  sorry

end gas_cost_l130_130880


namespace sara_quarters_l130_130809

theorem sara_quarters (initial_quarters : ℕ) (additional_quarters : ℕ) (total_quarters : ℕ) 
    (h1 : initial_quarters = 21) 
    (h2 : additional_quarters = 49) 
    (h3 : total_quarters = initial_quarters + additional_quarters) : 
    total_quarters = 70 :=
sorry

end sara_quarters_l130_130809


namespace polar_to_rectangular_coordinates_l130_130865

theorem polar_to_rectangular_coordinates (r θ : ℝ) (hr : r = 5) (hθ : θ = (3 * Real.pi) / 2) :
    (r * Real.cos θ, r * Real.sin θ) = (0, -5) :=
by
  rw [hr, hθ]
  simp [Real.cos, Real.sin]
  sorry

end polar_to_rectangular_coordinates_l130_130865


namespace correct_response_percentage_l130_130818

def number_of_students : List ℕ := [300, 1100, 100, 600, 400]
def total_students : ℕ := number_of_students.sum
def correct_response_students : ℕ := number_of_students.maximum.getD 0

theorem correct_response_percentage :
  (correct_response_students * 100 / total_students) = 44 := by
  sorry

end correct_response_percentage_l130_130818


namespace smallest_integer_l130_130501

theorem smallest_integer (x : ℕ) (n : ℕ) (h_pos : 0 < x)
  (h_gcd : Nat.gcd 30 n = x + 3)
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) : n = 70 :=
begin
  sorry
end

end smallest_integer_l130_130501


namespace shaded_area_is_correct_l130_130228

noncomputable def octagon_side_length := 3
noncomputable def octagon_area := 2 * (1 + Real.sqrt 2) * octagon_side_length^2
noncomputable def semicircle_radius := octagon_side_length / 2
noncomputable def semicircle_area := (1 / 2) * Real.pi * semicircle_radius^2
noncomputable def total_semicircle_area := 8 * semicircle_area
noncomputable def shaded_region_area := octagon_area - total_semicircle_area

theorem shaded_area_is_correct : shaded_region_area = 54 + 36 * Real.sqrt 2 - 9 * Real.pi :=
by
  -- Proof goes here, but we're inserting sorry to skip it
  sorry

end shaded_area_is_correct_l130_130228


namespace max_ab_l130_130430

theorem max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 8) : 
  ab ≤ 8 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2 * b = 8 ∧ ab = 8 :=
by
  sorry

end max_ab_l130_130430


namespace eval_expression_l130_130722

theorem eval_expression :
  (2011 * (2012 * 10001) * (2013 * 100010001)) - (2013 * (2011 * 10001) * (2012 * 100010001)) =
  -2 * 2012 * 2013 * 10001 * 100010001 :=
by
  sorry

end eval_expression_l130_130722


namespace find_four_numbers_l130_130893

theorem find_four_numbers 
    (a b c d : ℕ) 
    (h1 : b - a = c - b)  -- first three numbers form an arithmetic sequence
    (h2 : d / c = c / (b - a + b))  -- last three numbers form a geometric sequence
    (h3 : a + d = 16)  -- sum of first and last numbers is 16
    (h4 : b + (12 - b) = 12)  -- sum of the two middle numbers is 12
    : (a = 15 ∧ b = 9 ∧ c = 3 ∧ d = 1) ∨ (a = 0 ∧ b = 4 ∧ c = 8 ∧ d = 16) :=
by
  -- Proof will be provided here
  sorry

end find_four_numbers_l130_130893


namespace left_handed_and_like_scifi_count_l130_130991

-- Definitions based on the problem conditions
def total_members : ℕ := 30
def left_handed_members : ℕ := 12
def like_scifi_members : ℕ := 18
def right_handed_not_like_scifi : ℕ := 4

-- Main proof statement
theorem left_handed_and_like_scifi_count :
  ∃ x : ℕ, (left_handed_members - x) + (like_scifi_members - x) + x + right_handed_not_like_scifi = total_members ∧ x = 4 :=
by
  use 4
  sorry

end left_handed_and_like_scifi_count_l130_130991


namespace number_of_integers_covered_l130_130631

-- Define the number line and the length condition
def unit_length_cm (p : ℝ) := p = 1
def length_AB_cm (length : ℝ) := length = 2009

-- Statement of the proof problem in Lean
theorem number_of_integers_covered (ab_length : ℝ) (unit_length : ℝ) 
    (h1 : unit_length_cm unit_length) (h2 : length_AB_cm ab_length) :
    ∃ n : ℕ, n = 2009 ∨ n = 2010 :=
by
  sorry

end number_of_integers_covered_l130_130631


namespace originally_anticipated_profit_margin_l130_130364

theorem originally_anticipated_profit_margin (decrease_percent increase_percent : ℝ) (original_price current_price : ℝ) (selling_price : ℝ) :
  decrease_percent = 6.4 → 
  increase_percent = 8 → 
  original_price = 1 → 
  current_price = original_price - original_price * decrease_percent / 100 → 
  selling_price = original_price * (1 + x / 100) → 
  selling_price = current_price * (1 + (x + increase_percent) / 100) →
  x = 117 :=
by
  intros h_dec_perc h_inc_perc h_org_price h_cur_price h_selling_price_orig h_selling_price_cur
  sorry

end originally_anticipated_profit_margin_l130_130364


namespace total_widgets_sold_15_days_l130_130614

def widgets_sold (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * n

theorem total_widgets_sold_15_days :
  (Finset.range 15).sum widgets_sold = 359 :=
by
  sorry

end total_widgets_sold_15_days_l130_130614


namespace basketball_team_lineups_l130_130227

theorem basketball_team_lineups (n k : ℕ) (h_n : n = 20) (h_k : k = 5) :
  (∃ lineup : ℕ, lineup = 20 * Nat.choose 19 4 ∧ lineup = 77520) :=
by
  let point_guard_choice := 20
  let remaining_choices := Nat.choose 19 4
  have : point_guard_choice * remaining_choices = 77520 := by sorry
  use point_guard_choice * remaining_choices
  split
  · 
    exact rfl
    
  ·
    assumption

end basketball_team_lineups_l130_130227


namespace problem_statement_l130_130386

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l130_130386


namespace parabola_vertex_coords_l130_130672

theorem parabola_vertex_coords (a b c : ℝ) (ha : a = 1) (hb : b = -4) (hc : c = 3) :
    ∃ x y : ℝ, x = -b / (2 * a) ∧ y = (4 * a * c - b^2) / (4 * a) ∧ x = 2 ∧ y = -1 :=
by
  sorry

end parabola_vertex_coords_l130_130672


namespace geometric_series_sum_l130_130019

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end geometric_series_sum_l130_130019


namespace smallest_number_greater_than_l130_130240

theorem smallest_number_greater_than : 
  ∀ (S : Set ℝ), S = {0.8, 0.5, 0.3} → 
  (∃ x ∈ S, x > 0.4 ∧ (∀ y ∈ S, y > 0.4 → x ≤ y)) → 
  x = 0.5 :=
by
  sorry

end smallest_number_greater_than_l130_130240


namespace lower_limit_of_range_l130_130216

theorem lower_limit_of_range (x y : ℝ) (hx1 : 3 < x) (hx2 : x < 8) (hx3 : y < x) (hx4 : x < 10) (hx5 : x = 7) : 3 < y ∧ y ≤ 7 :=
by
  sorry

end lower_limit_of_range_l130_130216


namespace sum_of_integers_l130_130177

theorem sum_of_integers (m n : ℕ) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by
  sorry

end sum_of_integers_l130_130177


namespace arrangement_of_books_l130_130547

-- Conditions: 
-- 4 copies of Introduction to Geometry
-- 5 copies of Introduction to Algebra with 
-- 2 specific copies of Introduction to Geometry must be adjacent

def num_ways_to_arrange_books : Nat :=
  let total_books := 9                        -- Total books = 4 Geometry + 5 Algebra
  let fixed_unit_slots := 8                   -- Consider the 2 specific Geometry books as a single unit
  let ways_to_arrange_slots := factorial fixed_unit_slots
  let ways_to_arrange_fixed_unit := 2         -- Internal arrangement of the unit of 2 Geometry books
  ways_to_arrange_slots * ways_to_arrange_fixed_unit / factorial 3 / factorial 5

-- The proof statement:
theorem arrangement_of_books : num_ways_to_arrange_books = 112 := sorry

end arrangement_of_books_l130_130547


namespace lisa_and_robert_total_photos_l130_130476

def claire_photos : Nat := 10
def lisa_photos (c : Nat) : Nat := 3 * c
def robert_photos (c : Nat) : Nat := c + 20

theorem lisa_and_robert_total_photos :
  let c := claire_photos
  let l := lisa_photos c
  let r := robert_photos c
  l + r = 60 :=
by
  sorry

end lisa_and_robert_total_photos_l130_130476


namespace total_strength_college_l130_130527

-- Defining the conditions
def C : ℕ := 500
def B : ℕ := 600
def Both : ℕ := 220

-- Declaring the theorem
theorem total_strength_college : (C + B - Both) = 880 :=
by
  -- The proof is not required, put sorry
  sorry

end total_strength_college_l130_130527


namespace count_special_integers_in_range_l130_130866

theorem count_special_integers_in_range :
  let is_special (n : ℕ) := (n >= 1000) ∧ (n < 3000) ∧ (n % 10 = (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10))
  (finset.filter is_special (finset.range 3000)).card = 109 :=
by
  sorry

end count_special_integers_in_range_l130_130866


namespace sin_alpha_value_l130_130744

open Real


theorem sin_alpha_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : π / 2 < β ∧ β < π)
  (h_sin_alpha_beta : sin (α + β) = 3 / 5) (h_cos_beta : cos β = -5 / 13) :
  sin α = 33 / 65 := 
by
  sorry

end sin_alpha_value_l130_130744


namespace spring_problem_l130_130560

theorem spring_problem (x y : ℝ) : 
  (∀ x, y = 0.5 * x + 12) →
  (0.5 * 3 + 12 = 13.5) ∧
  (y = 0.5 * x + 12) ∧
  (0.5 * 5.5 + 12 = 14.75) ∧
  (20 = 0.5 * 16 + 12) :=
by 
  sorry

end spring_problem_l130_130560


namespace part1_C_value_part2_triangle_equality_l130_130792

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l130_130792


namespace exists_nat_a_b_l130_130936

theorem exists_nat_a_b (n : ℕ) (hn : 0 < n) : 
∃ a b : ℕ, 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n :=
by
  -- The proof steps would be filled here.
  sorry

end exists_nat_a_b_l130_130936


namespace rectangle_maximized_area_side_length_l130_130256

theorem rectangle_maximized_area_side_length
  (x y : ℝ)
  (h_perimeter : 2 * x + 2 * y = 40)
  (h_max_area : x * y = 100) :
  x = 10 :=
by
  sorry

end rectangle_maximized_area_side_length_l130_130256


namespace smallest_positive_integer_divisible_by_14_15_16_l130_130241

theorem smallest_positive_integer_divisible_by_14_15_16 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (16 ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) → n ≤ m) :=
  ∃ n : ℕ, n = 1680 ∧ ∀ m : ℕ, (14 ∣ m) ∧ (15 ∣ m) ∧ (16 ∣ m) ∧ m > 0 → n ≤ m

end smallest_positive_integer_divisible_by_14_15_16_l130_130241


namespace donuts_count_is_correct_l130_130710

-- Define the initial number of donuts
def initial_donuts : ℕ := 50

-- Define the number of donuts Bill eats
def eaten_by_bill : ℕ := 2

-- Define the number of donuts taken by the secretary
def taken_by_secretary : ℕ := 4

-- Calculate the remaining donuts after Bill and the secretary take their portions
def remaining_after_bill_and_secretary : ℕ := initial_donuts - eaten_by_bill - taken_by_secretary

-- Define the number of donuts stolen by coworkers (half of the remaining donuts)
def stolen_by_coworkers : ℕ := remaining_after_bill_and_secretary / 2

-- Define the number of donuts left for the meeting
def donuts_left_for_meeting : ℕ := remaining_after_bill_and_secretary - stolen_by_coworkers

-- The theorem to prove
theorem donuts_count_is_correct : donuts_left_for_meeting = 22 :=
by
  sorry

end donuts_count_is_correct_l130_130710


namespace division_result_l130_130680

theorem division_result : (8900 / 6) / 4 = 370.8333 :=
by sorry

end division_result_l130_130680


namespace point_on_y_axis_l130_130303

theorem point_on_y_axis (a : ℝ) 
  (h : (a - 2) = 0) : a = 2 := 
  by 
    sorry

end point_on_y_axis_l130_130303


namespace floor_pi_plus_four_l130_130409

theorem floor_pi_plus_four : Int.floor (Real.pi + 4) = 7 := by
  sorry

end floor_pi_plus_four_l130_130409


namespace infinite_primes_dividing_S_l130_130145

noncomputable def infinite_set_of_pos_integers (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S) ∧ ∀ n : ℕ, n ∈ S → n > 0

def set_of_sums (S : Set ℕ) : Set ℕ :=
  {t | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ t = x + y}

noncomputable def finitely_many_primes_condition (S : Set ℕ) (T : Set ℕ) : Prop :=
  {p : ℕ | Prime p ∧ p % 4 = 1 ∧ (∃ t ∈ T, p ∣ t)}.Finite

theorem infinite_primes_dividing_S (S : Set ℕ) (T := set_of_sums S)
  (hS : infinite_set_of_pos_integers S)
  (hT : finitely_many_primes_condition S T) :
  {p : ℕ | Prime p ∧ ∃ s ∈ S, p ∣ s}.Infinite := 
sorry

end infinite_primes_dividing_S_l130_130145


namespace initial_amount_spent_l130_130200

theorem initial_amount_spent
    (X : ℕ) -- initial amount of money to spend
    (sets_purchased : ℕ := 250) -- total sets purchased
    (sets_cost_20 : ℕ := 178) -- sets that cost $20 each
    (price_per_set : ℕ := 20) -- price of each set that cost $20
    (remaining_sets : ℕ := sets_purchased - sets_cost_20) -- remaining sets
    (spent_all : (X = sets_cost_20 * price_per_set + remaining_sets * 0)) -- spent all money, remaining sets assumed free to simplify as the exact price is not given or necessary
    : X = 3560 :=
    by
    sorry

end initial_amount_spent_l130_130200


namespace percent_filled_cone_l130_130071

noncomputable def volume_of_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r ^ 2 * h

noncomputable def volume_of_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r) ^ 2 * (2 / 3 * h)

theorem percent_filled_cone (r h : ℝ) :
  let V := volume_of_cone r h in
  let V_water := volume_of_water_filled_cone r h in
  (V_water / V) * 100 = 29.6296 := by
  sorry

end percent_filled_cone_l130_130071


namespace min_height_required_kingda_ka_l130_130626

-- Definitions of the given conditions
def brother_height : ℕ := 180
def mary_relative_height : ℚ := 2 / 3
def growth_needed : ℕ := 20

-- Definition and statement of the problem
def marys_height : ℚ := mary_relative_height * brother_height
def minimum_height_required : ℚ := marys_height + growth_needed

theorem min_height_required_kingda_ka :
  minimum_height_required = 140 := by
  sorry

end min_height_required_kingda_ka_l130_130626


namespace volume_filled_cone_l130_130068

theorem volume_filled_cone (r h : ℝ) : (2/3 : ℝ) * h > 0 → (r > 0) → 
  let V := (1/3) * π * r^2 * h,
      V' := (1/3) * π * ((2/3) * r)^2 * ((2/3) * h) in
  (V' / V : ℝ) = 0.296296 : ℝ := 
by
  intros h_pos r_pos
  let V := (1/3 : ℝ) * π * r^2 * h
  let V' := (1/3 : ℝ) * π * ((2/3 : ℝ) * r)^2 * ((2/3 : ℝ) * h)
  change (V' / V) = (8 / 27 : ℝ)
  sorry

end volume_filled_cone_l130_130068


namespace problem_statement_l130_130389

theorem problem_statement : 4 * 6 * 8 + 24 / 4 = 198 :=
by
  have h1 : 4 * 6 * 8 = 192 := by norm_num
  have h2 : 24 / 4 = 6 := by norm_num
  calc
    4 * 6 * 8 + 24 / 4 = 192 + 6        : by rw [h1, h2]
                    ... = 198           : by norm_num

end problem_statement_l130_130389


namespace smallest_multiple_14_15_16_l130_130242

theorem smallest_multiple_14_15_16 : 
  Nat.lcm (Nat.lcm 14 15) 16 = 1680 := by
  sorry

end smallest_multiple_14_15_16_l130_130242


namespace picture_area_l130_130909

theorem picture_area (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (3 * x + 4) * (y + 3) - x * y = 54 → x * y = 6 :=
by
  intros h
  sorry

end picture_area_l130_130909


namespace probability_0_to_1_l130_130255

variable {μ δ : ℝ}
variable (ξ : ℝ → ℝ) -- representing ξ as a function which we'd interpret as the random variable

noncomputable def normal_distribution (x : ℝ) := (1 / (δ * sqrt (2 * π))) * exp (- ((x - μ)^2) / (2 * δ^2))

theorem probability_0_to_1 :
  (∃ μ δ, ∀ x, ξ x = normal_distribution x) → (P(ξ < 1) = 0.5) → (P(ξ > 2) = 0.4) → P (0 < ξ < 1) = 0.1 :=
by
sorry

end probability_0_to_1_l130_130255


namespace percent_volume_filled_with_water_l130_130066

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_water_filled_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * (2 / 3 * r)^2 * (2 / 3 * h)

theorem percent_volume_filled_with_water (r h : ℝ) :
  volume_water_filled_cone r h / volume_cone r h = 8 / 27 :=
by
  -- the proof would go here
  sorry

end percent_volume_filled_with_water_l130_130066


namespace solution_set_for_inequality_l130_130824

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

end solution_set_for_inequality_l130_130824


namespace ratio_of_full_boxes_l130_130709

theorem ratio_of_full_boxes 
  (F H : ℕ)
  (boxes_count_eq : F + H = 20)
  (parsnips_count_eq : 20 * F + 10 * H = 350) :
  F / (F + H) = 3 / 4 := 
by
  -- proof will be placed here
  sorry

end ratio_of_full_boxes_l130_130709


namespace multiple_of_totient_l130_130930

theorem multiple_of_totient (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a : ℕ), ∀ (i : ℕ), 0 ≤ i ∧ i ≤ n → m ∣ Nat.totient (a + i) :=
by
sorry

end multiple_of_totient_l130_130930


namespace CodgerNeedsTenPairs_l130_130556

def CodgerHasThreeFeet : Prop := true

def ShoesSoldInPairs : Prop := true

def ShoesSoldInEvenNumberedPairs : Prop := true

def CodgerOwnsOneThreePieceSet : Prop := true

-- Main theorem stating Codger needs 10 pairs of shoes to have 7 complete 3-piece sets
theorem CodgerNeedsTenPairs (h1 : CodgerHasThreeFeet) (h2 : ShoesSoldInPairs)
  (h3 : ShoesSoldInEvenNumberedPairs) (h4 : CodgerOwnsOneThreePieceSet) : 
  ∃ pairsToBuy : ℕ, pairsToBuy = 10 := 
by {
  -- We have to prove codger needs 10 pairs of shoes to have 7 complete 3-piece sets
  sorry
}

end CodgerNeedsTenPairs_l130_130556


namespace find_number_l130_130872

theorem find_number :
  ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x ∧ x ≈ 24 :=
by
  have h : ∃ (x : ℝ), 0.6667 * x - 10 = 0.25 * x := sorry
  cases h with x hx
  use x
  split
  · exact hx
  · linarith

end find_number_l130_130872


namespace angles_set_equality_solution_l130_130095

theorem angles_set_equality_solution (α : ℝ) :
  ({Real.sin α, Real.sin (2 * α), Real.sin (3 * α)} = {Real.cos α, Real.cos (2 * α), Real.cos (3 * α)}) ↔ 
  (∃ (k : ℤ), 0 ≤ k ∧ k ≤ 7 ∧ α = (k * Real.pi / 2) + (Real.pi / 8)) := 
by
  sorry

end angles_set_equality_solution_l130_130095


namespace uneaten_chips_l130_130402

theorem uneaten_chips :
  ∀ (chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips : ℕ),
    (chips_per_cookie = 7) →
    (cookies_total = 12 * 4) →
    (half_cookies = cookies_total / 2) →
    (uneaten_cookies = cookies_total - half_cookies) →
    (uneaten_chips = uneaten_cookies * chips_per_cookie) →
    uneaten_chips = 168 :=
by
  intros chips_per_cookie cookies_total half_cookies uneaten_cookies uneaten_chips
  intros chips_per_cookie_eq cookies_total_eq half_cookies_eq uneaten_cookies_eq uneaten_chips_eq
  rw [chips_per_cookie_eq, cookies_total_eq, half_cookies_eq, uneaten_cookies_eq, uneaten_chips_eq]
  norm_num
  sorry

end uneaten_chips_l130_130402


namespace no_geometric_progression_11_12_13_l130_130862

theorem no_geometric_progression_11_12_13 :
  ∀ (b1 : ℝ) (q : ℝ) (k l n : ℕ), 
  (b1 * q ^ (k - 1) = 11) → 
  (b1 * q ^ (l - 1) = 12) → 
  (b1 * q ^ (n - 1) = 13) → 
  False :=
by
  intros b1 q k l n hk hl hn
  sorry

end no_geometric_progression_11_12_13_l130_130862


namespace ice_cream_ratio_l130_130801

-- Definitions based on the conditions
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := oli_scoops + 4

-- Statement to prove the ratio
theorem ice_cream_ratio :
  victoria_scoops / oli_scoops = 2 :=
by
  -- The exact proof strategy here is omitted with 'sorry'
  sorry

end ice_cream_ratio_l130_130801


namespace units_digit_base8_l130_130155

theorem units_digit_base8 (a b : ℕ) (h_a : a = 123) (h_b : b = 57) :
  let product := a * b
  let units_digit := product % 8
  units_digit = 7 := by
  sorry

end units_digit_base8_l130_130155


namespace curve_symmetry_l130_130611

theorem curve_symmetry :
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧
  ∀ (ρ θ' : ℝ), ρ = 4 * Real.sin (θ' - Real.pi / 3) ↔ ρ = 4 * Real.sin ((θ - θ') - Real.pi / 3) :=
sorry

end curve_symmetry_l130_130611


namespace max_table_sum_l130_130461

-- Define the conditions
def grid (x y : ℕ) := (x < 4) ∧ (y < 8)
def is_corner (x y : ℕ) := (x = 0 ∧ y = 0) ∨ (x = 0 ∧ y = 7) ∨ (x = 3 ∧ y = 0) ∨ (x = 3 ∧ y = 7)
def not_corner (x y : ℕ) := grid x y ∧ ¬is_corner x y
def cross_cells (x y : ℕ) := [(x, y), (x-1, y), (x+1, y), (x, y-1), (x, y+1)]

-- Formalize the theorem
theorem max_table_sum :
  ∀ (f : ℕ → ℕ → ℝ), (∀ x y, not_corner x y → ∑ c in (cross_cells x y), f (prod.fst c) (prod.snd c) ≤ 8) →
  ∑ x y, if not_corner x y then f x y else 0 = 96 :=
begin
  sorry
end

end max_table_sum_l130_130461


namespace find_C_l130_130201

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 :=
by
  sorry

end find_C_l130_130201


namespace gcd_lcm_product_l130_130108

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end gcd_lcm_product_l130_130108


namespace bacteria_count_correct_l130_130975

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 800

-- Define the doubling time in hours
def doubling_time : ℕ := 3

-- Define the function that calculates the number of bacteria after t hours
noncomputable def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * 2 ^ (t / doubling_time)

-- Define the target number of bacteria
def target_bacteria : ℕ := 51200

-- Define the specific time we want to prove the bacteria count equals the target
def specific_time : ℕ := 18

-- Prove that after 18 hours, there will be exactly 51,200 bacteria
theorem bacteria_count_correct : bacteria_after specific_time = target_bacteria :=
  sorry

end bacteria_count_correct_l130_130975


namespace xiao_wang_program_output_l130_130038

theorem xiao_wang_program_output (n : ℕ) (h : n = 8) : (n : ℝ) / (n^2 + 1) = 8 / 65 := by
  sorry

end xiao_wang_program_output_l130_130038


namespace calculate_ladder_cost_l130_130467

theorem calculate_ladder_cost (ladders1 ladders2 rungs1 rungs2 rung_cost : ℕ) : 
  (ladders1 = 10) → 
  (rungs1 = 50) → 
  (ladders2 = 20) → 
  (rungs2 = 60) → 
  (rung_cost = 2) → 
  (ladders1 * rungs1 + ladders2 * rungs2) * rung_cost = 3400 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3, h4, h5] 
  norm_num 
  sorry

end calculate_ladder_cost_l130_130467


namespace curve_properties_l130_130350

noncomputable def curve (x y : ℝ) : Prop := Real.sqrt x + Real.sqrt y = 1

theorem curve_properties :
  curve 1 0 ∧ curve 0 1 ∧ curve (1/4) (1/4) ∧ 
  (∀ p : ℝ × ℝ, curve p.1 p.2 → curve p.2 p.1) :=
by
  sorry

end curve_properties_l130_130350


namespace train_pass_time_l130_130843

-- Definitions based on the conditions
def train_length : ℕ := 360   -- Length of the train in meters
def platform_length : ℕ := 190 -- Length of the platform in meters
def speed_kmh : ℕ := 45       -- Speed of the train in km/h
def speed_ms : ℚ := speed_kmh * (1000 / 3600) -- Speed of the train in m/s

-- Total distance to be covered
def total_distance : ℕ := train_length + platform_length 

-- Time taken to pass the platform
def time_to_pass_platform : ℚ := total_distance / speed_ms

-- Proof that the time taken is 44 seconds
theorem train_pass_time : time_to_pass_platform = 44 := 
by 
  -- this is where the detailed proof would go
  sorry  

end train_pass_time_l130_130843


namespace mixed_operation_with_rationals_l130_130396

theorem mixed_operation_with_rationals :
  (- (2 / 21)) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := 
by 
  sorry

end mixed_operation_with_rationals_l130_130396


namespace dacid_average_l130_130230

noncomputable def average (a b : ℕ) : ℚ :=
(a + b) / 2

noncomputable def overall_average (a b c d e : ℕ) : ℚ :=
(a + b + c + d + e) / 5

theorem dacid_average :
  ∀ (english mathematics physics chemistry biology : ℕ),
  english = 86 →
  mathematics = 89 →
  physics = 82 →
  chemistry = 87 →
  biology = 81 →
  (average english mathematics < 90) ∧
  (average english physics < 90) ∧
  (average english chemistry < 90) ∧
  (average english biology < 90) ∧
  (average mathematics physics < 90) ∧
  (average mathematics chemistry < 90) ∧
  (average mathematics biology < 90) ∧
  (average physics chemistry < 90) ∧
  (average physics biology < 90) ∧
  (average chemistry biology < 90) ∧
  overall_average english mathematics physics chemistry biology = 85 := by
  intros english mathematics physics chemistry biology
  intros h_english h_mathematics h_physics h_chemistry h_biology
  simp [average, overall_average]
  rw [h_english, h_mathematics, h_physics, h_chemistry, h_biology]
  sorry

end dacid_average_l130_130230


namespace quotient_is_12_l130_130505

theorem quotient_is_12 (a b q : ℕ) (h1: q = a / b) (h2: q = a / 2) (h3: q = 6 * b) : q = 12 :=
by 
  sorry

end quotient_is_12_l130_130505


namespace boys_difference_twice_girls_l130_130820

theorem boys_difference_twice_girls :
  ∀ (total_students girls boys : ℕ),
  total_students = 68 →
  girls = 28 →
  boys = total_students - girls →
  2 * girls - boys = 16 :=
by
  intros total_students girls boys h1 h2 h3
  sorry

end boys_difference_twice_girls_l130_130820


namespace find_a_l130_130577

theorem find_a (a : ℝ) (h : 1 ∈ ({a + 2, (a + 1)^2, a^2 + 3a + 3} : set ℝ)) : a = 0 :=
sorry

end find_a_l130_130577


namespace rectangle_short_side_l130_130365

theorem rectangle_short_side
  (r : ℝ) (a_circle : ℝ) (a_rect : ℝ) (d : ℝ) (other_side : ℝ) :
  r = 6 →
  a_circle = Real.pi * r^2 →
  a_rect = 3 * a_circle →
  d = 2 * r →
  a_rect = d * other_side →
  other_side = 9 * Real.pi :=
by
  sorry

end rectangle_short_side_l130_130365


namespace no_real_m_for_parallel_lines_l130_130585

theorem no_real_m_for_parallel_lines : 
  ∀ (m : ℝ), ∃ (l1 l2 : ℝ × ℝ × ℝ), 
  (l1 = (2, (m + 1), 4)) ∧ (l2 = (m, 3, 4)) ∧ 
  ( ∀ (m : ℝ), -2 / (m + 1) = -m / 3 → false ) :=
by sorry

end no_real_m_for_parallel_lines_l130_130585


namespace simple_annual_interest_rate_l130_130381

-- Given definitions and conditions
def monthly_interest_payment := 225
def principal_amount := 30000
def annual_interest_payment := monthly_interest_payment * 12
def annual_interest_rate := annual_interest_payment / principal_amount

-- Theorem statement
theorem simple_annual_interest_rate :
  annual_interest_rate * 100 = 9 := by
sorry

end simple_annual_interest_rate_l130_130381


namespace danny_reaches_steve_house_in_31_minutes_l130_130719

theorem danny_reaches_steve_house_in_31_minutes:
  ∃ (t : ℝ), 2 * t - t = 15.5 * 2 ∧ t = 31 := sorry

end danny_reaches_steve_house_in_31_minutes_l130_130719


namespace fraction_problem_l130_130197

theorem fraction_problem : 
  (  (1/4 - 1/5) / (1/3 - 1/4)  ) = 3/5 :=
by
  sorry

end fraction_problem_l130_130197


namespace solve_for_M_plus_N_l130_130130

theorem solve_for_M_plus_N (M N : ℕ) (h1 : 4 * N = 588) (h2 : 4 * 63 = 7 * M) : M + N = 183 := by
  sorry

end solve_for_M_plus_N_l130_130130


namespace total_adults_across_all_three_buses_l130_130509

def total_passengers : Nat := 450
def bus_A_passengers : Nat := 120
def bus_B_passengers : Nat := 210
def bus_C_passengers : Nat := 120
def children_ratio_A : ℚ := 1/3
def children_ratio_B : ℚ := 2/5
def children_ratio_C : ℚ := 3/8

theorem total_adults_across_all_three_buses :
  let children_A := bus_A_passengers * children_ratio_A
  let children_B := bus_B_passengers * children_ratio_B
  let children_C := bus_C_passengers * children_ratio_C
  let adults_A := bus_A_passengers - children_A
  let adults_B := bus_B_passengers - children_B
  let adults_C := bus_C_passengers - children_C
  (adults_A + adults_B + adults_C) = 281 := by {
    -- The proof steps will go here
    sorry
}

end total_adults_across_all_three_buses_l130_130509


namespace avg_age_team_proof_l130_130075

-- Defining the known constants
def members : ℕ := 15
def avg_age_team : ℕ := 28
def captain_age : ℕ := avg_age_team + 4
def remaining_players : ℕ := members - 2
def avg_age_remaining : ℕ := avg_age_team - 2

-- Stating the problem to prove the average age remains 28
theorem avg_age_team_proof (W : ℕ) :
  28 = avg_age_team ∧
  members = 15 ∧
  captain_age = avg_age_team + 4 ∧
  remaining_players = members - 2 ∧
  avg_age_remaining = avg_age_team - 2 ∧
  28 * 15 = 26 * 13 + captain_age + W :=
sorry

end avg_age_team_proof_l130_130075


namespace geometric_sequence_a6_l130_130463

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 4) (h2 : a 4 = 2) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 6 = 4 :=
sorry

end geometric_sequence_a6_l130_130463


namespace gcd_lcm_problem_l130_130497

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l130_130497


namespace derivative_at_one_l130_130935

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : deriv f 1 = 2 * Real.exp 1 := by
sorry

end derivative_at_one_l130_130935


namespace c_left_days_before_completion_l130_130525

-- Definitions for the given conditions
def work_done_by_a_in_one_day := 1 / 30
def work_done_by_b_in_one_day := 1 / 30
def work_done_by_c_in_one_day := 1 / 40
def total_days := 12

-- Proof problem statement (to prove that c left 8 days before the completion)
theorem c_left_days_before_completion :
  ∃ x : ℝ, 
  (12 - x) * (7 / 60) + x * (1 / 15) = 1 → 
  x = 8 := sorry

end c_left_days_before_completion_l130_130525


namespace sum_of_midpoints_eq_15_l130_130670

theorem sum_of_midpoints_eq_15 (a b c d : ℝ) (h : a + b + c + d = 15) :
  (a + b) / 2 + (b + c) / 2 + (c + d) / 2 + (d + a) / 2 = 15 :=
by sorry

end sum_of_midpoints_eq_15_l130_130670


namespace find_perpendicular_slope_value_l130_130896

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value_l130_130896


namespace third_quadrant_condition_l130_130034

-- Define the conditions for the third quadrant
def in_third_quadrant (p: ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

-- Translate the problem statement to a Lean theorem
theorem third_quadrant_condition (a b : ℝ) (h1 : a + b < 0) (h2 : a * b > 0) : in_third_quadrant (a, b) :=
sorry

end third_quadrant_condition_l130_130034


namespace rectangle_perimeter_l130_130977

-- We first define the side lengths of the squares and their relationships
def b1 : ℕ := 3
def b2 : ℕ := 9
def b3 := b1 + b2
def b4 := 2 * b1 + b2
def b5 := 3 * b1 + 2 * b2
def b6 := 3 * b1 + 3 * b2
def b7 := 4 * b1 + 3 * b2

-- Dimensions of the rectangle
def L := 37
def W := 52

-- Theorem to prove the perimeter of the rectangle
theorem rectangle_perimeter : 2 * L + 2 * W = 178 := by
  -- Proof will be provided here
  sorry

end rectangle_perimeter_l130_130977


namespace total_supervisors_correct_l130_130502

-- Define the number of supervisors on each bus
def bus_supervisors : List ℕ := [4, 5, 3, 6, 7]

-- Define the total number of supervisors
def total_supervisors := bus_supervisors.sum

-- State the theorem to prove that the total number of supervisors is 25
theorem total_supervisors_correct : total_supervisors = 25 :=
by
  sorry -- Proof is to be completed

end total_supervisors_correct_l130_130502


namespace find_abc_l130_130147

theorem find_abc (a b c : ℕ) (h1 : c = b^2) (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 3 := 
by
  sorry

end find_abc_l130_130147


namespace wechat_balance_l130_130153

def transaction1 : ℤ := 48
def transaction2 : ℤ := -30
def transaction3 : ℤ := -50

theorem wechat_balance :
  transaction1 + transaction2 + transaction3 = -32 :=
by
  -- placeholder for proof
  sorry

end wechat_balance_l130_130153


namespace sqrt_ceil_eq_sqrt_sqrt_l130_130924

theorem sqrt_ceil_eq_sqrt_sqrt (a : ℝ) (h : a > 1) : 
  (Int.floor (Real.sqrt (Int.floor (Real.sqrt a)))) = (Int.floor (Real.sqrt (Real.sqrt a))) :=
sorry

end sqrt_ceil_eq_sqrt_sqrt_l130_130924


namespace cos_alpha_beta_value_l130_130711

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end cos_alpha_beta_value_l130_130711


namespace trajectory_midpoint_l130_130254

-- Defining the point A(-2, 0)
def A : ℝ × ℝ := (-2, 0)

-- Defining the curve equation
def curve (x y : ℝ) : Prop := 2 * y^2 = x

-- Coordinates of P based on the midpoint formula
def P (x y : ℝ) : ℝ × ℝ := (2 * x + 2, 2 * y)

-- The target trajectory equation
def trajectory_eqn (x y : ℝ) : Prop := x = 4 * y^2 - 1

-- The theorem to be proved
theorem trajectory_midpoint (x y : ℝ) :
  curve (2 * y) (2 * x + 2) → 
  trajectory_eqn x y :=
sorry

end trajectory_midpoint_l130_130254


namespace sophomores_in_program_l130_130775

theorem sophomores_in_program (total_students : ℕ) (not_sophomores_nor_juniors : ℕ) 
    (percentage_sophomores_debate : ℚ) (percentage_juniors_debate : ℚ) 
    (eq_debate_team : ℚ) (total_students := 40) 
    (not_sophomores_nor_juniors := 5) 
    (percentage_sophomores_debate := 0.20) 
    (percentage_juniors_debate := 0.25) 
    (eq_debate_team := (percentage_sophomores_debate * S = percentage_juniors_debate * J)) :
    ∀ (S J : ℚ), S + J = total_students - not_sophomores_nor_juniors → 
    (S = 5 * J / 4) → S = 175 / 9 := 
by 
  sorry

end sophomores_in_program_l130_130775


namespace distance_from_right_focus_to_line_is_sqrt5_l130_130664

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l130_130664


namespace find_triplets_l130_130564

theorem find_triplets (u v w : ℝ):
  (u + v * w = 12) ∧ 
  (v + w * u = 12) ∧ 
  (w + u * v = 12) ↔ 
  (u = 3 ∧ v = 3 ∧ w = 3) ∨ 
  (u = -4 ∧ v = -4 ∧ w = -4) ∨ 
  (u = 1 ∧ v = 1 ∧ w = 11) ∨ 
  (u = 11 ∧ v = 1 ∧ w = 1) ∨ 
  (u = 1 ∧ v = 11 ∧ w = 1) := 
sorry

end find_triplets_l130_130564


namespace area_of_given_polygon_l130_130158

def point := (ℝ × ℝ)

def vertices : List point := [(0,0), (5,0), (5,2), (3,2), (3,3), (2,3), (2,2), (0,2), (0,0)]

def polygon_area (vertices : List point) : ℝ := 
  -- Function to compute the area of the given polygon
  -- Implementation of the area computation is assumed to be correct
  sorry

theorem area_of_given_polygon : polygon_area vertices = 11 :=
sorry

end area_of_given_polygon_l130_130158


namespace prob_red_or_blue_l130_130183

open Nat

noncomputable def total_marbles : Nat := 90
noncomputable def prob_white : (ℚ) := 1 / 6
noncomputable def prob_green : (ℚ) := 1 / 5

theorem prob_red_or_blue :
  let prob_total := 1
  let prob_white_or_green := prob_white + prob_green
  let prob_red_blue := prob_total - prob_white_or_green
  prob_red_blue = 19 / 30 := by
    sorry

end prob_red_or_blue_l130_130183


namespace f_is_constant_l130_130741

noncomputable def f (x θ : ℝ) : ℝ :=
  (Real.cos (x - θ))^2 + (Real.cos x)^2 - 2 * Real.cos θ * Real.cos (x - θ) * Real.cos x

theorem f_is_constant (θ : ℝ) : ∀ x, f x θ = (Real.sin θ)^2 :=
by
  intro x
  sorry

end f_is_constant_l130_130741


namespace group_size_is_eight_l130_130173

/-- Theorem: The number of people in the group is 8 if the 
average weight increases by 6 kg when a new person replaces 
one weighing 45 kg, and the weight of the new person is 93 kg. -/
theorem group_size_is_eight
    (n : ℕ)
    (H₁ : 6 * n = 48)
    (H₂ : 93 - 45 = 48) :
    n = 8 :=
by
  sorry

end group_size_is_eight_l130_130173


namespace problem_1_problem_2_l130_130594

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := m * |x| - 2

theorem problem_1 : {x : ℝ | f x > 3} = {x : ℝ | x < -1 ∨ x > 5} :=
sorry

theorem problem_2 (m : ℝ) : (∀ x : ℝ, f x ≥ g x m) → m ≤ 1 :=
sorry

end problem_1_problem_2_l130_130594


namespace train_speed_correct_l130_130700

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ) : ℝ :=
  (train_length + bridge_length) / time_seconds

theorem train_speed_correct :
  train_speed (400 : ℝ) (300 : ℝ) (45 : ℝ) = 700 / 45 :=
by
  sorry

end train_speed_correct_l130_130700


namespace find_coins_l130_130989

-- Definitions based on conditions
structure Wallet where
  coin1 : ℕ
  coin2 : ℕ
  h_total_value : coin1 + coin2 = 15
  h_not_five : coin1 ≠ 5 ∨ coin2 ≠ 5

-- Theorem statement based on the proof problem
theorem find_coins (w : Wallet) : (w.coin1 = 5 ∧ w.coin2 = 10) ∨ (w.coin1 = 10 ∧ w.coin2 = 5) := by
  sorry

end find_coins_l130_130989


namespace geometric_sequence_product_l130_130464

theorem geometric_sequence_product (a b : ℝ) (h : 2 * b = a * 16) : a * b = 32 :=
sorry

end geometric_sequence_product_l130_130464


namespace apples_harvested_l130_130628

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end apples_harvested_l130_130628


namespace gcd_lcm_problem_l130_130496

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l130_130496


namespace find_a_l130_130944

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l130_130944


namespace cone_water_volume_percentage_l130_130058

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l130_130058


namespace inequality_abc_l130_130318

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by
  sorry

end inequality_abc_l130_130318


namespace distance_from_right_focus_to_line_l130_130648

theorem distance_from_right_focus_to_line : 
  let h := hyperbola,
      l := line,
      c := sqrt 5
  in ∃ P : ℝ × ℝ, (h P) ∧ (is_right_focus P) → (is_line l) ∧ (distance P l = c) := 
begin
  sorry
end

end distance_from_right_focus_to_line_l130_130648


namespace find_x_squared_plus_y_squared_l130_130291

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y + 7 = (x - 3)^2) (h2 : x + 7 = (y - 3)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 17 :=
by
  sorry  -- Proof to be provided

end find_x_squared_plus_y_squared_l130_130291


namespace tom_age_ratio_l130_130511

theorem tom_age_ratio (T : ℕ) (h1 : T = 3 * (3 : ℕ)) (h2 : T - 5 = 3 * ((T / 3) - 10)) : T / 5 = 9 := 
by
  sorry

end tom_age_ratio_l130_130511


namespace find_s_for_g_l130_130794

def g (x : ℝ) (s : ℝ) : ℝ := 3*x^4 - 2*x^3 + 2*x^2 + x + s

theorem find_s_for_g (s : ℝ) : g (-1) s = 0 ↔ s = -6 :=
by
  sorry

end find_s_for_g_l130_130794


namespace find_a_find_b_l130_130903

section Problem1

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^4 - 4 * x^3 + a * x^2 - 1

-- Condition 1: f is monotonically increasing on [0, 1]
def f_increasing_on_interval_01 (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≤ y → f x a ≤ f y a

-- Condition 2: f is monotonically decreasing on [1, 2]
def f_decreasing_on_interval_12 (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x ≤ y → f y a ≤ f x a

-- Proof of a part
theorem find_a : ∃ a, f_increasing_on_interval_01 a ∧ f_decreasing_on_interval_12 a ∧ a = 4 :=
  sorry

end Problem1

section Problem2

noncomputable def f_fixed (x : ℝ) : ℝ := x^4 - 4 * x^3 + 4 * x^2 - 1
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x^2 - 1

-- Condition for intersections
def intersect_at_two_points (b : ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f_fixed x1 = g x1 b ∧ f_fixed x2 = g x2 b

-- Proof of b part
theorem find_b : ∃ b, intersect_at_two_points b ∧ (b = 0 ∨ b = 4) :=
  sorry

end Problem2

end find_a_find_b_l130_130903


namespace nat_numbers_eq_floor_condition_l130_130563

theorem nat_numbers_eq_floor_condition (a b : ℕ):
  (⌊(a ^ 2 : ℚ) / b⌋₊ + ⌊(b ^ 2 : ℚ) / a⌋₊ = ⌊((a ^ 2 + b ^ 2) : ℚ) / (a * b)⌋₊ + a * b) →
  (b = a ^ 2 + 1) ∨ (a = b ^ 2 + 1) :=
by
  sorry

end nat_numbers_eq_floor_condition_l130_130563


namespace evaluate_expression_l130_130234

theorem evaluate_expression (a : ℤ) : ((a + 10) - a + 3) * ((a + 10) - a - 2) = 104 := by
  sorry

end evaluate_expression_l130_130234


namespace motorcyclist_average_speed_l130_130851

theorem motorcyclist_average_speed :
  ∀ (t : ℝ), 120 / t = 60 * 3 → 
  3 * t / 4 = 45 :=
by
  sorry

end motorcyclist_average_speed_l130_130851


namespace probability_three_diff_suits_l130_130637

theorem probability_three_diff_suits :
  let num_cards := 52
  let num_suits := 4
  let cards_per_suit := num_cards / num_suits
  -- Number of ways to choose 3 different cards from a deck
  let total_ways := finset.card (finset.comb_finset (finset.Ico 0 num_cards) 3)
  -- Number of ways to choose 1 card of each suit
  let ways_diff_suits := finset.card 
    (finset.product 
      (finset.product 
        (finset.Ico 0 cards_per_suit) 
        (finset.Ico cards_per_suit (2 * cards_per_suit))) 
      (finset.Ico (2 * cards_per_suit) num_cards))
  -- The probability is the ratio of these two numbers
  ways_diff_suits / total_ways = 169 / 425 := 
sorry

end probability_three_diff_suits_l130_130637


namespace initial_money_l130_130411

theorem initial_money (x : ℝ) (cupcake_cost total_cookie_cost total_cost money_left : ℝ) 
  (h1 : cupcake_cost = 10 * 1.5) 
  (h2 : total_cookie_cost = 5 * 3)
  (h3 : total_cost = cupcake_cost + total_cookie_cost)
  (h4 : money_left = 30)
  (h5 : 3 * x = total_cost + money_left) 
  : x = 20 := 
sorry

end initial_money_l130_130411


namespace driving_distance_l130_130478

def miles_per_gallon : ℕ := 20
def gallons_of_gas : ℕ := 5

theorem driving_distance :
  miles_per_gallon * gallons_of_gas = 100 :=
  sorry

end driving_distance_l130_130478


namespace ellipse_eccentricity_a_l130_130947

theorem ellipse_eccentricity_a (a : ℝ) (e1 e2 : ℝ)
  (h1 : a > 1)
  (h2 : e2 = sqrt 3 * e1)
  (h3 : e1 = 1 / 2)
  (h4 : ∀ x y : ℝ, x^2 / a^2 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / a)^2 + (p.2)^2 = 1))
  (h5 : ∀ x y : ℝ, x^2 / 4 + y^2 = 1 ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, (p.1 / 2)^2 + (p.2)^2 = 1)) :
  a = 2 * sqrt 3 / 3 :=
sorry

end ellipse_eccentricity_a_l130_130947


namespace ladder_cost_l130_130468

theorem ladder_cost (ladders1 ladders2 rung_count1 rung_count2 cost_per_rung : ℕ)
  (h1 : ladders1 = 10) (h2 : ladders2 = 20) (h3 : rung_count1 = 50) (h4 : rung_count2 = 60) (h5 : cost_per_rung = 2) :
  (ladders1 * rung_count1 + ladders2 * rung_count2) * cost_per_rung = 3400 :=
by 
  sorry

end ladder_cost_l130_130468


namespace xyz_value_l130_130742

theorem xyz_value (x y z : ℝ)
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
    x * y * z = 16 / 3 := by
    sorry

end xyz_value_l130_130742


namespace sum_of_reciprocals_l130_130671

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 1 / x + 1 / y = 8 / 75 :=
by
  sorry

end sum_of_reciprocals_l130_130671


namespace canonical_equations_of_line_intersection_l130_130035

theorem canonical_equations_of_line_intersection
  (x y z : ℝ)
  (h1 : 2 * x - 3 * y + z + 6 = 0)
  (h2 : x - 3 * y - 2 * z + 3 = 0) :
  (∃ (m n p x0 y0 z0 : ℝ), 
  m * (x + 3) = n * y ∧ n * y = p * z ∧ 
  m = 9 ∧ n = 5 ∧ p = -3 ∧ 
  x0 = -3 ∧ y0 = 0 ∧ z0 = 0) :=
sorry

end canonical_equations_of_line_intersection_l130_130035


namespace smallest_n_with_square_ending_in_2016_l130_130613

theorem smallest_n_with_square_ending_in_2016 : 
  ∃ n : ℕ, (n^2 % 10000 = 2016) ∧ (n = 996) :=
by
  sorry

end smallest_n_with_square_ending_in_2016_l130_130613


namespace father_current_age_l130_130538

namespace AgeProof

def daughter_age : ℕ := 10
def years_future : ℕ := 20

def father_age (D : ℕ) : ℕ := 4 * D

theorem father_current_age :
  ∀ D : ℕ, ∀ F : ℕ, (F = father_age D) →
  (F + years_future = 2 * (D + years_future)) →
  D = daughter_age →
  F = 40 :=
by
  intro D F h1 h2 h3
  sorry

end AgeProof

end father_current_age_l130_130538


namespace observations_count_correct_l130_130008

noncomputable def corrected_observations (n : ℕ) : ℕ :=
  if 36 * n + 22 = 36.5 * n then n else 0

theorem observations_count_correct :
  ∃ n : ℕ, 36 * n + 22 = 36.5 * n ∧ corrected_observations n = 44 :=
by {
  sorry
}

end observations_count_correct_l130_130008


namespace absolute_value_positive_l130_130030

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l130_130030


namespace max_angle_B_l130_130914

-- We define the necessary terms to state our problem
variables {A B C : Real} -- The angles of triangle ABC
variables {cot_A cot_B cot_C : Real} -- The cotangents of angles A, B, and C

-- The main theorem stating that given the conditions the maximum value of angle B is pi/3
theorem max_angle_B (h1 : cot_B = (cot_A + cot_C) / 2) (h2 : A + B + C = Real.pi) :
  B ≤ Real.pi / 3 := by
  sorry

end max_angle_B_l130_130914


namespace smallest_integer_y_solution_l130_130518

theorem smallest_integer_y_solution :
  ∃ y : ℤ, (∀ z : ℤ, (z / 4 + 3 / 7 > 9 / 4) → (z ≥ y)) ∧ (y = 8) := 
by
  sorry

end smallest_integer_y_solution_l130_130518


namespace find_x_l130_130171

theorem find_x (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 7 * x^2 + 14 * x * y = 2 * x^3 + 4 * x^2 * y + y^3) : 
  x = 7 :=
sorry

end find_x_l130_130171


namespace max_value_of_expression_l130_130795

theorem max_value_of_expression :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 → 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 :=
by sorry

end max_value_of_expression_l130_130795


namespace find_number_chosen_l130_130683

theorem find_number_chosen (x : ℤ) (h : 4 * x - 138 = 102) : x = 60 := sorry

end find_number_chosen_l130_130683


namespace oreo_shop_l130_130855

theorem oreo_shop (alpha_oreos alpha_milks beta_oreos : ℕ) (h1 : alpha_oreos = 6) (h2 : alpha_milks = 4) (h3 : beta_oreos = 6) :
  let total_ways :=
    (Nat.choose (alpha_oreos + alpha_milks) 3) +
    (Nat.choose (alpha_oreos + alpha_milks) 2) * beta_oreos +
    (Nat.choose (alpha_oreos + alpha_milks) 1) * (Nat.choose beta_oreos 2 + beta_oreos) +
    (Nat.choose beta_oreos 3 + beta_oreos * (beta_oreos - 1) + beta_oreos) in
  total_ways = 656 :=
by
  intro total_ways
  sorry

end oreo_shop_l130_130855


namespace leo_current_weight_l130_130911

theorem leo_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 140) : 
  L = 80 :=
by 
  sorry

end leo_current_weight_l130_130911


namespace common_ratio_of_geometric_sequence_l130_130985

variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b_n (n + 1) = b_n n * r

def arithmetic_to_geometric (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  b_n 0 = a_n 2 ∧ b_n 1 = a_n 3 ∧ b_n 2 = a_n 7

-- Mathematical Proof Problem
theorem common_ratio_of_geometric_sequence :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), d ≠ 0 →
  is_arithmetic_sequence a_n d →
  (∃ (b_n : ℕ → ℝ) (r : ℝ), arithmetic_to_geometric a_n b_n ∧ is_geometric_sequence b_n r) →
  ∃ r, r = 4 :=
sorry

end common_ratio_of_geometric_sequence_l130_130985


namespace gymnast_score_difference_l130_130418

theorem gymnast_score_difference 
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x2 + x3 + x4 + x5 = 36)
  (h2 : x1 + x2 + x3 + x4 = 36.8) :
  x1 - x5 = 0.8 :=
by sorry

end gymnast_score_difference_l130_130418


namespace integral_exp_neg_l130_130415

theorem integral_exp_neg : ∫ x in (Set.Ioi 0), Real.exp (-x) = 1 := sorry

end integral_exp_neg_l130_130415


namespace arithmetic_sequence_sum_l130_130257

variable {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d

-- Condition 2: Given property
def property (a : ℕ → ℝ) : Prop :=
a 7 + a 13 = 20

theorem arithmetic_sequence_sum (h_seq : is_arithmetic_sequence a) (h_prop : property a) :
  a 9 + a 10 + a 11 = 30 := 
sorry

end arithmetic_sequence_sum_l130_130257


namespace four_digit_numbers_with_property_l130_130275

theorem four_digit_numbers_with_property :
  ∃ N : ℕ, ∃ a : ℕ, N = 1000 * a + (N / 11) ∧ 1000 ≤ N ∧ N < 10000 ∧ 1 ≤ a ∧ a < 10 ∧ Nat.gcd (N - 1000 * a) 1000 = 1 := by
  sorry

end four_digit_numbers_with_property_l130_130275


namespace solution_to_problem_l130_130123

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end solution_to_problem_l130_130123


namespace middle_angle_of_triangle_l130_130689

theorem middle_angle_of_triangle (α β γ : ℝ) 
  (h1 : 0 < β) (h2 : β < 90) 
  (h3 : α ≤ β) (h4 : β ≤ γ) 
  (h5 : α + β + γ = 180) :
  True :=
by
  -- Proof would go here
  sorry

end middle_angle_of_triangle_l130_130689


namespace obtain_26_kg_of_sand_l130_130046

theorem obtain_26_kg_of_sand :
  ∃ (x y : ℕ), (37 - x = x + 3) ∧ (20 - y = y + 2) ∧ (x + y = 26) := by
  sorry

end obtain_26_kg_of_sand_l130_130046


namespace ellipse_semimajor_axis_value_l130_130956

theorem ellipse_semimajor_axis_value (a b c e1 e2 : ℝ) (h1 : a > 1)
  (h2 : ∀ x y : ℝ, (x^2 / 4) + y^2 = 1 → e2 = Real.sqrt 3 * e1)
  (h3 : ∀ x y : ℝ, (x^2 / a^2) + y^2 = 1)
  (h4 : e2 = Real.sqrt 3 * e1) :
  a = 2 * Real.sqrt 3 / 3 :=
by sorry

end ellipse_semimajor_axis_value_l130_130956


namespace overall_average_is_63_point_4_l130_130767

theorem overall_average_is_63_point_4 : 
  ∃ (n total_marks : ℕ) (avg_marks : ℚ), 
  n = 50 ∧ 
  (∃ (marks_group1 marks_group2 marks_group3 marks_remaining : ℕ), 
    marks_group1 = 6 * 95 ∧
    marks_group2 = 4 * 0 ∧
    marks_group3 = 10 * 80 ∧
    marks_remaining = (n - 20) * 60 ∧
    total_marks = marks_group1 + marks_group2 + marks_group3 + marks_remaining) ∧ 
  avg_marks = total_marks / n ∧ 
  avg_marks = 63.4 := 
by 
  sorry

end overall_average_is_63_point_4_l130_130767


namespace smallest_possible_value_of_d_l130_130316

theorem smallest_possible_value_of_d (c d : ℝ) (hc : 1 < c) (hd : c < d)
  (h_triangle1 : ¬(1 + c > d ∧ c + d > 1 ∧ 1 + d > c))
  (h_triangle2 : ¬(1 / c + 1 / d > 1 ∧ 1 / d + 1 > 1 / c ∧ 1 / c + 1 > 1 / d)) :
  d = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_d_l130_130316


namespace area_of_region_l130_130325

theorem area_of_region :
  let f := λ x : ℝ, 2 * x + 3
  let g := λ x : ℝ, x ^ 2
  let a := -1
  let b := 3
  (∫ x in a..b, f x - g x) = 32 / 3 := sorry

end area_of_region_l130_130325


namespace power_function_inequality_l130_130905

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem power_function_inequality (x : ℝ) (a : ℝ) : (x > 1) → (f x a < x) ↔ (a < 1) :=
by
  sorry

end power_function_inequality_l130_130905


namespace find_cost_price_of_article_l130_130215

theorem find_cost_price_of_article 
  (C : ℝ) 
  (h1 : 1.05 * C - 2 = 1.045 * C) 
  (h2 : 0.005 * C = 2) 
: C = 400 := 
by 
  sorry

end find_cost_price_of_article_l130_130215


namespace cone_filled_with_water_to_2_3_height_l130_130061

def cone_filled_volume_ratio
  (h : ℝ) (r : ℝ) : ℝ :=
  let V := (1 / 3) * Real.pi * r^2 * h in
  let h_water := (2 / 3) * h in
  let r_water := (2 / 3) * r in
  let V_water := (1 / 3) * Real.pi * (r_water^2) * h_water in
  let ratio := V_water / V in
  Float.of_real ratio

theorem cone_filled_with_water_to_2_3_height
  (h r : ℝ) :
  cone_filled_volume_ratio h r = 0.2963 :=
  sorry

end cone_filled_with_water_to_2_3_height_l130_130061


namespace total_payroll_calc_l130_130380

theorem total_payroll_calc
  (h : ℕ := 129)          -- pay per day for heavy operators
  (l : ℕ := 82)           -- pay per day for general laborers
  (n : ℕ := 31)           -- total number of people hired
  (g : ℕ := 1)            -- number of general laborers employed
  : (h * (n - g) + l * g) = 3952 := 
by
  sorry

end total_payroll_calc_l130_130380


namespace red_balls_count_l130_130337

theorem red_balls_count (white_balls_ratio : ℕ) (red_balls_ratio : ℕ) (total_white_balls : ℕ)
  (h_ratio : white_balls_ratio = 3 ∧ red_balls_ratio = 2)
  (h_white_balls : total_white_balls = 9) :
  ∃ (total_red_balls : ℕ), total_red_balls = 6 :=
by
  sorry

end red_balls_count_l130_130337


namespace painted_cubes_count_l130_130202

/-- A theorem to prove the number of painted small cubes in a larger cube. -/
theorem painted_cubes_count (total_cubes unpainted_cubes : ℕ) (a b : ℕ) :
  total_cubes = a * a * a →
  unpainted_cubes = (a - 2) * (a - 2) * (a - 2) →
  22 = unpainted_cubes →
  64 = total_cubes →
  ∃ m, m = total_cubes - unpainted_cubes ∧ m = 42 :=
by
  sorry

end painted_cubes_count_l130_130202


namespace complement_intersection_l130_130596

noncomputable def real_universal_set : Set ℝ := Set.univ

noncomputable def set_A (x : ℝ) : Prop := x + 1 < 0
def A : Set ℝ := {x | set_A x}

noncomputable def set_B (x : ℝ) : Prop := x - 3 < 0
def B : Set ℝ := {x | set_B x}

noncomputable def complement_A : Set ℝ := {x | ¬set_A x}

noncomputable def intersection (S₁ S₂ : Set ℝ) : Set ℝ := {x | x ∈ S₁ ∧ x ∈ S₂}

theorem complement_intersection :
  intersection complement_A B = {x | -1 ≤ x ∧ x < 3} :=
sorry

end complement_intersection_l130_130596


namespace sum_of_solutions_l130_130878

theorem sum_of_solutions (x : ℝ) (h : x + 16 / x = 12) : x = 8 ∨ x = 4 → 8 + 4 = 12 := by
  sorry

end sum_of_solutions_l130_130878


namespace ramesh_paid_price_l130_130966

-- Define the variables based on the conditions
variable (labelledPrice transportCost installationCost sellingPrice paidPrice : ℝ)

-- Define the specific values given in the problem
def discount : ℝ := 0.20 
def profitRate : ℝ := 0.10 
def actualSellingPrice : ℝ := 24475
def transportAmount : ℝ := 125
def installationAmount : ℝ := 250

-- Define the conditions given in the problem as Lean definitions
def selling_price_no_discount (P : ℝ) : ℝ := (1 + profitRate) * P
def discounted_price (P : ℝ) : ℝ := P * (1 - discount)
def total_cost (P : ℝ) : ℝ :=  discounted_price P + transportAmount + installationAmount

-- The problem is to prove that the price Ramesh paid for the refrigerator is Rs. 18175
theorem ramesh_paid_price : 
  ∀ (labelledPrice : ℝ), 
  selling_price_no_discount labelledPrice = actualSellingPrice → 
  paidPrice = total_cost labelledPrice → 
  paidPrice = 18175 := 
by
  intros labelledPrice h1 h2 
  sorry

end ramesh_paid_price_l130_130966


namespace min_value_of_expression_l130_130210

variable (a b c : ℝ)
variable (h1 : a + b + c = 1)
variable (h2 : 0 < a ∧ a < 1)
variable (h3 : 0 < b ∧ b < 1)
variable (h4 : 0 < c ∧ c < 1)
variable (h5 : 3 * a + 2 * b = 2)

theorem min_value_of_expression : (2 / a + 1 / (3 * b)) ≥ 16 / 3 := 
  sorry

end min_value_of_expression_l130_130210


namespace sum_of_first_five_terms_is_31_l130_130431

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31_l130_130431


namespace solution_set_of_inequality_l130_130825

theorem solution_set_of_inequality (x : ℝ) :
  (abs x * (x - 1) ≥ 0) ↔ (x ≥ 1 ∨ x = 0) := 
by
  sorry

end solution_set_of_inequality_l130_130825


namespace golden_ratio_expression_l130_130546

variables (R : ℝ)
noncomputable def divide_segment (R : ℝ) := R^(R^(R^2 + 1/R) + 1/R) + 1/R

theorem golden_ratio_expression :
  (R = (1 / (1 + R))) →
  divide_segment R = 2 :=
by
  sorry

end golden_ratio_expression_l130_130546


namespace geometric_to_arithmetic_l130_130129

theorem geometric_to_arithmetic (a_1 a_2 a_3 b_1 b_2 b_3: ℝ) (ha: a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ b_1 > 0 ∧ b_2 > 0 ∧ b_3 > 0)
  (h_geometric_a : ∃ q : ℝ, a_2 = a_1 * q ∧ a_3 = a_1 * q^2)
  (h_geometric_b : ∃ q₁ : ℝ, b_2 = b_1 * q₁ ∧ b_3 = b_1 * q₁^2)
  (h_sum : a_1 + a_2 + a_3 = b_1 + b_2 + b_3)
  (h_arithmetic : 2 * a_2 * b_2 = a_1 * b_1 + a_3 * b_3) : 
  a_2 = b_2 :=
by
  sorry

end geometric_to_arithmetic_l130_130129


namespace infinitely_many_n_divisible_by_2018_l130_130163

theorem infinitely_many_n_divisible_by_2018 :
  ∃ᶠ n : ℕ in Filter.atTop, 2018 ∣ (1 + 2^n + 3^n + 4^n) :=
sorry

end infinitely_many_n_divisible_by_2018_l130_130163


namespace max_red_tiles_l130_130376

theorem max_red_tiles (n : ℕ) (color : ℕ → ℕ → color) :
    (∀ i j, color i j ≠ color (i + 1) j ∧ color i j ≠ color i (j + 1) ∧ color i j ≠ color (i + 1) (j + 1) 
           ∧ color i j ≠ color (i - 1) j ∧ color i j ≠ color i (j - 1) ∧ color i j ≠ color (i - 1) (j - 1)) 
    → ∃ m ≤ 2500, ∀ i j, (color i j = red ↔ i * n + j < m) :=
sorry

end max_red_tiles_l130_130376


namespace output_for_input_8_is_8_over_65_l130_130037

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l130_130037


namespace general_term_arithmetic_sequence_l130_130895

theorem general_term_arithmetic_sequence (a_n : ℕ → ℚ) (d : ℚ) (h_seq : ∀ n, a_n n = a_n 0 + n * d)
  (h_geometric : (a_n 2)^2 = a_n 1 * a_n 6)
  (h_condition : 2 * a_n 0 + a_n 1 = 1)
  (h_d_nonzero : d ≠ 0) :
  ∀ n, a_n n = (5/3) - n := 
by
  sorry

end general_term_arithmetic_sequence_l130_130895


namespace pipe_fill_rate_l130_130803

variable (R_A R_B : ℝ)

theorem pipe_fill_rate :
  R_A = 1 / 32 →
  R_A + R_B = 1 / 6.4 →
  R_B / R_A = 4 :=
by
  intros hRA hSum
  have hRA_pos : R_A ≠ 0 := by linarith
  sorry

end pipe_fill_rate_l130_130803


namespace geometric_series_sum_l130_130025

theorem geometric_series_sum
  (a r : ℚ) (n : ℕ)
  (ha : a = 1/4)
  (hr : r = 1/4)
  (hn : n = 5) :
  (∑ i in finset.range n, a * r^i) = 341 / 1024 := by
  sorry

end geometric_series_sum_l130_130025


namespace chips_left_uneaten_l130_130406

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l130_130406


namespace range_of_a_l130_130601

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → (a ≤ 1 ∨ a ≥ 3) :=
sorry

end range_of_a_l130_130601


namespace sheila_earning_per_hour_l130_130045

def sheila_hours_per_day_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_per_day_tt : ℕ := 6
def sheila_days_tt : ℕ := 2
def sheila_total_earnings : ℕ := 432

theorem sheila_earning_per_hour : (sheila_total_earnings / (sheila_hours_per_day_mwf * sheila_days_mwf + sheila_hours_per_day_tt * sheila_days_tt)) = 12 := by
  sorry

end sheila_earning_per_hour_l130_130045


namespace jillian_distance_l130_130142

theorem jillian_distance : 
  ∀ (x y z : ℝ),
  (1 / 63) * x + (1 / 77) * y + (1 / 99) * z = 11 / 3 →
  (1 / 63) * z + (1 / 77) * y + (1 / 99) * x = 13 / 3 →
  x + y + z = 308 :=
by
  sorry

end jillian_distance_l130_130142


namespace top_and_bottom_edges_same_color_l130_130401

-- Define the vertices for top and bottom pentagonal faces
inductive Vertex
| A1 | A2 | A3 | A4 | A5
| B1 | B2 | B3 | B4 | B5

-- Define the edges
inductive Edge : Type
| TopEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) : Edge
| BottomEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge
| SideEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge

-- Define colors
inductive Color
| Red | Blue

-- Define a function that assigns a color to each edge
def edgeColor : Edge → Color := sorry

-- Define a function that checks if a triangle is monochromatic
def isMonochromatic (e1 e2 e3 : Edge) : Prop :=
  edgeColor e1 = edgeColor e2 ∧ edgeColor e2 = edgeColor e3

-- Define our main theorem statement
theorem top_and_bottom_edges_same_color (h : ∀ v1 v2 v3 : Vertex, ¬ isMonochromatic (Edge.TopEdge v1 v2 sorry sorry) (Edge.SideEdge v1 v3 sorry sorry) (Edge.BottomEdge v2 v3 sorry sorry)) : 
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → edgeColor (Edge.TopEdge v1 v2 sorry sorry) = edgeColor (Edge.TopEdge Vertex.A1 Vertex.A2 sorry sorry)) ∧
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → edgeColor (Edge.BottomEdge v1 v2 sorry sorry) = edgeColor (Edge.BottomEdge Vertex.B1 Vertex.B2 sorry sorry)) :=
sorry

end top_and_bottom_edges_same_color_l130_130401


namespace find_square_length_CD_l130_130964

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x - 2

def is_midpoint (mid C D : (ℝ × ℝ)) : Prop :=
  mid.1 = (C.1 + D.1) / 2 ∧ mid.2 = (C.2 + D.2) / 2

theorem find_square_length_CD (C D : ℝ × ℝ)
  (hC : C.2 = parabola C.1)
  (hD : D.2 = parabola D.1)
  (h_mid : is_midpoint (0,0) C D) :
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
sorry

end find_square_length_CD_l130_130964


namespace time_spent_on_spelling_l130_130091

-- Define the given conditions
def total_time : Nat := 60
def math_time : Nat := 15
def reading_time : Nat := 27

-- Define the question as a Lean theorem statement
theorem time_spent_on_spelling : total_time - math_time - reading_time = 18 := sorry

end time_spent_on_spelling_l130_130091


namespace oil_vinegar_new_ratio_l130_130773

theorem oil_vinegar_new_ratio (initial_oil initial_vinegar new_vinegar : ℕ) 
    (h1 : initial_oil / initial_vinegar = 3 / 1)
    (h2 : new_vinegar = (2 * initial_vinegar)) :
    initial_oil / new_vinegar = 3 / 2 :=
by
  sorry

end oil_vinegar_new_ratio_l130_130773


namespace circumcircle_radius_l130_130296

theorem circumcircle_radius (b A S : ℝ) (h_b : b = 2) 
  (h_A : A = 120 * Real.pi / 180) (h_S : S = Real.sqrt 3) : 
  ∃ R, R = 2 := 
by
  sorry

end circumcircle_radius_l130_130296


namespace polynomial_multiplication_equiv_l130_130154

theorem polynomial_multiplication_equiv (x : ℝ) : 
  (x^4 + 50*x^2 + 625)*(x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := 
by 
  sorry

end polynomial_multiplication_equiv_l130_130154


namespace find_m_for_one_real_solution_l130_130421

variables {m x : ℝ}

-- Given condition
def equation := (x + 4) * (x + 1) = m + 2 * x

-- The statement to prove
theorem find_m_for_one_real_solution : (∃ m : ℝ, m = 7 / 4 ∧ ∀ (x : ℝ), (x + 4) * (x + 1) = m + 2 * x) :=
by
  -- The proof starts here, which we will skip with sorry
  sorry

end find_m_for_one_real_solution_l130_130421


namespace lcm_of_36_and_45_l130_130238

theorem lcm_of_36_and_45 : Nat.lcm 36 45 = 180 := by
  sorry

end lcm_of_36_and_45_l130_130238


namespace output_for_input_8_is_8_over_65_l130_130036

def function_f (n : ℕ) : ℚ := n / (n^2 + 1)

theorem output_for_input_8_is_8_over_65 : function_f 8 = 8 / 65 := by
  sorry

end output_for_input_8_is_8_over_65_l130_130036


namespace four_digit_numbers_with_property_l130_130270

theorem four_digit_numbers_with_property :
  (∃ N a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                   N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) ∧
  ∀ (N : ℕ), (∃ a x : ℕ, 1000 ≤ N ∧ N ≤ 9999 ∧ 1 ≤ a ∧ a ≤ 9 ∧
                           N = 1000 * a + x ∧ x = 100 * a ∧ N = 11 * x) →
             ∃ n : ℕ, n = 9 :=
by
  sorry

end four_digit_numbers_with_property_l130_130270


namespace deg_d_eq_6_l130_130542

theorem deg_d_eq_6
  (f d q : Polynomial ℝ)
  (r : Polynomial ℝ)
  (hf : f.degree = 15)
  (hdq : (d * q + r) = f)
  (hq : q.degree = 9)
  (hr : r.degree = 4) :
  d.degree = 6 :=
by sorry

end deg_d_eq_6_l130_130542


namespace coordinates_of_point_l130_130484

theorem coordinates_of_point (a : ℝ) (h : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end coordinates_of_point_l130_130484


namespace initial_candies_l130_130160

-- Define initial variables and conditions
variable (x : ℕ)
variable (remaining_candies_after_first_day : ℕ)
variable (remaining_candies_after_second_day : ℕ)

-- Conditions as per given problem
def condition1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
def condition2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
def final_condition : remaining_candies_after_second_day = 10 := sorry

-- Goal: Prove that initially, Liam had 52 candies
theorem initial_candies : x = 52 := by
  have h1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
  have h2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
  have h3 : remaining_candies_after_second_day = 10 := sorry
    
  -- Combine conditions to solve for x
  sorry

end initial_candies_l130_130160


namespace smallest_c_such_that_one_in_range_l130_130111

theorem smallest_c_such_that_one_in_range :
  ∃ c : ℝ, (∀ x : ℝ, ∃ y : ℝ, y =  x^2 - 2 * x + c ∧ y = 1) ∧ c = 2 :=
by
  sorry

end smallest_c_such_that_one_in_range_l130_130111


namespace average_height_of_60_students_l130_130172

theorem average_height_of_60_students :
  (35 * 22 + 25 * 18) / 60 = 20.33 := 
sorry

end average_height_of_60_students_l130_130172


namespace solve_equation_l130_130490

theorem solve_equation : ∀ (x : ℝ), 2 * (x - 1) = 2 - (5 * x - 2) → x = 6 / 7 :=
by
  sorry

end solve_equation_l130_130490


namespace triangle_obtuse_l130_130433

variable {a b c : ℝ}

theorem triangle_obtuse (h : 2 * c^2 = 2 * a^2 + 2 * b^2 + a * b) :
  ∃ C : ℝ, 0 ≤ C ∧ C ≤ π ∧ Real.cos C = -1/4 ∧ C > Real.pi / 2 :=
by
  sorry

end triangle_obtuse_l130_130433


namespace sector_area_15deg_radius_6cm_l130_130823

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_15deg_radius_6cm :
  sector_area 6 (15 * Real.pi / 180) = 3 * Real.pi / 2 := by
  sorry

end sector_area_15deg_radius_6cm_l130_130823


namespace complex_number_multiplication_l130_130258

theorem complex_number_multiplication (i : ℂ) (hi : i * i = -1) : i * (1 + i) = -1 + i :=
by sorry

end complex_number_multiplication_l130_130258


namespace find_polynomial_R_l130_130529

-- Define the polynomials S(x), Q(x), and the remainder R(x)

noncomputable def S (x : ℝ) := 7 * x ^ 31 + 3 * x ^ 13 + 10 * x ^ 11 - 5 * x ^ 9 - 10 * x ^ 7 + 5 * x ^ 5 - 2
noncomputable def Q (x : ℝ) := x ^ 4 + x ^ 3 + x ^ 2 + x + 1
noncomputable def R (x : ℝ) := 13 * x ^ 3 + 5 * x ^ 2 + 12 * x + 3

-- Statement of the proof
theorem find_polynomial_R :
  ∃ (P : ℝ → ℝ), ∀ x : ℝ, S x = P x * Q x + R x := sorry

end find_polynomial_R_l130_130529


namespace subtract_fractions_l130_130813

theorem subtract_fractions : (18 / 42 - 3 / 8) = 3 / 56 :=
by
  sorry

end subtract_fractions_l130_130813


namespace geometric_series_sum_l130_130016

theorem geometric_series_sum :
  let a := (1/4 : ℚ)
  ∧ let r := (1/4 : ℚ)
  ∧ let n := (5 : ℕ)
  → ∑ i in finset.range n, a * (r ^ i) = 341 / 1024 :=
by
  intro a r n
  apply sorry

end geometric_series_sum_l130_130016


namespace find_a_inverse_function_l130_130359

theorem find_a_inverse_function
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x y, y = f x ↔ x = a * y)
  (h2 : f 4 = 2) :
  a = 2 := 
sorry

end find_a_inverse_function_l130_130359


namespace James_comics_l130_130310

theorem James_comics (days_in_year : ℕ) (years : ℕ) (writes_every_other_day : ℕ) (no_leap_years : ℕ) 
  (h1 : days_in_year = 365) (h2 : years = 4) (h3 : writes_every_other_day = 2) : 
  (days_in_year * years) / writes_every_other_day = 730 := 
by
  sorry

end James_comics_l130_130310


namespace top_square_is_9_l130_130363

def initial_grid : List (List ℕ) := 
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

def fold_step_1 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col3 := grid.map (fun row => row.get! 2)
  let col2 := grid.map (fun row => row.get! 1)
  [[col1.get! 0, col3.get! 0, col2.get! 0],
   [col1.get! 1, col3.get! 1, col2.get! 1],
   [col1.get! 2, col3.get! 2, col2.get! 2]]

def fold_step_2 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col2 := grid.map (fun row => row.get! 1)
  let col3 := grid.map (fun row => row.get! 2)
  [[col2.get! 0, col1.get! 0, col3.get! 0],
   [col2.get! 1, col1.get! 1, col3.get! 1],
   [col2.get! 2, col1.get! 2, col3.get! 2]]

def fold_step_3 (grid : List (List ℕ)) : List (List ℕ) :=
  let row1 := grid.get! 0
  let row2 := grid.get! 1
  let row3 := grid.get! 2
  [row3, row2, row1]

def folded_grid : List (List ℕ) :=
  fold_step_3 (fold_step_2 (fold_step_1 initial_grid))

theorem top_square_is_9 : folded_grid.get! 0 = [9, 7, 8] :=
  sorry

end top_square_is_9_l130_130363


namespace birthday_candles_l130_130712

def number_of_red_candles : ℕ := 18
def number_of_green_candles : ℕ := 37
def number_of_yellow_candles := number_of_red_candles / 2
def total_age : ℕ := 85
def total_candles_so_far := number_of_red_candles + number_of_yellow_candles + number_of_green_candles
def number_of_blue_candles := total_age - total_candles_so_far

theorem birthday_candles :
  number_of_yellow_candles = 9 ∧
  number_of_blue_candles = 21 ∧
  (number_of_red_candles + number_of_yellow_candles + number_of_green_candles + number_of_blue_candles) = total_age :=
by
  sorry

end birthday_candles_l130_130712


namespace composite_proposition_l130_130127

theorem composite_proposition :
  (∀ x : ℝ, x^2 ≥ 0) ∧ ¬ (1 < 0) :=
by
  sorry

end composite_proposition_l130_130127


namespace am_gm_example_l130_130429

open Real

theorem am_gm_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1)^3 / b + (b + 1)^3 / c + (c + 1)^3 / a ≥ 81 / 4 := 
by 
  sorry

end am_gm_example_l130_130429


namespace length_of_AB_l130_130633

noncomputable def ratio3to5 (AP PB : ℝ) : Prop := AP / PB = 3 / 5
noncomputable def ratio4to5 (AQ QB : ℝ) : Prop := AQ / QB = 4 / 5
noncomputable def pointDistances (P Q : ℝ) : Prop := P - Q = 3

theorem length_of_AB (A B P Q : ℝ) (P_on_AB : P > A ∧ P < B) (Q_on_AB : Q > A ∧ Q < B)
  (middle_side : P < (A + B) / 2 ∧ Q < (A + B) / 2)
  (h1 : ratio3to5 (P - A) (B - P))
  (h2 : ratio4to5 (Q - A) (B - Q))
  (h3 : pointDistances P Q) : B - A = 43.2 := 
sorry

end length_of_AB_l130_130633


namespace not_always_true_inequality_l130_130088

theorem not_always_true_inequality (x : ℝ) (hx : x > 0) : 2^x ≤ x^2 := sorry

end not_always_true_inequality_l130_130088


namespace pow_mod_eq_l130_130192

theorem pow_mod_eq (h : 101 % 100 = 1) : (101 ^ 50) % 100 = 1 :=
by
  -- Proof omitted
  sorry

end pow_mod_eq_l130_130192


namespace sara_grew_4_onions_l130_130968

def onions_grown_by_sally : Nat := 5
def onions_grown_by_fred : Nat := 9
def total_onions_grown : Nat := 18

def onions_grown_by_sara : Nat :=
  total_onions_grown - (onions_grown_by_sally + onions_grown_by_fred)

theorem sara_grew_4_onions :
  onions_grown_by_sara = 4 :=
by
  sorry

end sara_grew_4_onions_l130_130968


namespace correct_function_at_x_equals_1_l130_130260

noncomputable def candidate_A (x : ℝ) : ℝ := (x - 1)^3 + 3 * (x - 1)
noncomputable def candidate_B (x : ℝ) : ℝ := 2 * (x - 1)^2
noncomputable def candidate_C (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def candidate_D (x : ℝ) : ℝ := x - 1

theorem correct_function_at_x_equals_1 :
  (deriv candidate_A 1 = 3) ∧ 
  (deriv candidate_B 1 ≠ 3) ∧ 
  (deriv candidate_C 1 ≠ 3) ∧ 
  (deriv candidate_D 1 ≠ 3) := 
by
  sorry

end correct_function_at_x_equals_1_l130_130260


namespace complementary_not_supplementary_l130_130345

theorem complementary_not_supplementary (α β : ℝ) (h₁ : α + β = 90) (h₂ : α + β ≠ 180) : (α + β = 180) = false :=
by 
  sorry

end complementary_not_supplementary_l130_130345


namespace value_of_a_squared_plus_b_squared_plus_2ab_l130_130600

theorem value_of_a_squared_plus_b_squared_plus_2ab (a b : ℝ) (h : a + b = -1) :
  a^2 + b^2 + 2 * a * b = 1 :=
by sorry

end value_of_a_squared_plus_b_squared_plus_2ab_l130_130600


namespace total_area_of_frequency_histogram_l130_130180

theorem total_area_of_frequency_histogram (f : ℝ → ℝ) (h_f : ∀ x, 0 ≤ f x ∧ f x ≤ 1) (integral_f_one : ∫ x, f x = 1) :
  ∫ x, f x = 1 := 
sorry

end total_area_of_frequency_histogram_l130_130180


namespace students_not_picked_l130_130007

def total_students : ℕ := 58
def number_of_groups : ℕ := 8
def students_per_group : ℕ := 6

theorem students_not_picked :
  total_students - (number_of_groups * students_per_group) = 10 := by 
  sorry

end students_not_picked_l130_130007


namespace max_ab_min_fraction_l130_130739

-- Question 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : ab ≤ 25/21 := sorry

-- Question 2: Minimum value of (3/a + 7/b)
theorem min_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : 3/a + 7/b ≥ 10 := sorry

end max_ab_min_fraction_l130_130739


namespace count_non_integer_angles_l130_130620

open Int

def interior_angle (n : ℕ) : ℕ := 180 * (n - 2) / n

def is_integer_angle (n : ℕ) : Prop := 180 * (n - 2) % n = 0

theorem count_non_integer_angles : ∃ (count : ℕ), count = 2 ∧ ∀ n, 3 ≤ n ∧ n < 12 → is_integer_angle n ↔ ¬ (count = count + 1) :=
sorry

end count_non_integer_angles_l130_130620


namespace question_mark_value_l130_130207

theorem question_mark_value :
  ∀ (x : ℕ), ( ( (5568: ℝ) / (x: ℝ) )^(1/3: ℝ) + ( (72: ℝ) * (2: ℝ) )^(1/2: ℝ) = (256: ℝ)^(1/2: ℝ) ) → x = 87 :=
by
  intro x
  intro h
  sorry

end question_mark_value_l130_130207


namespace range_of_b_l130_130586

noncomputable def set_A : Set ℝ := {x | -2 < x ∧ x < 1/3}
noncomputable def set_B (b : ℝ) : Set ℝ := {x | x^2 - 4*b*x + 3*b^2 < 0}

theorem range_of_b (b : ℝ) : 
  (set_A ∩ set_B b = ∅) ↔ (b = 0 ∨ b ≥ 1/3 ∨ b ≤ -2) :=
sorry

end range_of_b_l130_130586


namespace slope_of_parallel_line_l130_130194

theorem slope_of_parallel_line (a b c : ℝ) (h: 3*a + 6*b = -24) :
  ∃ m : ℝ, (a * 3 + b * 6 = c) → m = -1/2 :=
by
  sorry

end slope_of_parallel_line_l130_130194


namespace four_digit_num_condition_l130_130286

theorem four_digit_num_condition :
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ x = 100*a ∧ n = 1000*a + x :=
by sorry

end four_digit_num_condition_l130_130286


namespace volume_of_region_l130_130244

-- Define the conditions
def condition1 (x y z : ℝ) := abs (x + y + 2 * z) + abs (x + y - 2 * z) ≤ 12
def condition2 (x : ℝ) := x ≥ 0
def condition3 (y : ℝ) := y ≥ 0
def condition4 (z : ℝ) := z ≥ 0

-- Define the volume function
def volume (x y z : ℝ) := 18 * 3

-- Proof statement
theorem volume_of_region : ∀ (x y z : ℝ),
  condition1 x y z →
  condition2 x →
  condition3 y →
  condition4 z →
  volume x y z = 54 := by
  sorry

end volume_of_region_l130_130244


namespace men_absent_l130_130079

theorem men_absent (n : ℕ) (d1 d2 : ℕ) (x : ℕ) 
  (h1 : n = 22) 
  (h2 : d1 = 20) 
  (h3 : d2 = 22) 
  (hc : n * d1 = (n - x) * d2) : 
  x = 2 := 
by {
  sorry
}

end men_absent_l130_130079


namespace water_volume_percentage_l130_130060

variables {h r : ℝ}
def cone_volume (h r : ℝ) : ℝ := (1 / 3) * π * r^2 * h
def water_volume (h r : ℝ) : ℝ := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h)

theorem water_volume_percentage :
  ((water_volume h r) / (cone_volume h r)) = 8 / 27 :=
by
  sorry

end water_volume_percentage_l130_130060


namespace cars_with_neither_l130_130630

theorem cars_with_neither (total_cars air_bag power_windows both : ℕ) 
                          (h1 : total_cars = 65) (h2 : air_bag = 45)
                          (h3 : power_windows = 30) (h4 : both = 12) : 
                          (total_cars - (air_bag + power_windows - both) = 2) :=
by
  sorry

end cars_with_neither_l130_130630


namespace remainder_correct_l130_130731

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 8 - 2 * x ^ 5 + 5 * x ^ 3 - 9
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) : ℝ := 29 * x - 32

theorem remainder_correct (x : ℝ) :
  ∃ q : ℝ → ℝ, p x = d x * q x + r x :=
sorry

end remainder_correct_l130_130731


namespace rocket_max_speed_l130_130998

theorem rocket_max_speed (M m : ℝ) (h : 2000 * Real.log (1 + M / m) = 12000) : 
  M / m = Real.exp 6 - 1 := 
by {
  sorry
}

end rocket_max_speed_l130_130998


namespace solve_grape_rate_l130_130706

noncomputable def grape_rate (G : ℝ) : Prop :=
  11 * G + 7 * 50 = 1428

theorem solve_grape_rate : ∃ G : ℝ, grape_rate G ∧ G = 98 :=
by
  exists 98
  sorry

end solve_grape_rate_l130_130706


namespace solve_problem_l130_130098

open Nat

theorem solve_problem :
  ∃ (n p : ℕ), p.Prime ∧ n > 0 ∧ ∃ k : ℤ, p^2 + 7^n = k^2 ∧ (n, p) = (1, 3) := 
by
  sorry

end solve_problem_l130_130098


namespace x_y_divisible_by_7_l130_130916

theorem x_y_divisible_by_7
  (x y a b : ℤ)
  (hx : 3 * x + 4 * y = a ^ 2)
  (hy : 4 * x + 3 * y = b ^ 2)
  (hx_pos : x > 0) (hy_pos : y > 0) :
  7 ∣ x ∧ 7 ∣ y :=
by
  sorry

end x_y_divisible_by_7_l130_130916


namespace distance_from_focus_to_line_l130_130652

def hyperbola : Type := {x : ℝ // (x^2 / 4) - (x^2 / 5) = 1}
def line : Type := {p : ℝ × ℝ // 1 * p.1 + 2 * p.2 - 8 = 0}

theorem distance_from_focus_to_line (F : ℝ × ℝ) (L : line) : 
  let d := (abs ((1 * F.1 + 2 * F.2 - 8) / (sqrt(1^2 + 2^2)))) in
  F = (3, 0) → d = sqrt 5 :=
sorry

end distance_from_focus_to_line_l130_130652


namespace inequality_proof_l130_130591

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : a3 > 0) 
  (h4 : b1 ≥ b2) (h5 : b2 ≥ b3) (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) : 
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := sorry

end inequality_proof_l130_130591


namespace probability_laurent_greater_chloe_l130_130089

noncomputable def chloe_and_laurent_probability : ℝ :=
  let chloe_dist := set.Icc (0 : ℝ) 1000
  let laurent_dist := set.Icc (500 : ℝ) 3000
  let total_area := 1000 * 2500
  let favorable_area := (1/2 : ℝ) * 500 * 500 + 1000 * 2000
  favorable_area / total_area

theorem probability_laurent_greater_chloe : chloe_and_laurent_probability = 0.85 := 
  by
    sorry

end probability_laurent_greater_chloe_l130_130089


namespace circles_intersect_line_l130_130010

theorem circles_intersect_line (m c : ℝ)
  (hA : (1 : ℝ) - 3 + c = 0)
  (hB : 1 = -(m - 1) / (-4)) :
  m + c = -1 :=
by
  sorry

end circles_intersect_line_l130_130010


namespace highest_power_of_3_l130_130492

-- Define the integer M formed by concatenating the 3-digit numbers from 100 to 250
def M : ℕ := sorry  -- We should define it in a way that represents the concatenation

-- Define a proof that the highest power of 3 that divides M is 3^1
theorem highest_power_of_3 (n : ℕ) (h : M = n) : ∃ m : ℕ, 3^m ∣ n ∧ ¬ (3^(m + 1) ∣ n) ∧ m = 1 :=
by sorry  -- We will not provide proofs; we're only writing the statement

end highest_power_of_3_l130_130492


namespace find_angle_A_l130_130140

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hB : B = Real.pi / 3) : 
  A = Real.pi / 4 := 
sorry

end find_angle_A_l130_130140


namespace smallest_n_correct_l130_130621

/-- The first term of the geometric sequence. -/
def a₁ : ℚ := 5 / 6

/-- The second term of the geometric sequence. -/
def a₂ : ℚ := 25

/-- The common ratio for the geometric sequence. -/
def r : ℚ := a₂ / a₁

/-- The nth term of the geometric sequence. -/
def a_n (n : ℕ) : ℚ := a₁ * r^(n - 1)

/-- The smallest n such that the nth term is divisible by 10^7. -/
def smallest_n : ℕ := 8

theorem smallest_n_correct :
  ∀ n : ℕ, (a₁ * r^(n - 1)) ∣ (10^7 : ℚ) ↔ n = smallest_n := 
sorry

end smallest_n_correct_l130_130621


namespace area_of_triangle_OAB_l130_130932

noncomputable def parabola (x : ℝ) : ℝ := sqrt (3 * x)

noncomputable def focus : ℝ × ℝ := (3 / 4, 0)

noncomputable def line_through_focus (x : ℝ) : ℝ := (sqrt 3 / 3) * (x - 3 / 4)

noncomputable def points_of_intersection : set (ℝ × ℝ) :=
  { (x, y) | y = parabola x ∧ y = line_through_focus x }

theorem area_of_triangle_OAB :
  let A := classical.some (points_of_intersection.some_spec.1)
  let B := classical.some (points_of_intersection.some_spec.2)
  ∃ (area : ℝ), area = 9 / 4 :=
sorry

end area_of_triangle_OAB_l130_130932


namespace men_in_first_group_l130_130168

theorem men_in_first_group (M : ℕ) 
  (h1 : (M * 25 : ℝ) = (15 * 26.666666666666668 : ℝ)) : 
  M = 16 := 
by 
  sorry

end men_in_first_group_l130_130168


namespace min_correct_answers_l130_130772

theorem min_correct_answers (x : ℕ) (hx : 10 * x - 5 * (30 - x) > 90) : x ≥ 17 :=
by {
  -- calculations and solution steps go here.
  sorry
}

end min_correct_answers_l130_130772


namespace length_of_train_l130_130701

-- Definitions for the given conditions:
def speed : ℝ := 60   -- in kmph
def time : ℝ := 20    -- in seconds
def platform_length : ℝ := 213.36  -- in meters

-- Conversion factor from km/h to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Total distance covered by train while crossing the platform
noncomputable def total_distance (speed_in_kmph : ℝ) (time_in_seconds : ℝ) : ℝ := 
  (kmph_to_mps speed_in_kmph) * time_in_seconds

-- Length of the train
noncomputable def train_length (total_distance_covered : ℝ) (platform_len : ℝ) : ℝ :=
  total_distance_covered - platform_len

-- Expected length of the train
def expected_train_length : ℝ := 120.04

-- Theorem to prove the length of the train given the conditions
theorem length_of_train : 
  train_length (total_distance speed time) platform_length = expected_train_length :=
by 
  sorry

end length_of_train_l130_130701


namespace first_shaded_square_each_column_l130_130544

/-- A rectangular board with 10 columns, numbered starting from 
    1 to the nth square left-to-right and top-to-bottom. The student shades squares 
    that are perfect squares. Prove that the first shaded square ensuring there's at least 
    one shaded square in each of the 10 columns is 400. -/
theorem first_shaded_square_each_column : 
  (∃ n, (∀ k, 1 ≤ k ∧ k ≤ 10 → ∃ m, m^2 ≡ k [MOD 10] ∧ m^2 ≤ n) ∧ n = 400) :=
sorry

end first_shaded_square_each_column_l130_130544


namespace polygon_sides_l130_130912

theorem polygon_sides (a : ℝ) (n : ℕ) (h1 : a = 140) (h2 : 180 * (n-2) = n * a) : n = 9 := 
by sorry

end polygon_sides_l130_130912


namespace probability_home_appliance_correct_maximum_m_profitable_l130_130093

namespace Promotions

open Nat

def probability_at_least_one_home_appliance : ℝ :=
  1 - (choose 6 3 : ℕ) / (choose 8 3 : ℕ)

theorem probability_home_appliance_correct : probability_at_least_one_home_appliance = 9 / 14 :=
  sorry

def expected_value_of_lottery (m : ℝ) : ℝ :=
  0 * (8 / 27) +
  m * (4 / 9) +
  3 * m * (2 / 9) +
  6 * m * (1 / 27)

theorem maximum_m_profitable (m : ℝ) (h : expected_value_of_lottery m ≤ 100) : m ≤ 75 :=
  sorry

end Promotions

end probability_home_appliance_correct_maximum_m_profitable_l130_130093


namespace find_a_l130_130576

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}) (h1 : 1 ∈ A) : a = -1 :=
by
  sorry

end find_a_l130_130576


namespace sum_of_roots_eq_three_l130_130348

theorem sum_of_roots_eq_three {a b : ℝ} (h₁ : a * (-3)^3 + (a + 3 * b) * (-3)^2 + (b - 4 * a) * (-3) + (11 - a) = 0)
  (h₂ : a * 2^3 + (a + 3 * b) * 2^2 + (b - 4 * a) * 2 + (11 - a) = 0)
  (h₃ : a * 4^3 + (a + 3 * b) * 4^2 + (b - 4 * a) * 4 + (11 - a) = 0) :
  (-3) + 2 + 4 = 3 :=
by
  sorry

end sum_of_roots_eq_three_l130_130348


namespace probability_of_four_green_marbles_l130_130635

-- Define the conditions: 10 green marbles, 5 purple marbles
def green_marbles : ℕ := 10
def purple_marbles : ℕ := 5
def total_marbles : ℕ := green_marbles + purple_marbles
def draws : ℕ := 7
def green_probability : ℚ := green_marbles / total_marbles
def purple_probability : ℚ := purple_marbles / total_marbles
def success_draws : ℕ := 4

-- Calculate the binomial coefficient
def binomial_coefficient : ℕ := nat.choose draws success_draws

-- Calculate the probability
def probability : ℚ := binomial_coefficient * (green_probability ^ success_draws) * (purple_probability ^ (draws - success_draws))

-- Prove that the resulting probability is approximately 0.256
theorem probability_of_four_green_marbles :
  probability ≈ 0.256 := 
begin
  -- We calculate the exact value 
  have exact_value : probability = 35 * ((2/3)^4) * ((1/3)^3), 
  {
    sorry
  },
  -- Numerically, this simplifies approximately to 0.256
  have numerical_value : 35 * ((2/3)^4) * ((1/3)^3) ≈ 0.256, 
  {
    sorry
  },
  exact eq.trans exact_value numerical_value
end

end probability_of_four_green_marbles_l130_130635


namespace set_complement_intersection_l130_130934

open Set

variable (U A B : Set ℕ)

theorem set_complement_intersection :
  U = {2, 3, 5, 7, 8} →
  A = {2, 8} →
  B = {3, 5, 8} →
  (U \ A) ∩ B = {3, 5} :=
by
  intros
  sorry

end set_complement_intersection_l130_130934


namespace number_of_red_balloons_l130_130674

-- Definitions for conditions
def balloons_total : ℕ := 85
def at_least_one_red (red blue : ℕ) : Prop := red ≥ 1 ∧ red + blue = balloons_total
def every_pair_has_blue (red blue : ℕ) : Prop := ∀ r1 r2, r1 < red → r2 < red → red = 1

-- Theorem to be proved
theorem number_of_red_balloons (red blue : ℕ) 
  (total : red + blue = balloons_total)
  (at_least_one : at_least_one_red red blue)
  (pair_condition : every_pair_has_blue red blue) : red = 1 :=
sorry

end number_of_red_balloons_l130_130674


namespace pure_imaginary_solution_l130_130455

theorem pure_imaginary_solution (b : ℝ) (z : ℂ) 
  (H : z = (b + Complex.I) / (2 + Complex.I))
  (H_imaginary : z.im = z ∧ z.re = 0) :
  b = -1 / 2 := 
by 
  sorry

end pure_imaginary_solution_l130_130455


namespace cone_water_volume_percentage_l130_130057

theorem cone_water_volume_percentage (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let V := (1 / 3) * π * r^2 * h in
  let V_w := (1 / 3) * π * (2 / 3 * r)^2 * (2 / 3 * h) in
  (V_w / V) ≈ 0.296296 := 
by
  sorry

end cone_water_volume_percentage_l130_130057


namespace product_of_roots_l130_130877

theorem product_of_roots (a b c : ℤ) (h_eq : a = 24 ∧ b = 60 ∧ c = -600) :
  ∀ x : ℂ, (a * x^2 + b * x + c = 0) → (x * (-b - x) = -25) := sorry

end product_of_roots_l130_130877


namespace max_marks_l130_130524

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 165): M = 500 :=
by
  sorry

end max_marks_l130_130524


namespace find_h_l130_130763

theorem find_h (x : ℝ) : 
  ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
sorry

end find_h_l130_130763


namespace aarti_three_times_work_l130_130684

theorem aarti_three_times_work (d : ℕ) (h : d = 5) : 3 * d = 15 :=
by
  sorry

end aarti_three_times_work_l130_130684


namespace tan_ratio_given_sin_equation_l130_130436

theorem tan_ratio_given_sin_equation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.sin (2*α + β) = (3/2) * Real.sin β) : 
  Real.tan (α + β) / Real.tan α = 5 :=
by
  -- Proof goes here
  sorry

end tan_ratio_given_sin_equation_l130_130436


namespace painting_time_eq_l130_130220

theorem painting_time_eq (t : ℝ) :
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 ↔ t = 8 / 3 :=
by
  sorry

end painting_time_eq_l130_130220


namespace parsnip_box_fullness_l130_130385

theorem parsnip_box_fullness (capacity : ℕ) (fraction_full : ℚ) (avg_boxes : ℕ) (avg_parsnips : ℕ) :
  capacity = 20 →
  fraction_full = 3 / 4 →
  avg_boxes = 20 →
  avg_parsnips = 350 →
  ∃ (full_boxes : ℕ) (non_full_boxes : ℕ) (parsnips_in_full_boxes : ℕ) (parsnips_in_non_full_boxes : ℕ)
    (avg_fullness_non_full_boxes : ℕ),
    full_boxes = fraction_full * avg_boxes ∧
    non_full_boxes = avg_boxes - full_boxes ∧
    parsnips_in_full_boxes = full_boxes * capacity ∧
    parsnips_in_non_full_boxes = avg_parsnips - parsnips_in_full_boxes ∧
    avg_fullness_non_full_boxes = parsnips_in_non_full_boxes / non_full_boxes ∧
    avg_fullness_non_full_boxes = 10 :=
by
  sorry

end parsnip_box_fullness_l130_130385


namespace solve_for_a_l130_130941

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := 
  (Real.sqrt (a^2 - b^2)) / a

noncomputable def solve_ellipse_parameters (a1 e1 e2 : ℝ) :=
  let c1 := (a1 / 2) in
  let a1_squared := 4 * (a1^2 - 1) in
  a1 = sqrt (4 / 3)

theorem solve_for_a 
  (a1 a2 b2 : ℝ)
  (h1 : a1 > 1)
  (h2 : a2 = 2)
  (h3 : b2 = 1)
  (e2 = sqrt 3 * e1)
  (e1 = 1 / 2)
  : a = 2 * sqrt 3 / 3 :=
by
  -- Insert proof here
  sorry

end solve_for_a_l130_130941


namespace quadratic_inequality_solution_l130_130322

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → -3 * x^2 + 8 * x + 1 < 0 :=
by
  intro x
  intro h
  sorry

end quadratic_inequality_solution_l130_130322


namespace number_of_buses_required_l130_130342

def total_seats : ℕ := 28
def students_per_bus : ℝ := 14.0

theorem number_of_buses_required :
  (total_seats / students_per_bus) = 2 := 
by
  -- The actual proof is intentionally left out.
  sorry

end number_of_buses_required_l130_130342


namespace solve_linear_system_l130_130167

/-- Let x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈ be real numbers that satisfy the following system of equations:
1. x₁ + x₂ + x₃ = 6
2. x₂ + x₃ + x₄ = 9
3. x₃ + x₄ + x₅ = 3
4. x₄ + x₅ + x₆ = -3
5. x₅ + x₆ + x₇ = -9
6. x₆ + x₇ + x₈ = -6
7. x₇ + x₈ + x₁ = -2
8. x₈ + x₁ + x₂ = 2
Prove that the solution is
  x₁ = 1, x₂ = 2, x₃ = 3, x₄ = 4, x₅ = -4, x₆ = -3, x₇ = -2, x₈ = -1
-/
theorem solve_linear_system :
  ∀ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ),
  x₁ + x₂ + x₃ = 6 →
  x₂ + x₃ + x₄ = 9 →
  x₃ + x₄ + x₅ = 3 →
  x₄ + x₅ + x₆ = -3 →
  x₅ + x₆ + x₇ = -9 →
  x₆ + x₇ + x₈ = -6 →
  x₇ + x₈ + x₁ = -2 →
  x₈ + x₁ + x₂ = 2 →
  x₁ = 1 ∧ x₂ = 2 ∧ x₃ = 3 ∧ x₄ = 4 ∧ x₅ = -4 ∧ x₆ = -3 ∧ x₇ = -2 ∧ x₈ = -1 :=
by
  intros x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ h1 h2 h3 h4 h5 h6 h7 h8
  -- Here, the proof steps would go
  sorry

end solve_linear_system_l130_130167


namespace sphere_volume_from_surface_area_l130_130900

theorem sphere_volume_from_surface_area (S : ℝ) (V : ℝ) (R : ℝ) (h1 : S = 36 * Real.pi) (h2 : S = 4 * Real.pi * R ^ 2) (h3 : V = (4 / 3) * Real.pi * R ^ 3) : V = 36 * Real.pi :=
by
  sorry

end sphere_volume_from_surface_area_l130_130900


namespace f_f_f_f_f_3_eq_4_l130_130622

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_f_f_f_f_3_eq_4 : f (f (f (f (f 3)))) = 4 := 
  sorry

end f_f_f_f_f_3_eq_4_l130_130622


namespace calculate_minimal_total_cost_l130_130765

structure GardenSection where
  area : ℕ
  flower_cost : ℚ

def garden := [
  GardenSection.mk 10 2.75, -- Orchids
  GardenSection.mk 14 2.25, -- Violets
  GardenSection.mk 14 1.50, -- Hyacinths
  GardenSection.mk 15 1.25, -- Tulips
  GardenSection.mk 25 0.75  -- Sunflowers
]

def total_cost (sections : List GardenSection) : ℚ :=
  sections.foldr (λ s acc => s.area * s.flower_cost + acc) 0

theorem calculate_minimal_total_cost :
  total_cost garden = 117.5 := by
  sorry

end calculate_minimal_total_cost_l130_130765


namespace find_length_of_rod_l130_130133

-- Constants representing the given conditions
def weight_6m_rod : ℝ := 6.1
def length_6m_rod : ℝ := 6
def weight_unknown_rod : ℝ := 12.2

-- Proof statement ensuring the length of the rod that weighs 12.2 kg is 12 meters
theorem find_length_of_rod (L : ℝ) (h : weight_6m_rod / length_6m_rod = weight_unknown_rod / L) : 
  L = 12 := by
  sorry

end find_length_of_rod_l130_130133


namespace even_sum_probability_l130_130512

-- Definitions of the conditions
def balls : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Function to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Function to calculate the sum of two numbers
def sum (a b : ℕ) : ℕ := a + b

-- Function to calculate the probability
noncomputable def probability_even_sum : ℚ := 
  let total_outcomes := 12 * 11 in
  let favorable_outcomes := 30 + 30 in
  favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem even_sum_probability :
  probability_even_sum = 5 / 11 :=
  by
  sorry

end even_sum_probability_l130_130512


namespace monkey_reaches_tree_top_in_hours_l130_130523

-- Definitions based on conditions
def height_of_tree : ℕ := 22
def hop_per_hour : ℕ := 3
def slip_per_hour : ℕ := 2
def effective_climb_per_hour : ℕ := hop_per_hour - slip_per_hour

-- The theorem we want to prove
theorem monkey_reaches_tree_top_in_hours
  (height_of_tree hop_per_hour slip_per_hour : ℕ)
  (h1 : height_of_tree = 22)
  (h2 : hop_per_hour = 3)
  (h3 : slip_per_hour = 2) :
  ∃ t : ℕ, t = 22 ∧ effective_climb_per_hour * (t - 1) + hop_per_hour = height_of_tree := by
  sorry

end monkey_reaches_tree_top_in_hours_l130_130523


namespace find_a_l130_130579

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end find_a_l130_130579


namespace count_valid_numbers_l130_130272
noncomputable def valid_four_digit_numbers : ℕ :=
  let N := (λ a x : ℕ, 1000 * a + x) in
  let x := (λ a : ℕ, 100 * a) in
  finset.card $ finset.filter (λ a, 1 ≤ a ∧ a ≤ 9) (@finset.range _ _ (nat.lt_succ_self 9))

theorem count_valid_numbers : valid_four_digit_numbers = 9 :=
sorry

end count_valid_numbers_l130_130272


namespace sqrt_64_eq_pm_8_l130_130344

theorem sqrt_64_eq_pm_8 : ∃x : ℤ, x^2 = 64 ∧ (x = 8 ∨ x = -8) :=
by
  sorry

end sqrt_64_eq_pm_8_l130_130344


namespace abs_sum_factors_l130_130042

theorem abs_sum_factors (a b c d : ℤ) : 
  (6 * x ^ 2 + x - 12 = (a * x + b) * (c * x + d)) →
  (|a| + |b| + |c| + |d| = 12) :=
by
  intros h
  sorry

end abs_sum_factors_l130_130042


namespace pedro_furniture_area_l130_130481

theorem pedro_furniture_area :
  let width : ℝ := 2
  let length : ℝ := 2.5
  let door_arc_area := (1 / 4) * Real.pi * (0.5 ^ 2)
  let window_arc_area := 2 * (1 / 2) * Real.pi * (0.5 ^ 2)
  let room_area := width * length
  room_area - door_arc_area - window_arc_area = (80 - 9 * Real.pi) / 16 := 
by
  sorry

end pedro_furniture_area_l130_130481


namespace find_a_from_perpendicular_lines_l130_130899

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l130_130899


namespace arithmetic_sequence_second_term_l130_130603

theorem arithmetic_sequence_second_term (S₃: ℕ) (a₁: ℕ) (h1: S₃ = 9) (h2: a₁ = 1) : 
∃ d a₂, 3 * a₁ + 3 * d = S₃ ∧ a₂ = a₁ + d ∧ a₂ = 3 :=
by
  sorry

end arithmetic_sequence_second_term_l130_130603


namespace max_integer_a_real_roots_l130_130249

theorem max_integer_a_real_roots :
  ∀ (a : ℤ), (∃ (x : ℝ), (a + 1 : ℝ) * x^2 - 2 * x + 3 = 0) → a ≤ -2 :=
by
  sorry

end max_integer_a_real_roots_l130_130249


namespace geometric_sequence_third_term_l130_130694

theorem geometric_sequence_third_term (r : ℕ) (a : ℕ) (h1 : a = 6) (h2 : a * r^3 = 384) : a * r^2 = 96 :=
by
  sorry

end geometric_sequence_third_term_l130_130694


namespace max_value_sum_l130_130583

variable (n : ℕ) (x : Fin n → ℝ)

theorem max_value_sum 
  (h1 : ∀ i, 0 ≤ x i)
  (h2 : 2 ≤ n)
  (h3 : (Finset.univ : Finset (Fin n)).sum x = 1) :
  ∃ max_val, max_val = (1 / 4) :=
sorry

end max_value_sum_l130_130583


namespace fraction_dad_roasted_l130_130927

theorem fraction_dad_roasted :
  ∀ (dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roast = joe_marshmallows / 2 →
    total_roast = 49 →
    dad_roast = total_roast - joe_roast →
    (dad_roast : ℚ) / (dad_marshmallows : ℚ) = 1 / 3 :=
by
  intros dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast
  intro h_dad_marshmallows
  intro h_joe_marshmallows
  intro h_joe_roast
  intro h_total_roast
  intro h_dad_roast
  sorry

end fraction_dad_roasted_l130_130927


namespace surveys_from_retired_is_12_l130_130559

-- Define the given conditions
def ratio_retired : ℕ := 2
def ratio_current : ℕ := 8
def ratio_students : ℕ := 40
def total_surveys : ℕ := 300
def total_ratio : ℕ := ratio_retired + ratio_current + ratio_students

-- Calculate the expected number of surveys from retired faculty
def number_of_surveys_retired : ℕ := total_surveys * ratio_retired / total_ratio

-- Lean 4 statement for proof
theorem surveys_from_retired_is_12 :
  number_of_surveys_retired = 12 :=
by
  -- Proof to be filled in
  sorry

end surveys_from_retired_is_12_l130_130559


namespace school_A_original_students_l130_130997

theorem school_A_original_students 
  (x y : ℕ) 
  (h1 : x + y = 864) 
  (h2 : x - 32 = y + 80) : 
  x = 488 := 
by 
  sorry

end school_A_original_students_l130_130997


namespace remainder_of_S_mod_1000_l130_130933

def digit_contribution (d pos : ℕ) : ℕ := (d * d) * pos

def sum_of_digits_with_no_repeats : ℕ :=
  let thousands := (16 + 25 + 36 + 49 + 64 + 81) * (9 * 8 * 7) * 1000
  let hundreds := (16 + 25 + 36 + 49 + 64 + 81) * (8 * 7 * 6) * 100
  let tens := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 10
  let units := (0 + 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 - (16 + 25 + 36 + 49 + 64 + 81)) * 6 * 5 * 1
  thousands + hundreds + tens + units

theorem remainder_of_S_mod_1000 : (sum_of_digits_with_no_repeats % 1000) = 220 :=
  by
  sorry

end remainder_of_S_mod_1000_l130_130933


namespace absolute_value_positive_l130_130031

theorem absolute_value_positive (a : ℝ) (h : a ≠ 0) : |a| > 0 := by
  sorry

end absolute_value_positive_l130_130031


namespace leila_toys_l130_130784

theorem leila_toys:
  ∀ (x : ℕ),
  (∀ l m : ℕ, l = 2 * x ∧ m = 3 * 19 ∧ m = l + 7 → x = 25) :=
by
  sorry

end leila_toys_l130_130784


namespace history_homework_time_l130_130804

def total_time := 180
def math_homework := 45
def english_homework := 30
def science_homework := 50
def special_project := 30

theorem history_homework_time : total_time - (math_homework + english_homework + science_homework + special_project) = 25 := by
  sorry

end history_homework_time_l130_130804


namespace parabola_symmetry_l130_130894

theorem parabola_symmetry (a h m : ℝ) (A_on_parabola : 4 = a * (-1 - 3)^2 + h) (B_on_parabola : 4 = a * (m - 3)^2 + h) : 
  m = 7 :=
by 
  sorry

end parabola_symmetry_l130_130894


namespace steven_set_aside_pears_l130_130970

theorem steven_set_aside_pears :
  ∀ (apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape : ℕ),
    apples = 4 →
    grapes = 9 →
    neededSeeds = 60 →
    seedPerApple = 6 →
    seedPerPear = 2 →
    seedPerGrape = 3 →
    (neededSeeds - 3) = (apples * seedPerApple + grapes * seedPerGrape + pears * seedPerPear) →
    pears = 3 :=
by
  intros apples pears grapes neededSeeds seedPerApple seedPerPear seedPerGrape
  intros h_apple h_grape h_needed h_seedApple h_seedPear h_seedGrape
  intros h_totalSeeds
  sorry

end steven_set_aside_pears_l130_130970


namespace inscribe_2n_gon_l130_130883

-- Define the problem conditions in Lean:

variables (n : ℕ) (n_gt_0 : n > 0)
-- n is a natural number, and it is greater than 0

def lines : list (ℝ → ℝ) := sorry -- Placeholder for the list of \(2n-1\) lines
def circle_center : ℝ × ℝ := (0, 0) -- Assuming the circle is centered at origin for simplification
def circle_radius : ℝ := 1 -- Assuming radius of the circle is 1 for simplification
def point_K : ℝ × ℝ := sorry -- Placeholder for point K inside the circle

-- The main theorem to prove

theorem inscribe_2n_gon 
  (lines : list (ℝ → ℝ)) -- \(2n-1\) lines given on the plane
  (h_len_lines : lines.length = 2 * n - 1) -- given lines are exactly \(2n-1\)
  (circle_center : ℝ × ℝ) -- the coordinates of the circle center
  (circle_radius : ℝ) --  the radius of circle
  (point_K : ℝ × ℝ) -- Point K inside the circle
  (h_inside : (point_K.1 - circle_center.1)^2 + (point_K.2 - circle_center.2)^2 < circle_radius^2): -- K inside the circle
  ∃ polygon : list (ℝ × ℝ), -- The polygon vertices on the plane
    (polygon.length = 2 * n) ∧ -- polygon is a \(2n\)-gon
    (∃ i, 1 ≤ i ∧ i ≤ 2 * n ∧ (polygon.nth (i % (2 * n)) = point_K ∨ polygon.nth ((i-1) % (2 * n)) = point_K)) ∧ -- one side passes through K
    ∀ j, (1 ≤ j ∧ j < 2 * n) → parallel_to lines (polygon.nth (j % (2 * n))) (polygon.nth ((j+1) % (2 * n))) -- sides are parallel to the lines
      parse sorry -- Placeholder for parsing parallel_to
  
-- Placeholder for the function that checks if two line segments are parallel to any of the given lines.
def parallel_to : list (ℝ → ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry


end inscribe_2n_gon_l130_130883


namespace sum_geometric_series_is_correct_l130_130013

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end sum_geometric_series_is_correct_l130_130013


namespace josh_marbles_l130_130781

theorem josh_marbles (initial_marbles lost_marbles : ℕ) (h_initial : initial_marbles = 9) (h_lost : lost_marbles = 5) :
  initial_marbles - lost_marbles = 4 :=
by
  sorry

end josh_marbles_l130_130781


namespace kim_pairs_of_shoes_l130_130618

theorem kim_pairs_of_shoes : ∃ n : ℕ, 2 * n + 1 = 14 ∧ (1 : ℚ) / (2 * n - 1) = (0.07692307692307693 : ℚ) :=
by
  sorry

end kim_pairs_of_shoes_l130_130618


namespace min_value_expression_geq_twosqrt3_l130_130748

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  (1/(x-1)) + (3/(y-1))

theorem min_value_expression_geq_twosqrt3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1/x) + (1/y) = 1) : 
  min_value_expression x y >= 2 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_geq_twosqrt3_l130_130748


namespace arithmetic_mean_of_multiples_of_5_l130_130999

-- Define the sequence and its properties
def a₁ : ℕ := 10
def d : ℕ := 5
def aₙ : ℕ := 95

-- Find number of terms in the sequence
def n: ℕ := (aₙ - a₁) / d + 1

-- Define the sum of the sequence
def S := n * (a₁ + aₙ) / 2

-- Define the arithmetic mean
def arithmetic_mean := S / n

-- Prove the arithmetic mean
theorem arithmetic_mean_of_multiples_of_5 : arithmetic_mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_multiples_of_5_l130_130999


namespace rectangle_similarity_l130_130938

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def is_congruent (A B : Rectangle) : Prop :=
  A.length = B.length ∧ A.width = B.width

def is_similar (A B : Rectangle) : Prop :=
  A.length / A.width = B.length / B.width

theorem rectangle_similarity (A B : Rectangle)
  (h1 : ∀ P, is_congruent P A → ∃ Q, is_similar Q B)
  : ∀ P, is_congruent P B → ∃ Q, is_similar Q A :=
by sorry

end rectangle_similarity_l130_130938


namespace cone_water_fill_percentage_l130_130073

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l130_130073


namespace ram_krish_task_completion_l130_130965

theorem ram_krish_task_completion
  (ram_days : ℝ)
  (krish_efficiency_factor : ℝ)
  (task_time : ℝ) 
  (H1 : krish_efficiency_factor = 2)
  (H2 : ram_days = 27) 
  (H3 : task_time = 9) :
  (1 / task_time) = (1 / ram_days + 1 / (ram_days / krish_efficiency_factor)) := 
sorry

end ram_krish_task_completion_l130_130965


namespace monthly_salary_l130_130354

theorem monthly_salary (S : ℝ) (E : ℝ) 
  (h1 : S - 1.20 * E = 220)
  (h2 : E = 0.80 * S) :
  S = 5500 :=
by
  sorry

end monthly_salary_l130_130354


namespace valid_division_l130_130164

theorem valid_division (A B C E F G H K : ℕ) (hA : A = 7) (hB : B = 1) (hC : C = 2)
    (hE : E = 6) (hF : F = 8) (hG : G = 5) (hH : H = 4) (hK : K = 9) :
    (A * 10 + B) / ((C * 100 + A * 10 + B) / 100 + E + B * F * D) = 71 / 271 :=
by {
  sorry
}

end valid_division_l130_130164


namespace min_x_plus_y_l130_130292

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) :
  x + y ≥ 16 :=
sorry

end min_x_plus_y_l130_130292


namespace rectangle_area_l130_130187

theorem rectangle_area (x y : ℝ) (hx : 3 * y = 7 * x) (hp : 2 * (x + y) = 40) :
  x * y = 84 := by
  sorry

end rectangle_area_l130_130187


namespace distance_from_right_focus_to_line_is_sqrt5_l130_130663

-- Define the point (3, 0)
def point : ℝ × ℝ := (3, 0)

-- Define the line equation x + 2y - 8 = 0
def line (x y : ℝ) : ℝ := x + 2 * y - 8

-- Define the distance formula from a point to a line
def distance_point_to_line (p : ℝ × ℝ) (l : ℝ × ℝ → ℝ) : ℝ :=
  abs (l p.1 p.2) / sqrt (1^2 + 2^2)  -- parameters of the line equation Ax + By + C=0, where A=1, B=2

-- Prove the distance is √5
theorem distance_from_right_focus_to_line_is_sqrt5 :
  distance_point_to_line point line = sqrt 5 := by
  sorry

end distance_from_right_focus_to_line_is_sqrt5_l130_130663


namespace clara_weight_l130_130987

-- Define the weights of Alice and Clara
variables (a c : ℕ)

-- Define the conditions given in the problem
def condition1 := a + c = 240
def condition2 := c - a = c / 3

-- The theorem to prove Clara's weight given the conditions
theorem clara_weight : condition1 a c → condition2 a c → c = 144 :=
by
  intros h1 h2
  sorry

end clara_weight_l130_130987


namespace trapezoid_area_l130_130976

-- Definitions based on the given conditions
variable (BD AC h : ℝ)
variable (BD_perpendicular_AC : BD * AC = 0)
variable (BD_val : BD = 13)
variable (h_val : h = 12)

-- Statement of the theorem to prove the area of the trapezoid
theorem trapezoid_area (BD AC h : ℝ)
  (BD_perpendicular_AC : BD * AC = 0)
  (BD_val : BD = 13)
  (h_val : h = 12) :
  0.5 * 13 * 12 = 1014 / 5 := sorry

end trapezoid_area_l130_130976


namespace isosceles_triangle_interior_angles_l130_130259

theorem isosceles_triangle_interior_angles (a b c : ℝ) 
  (h1 : b = c) (h2 : a + b + c = 180) (exterior : a + 40 = 180 ∨ b + 40 = 140) :
  (a = 40 ∧ b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40 ∧ c = 40) :=
by
  sorry

end isosceles_triangle_interior_angles_l130_130259


namespace canoe_rental_cost_l130_130834

theorem canoe_rental_cost (C : ℕ) (K : ℕ) :
  18 * K + C * (K + 5) = 405 → 
  3 * K = 2 * (K + 5) → 
  C = 15 :=
by
  intros revenue_eq ratio_eq
  sorry

end canoe_rental_cost_l130_130834


namespace ice_cream_cost_l130_130850

theorem ice_cream_cost
  (num_pennies : ℕ) (num_nickels : ℕ) (num_dimes : ℕ) (num_quarters : ℕ) 
  (leftover_cents : ℤ) (num_family_members : ℕ)
  (h_pennies : num_pennies = 123)
  (h_nickels : num_nickels = 85)
  (h_dimes : num_dimes = 35)
  (h_quarters : num_quarters = 26)
  (h_leftover : leftover_cents = 48)
  (h_members : num_family_members = 5) :
  (123 * 0.01 + 85 * 0.05 + 35 * 0.1 + 26 * 0.25 - 0.48) / 5 = 3 :=
by
  sorry

end ice_cream_cost_l130_130850


namespace inequality_implication_l130_130178

theorem inequality_implication (x : ℝ) : 3 * x + 4 < 5 * x - 6 → x > 5 := 
by {
  sorry
}

end inequality_implication_l130_130178


namespace period_of_f_l130_130435

noncomputable def f (x : ℝ) : ℝ := sorry

theorem period_of_f (a : ℝ) (h : a ≠ 0) (H : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * |a| :=
by
  sorry

end period_of_f_l130_130435


namespace kate_money_ratio_l130_130312

-- Define the cost of the pen and the amount Kate needs
def pen_cost : ℕ := 30
def additional_money_needed : ℕ := 20

-- Define the amount of money Kate has
def kate_savings : ℕ := pen_cost - additional_money_needed

-- Define the ratio of Kate's money to the cost of the pen
def ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- The target property: the ratio of Kate's savings to the cost of the pen
theorem kate_money_ratio : ratio kate_savings pen_cost = (1, 3) :=
by
  sorry

end kate_money_ratio_l130_130312


namespace find_a_l130_130945

def ellipse1 (a : ℝ) : Prop := ∀ x y : ℝ, x^2 / a^2 + y^2 = 1
def ellipse2 : Prop := ∀ x y : ℝ, x^2 / 4 + y^2 = 1

def eccentricity (a b c : ℝ) : ℝ := c / a

def eccentricity_relation (e1 e2 : ℝ) : Prop := e2 = real.sqrt 3 * e1

theorem find_a (a e1 e2 : ℝ) 
  (h_cond1 : 1 < a)
  (h_cond2 : ellipse1 a)
  (h_cond3 : ellipse2)
  (h_e2 : e2 = real.sqrt 3 * e1)
  (h_e1 : e1 = 1 / 2)  
  (h_e2_def : e2 = eccentricity 2 1 (real.sqrt (4 - 1))) : 
  a = 2 * real.sqrt 3 / 3 :=
sorry

end find_a_l130_130945


namespace xy_series_16_l130_130867

noncomputable def series (x y : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * (x * y)^n

theorem xy_series_16 (x y : ℝ) (h_series : series x y = 16) (h_abs : |x * y| < 1) :
  (x = 3 / 4 ∧ (y = 1 ∨ y = -1)) :=
sorry

end xy_series_16_l130_130867
