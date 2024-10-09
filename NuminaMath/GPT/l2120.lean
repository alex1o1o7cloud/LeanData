import Mathlib

namespace min_tangent_length_l2120_212019

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end min_tangent_length_l2120_212019


namespace most_probable_standard_parts_in_batch_l2120_212055

theorem most_probable_standard_parts_in_batch :
  let q := 0.075
  let p := 1 - q
  let n := 39
  ∃ k₀ : ℤ, 36 ≤ k₀ ∧ k₀ ≤ 37 := 
by
  sorry

end most_probable_standard_parts_in_batch_l2120_212055


namespace min_value_inequality_l2120_212021

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  x^2 + 4 * x * y + 9 * y^2 + 6 * y * z + 8 * z^2 + 3 * x * w + 4 * w^2

theorem min_value_inequality 
  (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_prod : x * y * z * w = 3) : 
  min_value x y z w ≥ 81.25 := 
sorry

end min_value_inequality_l2120_212021


namespace negation_of_p_l2120_212046

-- Given conditions
def p : Prop := ∃ x : ℝ, x^2 + 3 * x = 4

-- The proof problem to be solved 
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^2 + 3 * x ≠ 4 := by
  sorry

end negation_of_p_l2120_212046


namespace tennis_balls_ordered_l2120_212010

def original_white_balls : ℕ := sorry
def original_yellow_balls_with_error : ℕ := sorry

theorem tennis_balls_ordered 
  (W Y : ℕ)
  (h1 : W = Y)
  (h2 : Y + 70 = original_yellow_balls_with_error)
  (h3 : W = 8 / 13 * (Y + 70)):
  W + Y = 224 := sorry

end tennis_balls_ordered_l2120_212010


namespace area_of_rectangular_field_l2120_212038

theorem area_of_rectangular_field (length width : ℝ) (h_length: length = 5.9) (h_width: width = 3) : 
  length * width = 17.7 := 
by
  sorry

end area_of_rectangular_field_l2120_212038


namespace certain_number_is_10000_l2120_212025

theorem certain_number_is_10000 (n : ℕ) (h1 : n - 999 = 9001) : n = 10000 :=
by
  sorry

end certain_number_is_10000_l2120_212025


namespace parallelepiped_length_l2120_212093

theorem parallelepiped_length (n : ℕ) :
  (n - 2) * (n - 4) * (n - 6) = 2 * n * (n - 2) * (n - 4) / 3 →
  n = 18 :=
by
  intros h
  sorry

end parallelepiped_length_l2120_212093


namespace john_pool_cleanings_per_month_l2120_212049

noncomputable def tip_percent : ℝ := 0.10
noncomputable def cost_per_cleaning : ℝ := 150
noncomputable def total_cost_per_cleaning : ℝ := cost_per_cleaning + (tip_percent * cost_per_cleaning)
noncomputable def chemical_cost_bi_monthly : ℝ := 200
noncomputable def monthly_chemical_cost : ℝ := 2 * chemical_cost_bi_monthly
noncomputable def total_monthly_pool_cost : ℝ := 2050
noncomputable def total_cleaning_cost : ℝ := total_monthly_pool_cost - monthly_chemical_cost

theorem john_pool_cleanings_per_month : total_cleaning_cost / total_cost_per_cleaning = 10 := by
  sorry

end john_pool_cleanings_per_month_l2120_212049


namespace rectangle_area_from_diagonal_l2120_212085

theorem rectangle_area_from_diagonal (x : ℝ) (w : ℝ) (h_lw : 3 * w = 3 * w) (h_diag : x^2 = 10 * w^2) : 
    (3 * w^2 = (3 / 10) * x^2) :=
by 
sorry

end rectangle_area_from_diagonal_l2120_212085


namespace product_is_correct_l2120_212072

-- Define the variables and conditions
variables {a b c d : ℚ}

-- State the conditions
def conditions (a b c d : ℚ) :=
  3 * a + 2 * b + 4 * c + 6 * d = 36 ∧
  4 * (d + c) = b ∧
  4 * b + 2 * c = a ∧
  c - 2 = d

-- The theorem statement
theorem product_is_correct (a b c d : ℚ) (h : conditions a b c d) :
  a * b * c * d = -315 / 32 :=
sorry

end product_is_correct_l2120_212072


namespace value_of_a_l2120_212005

theorem value_of_a (a : ℕ) (A_a B_a : ℕ)
  (h1 : A_a = 10)
  (h2 : B_a = 11)
  (h3 : 2 * a^2 + 10 * a + 3 + 5 * a^2 + 7 * a + 8 = 8 * a^2 + 4 * a + 11) :
  a = 13 :=
sorry

end value_of_a_l2120_212005


namespace train_speed_conversion_l2120_212043

def km_per_hour_to_m_per_s (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

theorem train_speed_conversion (speed_kmph : ℕ) (h : speed_kmph = 108) :
  km_per_hour_to_m_per_s speed_kmph = 30 :=
by
  rw [h]
  sorry

end train_speed_conversion_l2120_212043


namespace compute_P_part_l2120_212080

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem compute_P_part (a b c d : ℝ) 
  (H1 : P 1 a b c d = 1993) 
  (H2 : P 2 a b c d = 3986) 
  (H3 : P 3 a b c d = 5979) : 
  (1 / 4) * (P 11 a b c d + P (-7) a b c d) = 4693 :=
by
  sorry

end compute_P_part_l2120_212080


namespace find_equation_of_line_l_l2120_212023

-- Define the conditions
def point_P : ℝ × ℝ := (2, 3)

noncomputable def angle_of_inclination : ℝ := 2 * Real.pi / 3

def intercept_condition (a b : ℝ) : Prop := a + b = 0

-- The proof statement
theorem find_equation_of_line_l :
  ∃ (k : ℝ), k = Real.tan angle_of_inclination ∧
  ∃ (C : ℝ), ∀ (x y : ℝ), (y - 3 = k * (x - 2)) ∧ C = (3 + 2 * (Real.sqrt 3)) ∨ 
             (intercept_condition (x / point_P.1) (y / point_P.2) ∧ C = 1) ∨ 
             -- The standard forms of the line equation
             ((Real.sqrt 3 * x + y - C = 0) ∨ (x - y + 1 = 0)) :=
sorry

end find_equation_of_line_l_l2120_212023


namespace graph_representation_l2120_212017

theorem graph_representation {x y : ℝ} (h : x^2 * (x - y - 2) = y^2 * (x - y - 2)) :
  ( ∃ a : ℝ, ∀ (x : ℝ), y = a * x ) ∨ 
  ( ∃ b : ℝ, ∀ (x : ℝ), y = b * x ) ∨ 
  ( ∃ c : ℝ, ∀ (x : ℝ), y = x - 2 ) ∧ 
  (¬ ∃ d : ℝ, ∀ (x : ℝ), y = d * x ∧ y = d * x - 2) :=
sorry

end graph_representation_l2120_212017


namespace common_ratio_of_geo_seq_l2120_212035

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem common_ratio_of_geo_seq :
  (∀ n, 0 < a n) →
  geometric_sequence a q →
  a 6 = a 5 + 2 * a 4 →
  q = 2 :=
by
  intros
  sorry

end common_ratio_of_geo_seq_l2120_212035


namespace sale_in_first_month_is_5420_l2120_212013

-- Definitions of the sales in months 2 to 6
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month4 : ℕ := 6350
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470

-- Definition of the average sale goal
def average_sale_goal : ℕ := 6100

-- Calculating the total needed sales to achieve the average sale goal
def total_required_sales := 6 * average_sale_goal

-- Known sales for months 2 to 6
def known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6

-- Definition of the sale in the first month
def sale_month1 := total_required_sales - known_sales

-- The proof statement that the sale in the first month is 5420
theorem sale_in_first_month_is_5420 : sale_month1 = 5420 := by
  sorry

end sale_in_first_month_is_5420_l2120_212013


namespace negation_proposition_l2120_212054

theorem negation_proposition :
  ¬(∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_proposition_l2120_212054


namespace find_digit_B_l2120_212051

theorem find_digit_B (A B : ℕ) (h1 : A3B = 100 * A + 30 + B)
  (h2 : 0 ≤ A ∧ A ≤ 9)
  (h3 : 0 ≤ B ∧ B ≤ 9)
  (h4 : A3B - 41 = 591) : 
  B = 2 := 
by sorry

end find_digit_B_l2120_212051


namespace one_gt_one_others_lt_one_l2120_212036

theorem one_gt_one_others_lt_one 
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_prod : a * b * c = 1)
  (h_ineq : a + b + c > (1 / a) + (1 / b) + (1 / c)) :
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ (b > 1 ∧ a < 1 ∧ c < 1) ∨ (c > 1 ∧ a < 1 ∧ b < 1) :=
sorry

end one_gt_one_others_lt_one_l2120_212036


namespace triangle_base_length_l2120_212040

theorem triangle_base_length :
  ∀ (base height area : ℕ), height = 4 → area = 16 → area = (base * height) / 2 → base = 8 :=
by
  intros base height area h_height h_area h_formula
  sorry

end triangle_base_length_l2120_212040


namespace negation_equivalence_l2120_212024

theorem negation_equivalence : (¬ ∃ x : ℝ, x^3 - x^2 + 1 > 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) :=
  sorry

end negation_equivalence_l2120_212024


namespace probability_of_gui_field_in_za_field_l2120_212070

noncomputable def area_gui_field (base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * base * height

noncomputable def area_za_field (small_base large_base height : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (small_base + large_base) * height

theorem probability_of_gui_field_in_za_field :
  let b1 := 10
  let b2 := 20
  let h1 := 10
  let base_gui := 8
  let height_gui := 5
  let za_area := area_za_field b1 b2 h1
  let gui_area := area_gui_field base_gui height_gui
  (gui_area / za_area) = (2 / 15 : ℚ) := by
    sorry

end probability_of_gui_field_in_za_field_l2120_212070


namespace tangent_ellipse_hyperbola_l2120_212066

theorem tangent_ellipse_hyperbola (n : ℝ) :
  (∀ x y : ℝ, x^2 + 9 * y^2 = 9 ↔ x^2 - n * (y - 1)^2 = 4) →
  n = 9 / 5 :=
by sorry

end tangent_ellipse_hyperbola_l2120_212066


namespace minimum_value_of_fractions_l2120_212084

theorem minimum_value_of_fractions (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) : 
  ∃ a b, (0 < a) ∧ (0 < b) ∧ (1 / a + 1 / b = 1) ∧ (∃ t, ∀ x y, (0 < x) ∧ (0 < y) ∧ (1 / x + 1 / y = 1) -> t = (1 / (x - 1) + 4 / (y - 1))) := 
sorry

end minimum_value_of_fractions_l2120_212084


namespace nested_inverse_value_l2120_212063

def f (x : ℝ) : ℝ := 5 * x + 6

noncomputable def f_inv (y : ℝ) : ℝ := (y - 6) / 5

theorem nested_inverse_value :
  f_inv (f_inv 16) = -4/5 :=
by
  sorry

end nested_inverse_value_l2120_212063


namespace max_blue_points_l2120_212074

-- We define the number of spheres and the categorization of red and green spheres
def number_of_spheres : ℕ := 2016

-- Definition of the number of red spheres
def red_spheres (r : ℕ) : Prop := r <= number_of_spheres

-- Definition of the number of green spheres as the complement of red spheres
def green_spheres (r : ℕ) : ℕ := number_of_spheres - r

-- Definition of the number of blue points as the intersection of red and green spheres
def blue_points (r : ℕ) : ℕ := r * green_spheres r

-- Theorem: Given the conditions, the maximum number of blue points is 1016064
theorem max_blue_points : ∃ r : ℕ, red_spheres r ∧ blue_points r = 1016064 := by
  sorry

end max_blue_points_l2120_212074


namespace right_triangle_area_eq_8_over_3_l2120_212068

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end right_triangle_area_eq_8_over_3_l2120_212068


namespace jacqueline_erasers_l2120_212004

def num_boxes : ℕ := 4
def erasers_per_box : ℕ := 10
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end jacqueline_erasers_l2120_212004


namespace theta_third_quadrant_l2120_212077

theorem theta_third_quadrant (θ : ℝ) (h1 : Real.sin θ < 0) (h2 : Real.tan θ > 0) : 
  π < θ ∧ θ < 3 * π / 2 :=
by 
  sorry

end theta_third_quadrant_l2120_212077


namespace find_integer_l2120_212088

theorem find_integer (x : ℕ) (h : (4 * x) ^ 2 - 3 * x = 1764) : x = 18 := 
by 
  sorry

end find_integer_l2120_212088


namespace daily_wage_of_c_l2120_212009

-- Define the conditions
variables (a b c : ℝ)
variables (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5)
variables (h_days : 6 * a + 9 * b + 4 * c = 1702)

-- Define the proof problem; to prove c = 115
theorem daily_wage_of_c (h_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) (h_days : 6 * a + 9 * b + 4 * c = 1702) : 
  c = 115 :=
sorry

end daily_wage_of_c_l2120_212009


namespace sin_c_eq_tan_b_find_side_length_c_l2120_212094

-- (1) Prove that sinC = tanB
theorem sin_c_eq_tan_b {a b c : ℝ} {C : ℝ} (h1 : a / b = 1 + Real.cos C) : 
  Real.sin C = Real.tan B := by
  sorry

-- (2) If given conditions, find the value of c
theorem find_side_length_c {a b c : ℝ} {B C : ℝ} 
  (h1 : Real.cos B = 2 * Real.sqrt 7 / 7)
  (h2 : 0 < C ∧ C < Real.pi / 2)
  (h3 : 1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) 
  : c = Real.sqrt 7 := by
  sorry

end sin_c_eq_tan_b_find_side_length_c_l2120_212094


namespace longest_side_of_triangle_l2120_212090

variable (x y : ℝ)

def side1 := 10
def side2 := 2*y + 3
def side3 := 3*x + 2

theorem longest_side_of_triangle
  (h_perimeter : side1 + side2 + side3 = 45)
  (h_side2_pos : side2 > 0)
  (h_side3_pos : side3 > 0) :
  side3 = 32 :=
sorry

end longest_side_of_triangle_l2120_212090


namespace find_two_digit_number_with_cubic_ending_in_9_l2120_212008

theorem find_two_digit_number_with_cubic_ending_in_9:
  ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ n^3 % 10 = 9 ∧ n = 19 := 
by
  sorry

end find_two_digit_number_with_cubic_ending_in_9_l2120_212008


namespace factorization_a_minus_b_l2120_212029

theorem factorization_a_minus_b (a b : ℤ) (y : ℝ) 
  (h1 : 3 * y ^ 2 - 7 * y - 6 = (3 * y + a) * (y + b)) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) : 
  a - b = 5 :=
sorry

end factorization_a_minus_b_l2120_212029


namespace inequality_proof_l2120_212065

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := 
sorry

end inequality_proof_l2120_212065


namespace intersection_of_A_and_B_l2120_212061

-- Definitions based on conditions
def A : Set ℝ := { x | x + 2 = 0 }
def B : Set ℝ := { x | x^2 - 4 = 0 }

-- Theorem statement proving the question == answer given conditions
theorem intersection_of_A_and_B : A ∩ B = {-2} :=
by 
  sorry

end intersection_of_A_and_B_l2120_212061


namespace quadratic_root_unique_l2120_212027

theorem quadratic_root_unique 
  (a b c : ℝ)
  (hf1 : b^2 - 4 * a * c = 0)
  (hf2 : (b - 30 * a)^2 - 4 * a * (17 * a - 7 * b + c) = 0)
  (ha_pos : a ≠ 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ x = -11 := 
by
  sorry

end quadratic_root_unique_l2120_212027


namespace city_raised_money_for_charity_l2120_212081

-- Definitions based on conditions from part a)
def price_regular_duck : ℝ := 3.0
def price_large_duck : ℝ := 5.0
def number_regular_ducks_sold : ℕ := 221
def number_large_ducks_sold : ℕ := 185

-- Definition to represent the main theorem: Total money raised
noncomputable def total_money_raised : ℝ :=
  price_regular_duck * number_regular_ducks_sold + price_large_duck * number_large_ducks_sold

-- Theorem to prove that the total money raised is $1588.00
theorem city_raised_money_for_charity : total_money_raised = 1588.0 := by
  sorry

end city_raised_money_for_charity_l2120_212081


namespace train_bus_difference_l2120_212001

variable (T : ℝ)  -- T is the cost of a train ride

-- conditions
def cond1 := T + 1.50 = 9.85
def cond2 := 1.50 = 1.50

theorem train_bus_difference (h1 : cond1 T) (h2 : cond2) : T - 1.50 = 6.85 := 
sorry

end train_bus_difference_l2120_212001


namespace prime_divides_diff_of_cubes_l2120_212041

theorem prime_divides_diff_of_cubes (a b c : ℕ) [Fact (Nat.Prime a)] [Fact (Nat.Prime b)]
  (h1 : c ∣ (a + b)) (h2 : c ∣ (a * b)) : c ∣ (a^3 - b^3) :=
by
  sorry

end prime_divides_diff_of_cubes_l2120_212041


namespace sphere_volume_of_hexagonal_prism_l2120_212034

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

end sphere_volume_of_hexagonal_prism_l2120_212034


namespace candy_bar_split_l2120_212099
noncomputable def split (total: ℝ) (people: ℝ): ℝ := total / people

theorem candy_bar_split: split 5.0 3.0 = 1.67 :=
by
  sorry

end candy_bar_split_l2120_212099


namespace max_sum_of_arithmetic_sequence_l2120_212037

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (a1 : a 1 = 29) 
  (S10_eq_S20 : S 10 = S 20) :
  (∃ n, ∀ m, S n ≥ S m) ∧ ∃ n, (S n = S 15) :=
sorry

end max_sum_of_arithmetic_sequence_l2120_212037


namespace rectangle_perimeter_l2120_212064

theorem rectangle_perimeter 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℕ)
  (relatively_prime : Nat.gcd (a_4 + a_7 + a_9) (a_2 + a_8 + a_6) = 1)
  (h1 : a_1 + a_2 = a_4)
  (h2 : a_1 + a_4 = a_5)
  (h3 : a_4 + a_5 = a_7)
  (h4 : a_5 + a_7 = a_9)
  (h5 : a_2 + a_4 + a_7 = a_8)
  (h6 : a_2 + a_8 = a_6)
  (h7 : a_1 + a_5 + a_9 = a_3)
  (h8 : a_3 + a_6 = a_8 + a_7) :
  2 * ((a_4 + a_7 + a_9) + (a_2 + a_8 + a_6)) = 164 := 
sorry -- proof omitted

end rectangle_perimeter_l2120_212064


namespace vowel_initial_probability_is_correct_l2120_212045

-- Given conditions as definitions
def total_students : ℕ := 34
def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def vowels_count_per_vowel : ℕ := 2
def total_vowels_count := vowels.length * vowels_count_per_vowel

-- The probabilistic statement we want to prove
def vowel_probability : ℚ := total_vowels_count / total_students

-- The final statement to prove
theorem vowel_initial_probability_is_correct :
  vowel_probability = 6 / 17 :=
by
  unfold vowel_probability total_vowels_count
  -- Simplification to verify our statement.
  sorry

end vowel_initial_probability_is_correct_l2120_212045


namespace tens_digit_36_pow_12_l2120_212032

theorem tens_digit_36_pow_12 : ((36 ^ 12) % 100) / 10 % 10 = 1 := 
by 
sorry

end tens_digit_36_pow_12_l2120_212032


namespace sum_not_prime_30_l2120_212089

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_not_prime_30 (p1 p2 : ℕ) (hp1 : is_prime p1) (hp2 : is_prime p2) (h : p1 + p2 = 30) : false :=
sorry

end sum_not_prime_30_l2120_212089


namespace min_value_xyz_l2120_212044

-- Definition of the problem
theorem min_value_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 108):
  x^2 + 9 * x * y + 9 * y^2 + 3 * z^2 ≥ 324 :=
sorry

end min_value_xyz_l2120_212044


namespace opposite_of_neg_two_l2120_212050

theorem opposite_of_neg_two : -(-2) = 2 := 
by
  sorry

end opposite_of_neg_two_l2120_212050


namespace standard_circle_equation_passing_through_P_l2120_212028

-- Define the condition that a point P is a solution to the system of equations derived from the line
def PointPCondition (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1 = 0) ∧ (3 * x - 2 * y + 5 = 0)

-- Define the center and radius of the given circle C
def CenterCircleC : ℝ × ℝ := (2, -3)
def RadiusCircleC : ℝ := 4  -- Since the radius squared is 16

-- Define the condition that a point is on a circle with a given center and radius
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.fst)^2 + (y + center.snd)^2 = radius^2

-- State the problem
theorem standard_circle_equation_passing_through_P :
  ∃ (x y : ℝ), PointPCondition x y ∧ OnCircle CenterCircleC 5 x y :=
sorry

end standard_circle_equation_passing_through_P_l2120_212028


namespace divided_number_l2120_212073

theorem divided_number (x y : ℕ) (h1 : 7 * x + 5 * y = 146) (h2 : y = 11) : x + y = 24 :=
sorry

end divided_number_l2120_212073


namespace gcd_lcm_product_l2120_212020

theorem gcd_lcm_product (a b : ℕ) (ha : a = 100) (hb : b = 120) :
  Nat.gcd a b * Nat.lcm a b = 12000 := by
  sorry

end gcd_lcm_product_l2120_212020


namespace sum_angles_star_l2120_212026

theorem sum_angles_star (β : ℝ) (h : β = 90) : 
  8 * β = 720 :=
by
  sorry

end sum_angles_star_l2120_212026


namespace fraction_to_decimal_l2120_212083

theorem fraction_to_decimal :
  (45 : ℚ) / (5 ^ 3) = 0.360 :=
by
  sorry

end fraction_to_decimal_l2120_212083


namespace number_of_packs_l2120_212042

theorem number_of_packs (total_towels towels_per_pack : ℕ) (h1 : total_towels = 27) (h2 : towels_per_pack = 3) :
  total_towels / towels_per_pack = 9 :=
by
  sorry

end number_of_packs_l2120_212042


namespace simultaneous_messengers_l2120_212052

theorem simultaneous_messengers (m n : ℕ) (h : m * n = 2010) : 
  m ≠ n → ((m, n) = (1, 2010) ∨ (m, n) = (2, 1005) ∨ (m, n) = (3, 670) ∨ 
          (m, n) = (5, 402) ∨ (m, n) = (6, 335) ∨ (m, n) = (10, 201) ∨ 
          (m, n) = (15, 134) ∨ (m, n) = (30, 67)) :=
sorry

end simultaneous_messengers_l2120_212052


namespace exponential_increasing_l2120_212076

theorem exponential_increasing (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) :
  (∀ x y : ℝ, x < y → a^x < a^y) ↔ a > 1 :=
by
  sorry

end exponential_increasing_l2120_212076


namespace left_square_side_length_l2120_212095

theorem left_square_side_length (x : ℕ) (h1 : ∀ y : ℕ, y = x + 17)
                                (h2 : ∀ z : ℕ, z = x + 11)
                                (h3 : 3 * x + 28 = 52) : x = 8 :=
by
  sorry

end left_square_side_length_l2120_212095


namespace average_star_rating_l2120_212091

/-- Define specific constants for the problem. --/
def reviews_5_star := 6
def reviews_4_star := 7
def reviews_3_star := 4
def reviews_2_star := 1
def total_reviews := 18

/-- Calculate the total stars given the number of each type of review. --/
def total_stars : ℕ := 
  (reviews_5_star * 5) + 
  (reviews_4_star * 4) + 
  (reviews_3_star * 3) + 
  (reviews_2_star * 2)

/-- Prove that the average star rating is 4. --/
theorem average_star_rating : total_stars / total_reviews = 4 := by 
  sorry

end average_star_rating_l2120_212091


namespace sum_of_squares_due_to_regression_eq_72_l2120_212060

theorem sum_of_squares_due_to_regression_eq_72
    (total_squared_deviations : ℝ)
    (correlation_coefficient : ℝ)
    (h1 : total_squared_deviations = 120)
    (h2 : correlation_coefficient = 0.6)
    : total_squared_deviations * correlation_coefficient^2 = 72 :=
by
  -- Proof goes here
  sorry

end sum_of_squares_due_to_regression_eq_72_l2120_212060


namespace custom_deck_card_selection_l2120_212012

theorem custom_deck_card_selection :
  let cards := 60
  let suits := 4
  let cards_per_suit := 15
  let red_suits := 2
  let black_suits := 2
  -- Total number of ways to pick two cards with the second of a different color
  ∃ (ways : ℕ), ways = 60 * 30 ∧ ways = 1800 := by
  sorry

end custom_deck_card_selection_l2120_212012


namespace isosceles_trapezoid_ratio_ab_cd_l2120_212014

theorem isosceles_trapezoid_ratio_ab_cd (AB CD : ℝ) (P : ℝ → ℝ → Prop)
  (area1 area2 area3 area4 : ℝ)
  (h1 : AB > CD)
  (h2 : area1 = 5)
  (h3 : area2 = 7)
  (h4 : area3 = 3)
  (h5 : area4 = 9) :
  AB / CD = 1 + 2 * Real.sqrt 2 :=
sorry

end isosceles_trapezoid_ratio_ab_cd_l2120_212014


namespace find_b6b8_l2120_212062

-- Define sequences {a_n} (arithmetic sequence) and {b_n} (geometric sequence)
variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Given conditions
axiom h1 : ∀ n m : ℕ, a m = a n + (m - n) * (a (n + 1) - a n) -- Arithmetic sequence property
axiom h2 : 2 * a 3 - (a 7) ^ 2 + 2 * a 11 = 0
axiom h3 : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1 -- Geometric sequence property
axiom h4 : b 7 = a 7
axiom h5 : ∀ n : ℕ, b n > 0                 -- Assuming b_n has positive terms
axiom h6 : ∀ n : ℕ, a n > 0                 -- Positive terms in sequence a_n

-- Proof objective
theorem find_b6b8 : b 6 * b 8 = 16 :=
by sorry

end find_b6b8_l2120_212062


namespace smallest_AAB_value_exists_l2120_212039

def is_consecutive_digits (A B : ℕ) : Prop :=
  (B = A + 1 ∨ A = B + 1) ∧ 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9

def two_digit_to_int (A B : ℕ) : ℕ :=
  10 * A + B

def three_digit_to_int (A B : ℕ) : ℕ :=
  110 * A + B

theorem smallest_AAB_value_exists :
  ∃ (A B: ℕ), is_consecutive_digits A B ∧ two_digit_to_int A B = (1 / 7 : ℝ) * ↑(three_digit_to_int A B) ∧ three_digit_to_int A B = 889 :=
sorry

end smallest_AAB_value_exists_l2120_212039


namespace average_speed_palindrome_l2120_212096

open Nat

theorem average_speed_palindrome :
  ∀ (initial final : ℕ) (time : ℕ), (initial = 12321) →
    (final = 12421) →
    (time = 3) →
    (∃ speed : ℚ, speed = (final - initial) / time ∧ speed = 33.33) :=
by
  intros initial final time h_initial h_final h_time
  sorry

end average_speed_palindrome_l2120_212096


namespace expected_attempts_for_10_suitcases_l2120_212097

noncomputable def expected_attempts (n : ℕ) : ℝ :=
  (1 / 2) * (n * (n + 1) / 2) + (n / 2) - (Real.log n + 0.577)

theorem expected_attempts_for_10_suitcases :
  abs (expected_attempts 10 - 29.62) < 1 :=
by
  sorry

end expected_attempts_for_10_suitcases_l2120_212097


namespace smallest_base_for_100_l2120_212011

theorem smallest_base_for_100 :
  ∃ b : ℕ, b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
sorry

end smallest_base_for_100_l2120_212011


namespace find_a_of_inequality_solution_l2120_212048

theorem find_a_of_inequality_solution (a : ℝ) :
  (∀ x : ℝ, -3 < ax - 2 ∧ ax - 2 < 3 ↔ -5/3 < x ∧ x < 1/3) →
  a = -3 := by
  sorry

end find_a_of_inequality_solution_l2120_212048


namespace product_of_abcd_l2120_212092

theorem product_of_abcd :
  ∃ (a b c d : ℚ), 
    3 * a + 4 * b + 6 * c + 8 * d = 42 ∧ 
    4 * (d + c) = b ∧ 
    4 * b + 2 * c = a ∧ 
    c - 2 = d ∧ 
    a * b * c * d = (367 * 76 * 93 * -55) / (37^2 * 74^2) :=
sorry

end product_of_abcd_l2120_212092


namespace exists_smaller_circle_with_at_least_as_many_lattice_points_l2120_212030

theorem exists_smaller_circle_with_at_least_as_many_lattice_points
  (R : ℝ) (hR : 0 < R) :
  ∃ R' : ℝ, (R' < R) ∧ (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 → ∃ (x' y' : ℤ), (x')^2 + (y')^2 ≤ (R')^2) := sorry

end exists_smaller_circle_with_at_least_as_many_lattice_points_l2120_212030


namespace equivalent_fraction_l2120_212098

theorem equivalent_fraction :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) :=
by 
  sorry

end equivalent_fraction_l2120_212098


namespace no_such_m_exists_l2120_212071

theorem no_such_m_exists : ¬ ∃ m : ℝ, ∀ x : ℝ, m * x^2 - 2 * x - m + 1 < 0 :=
sorry

end no_such_m_exists_l2120_212071


namespace number_of_lines_intersecting_circle_l2120_212007

theorem number_of_lines_intersecting_circle : 
  ∃ l : ℕ, 
  (∀ a b x y : ℤ, (x^2 + y^2 = 50 ∧ (x / a + y / b = 1))) → 
  (∃ n : ℕ, n = 60) :=
sorry

end number_of_lines_intersecting_circle_l2120_212007


namespace books_sold_in_february_l2120_212059

theorem books_sold_in_february (F : ℕ) 
  (h_avg : (15 + F + 17) / 3 = 16): 
  F = 16 := 
by 
  sorry

end books_sold_in_february_l2120_212059


namespace salad_dressing_oil_percentage_l2120_212022

theorem salad_dressing_oil_percentage 
  (vinegar_P : ℝ) (vinegar_Q : ℝ) (oil_Q : ℝ)
  (new_vinegar : ℝ) (proportion_P : ℝ) :
  vinegar_P = 0.30 ∧ vinegar_Q = 0.10 ∧ oil_Q = 0.90 ∧ new_vinegar = 0.12 ∧ proportion_P = 0.10 →
  (1 - vinegar_P) = 0.70 :=
by
  intro h
  sorry

end salad_dressing_oil_percentage_l2120_212022


namespace count_difference_l2120_212087

-- Given definitions
def count_six_digit_numbers_in_ascending_order_by_digits : ℕ := by
  -- Calculation using binomial coefficient
  exact Nat.choose 9 6

def count_six_digit_numbers_with_one : ℕ := by
  -- Calculation using binomial coefficient with fixed '1' in one position
  exact Nat.choose 8 5

def count_six_digit_numbers_without_one : ℕ := by
  -- Calculation subtracting with and without 1
  exact count_six_digit_numbers_in_ascending_order_by_digits - count_six_digit_numbers_with_one

-- Theorem to prove
theorem count_difference : 
  count_six_digit_numbers_with_one - count_six_digit_numbers_without_one = 28 :=
by
  sorry

end count_difference_l2120_212087


namespace perfect_square_pattern_l2120_212082

theorem perfect_square_pattern {a b n : ℕ} (h₁ : 1000 ≤ n ∧ n ≤ 9999)
  (h₂ : n = (10 * a + b) ^ 2)
  (h₃ : n = 1100 * a + 11 * b) : n = 7744 :=
  sorry

end perfect_square_pattern_l2120_212082


namespace smallest_k_for_divisibility_l2120_212067

theorem smallest_k_for_divisibility : (∃ k : ℕ, ∀ z : ℂ, z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^k - 1 ∧ (∀ m : ℕ, m < k → ∃ z : ℂ, ¬(z^8 + z^7 + z^4 + z^3 + z^2 + z + 1 ∣ z^m - 1))) ↔ k = 14 := sorry

end smallest_k_for_divisibility_l2120_212067


namespace cookies_per_tray_l2120_212086

def num_trays : ℕ := 4
def num_packs : ℕ := 8
def cookies_per_pack : ℕ := 12
def total_cookies : ℕ := num_packs * cookies_per_pack

theorem cookies_per_tray : total_cookies / num_trays = 24 := by
  sorry

end cookies_per_tray_l2120_212086


namespace chocolates_remaining_l2120_212047

theorem chocolates_remaining 
  (total_chocolates : ℕ)
  (ate_day1 : ℕ) (ate_day2 : ℕ) (ate_day3 : ℕ) (ate_day4 : ℕ) (ate_day5 : ℕ) (remaining_chocolates : ℕ) 
  (h_total : total_chocolates = 48)
  (h_day1 : ate_day1 = 6) 
  (h_day2 : ate_day2 = 2 * ate_day1 + 2) 
  (h_day3 : ate_day3 = ate_day1 - 3) 
  (h_day4 : ate_day4 = 2 * ate_day3 + 1) 
  (h_day5 : ate_day5 = ate_day2 / 2) 
  (h_rem : remaining_chocolates = total_chocolates - (ate_day1 + ate_day2 + ate_day3 + ate_day4 + ate_day5)) :
  remaining_chocolates = 14 :=
sorry

end chocolates_remaining_l2120_212047


namespace initial_pencils_count_l2120_212053

theorem initial_pencils_count (pencils_taken : ℕ) (pencils_left : ℕ) (h1 : pencils_taken = 4) (h2 : pencils_left = 75) : 
  pencils_left + pencils_taken = 79 :=
by
  sorry

end initial_pencils_count_l2120_212053


namespace students_enjoy_both_music_and_sports_l2120_212069

theorem students_enjoy_both_music_and_sports :
  ∀ (T M S N B : ℕ), T = 55 → M = 35 → S = 45 → N = 4 → B = M + S - (T - N) → B = 29 :=
by
  intros T M S N B hT hM hS hN hB
  rw [hT, hM, hS, hN] at hB
  exact hB

end students_enjoy_both_music_and_sports_l2120_212069


namespace total_sum_vowels_l2120_212058

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end total_sum_vowels_l2120_212058


namespace problem_statement_l2120_212000

def setS : Set (ℝ × ℝ) := {p | p.1 * p.2 > 0}
def setT : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0}

theorem problem_statement : setS ∪ setT = setS ∧ setS ∩ setT = setT :=
by
  -- To be proved
  sorry

end problem_statement_l2120_212000


namespace domain_ln_l2120_212003

def domain_of_ln (x : ℝ) : Prop := x^2 - x > 0

theorem domain_ln (x : ℝ) :
  domain_of_ln x ↔ (x < 0 ∨ x > 1) :=
by sorry

end domain_ln_l2120_212003


namespace Mary_chewing_gums_count_l2120_212002

variable (Mary_gums Sam_gums Sue_gums : ℕ)

-- Define the given conditions
axiom Sam_chewing_gums : Sam_gums = 10
axiom Sue_chewing_gums : Sue_gums = 15
axiom Total_chewing_gums : Mary_gums + Sam_gums + Sue_gums = 30

theorem Mary_chewing_gums_count : Mary_gums = 5 := by
  sorry

end Mary_chewing_gums_count_l2120_212002


namespace registered_voters_democrats_l2120_212056

variables (D R : ℝ)

theorem registered_voters_democrats :
  (D + R = 100) →
  (0.80 * D + 0.30 * R = 65) →
  D = 70 :=
by
  intros h1 h2
  sorry

end registered_voters_democrats_l2120_212056


namespace two_numbers_product_l2120_212075

theorem two_numbers_product (x y : ℕ) 
  (h1 : x + y = 90) 
  (h2 : x - y = 10) : x * y = 2000 :=
by
  sorry

end two_numbers_product_l2120_212075


namespace grooming_time_5_dogs_3_cats_l2120_212031

theorem grooming_time_5_dogs_3_cats :
  (2.5 * 5 + 0.5 * 3) * 60 = 840 :=
by
  -- Prove that grooming 5 dogs and 3 cats takes 840 minutes.
  sorry

end grooming_time_5_dogs_3_cats_l2120_212031


namespace find_other_number_l2120_212033

theorem find_other_number (w : ℕ) (x : ℕ) 
    (h1 : w = 468)
    (h2 : x * w = 2^4 * 3^3 * 13^3) 
    : x = 2028 :=
by
  sorry

end find_other_number_l2120_212033


namespace max_marks_l2120_212016

theorem max_marks (M: ℝ) (h1: 0.95 * M = 285):
  M = 300 :=
by
  sorry

end max_marks_l2120_212016


namespace cosine_lt_sine_neg_four_l2120_212018

theorem cosine_lt_sine_neg_four : ∀ (m n : ℝ), m = Real.cos (-4) → n = Real.sin (-4) → m < n :=
by
  intros m n hm hn
  rw [hm, hn]
  sorry

end cosine_lt_sine_neg_four_l2120_212018


namespace mn_eq_one_l2120_212057

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end mn_eq_one_l2120_212057


namespace circle_equation_passing_through_P_l2120_212078

-- Define the problem conditions
def P : ℝ × ℝ := (3, 1)
def l₁ (x y : ℝ) := x + 2 * y + 3 = 0
def l₂ (x y : ℝ) := x + 2 * y - 7 = 0

-- The main theorem statement
theorem circle_equation_passing_through_P :
  ∃ (α β : ℝ), 
    ((α = 4 ∧ β = -1) ∨ (α = 4 / 5 ∧ β = 3 / 5)) ∧ 
    ((x - α)^2 + (y - β)^2 = 5) :=
  sorry

end circle_equation_passing_through_P_l2120_212078


namespace tim_runs_more_than_sarah_l2120_212015

-- Definitions based on the conditions
def street_width : ℕ := 25
def side_length : ℕ := 450

-- Perimeters of the paths
def sarah_perimeter : ℕ := 4 * side_length
def tim_perimeter : ℕ := 4 * (side_length + 2 * street_width)

-- The theorem to prove
theorem tim_runs_more_than_sarah : tim_perimeter - sarah_perimeter = 200 := by
  -- The proof will be filled in here
  sorry

end tim_runs_more_than_sarah_l2120_212015


namespace min_exponent_binomial_l2120_212006

theorem min_exponent_binomial (n : ℕ) (h1 : n > 0)
  (h2 : ∃ r : ℕ, (n.choose r) / (n.choose (r + 1)) = 5 / 7) : n = 11 :=
by {
-- Note: We are merely stating the theorem here according to the instructions,
-- the proof body is omitted and hence the use of 'sorry'.
sorry
}

end min_exponent_binomial_l2120_212006


namespace total_travel_time_l2120_212079

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l2120_212079
