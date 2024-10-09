import Mathlib

namespace points_on_line_any_real_n_l2111_211113

theorem points_on_line_any_real_n (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 1 = 2 * (n + 0.5) + 5) : 
  True :=
by
  sorry

end points_on_line_any_real_n_l2111_211113


namespace min_value_of_y_l2111_211103

theorem min_value_of_y (x : ℝ) : ∃ x0 : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ (4 * x0^2 + 8 * x0 + 16 = 12) :=
sorry

end min_value_of_y_l2111_211103


namespace rectangular_plot_area_l2111_211105

-- Define the conditions
def breadth := 11  -- breadth in meters
def length := 3 * breadth  -- length is thrice the breadth

-- Define the function to calculate area
def area (length breadth : ℕ) := length * breadth

-- The theorem to prove
theorem rectangular_plot_area : area length breadth = 363 := by
  sorry

end rectangular_plot_area_l2111_211105


namespace abs_eq_5_iff_l2111_211102

theorem abs_eq_5_iff (a : ℝ) : |a| = 5 ↔ a = 5 ∨ a = -5 :=
by
  sorry

end abs_eq_5_iff_l2111_211102


namespace sufficient_not_necessary_l2111_211199

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → a^2 > 1) ∧ ¬(a^2 > 1 → a > 1) :=
by {
  sorry
}

end sufficient_not_necessary_l2111_211199


namespace middle_number_l2111_211146

theorem middle_number (x y z : ℤ) 
  (h1 : x + y = 21)
  (h2 : x + z = 25)
  (h3 : y + z = 28)
  (h4 : x < y)
  (h5 : y < z) : 
  y = 12 :=
sorry

end middle_number_l2111_211146


namespace limit_r_l2111_211166

noncomputable def L (m : ℝ) : ℝ := (m - Real.sqrt (m^2 + 24)) / 2

noncomputable def r (m : ℝ) : ℝ := (L (-m) - L m) / m

theorem limit_r (h : ∀ m : ℝ, m ≠ 0) : Filter.Tendsto r (nhds 0) (nhds (-1)) :=
sorry

end limit_r_l2111_211166


namespace bobs_password_probability_l2111_211133

theorem bobs_password_probability :
  (5 / 10) * (5 / 10) * 1 * (9 / 10) = 9 / 40 :=
by
  sorry

end bobs_password_probability_l2111_211133


namespace find_integer_pairs_l2111_211187

theorem find_integer_pairs (a b : ℤ) : 
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → d ∣ (a^n + b^n + 1)) → 
  (∃ k₁ k₂ : ℤ, ((a = 2 * k₁) ∧ (b = 2 * k₂ + 1)) ∨ ((a = 3 * k₁ + 1) ∧ (b = 3 * k₂ + 1))) :=
by
  sorry

end find_integer_pairs_l2111_211187


namespace problem_l2111_211107

theorem problem 
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2010)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2010)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2010)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 996 / 1005 :=
sorry

end problem_l2111_211107


namespace arithmetic_sequence_sum_condition_l2111_211164

variable (a : ℕ → ℤ)

theorem arithmetic_sequence_sum_condition (h1 : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) : 
  a 2 + a 10 = 120 :=
sorry

end arithmetic_sequence_sum_condition_l2111_211164


namespace max_marks_l2111_211134

theorem max_marks (M : ℝ) : 0.33 * M = 59 + 40 → M = 300 :=
by
  sorry

end max_marks_l2111_211134


namespace max_profit_at_35_l2111_211181

-- Define the conditions
def unit_purchase_price : ℝ := 20
def base_selling_price : ℝ := 30
def base_sales_volume : ℕ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease_per_dollar : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - unit_purchase_price) * (base_sales_volume - sales_volume_decrease_per_dollar * (x - base_selling_price))

-- Lean statement to prove that the selling price which maximizes the profit is 35
theorem max_profit_at_35 : ∃ x : ℝ, x = 35 ∧ ∀ y : ℝ, profit y ≤ profit 35 := 
  sorry

end max_profit_at_35_l2111_211181


namespace problem_1_problem_2_problem_3_problem_4_l2111_211135

-- Given conditions
variable {T : Type} -- Type representing teachers
variable {S : Type} -- Type representing students

def arrangements_ends (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_not_next_to_each_other (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

def arrangements_two_between (teachers : List T) (students : List S) : ℕ :=
  sorry -- Implementation skipped

-- Statements to prove

-- 1. Prove that if teachers A and B must stand at the two ends, there are 48 different arrangements
theorem problem_1 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_ends teachers students = 48 :=
  sorry

-- 2. Prove that if teachers A and B must stand next to each other, there are 240 different arrangements
theorem problem_2 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_next_to_each_other teachers students = 240 :=
  sorry 

-- 3. Prove that if teachers A and B cannot stand next to each other, there are 480 different arrangements
theorem problem_3 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_not_next_to_each_other teachers students = 480 :=
  sorry 

-- 4. Prove that if there must be two students standing between teachers A and B, there are 144 different arrangements
theorem problem_4 {teachers : List T} {students : List S} (h : teachers.length = 2) (s : students.length = 4) :
  arrangements_two_between teachers students = 144 :=
  sorry

end problem_1_problem_2_problem_3_problem_4_l2111_211135


namespace smallest_positive_and_largest_negative_l2111_211158

theorem smallest_positive_and_largest_negative:
  (∃ (a : ℤ), a > 0 ∧ ∀ (b : ℤ), b > 0 → b ≥ a ∧ a = 1) ∧
  (∃ (c : ℤ), c < 0 ∧ ∀ (d : ℤ), d < 0 → d ≤ c ∧ c = -1) :=
by
  sorry

end smallest_positive_and_largest_negative_l2111_211158


namespace initial_percentage_alcohol_l2111_211123

variables (P : ℝ) (initial_volume : ℝ) (added_volume : ℝ) (total_volume : ℝ) (final_percentage : ℝ) (init_percentage : ℝ)

theorem initial_percentage_alcohol (h1 : initial_volume = 6)
                                  (h2 : added_volume = 3)
                                  (h3 : total_volume = initial_volume + added_volume)
                                  (h4 : final_percentage = 50)
                                  (h5 : init_percentage = 100 * (initial_volume * P / 100 + added_volume) / total_volume)
                                  : P = 25 :=
by {
  sorry
}

end initial_percentage_alcohol_l2111_211123


namespace cos_C_in_acute_triangle_l2111_211176

theorem cos_C_in_acute_triangle 
  (a b c : ℝ) (A B C : ℝ) 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides_angles : a * Real.cos B = 4 * c * Real.sin C - b * Real.cos A) 
  : Real.cos C = Real.sqrt 15 / 4 := 
sorry

end cos_C_in_acute_triangle_l2111_211176


namespace maximal_difference_of_areas_l2111_211129

-- Given:
-- A circle of radius R
-- A chord of length 2x is drawn perpendicular to the diameter of the circle
-- The endpoints of this chord are connected to the endpoints of the diameter
-- We need to prove that under these conditions, the length of the chord 2x that maximizes the difference in areas of the triangles is R √ 2

theorem maximal_difference_of_areas (R x : ℝ) (h : 2 * x = R * Real.sqrt 2) :
  2 * x = R * Real.sqrt 2 :=
by
  sorry

end maximal_difference_of_areas_l2111_211129


namespace beetle_total_distance_l2111_211196

theorem beetle_total_distance (r : ℝ) (r_eq : r = 75) : (2 * r + r + r) = 300 := 
by
  sorry

end beetle_total_distance_l2111_211196


namespace carbon_neutrality_l2111_211121

theorem carbon_neutrality (a b : ℝ) (t : ℕ) (ha : a > 0)
  (h1 : S = a * b ^ t)
  (h2 : a * b ^ 7 = 4 * a / 5)
  (h3 : a / 4 = S) :
  t = 42 := 
sorry

end carbon_neutrality_l2111_211121


namespace cylinder_radius_in_cone_l2111_211125

-- Define the conditions
def cone_diameter := 18
def cone_height := 20
def cylinder_height_eq_diameter {r : ℝ} := 2 * r

-- Define the theorem to prove
theorem cylinder_radius_in_cone : ∃ r : ℝ, r = 90 / 19 ∧ (20 - 2 * r) / r = 20 / 9 :=
by
  sorry

end cylinder_radius_in_cone_l2111_211125


namespace daily_production_n_l2111_211131

theorem daily_production_n (n : ℕ) 
  (h1 : (60 * n) / n = 60)
  (h2 : (60 * n + 90) / (n + 1) = 65) : 
  n = 5 :=
by
  -- Proof goes here
  sorry

end daily_production_n_l2111_211131


namespace longest_side_length_l2111_211114

-- Define the sides of the triangle
def side_a : ℕ := 9
def side_b (x : ℕ) : ℕ := 2 * x + 3
def side_c (x : ℕ) : ℕ := 3 * x - 2

-- Define the perimeter condition
def perimeter_condition (x : ℕ) : Prop := side_a + side_b x + side_c x = 45

-- Main theorem statement: Length of the longest side is 19
theorem longest_side_length (x : ℕ) (h : perimeter_condition x) : side_b x = 19 ∨ side_c x = 19 :=
sorry

end longest_side_length_l2111_211114


namespace James_bought_3_CDs_l2111_211109

theorem James_bought_3_CDs :
  ∃ (cd1 cd2 cd3 : ℝ), cd1 = 1.5 ∧ cd2 = 1.5 ∧ cd3 = 2 * cd1 ∧ cd1 + cd2 + cd3 = 6 ∧ 3 = 3 :=
by
  sorry

end James_bought_3_CDs_l2111_211109


namespace area_of_shaded_trapezoid_l2111_211117

-- Definitions of conditions:
def side_lengths : List ℕ := [1, 3, 5, 7]
def total_base : ℕ := side_lengths.sum
def height_largest_square : ℕ := 7
def ratio : ℚ := height_largest_square / total_base

def height_at_end (n : ℕ) : ℚ := ratio * n
def lower_base_height : ℚ := height_at_end 4
def upper_base_height : ℚ := height_at_end 9
def trapezoid_height : ℕ := 2

-- Main theorem:
theorem area_of_shaded_trapezoid :
  (1 / 2) * (lower_base_height + upper_base_height) * trapezoid_height = 91 / 8 :=
by
  sorry

end area_of_shaded_trapezoid_l2111_211117


namespace solve_arcsin_arccos_l2111_211177

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end solve_arcsin_arccos_l2111_211177


namespace correct_mark_l2111_211153

theorem correct_mark (x : ℝ) (n : ℝ) (avg_increase : ℝ) :
  n = 40 → avg_increase = 1 / 2 → (83 - x) / n = avg_increase → x = 63 :=
by
  intros h1 h2 h3
  sorry

end correct_mark_l2111_211153


namespace find_x_l2111_211151

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (8, 1/2 * x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 1)

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : vector_a x = (8, 1/2 * x)) 
(h3 : vector_b x = (x, 1)) 
(h4 : ∀ k : ℝ, (vector_a x).1 = k * (vector_b x).1 ∧ 
                       (vector_a x).2 = k * (vector_b x).2) : 
                       x = 4 := sorry

end find_x_l2111_211151


namespace age_product_difference_l2111_211127

theorem age_product_difference 
  (age_today : ℕ) 
  (Arnold_age : age_today = 6) 
  (Danny_age : age_today = 6) : 
  (7 * 7) - (6 * 6) = 13 := 
by
  sorry

end age_product_difference_l2111_211127


namespace ellipse_general_equation_l2111_211179

theorem ellipse_general_equation (x y : ℝ) (α : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) :
  x^2 / 25 + y^2 / 9 = 1 :=
sorry

end ellipse_general_equation_l2111_211179


namespace scientific_notation_of_213_million_l2111_211167

theorem scientific_notation_of_213_million : ∃ (n : ℝ), (213000000 : ℝ) = 2.13 * 10^8 :=
by
  sorry

end scientific_notation_of_213_million_l2111_211167


namespace meaningful_expression_l2111_211122

theorem meaningful_expression (x : ℝ) : (∃ y, y = 5 / (Real.sqrt (x + 1))) ↔ x > -1 :=
by
  sorry

end meaningful_expression_l2111_211122


namespace find_a2_a3_sequence_constant_general_formula_l2111_211152

-- Definition of the sequence and its sum Sn
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
axiom a1_eq : a 1 = 2
axiom S_eq : ∀ n, S (n + 1) = 4 * a n - 2

-- Prove that a_2 = 4 and a_3 = 8
theorem find_a2_a3 : a 2 = 4 ∧ a 3 = 8 :=
sorry

-- Prove that the sequence {a_n - 2a_{n-1}} is constant
theorem sequence_constant {n : ℕ} (hn : n ≥ 2) :
  ∃ c, ∀ k ≥ 2, a k - 2 * a (k - 1) = c :=
sorry

-- Find the general formula for the sequence
theorem general_formula :
  ∀ n, a n = 2^n :=
sorry

end find_a2_a3_sequence_constant_general_formula_l2111_211152


namespace interest_after_4_years_l2111_211128
-- Importing the necessary library

-- Definitions based on the conditions
def initial_amount : ℝ := 1500
def annual_interest_rate : ℝ := 0.12
def number_of_years : ℕ := 4

-- Calculating the total amount after 4 years using compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Calculating the interest earned
def interest_earned : ℝ :=
  compound_interest initial_amount annual_interest_rate number_of_years - initial_amount

-- The Lean statement to prove the interest earned is $859.25
theorem interest_after_4_years : interest_earned = 859.25 :=
by
  sorry

end interest_after_4_years_l2111_211128


namespace Ben_ate_25_percent_of_cake_l2111_211136

theorem Ben_ate_25_percent_of_cake (R B : ℕ) (h_ratio : R / B = 3 / 1) : B / (R + B) * 100 = 25 := by
  sorry

end Ben_ate_25_percent_of_cake_l2111_211136


namespace factorize_m_sq_minus_one_l2111_211169

theorem factorize_m_sq_minus_one (m : ℝ) : m^2 - 1 = (m + 1) * (m - 1) := 
by
  sorry

end factorize_m_sq_minus_one_l2111_211169


namespace socks_ratio_l2111_211184

theorem socks_ratio 
  (g : ℕ) -- number of pairs of green socks
  (y : ℝ) -- price per pair of green socks
  (h1 : y > 0) -- price per pair of green socks is positive
  (h2 : 3 * g * y + 3 * y = 1.2 * (9 * y + g * y)) -- swapping resulted in a 20% increase in the bill
  : 3 / g = 3 / 4 :=
by sorry

end socks_ratio_l2111_211184


namespace union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l2111_211161

def U : Set ℕ := { x | 1 ≤ x ∧ x < 9 }
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}
def C (S : Set ℕ) : Set ℕ := U \ S

theorem union_A_B : A ∪ B = {1, 2, 3, 4, 5, 6} := 
by {
  -- proof here
  sorry
}

theorem inter_A_B : A ∩ B = {3} := 
by {
  -- proof here
  sorry
}

theorem C_U_union_A_B : C (A ∪ B) = {7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_inter_A_B : C (A ∩ B) = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_A : C A = {4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem C_U_B : C B = {1, 2, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem union_C_U_A_C_U_B : C A ∪ C B = {1, 2, 4, 5, 6, 7, 8} := 
by {
  -- proof here
  sorry
}

theorem inter_C_U_A_C_U_B : C A ∩ C B = {7, 8} := 
by {
  -- proof here
  sorry
}

end union_A_B_inter_A_B_C_U_union_A_B_C_U_inter_A_B_C_U_A_C_U_B_union_C_U_A_C_U_B_inter_C_U_A_C_U_B_l2111_211161


namespace slope_of_tangent_line_l2111_211174

theorem slope_of_tangent_line 
  (center point : ℝ × ℝ) 
  (h_center : center = (5, 3)) 
  (h_point : point = (8, 8)) 
  : (∃ m : ℚ, m = -3/5) :=
sorry

end slope_of_tangent_line_l2111_211174


namespace total_tax_in_cents_l2111_211147

-- Declare the main variables and constants
def wage_per_hour_cents : ℕ := 2500
def local_tax_rate : ℝ := 0.02
def state_tax_rate : ℝ := 0.005

-- Define the total tax calculation as a proof statement
theorem total_tax_in_cents :
  local_tax_rate * wage_per_hour_cents + state_tax_rate * wage_per_hour_cents = 62.5 :=
by sorry

end total_tax_in_cents_l2111_211147


namespace isosceles_triangle_perimeter_l2111_211145

theorem isosceles_triangle_perimeter (a b c : ℕ) 
  (h1 : (a = 2 ∧ b = 4 ∧ c = 4) ∨ (a = 4 ∧ b = 2 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 2)) 
  (h2 : a + b > c ∧ a + c > b ∧ b + c > a) : a + b + c = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l2111_211145


namespace find_principal_amount_l2111_211175

theorem find_principal_amount :
  ∃ P : ℝ, P * (1 + 0.05) ^ 4 = 9724.05 ∧ P = 8000 :=
by
  sorry

end find_principal_amount_l2111_211175


namespace quadratic_coeff_sum_l2111_211124

theorem quadratic_coeff_sum {a b c : ℝ} (h1 : ∀ x, a * x^2 + b * x + c = a * (x - 1) * (x - 5))
    (h2 : a * 3^2 + b * 3 + c = 36) : a + b + c = 0 :=
by
  sorry

end quadratic_coeff_sum_l2111_211124


namespace evaluate_expression_l2111_211148

variable {x y : ℝ}

theorem evaluate_expression (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y ^ 2) :
  (x - 1 / x ^ 2) * (y + 2 / y) = 2 * x ^ (5 / 2) - 1 / x := 
by
  sorry

end evaluate_expression_l2111_211148


namespace cara_optimal_reroll_two_dice_probability_l2111_211150

def probability_reroll_two_dice : ℚ :=
  -- Probability derived from Cara's optimal reroll decisions
  5 / 27

theorem cara_optimal_reroll_two_dice_probability :
  cara_probability_optimal_reroll_two_dice = 5 / 27 := by sorry

end cara_optimal_reroll_two_dice_probability_l2111_211150


namespace square_diagonal_l2111_211144

theorem square_diagonal (P : ℝ) (d : ℝ) (hP : P = 200 * Real.sqrt 2) :
  d = 100 :=
by
  sorry

end square_diagonal_l2111_211144


namespace n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l2111_211168

theorem n_to_power_eight_plus_n_to_power_seven_plus_one_prime (n : ℕ) (hn_pos : n > 0) :
  (Nat.Prime (n^8 + n^7 + 1)) → (n = 1) :=
by
  sorry

end n_to_power_eight_plus_n_to_power_seven_plus_one_prime_l2111_211168


namespace find_monotonic_intervals_max_min_on_interval_l2111_211178

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

noncomputable def f' (x : ℝ) : ℝ := (Real.cos x - Real.sin x) * Real.exp x - 1

theorem find_monotonic_intervals (k : ℤ) : 
  ((2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi) → 0 < (f' x)) ∧
  ((2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi) → (f' x) < 0) :=
sorry

theorem max_min_on_interval : 
  (∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → f 0 = 1 ∧ f (2 * Real.pi / 3) =  -((1/2) * Real.exp (2/3 * Real.pi)) - (2 * Real.pi / 3)) :=
sorry

end find_monotonic_intervals_max_min_on_interval_l2111_211178


namespace partI_inequality_partII_inequality_l2111_211119

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Part (Ⅰ): Prove f(x) ≤ x + 1 for 1 ≤ x ≤ 5
theorem partI_inequality (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 5) : f x ≤ x + 1 := by
  sorry

-- Part (Ⅱ): Prove (a^2)/(a+1) + (b^2)/(b+1) ≥ 1 when a + b = 2 and a > 0, b > 0
theorem partII_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) : 
    (a^2) / (a + 1) + (b^2) / (b + 1) ≥ 1 := by
  sorry

end partI_inequality_partII_inequality_l2111_211119


namespace angle_value_l2111_211162

theorem angle_value (x : ℝ) (h₁ : (90 : ℝ) = 44 + x) : x = 46 :=
by
  sorry

end angle_value_l2111_211162


namespace solve_equation1_solve_equation2_l2111_211130

-- Statement for the first equation: x^2 - 16 = 0
theorem solve_equation1 (x : ℝ) : x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 :=
by sorry

-- Statement for the second equation: (x + 10)^3 + 27 = 0
theorem solve_equation2 (x : ℝ) : (x + 10)^3 + 27 = 0 ↔ x = -13 :=
by sorry

end solve_equation1_solve_equation2_l2111_211130


namespace max_sum_abc_l2111_211180

theorem max_sum_abc (a b c : ℝ) (h1 : 1 ≤ a) (h2 : 1 ≤ b) (h3 : 1 ≤ c) 
  (h4 : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end max_sum_abc_l2111_211180


namespace real_solutions_count_is_two_l2111_211111

def equation_has_two_real_solutions (a b c : ℝ) : Prop :=
  (3*a^2 - 8*b + 2 = c) → (∀ x : ℝ, 3*x^2 - 8*x + 2 = 0) → ∃! x₁ x₂ : ℝ, (3*x₁^2 - 8*x₁ + 2 = 0) ∧ (3*x₂^2 - 8*x₂ + 2 = 0)

theorem real_solutions_count_is_two : equation_has_two_real_solutions (3 : ℝ) (-8 : ℝ) (2 : ℝ) := by
  sorry

end real_solutions_count_is_two_l2111_211111


namespace maximum_value_l2111_211173

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem maximum_value : ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ≤ f 1 :=
by
  intros x hx
  sorry

end maximum_value_l2111_211173


namespace true_and_false_propositions_l2111_211159

theorem true_and_false_propositions (p q : Prop) 
  (hp : p = true) (hq : q = false) : (¬q) = true :=
by
  sorry

end true_and_false_propositions_l2111_211159


namespace tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l2111_211171

theorem tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m (m : ℝ) (h : Real.cos (80 * Real.pi / 180) = m) :
    Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2) / m) :=
by
  -- proof goes here
  sorry

end tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l2111_211171


namespace team_members_count_l2111_211104

theorem team_members_count (x : ℕ) (h1 : 3 * x + 2 * x = 33 ∨ 4 * x + 2 * x = 33) : x = 6 := by
  sorry

end team_members_count_l2111_211104


namespace marching_band_members_l2111_211139

theorem marching_band_members (B W P : ℕ) (h1 : P = 4 * W) (h2 : W = 2 * B) (h3 : B = 10) : B + W + P = 110 :=
by
  sorry

end marching_band_members_l2111_211139


namespace angle_in_triangle_l2111_211192

theorem angle_in_triangle (A B C x : ℝ) (hA : A = 40)
    (hB : B = 3 * x) (hC : C = x) (h_sum : A + B + C = 180) : x = 35 :=
by
  sorry

end angle_in_triangle_l2111_211192


namespace matchstick_problem_l2111_211165

theorem matchstick_problem (n : ℕ) (T : ℕ → ℕ) :
  (∀ n, T n = 4 + 9 * (n - 1)) ∧ n = 15 → T n = 151 :=
by
  sorry

end matchstick_problem_l2111_211165


namespace CALI_area_is_180_l2111_211141

-- all the conditions used in Lean definitions
def is_square (s : ℕ) : Prop := (s > 0)

def are_midpoints (T O W N B E R K : ℕ) : Prop := 
  (T = (B + E) / 2) ∧ (O = (E + R) / 2) ∧ (W = (R + K) / 2) ∧ (N = (K + B) / 2)

def is_parallel (CA BO : ℕ) : Prop :=
  CA = BO 

-- the condition indicates the length of each side of the square BERK is 10
def side_length_of_BERK : ℕ := 10

-- definition of lengths and condition
def BERK_lengths (BERK_side_length : ℕ) (BERK_diag_length : ℕ): Prop :=
  BERK_side_length = side_length_of_BERK ∧ BERK_diag_length = BERK_side_length * (2^(1/2))

def CALI_area_of_length (length: ℕ): ℕ := length^2

theorem CALI_area_is_180 
(BERK_side_length BERK_diag_length : ℕ)
(CALI_length : ℕ)
(T O W N B E R K CA BO : ℕ)
(h1 : is_square BERK_side_length)
(h2 : are_midpoints T O W N B E R K)
(h3 : is_parallel CA BO)
(h4 : BERK_lengths BERK_side_length BERK_diag_length)
(h5 : CA = CA)
: CALI_area_of_length 15 = 180 :=
sorry

end CALI_area_is_180_l2111_211141


namespace pet_store_cages_l2111_211170

-- Definitions and conditions
def initial_puppies : ℕ := 56
def sold_puppies : ℕ := 24
def puppies_per_cage : ℕ := 4
def remaining_puppies : ℕ := initial_puppies - sold_puppies
def cages_used : ℕ := remaining_puppies / puppies_per_cage

-- Theorem statement
theorem pet_store_cages : cages_used = 8 := by sorry

end pet_store_cages_l2111_211170


namespace dickens_birth_day_l2111_211112

def is_leap_year (year : ℕ) : Prop :=
  (year % 400 = 0) ∨ (year % 4 = 0 ∧ year % 100 ≠ 0)

theorem dickens_birth_day :
  let day_of_week_2012 := 2 -- 0: Sunday, 1: Monday, ..., 2: Tuesday
  let years := 200
  let regular_years := 151
  let leap_years := 49
  let days_shift := regular_years + 2 * leap_years
  let day_of_week_birth := (day_of_week_2012 + days_shift) % 7
  day_of_week_birth = 5 -- 5: Friday
:= 
sorry -- proof not supplied

end dickens_birth_day_l2111_211112


namespace train_speed_is_72_l2111_211195

def distance : ℕ := 24
def time_minutes : ℕ := 20
def time_hours : ℚ := time_minutes / 60
def speed := distance / time_hours

theorem train_speed_is_72 :
  speed = 72 := by
  sorry

end train_speed_is_72_l2111_211195


namespace intersection_point_at_neg4_l2111_211142

def f (x : Int) (b : Int) : Int := 4 * x + b
def f_inv (y : Int) (b : Int) : Int := (y - b) / 4

theorem intersection_point_at_neg4 (a b : Int) (h1 : f (-4) b = a) (h2 : f_inv (-4) b = a) : a = -4 := 
by 
  sorry

end intersection_point_at_neg4_l2111_211142


namespace common_difference_is_4_l2111_211132

variable (a : ℕ → ℤ) (d : ℤ)

-- Conditions of the problem
def arithmetic_sequence := ∀ n m : ℕ, a n = a m + (n - m) * d

axiom a7_eq_25 : a 7 = 25
axiom a4_eq_13 : a 4 = 13

-- The theorem to prove
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end common_difference_is_4_l2111_211132


namespace find_positive_integer_solutions_l2111_211138

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2^x + 3^y = z^2 ↔ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 4 ∧ y = 2 ∧ z = 5) := 
sorry

end find_positive_integer_solutions_l2111_211138


namespace root_in_interval_l2111_211198

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 2

theorem root_in_interval : ∃ x ∈ Set.Ioo (3 : ℝ) (4 : ℝ), f x = 0 := sorry

end root_in_interval_l2111_211198


namespace train_cross_pole_time_l2111_211172

noncomputable def L_train : ℝ := 300 -- Length of the train in meters
noncomputable def L_platform : ℝ := 870 -- Length of the platform in meters
noncomputable def t_platform : ℝ := 39 -- Time to cross the platform in seconds

theorem train_cross_pole_time
  (L_train : ℝ)
  (L_platform : ℝ)
  (t_platform : ℝ)
  (D : ℝ := L_train + L_platform)
  (v : ℝ := D / t_platform)
  (t_pole : ℝ := L_train / v) :
  t_pole = 10 :=
by sorry

end train_cross_pole_time_l2111_211172


namespace quadrilateral_centroid_perimeter_l2111_211116

-- Definition for the side length of the square and distances for points Q
def side_length : ℝ := 40
def EQ_dist : ℝ := 18
def FQ_dist : ℝ := 34

-- Theorem statement: Perimeter of the quadrilateral formed by centroids
theorem quadrilateral_centroid_perimeter :
  let centroid_perimeter := (4 * ((2 / 3) * side_length))
  centroid_perimeter = (320 / 3) := by
  sorry

end quadrilateral_centroid_perimeter_l2111_211116


namespace find_milk_ounces_l2111_211189

def bathroom_limit : ℕ := 32
def grape_juice_ounces : ℕ := 16
def water_ounces : ℕ := 8
def total_liquid_limit : ℕ := bathroom_limit
def total_liquid_intake : ℕ := grape_juice_ounces + water_ounces
def milk_ounces := total_liquid_limit - total_liquid_intake

theorem find_milk_ounces : milk_ounces = 8 := by
  sorry

end find_milk_ounces_l2111_211189


namespace area_of_square_B_l2111_211106

theorem area_of_square_B (c : ℝ) (hA : ∃ sA, sA * sA = 2 * c^2) (hB : ∃ sA, exists sB, sB * sB = 3 * (sA * sA)) : 
∃ sB, sB * sB = 6 * c^2 :=
by
  sorry

end area_of_square_B_l2111_211106


namespace sequence_b_n_l2111_211154

theorem sequence_b_n (b : ℕ → ℝ) 
  (h1 : b 1 = 3)
  (h2 : ∀ n ≥ 1, (b (n + 1))^3 = 27 * (b n)^3) :
  b 50 = 3^50 :=
sorry

end sequence_b_n_l2111_211154


namespace a5_eq_neg3_l2111_211185

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sequence with given conditions
def a (n : ℕ) : ℤ :=
  if n = 2 then -5
  else if n = 8 then 1
  else sorry  -- Placeholder for other values

axiom a3_eq_neg5 : a 2 = -5
axiom a9_eq_1 : a 8 = 1
axiom a_is_arithmetic : is_arithmetic_sequence a

-- Statement to prove
theorem a5_eq_neg3 : a 4 = -3 :=
by
  sorry

end a5_eq_neg3_l2111_211185


namespace triangle_PQR_area_l2111_211108

/-

Define the points P, Q, and R.
Define a function to calculate the area of a triangle given three points.
Then write a theorem to state that the area of triangle PQR is 12.

-/

structure Point where
  x : ℕ
  y : ℕ

def P : Point := ⟨2, 6⟩
def Q : Point := ⟨2, 2⟩
def R : Point := ⟨8, 5⟩

def area (A B C : Point) : ℚ :=
  abs ((A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2)

theorem triangle_PQR_area : area P Q R = 12 := by
  /- 
    The proof should involve calculating the area using the given points.
   -/
  sorry

end triangle_PQR_area_l2111_211108


namespace marcus_brought_30_peanut_butter_cookies_l2111_211186

/-- Jenny brought in 40 peanut butter cookies. -/
def jenny_peanut_butter_cookies := 40

/-- Jenny brought in 50 chocolate chip cookies. -/
def jenny_chocolate_chip_cookies := 50

/-- Marcus brought in 20 lemon cookies. -/
def marcus_lemon_cookies := 20

/-- The total number of non-peanut butter cookies is the sum of chocolate chip and lemon cookies. -/
def non_peanut_butter_cookies := jenny_chocolate_chip_cookies + marcus_lemon_cookies

/-- The total number of peanut butter cookies is Jenny's plus Marcus'. -/
def total_peanut_butter_cookies (marcus_peanut_butter_cookies : ℕ) := jenny_peanut_butter_cookies + marcus_peanut_butter_cookies

/-- If Renee has a 50% chance of picking a peanut butter cookie, the number of peanut butter cookies must equal the number of non-peanut butter cookies. -/
theorem marcus_brought_30_peanut_butter_cookies (x : ℕ) : total_peanut_butter_cookies x = non_peanut_butter_cookies → x = 30 :=
by
  sorry

end marcus_brought_30_peanut_butter_cookies_l2111_211186


namespace hens_ratio_l2111_211156

theorem hens_ratio
  (total_chickens : ℕ)
  (fraction_roosters : ℚ)
  (chickens_not_laying : ℕ)
  (h : total_chickens = 80)
  (fr : fraction_roosters = 1/4)
  (cnl : chickens_not_laying = 35) :
  (total_chickens * (1 - fraction_roosters) - chickens_not_laying) / (total_chickens * (1 - fraction_roosters)) = 5 / 12 :=
by
  sorry

end hens_ratio_l2111_211156


namespace fido_reachable_area_l2111_211191

theorem fido_reachable_area (r : ℝ) (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0)
  (h_leash : ∃ (r : ℝ), r > 0) (h_fraction : (a : ℝ) / b * π = π) : a * b = 1 :=
by
  sorry

end fido_reachable_area_l2111_211191


namespace not_possible_to_partition_into_groups_of_5_with_remainder_3_l2111_211182

theorem not_possible_to_partition_into_groups_of_5_with_remainder_3 (m : ℤ) :
  ¬ (m^2 % 5 = 3) :=
by sorry

end not_possible_to_partition_into_groups_of_5_with_remainder_3_l2111_211182


namespace isolate_y_l2111_211160

theorem isolate_y (x y : ℝ) (h : 3 * x - 2 * y = 6) : y = 3 * x / 2 - 3 :=
sorry

end isolate_y_l2111_211160


namespace range_of_a_l2111_211120

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 :=
sorry

end range_of_a_l2111_211120


namespace smallest_fraction_numerator_l2111_211197

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), a ≥ 10 ∧ a ≤ 99 ∧ b ≥ 10 ∧ b ≤ 99 ∧ (4 * b < 9 * a) ∧ 
  (∀ (a' b' : ℕ), a' ≥ 10 ∧ a' ≤ 99 ∧ b' ≥ 10 ∧ b' ≤ 99 ∧ (4 * b' < 9 * a') → b * a' ≥ a * b') ∧ a = 41 :=
sorry

end smallest_fraction_numerator_l2111_211197


namespace contradiction_proof_l2111_211155

theorem contradiction_proof (a b : ℝ) : a + b = 12 → ¬ (a < 6 ∧ b < 6) :=
by
  intro h
  intro h_contra
  sorry

end contradiction_proof_l2111_211155


namespace committee_count_is_correct_l2111_211137

-- Definitions of the problem conditions
def total_people : ℕ := 10
def committee_size : ℕ := 5
def remaining_people := total_people - 1
def members_to_choose := committee_size - 1

-- The combinatorial function for selecting committee members
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def number_of_ways_to_form_committee : ℕ :=
  binomial remaining_people members_to_choose

-- Statement of the problem to prove the number of ways is 126
theorem committee_count_is_correct :
  number_of_ways_to_form_committee = 126 :=
by
  sorry

end committee_count_is_correct_l2111_211137


namespace rectangle_height_l2111_211100

theorem rectangle_height (y : ℝ) (h_pos : 0 < y) 
  (h_area : let length := 5 - (-3)
            let height := y - (-2)
            length * height = 112) : y = 12 := 
by 
  -- The proof is omitted
  sorry

end rectangle_height_l2111_211100


namespace large_buckets_needed_l2111_211149

def capacity_large_bucket (S: ℚ) : ℚ := 2 * S + 3

theorem large_buckets_needed (n : ℕ) (L S : ℚ) (h1 : L = capacity_large_bucket S) (h2 : L = 4) (h3 : 2 * S + n * L = 63)
: n = 16 := sorry

end large_buckets_needed_l2111_211149


namespace circle_tangent_radius_l2111_211101

theorem circle_tangent_radius (k : ℝ) (r : ℝ) (hk : k > 4) 
  (h_tangent1 : dist (0, k) (x, x) = r)
  (h_tangent2 : dist (0, k) (x, -x) = r) 
  (h_tangent3 : dist (0, k) (x, 4) = r) : 
  r = 4 * Real.sqrt 2 := 
sorry

end circle_tangent_radius_l2111_211101


namespace part_a_part_b_l2111_211183

noncomputable def withdraw_rubles_after_one_year
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) : ℚ :=
  let deposit_in_dollars := initial_deposit / initial_rate
  let interest_earned := deposit_in_dollars * annual_yield
  let total_in_dollars := deposit_in_dollars + interest_earned
  let broker_fee := interest_earned * broker_commission
  let amount_after_fee := total_in_dollars - broker_fee
  let total_in_rubles := amount_after_fee * final_rate
  let conversion_fee := total_in_rubles * conversion_commission
  total_in_rubles - conversion_fee

theorem part_a
  (initial_deposit : ℤ) (initial_rate : ℤ) (annual_yield : ℚ)
  (final_rate : ℤ) (conversion_commission : ℚ) (broker_commission : ℚ) :
  withdraw_rubles_after_one_year initial_deposit initial_rate annual_yield final_rate conversion_commission broker_commission =
  16476.8 := sorry

def effective_yield (initial_rubles final_rubles : ℚ) : ℚ :=
  (final_rubles / initial_rubles - 1) * 100

theorem part_b
  (initial_deposit : ℤ) (final_rubles : ℚ) :
  effective_yield initial_deposit final_rubles = 64.77 := sorry

end part_a_part_b_l2111_211183


namespace bc_fraction_ad_l2111_211126

theorem bc_fraction_ad
  (B C E A D : Type)
  (on_AD : ∀ P : Type, P = B ∨ P = C ∨ P = E)
  (AB BD AC CD DE EA: ℝ)
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : DE = 2 * EA)

  : ∃ BC AD: ℝ, BC = 1 / 12 * AD := 
sorry -- Proof is omitted

end bc_fraction_ad_l2111_211126


namespace find_k_inv_h_of_10_l2111_211140

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10_l2111_211140


namespace area_of_inscribed_rectangle_l2111_211188

theorem area_of_inscribed_rectangle 
    (DA : ℝ) 
    (GD HD : ℝ) 
    (rectangle_inscribed : ∀ (A B C D G H : Type), true) 
    (radius : ℝ) 
    (GH : ℝ):
    DA = 20 ∧ GD = 5 ∧ HD = 5 ∧ GH = GD + DA + HD ∧ radius = GH / 2 → 
    200 * Real.sqrt 2 = DA * (Real.sqrt (radius^2 - (GD^2))) :=
by
  sorry

end area_of_inscribed_rectangle_l2111_211188


namespace revenue_increase_l2111_211110

theorem revenue_increase (P Q : ℝ) :
    let R := P * Q
    let P_new := 1.7 * P
    let Q_new := 0.8 * Q
    let R_new := P_new * Q_new
    R_new = 1.36 * R :=
sorry

end revenue_increase_l2111_211110


namespace intersection_points_on_circle_l2111_211190

theorem intersection_points_on_circle (u : ℝ) :
  ∃ (r : ℝ), ∀ (x y : ℝ), (u * x - 3 * y - 2 * u = 0) ∧ (2 * x - 3 * u * y + u = 0) → (x^2 + y^2 = r^2) :=
sorry

end intersection_points_on_circle_l2111_211190


namespace find_n_divisible_by_6_l2111_211194

theorem find_n_divisible_by_6 (n : Nat) : (71230 + n) % 6 = 0 ↔ n = 2 ∨ n = 8 := by
  sorry

end find_n_divisible_by_6_l2111_211194


namespace problem_solution_l2111_211118

-- Define the problem
noncomputable def a_b_sum : ℝ := 
  let a := 5
  let b := 3
  a + b

-- Theorem statement
theorem problem_solution (a b i : ℝ) (h1 : a + b * i = (11 - 7 * i) / (1 - 2 * i)) (hi : i * i = -1) :
  a + b = 8 :=
by sorry

end problem_solution_l2111_211118


namespace solve_for_n_l2111_211143

theorem solve_for_n (n : ℕ) (h : 9^n * 9^n * 9^n * 9^n = 81^n) : n = 0 :=
by
  sorry

end solve_for_n_l2111_211143


namespace digit_makes_divisible_by_nine_l2111_211157

theorem digit_makes_divisible_by_nine (A : ℕ) : (7 + A + 4 + 6) % 9 = 0 ↔ A = 1 :=
by
  sorry

end digit_makes_divisible_by_nine_l2111_211157


namespace increase_speed_to_pass_correctly_l2111_211115

theorem increase_speed_to_pass_correctly
  (x a : ℝ)
  (ha1 : 50 < a)
  (hx1 : (a - 40) * x = 30)
  (hx2 : (a + 50) * x = 210) :
  a - 50 = 5 :=
by
  sorry

end increase_speed_to_pass_correctly_l2111_211115


namespace fresh_grapes_weight_eq_l2111_211193

-- Definitions of the conditions from a)
def fresh_grapes_water_percent : ℝ := 0.80
def dried_grapes_water_percent : ℝ := 0.20
def dried_grapes_weight : ℝ := 10
def fresh_grapes_non_water_percent : ℝ := 1 - fresh_grapes_water_percent
def dried_grapes_non_water_percent : ℝ := 1 - dried_grapes_water_percent

-- Proving the weight of fresh grapes
theorem fresh_grapes_weight_eq :
  let F := (dried_grapes_non_water_percent * dried_grapes_weight) / fresh_grapes_non_water_percent
  F = 40 := by
  -- The proof has been omitted
  sorry

end fresh_grapes_weight_eq_l2111_211193


namespace time_after_9876_seconds_l2111_211163

-- Define the initial time in seconds
def initial_seconds : ℕ := 6 * 3600

-- Define the elapsed time in seconds
def elapsed_seconds : ℕ := 9876

-- Convert given time in seconds to hours, minutes, and seconds
def time_in_hms (total_seconds : ℕ) : (ℕ × ℕ × ℕ) :=
  let hours := total_seconds / 3600
  let minutes := (total_seconds % 3600) / 60
  let seconds := total_seconds % 60
  (hours, minutes, seconds)

-- Define the final time in 24-hour format (08:44:36)
def final_time : (ℕ × ℕ × ℕ) := (8, 44, 36)

-- The question's proof statement
theorem time_after_9876_seconds : 
  time_in_hms (initial_seconds + elapsed_seconds) = final_time :=
sorry

end time_after_9876_seconds_l2111_211163
